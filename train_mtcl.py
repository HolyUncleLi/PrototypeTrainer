# --- train_mtcl.py ---

import os, sys
import json
import argparse
import warnings
import numpy as np
import sklearn.metrics as skmet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from models.protop_cross import ProtoPNet
import torch.nn.functional as F

CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        # *** 重要提示 ***
        # 确保您的 EEGDataLoader 对每个输入的epoch(长度30000的信号)进行了归一化。
        # 例如，Z-score标准化: new_signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        # 这是提升模型性能至关重要的一步。

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        class_weight = torch.tensor(CLASS_WEIGHT).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'],
                                    weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_path = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

        # *** 关键改动: 调整损失权重 ***
        self.lambdas = {
            'cls': self.cfg['classifier']['class_lambda'],  # 保持为 1.0 或配置文件中的值
            'dist': 0.1,  # 从 1.0 降低，鼓励模型优先学习分类
            'identity': 0.05,  # 从 1.0 降低，作为次要的正则化项
            'freq': 0.1,  # 频率引导损失的权重
            'gabor_l1': 1e-4,  # Gabor基础原型振幅的L1稀疏权重
            'fourier_l1': 1e-4,  # Fourier基础原型振幅的L1稀疏权重
            'mix_l1': 1e-3  # 混合权重的L1稀疏权重
        }
        print(f"[INFO] Using loss lambdas: {self.lambdas}")

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        if len(self.args.gpu.split(",")) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True,
                                  num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def activate_train_mode(self):
        self.model.train()

    def compute_v2_loss(self, outputs, labels):
        self.loss_ensemble = {}

        # 1. 分类损失 (Cross-Entropy)
        cross_entropy = self.criterion(outputs, labels)
        self.loss_ensemble['cross_entropy'] = self.lambdas['cls'] * cross_entropy

        # 2. ProtoPNet 距离损失
        min_dist = self.model.module.min_distance if isinstance(self.model,
                                                                nn.DataParallel) else self.model.min_distance
        dist_loss = torch.mean(torch.min(min_dist, dim=1).values)
        self.loss_ensemble['dist_loss'] = self.lambdas['dist'] * dist_loss
        identity_loss = torch.mean(torch.min(min_dist, dim=0).values)
        self.loss_ensemble['identity_loss'] = self.lambdas['identity'] * identity_loss

        # 3. 频率引导损失
        gabor_targets = torch.cat([
            14 + 2 * torch.rand(5), 2 + 2 * torch.rand(5),
            10 + 2 * torch.rand(5), 25 + 5 * torch.rand(5)
        ]).to(self.device)
        fourier_targets = torch.cat([
            2 + 2 * torch.rand(10), 20 + 10 * torch.rand(10)
        ]).to(self.device)

        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        l_freq_gabor = F.mse_loss(model_module.gabor_basis_bank.f, gabor_targets)
        l_freq_fourier = F.mse_loss(model_module.fourier_basis_bank.f, fourier_targets)
        self.loss_ensemble['gabor_loss'] = self.lambdas['freq'] * l_freq_gabor
        self.loss_ensemble['Fourier_loss'] = self.lambdas['freq'] * l_freq_fourier

        # 4. 稀疏性损失
        l_gabor_l1 = torch.norm(model_module.gabor_basis_bank.A, p=1)
        l_fourier_l1 = torch.norm(model_module.fourier_basis_bank.A, p=1)
        self.loss_ensemble['Gabor_l1'] = self.lambdas['gabor_l1'] * l_gabor_l1
        self.loss_ensemble['Fourier_l1'] = self.lambdas['fourier_l1'] * l_fourier_l1
        mix_l1_loss = torch.norm(model_module.mixing_weights, p=1)
        self.loss_ensemble['weight_loss'] = self.lambdas['mix_l1'] * mix_l1_loss

        total_loss = sum(self.loss_ensemble.values())
        return total_loss

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0
        self.model.train()

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            loss = self.compute_v2_loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            # 使用 | 分隔符打印所有 loss
            progress_bar(i, len(self.loader_dict['train']),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | cls: %.3f | dist: %.3f | id: %.3f | gabor_f: %.3f | fourier_f: %.3f | G_l1: %.3f | F_l1: %.3f | mix_l1: %.3f'
                         % (train_loss / (i + 1), 100. * correct / total, correct, total,
                            self.loss_ensemble['cross_entropy'].item(),
                            self.loss_ensemble['dist_loss'].item(),
                            self.loss_ensemble['identity_loss'].item(),
                            self.loss_ensemble['gabor_loss'].item(),
                            self.loss_ensemble['Fourier_loss'].item(),
                            self.loss_ensemble['Gabor_l1'].item(),
                            self.loss_ensemble['Fourier_l1'].item(),
                            self.loss_ensemble['weight_loss'].item()))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss, val_mf1 = self.evaluate(mode='val')
                self.early_stopping(val_mf1, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            loss = self.compute_v2_loss(outputs, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), f'Evaluating {mode} set...')

        y_pred_argmax = np.argmax(y_pred, 1)
        result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True, zero_division=0)
        mf1 = round(result_dict['macro avg']['f1-score'] * 100, 2)
        accuracy = round(100. * correct / total, 2)

        print(
            f'\n{mode.capitalize()} Results | Acc: {accuracy}% ({correct}/{total}) | MF1: {mf1} | Loss: {eval_loss / len(self.loader_dict[mode]):.4f}')

        if mode == 'val':
            return 100. * correct / total, eval_loss, mf1
        elif mode == 'test':
            return y_true, y_pred, mf1
        else:
            raise NotImplementedError

    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                print("[INFO] Early stopping triggered.")
                break

        print("[INFO] Loading best model for final evaluation...")
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred, mf1 = self.evaluate(mode='test')
        print('')
        return y_true, y_pred


def main():
    # (main 函数保持不变)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)
        break


if __name__ == "__main__":
    main()