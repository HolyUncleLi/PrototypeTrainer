import os, sys
import json
import argparse
import warnings
import numpy as np  # 确保导入numpy
import sklearn.metrics as skmet  # 确保导入sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
# from models.protop_fusion import ProtoPNet
from models.protop_cross import ProtoPNet
import torch.nn.functional as F

CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config['feature_pyramid']
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        class_weight = torch.tensor(CLASS_WEIGHT).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)  # 原代码漏了 weight=
        self.activate_train_mode()
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'],
                                    weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_path = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path,
                                            ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])

        # NEW: 为新模型损失函数定义超参数
        self.lambdas = {
            'cls': self.cfg['classifier']['class_lambda'],
            'dist': self.cfg['classifier']['dist_lambda'],
            'identity': self.cfg['classifier']['identity_lambda'],
            'freq': 0.1,  # 频率引导损失的权重
            'gabor_l1': 1e-4,  # Gabor基础原型振幅的L1稀疏权重
            'fourier_l1': 1e-4,  # Fourier基础原型振幅的L1稀疏权重
            'mix_l1': 1e-3  # 混合权重的L1稀疏权重 (代替了旧的weight_loss)
        }

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        # ... (加载预训练模型的逻辑保持不变) ...
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

    # ====================================================================
    # NEW: 为 ProtoPNetV2 设计的全新损失函数
    # ====================================================================
    def compute_v2_loss(self, outputs, labels):
        self.loss_ensemble = {}

        # 1. 分类损失 (Cross-Entropy)
        cross_entropy = self.criterion(outputs, labels)
        self.loss_ensemble['cross_entropy'] = self.lambdas['cls'] * cross_entropy

        # 2. ProtoPNet 距离损失 (保留了您原有的优秀设计)
        #    假设模型返回的是整体预测，min_distance 是 [B, P]
        min_dist = self.model.module.min_distance
        #    聚类损失：让每个样本靠近至少一个原型
        dist_loss = torch.mean(torch.min(min_dist, dim=1).values)
        self.loss_ensemble['dist_loss'] = self.lambdas['dist'] * dist_loss
        #    分离损失：让每个原型至少对一个样本是最近的
        identity_loss = torch.mean(torch.min(min_dist, dim=0).values)
        self.loss_ensemble['identity_loss'] = self.lambdas['identity'] * identity_loss

        # 3. 频率引导损失 (新模型的核心)
        #    为Gabor和Fourier基础原型定义目标频率 (Hz)
        gabor_targets = torch.cat([
            14 + 2 * torch.rand(5),  # 纺锤波 (Spindles)
            2 + 2 * torch.rand(5),  # 慢波 (Delta)
            10 + 2 * torch.rand(5),  # Alpha波
            25 + 5 * torch.rand(5)  # Beta/Gamma波
        ]).to(self.device)
        fourier_targets = torch.cat([
            2 + 2 * torch.rand(10),  # Delta/Theta波
            20 + 10 * torch.rand(10)  # Alpha/Beta波
        ]).to(self.device)

        # 计算学习到的频率与目标频率的差距
        l_freq_gabor = F.mse_loss(self.model.module.gabor_basis_bank.f, gabor_targets)
        l_freq_fourier = F.mse_loss(self.model.module.fourier_basis_bank.f, fourier_targets)
        self.loss_ensemble['gabor_loss'] = self.lambdas['freq'] * l_freq_gabor
        self.loss_ensemble['Fourier_loss'] = self.lambdas['freq'] * l_freq_fourier

        # 4. 稀疏性损失
        #    基础原型振幅稀疏：鼓励模型只使用少数几个基础原型
        l_gabor_l1 = torch.norm(self.model.module.gabor_basis_bank.A, p=1)
        l_fourier_l1 = torch.norm(self.model.module.fourier_basis_bank.A, p=1)
        self.loss_ensemble['Gabor_l1'] = self.lambdas['gabor_l1'] * l_gabor_l1
        self.loss_ensemble['Fourier_l1'] = self.lambdas['fourier_l1'] * l_fourier_l1

        #    混合权重稀疏：让每个复合原型由少数基础原型构成，增强可解释性
        #    我们用这项代替了原来对fc.weight的L1损失
        mix_l1_loss = torch.norm(self.model.module.mixing_weights, p=1)
        self.loss_ensemble['weight_loss'] = self.lambdas['mix_l1'] * mix_l1_loss

        # 计算总损失
        total_loss = (self.loss_ensemble['cross_entropy'] +
                      self.loss_ensemble['dist_loss'] +
                      self.loss_ensemble['identity_loss'] +
                      self.loss_ensemble['gabor_loss'] +
                      self.loss_ensemble['Fourier_loss'] +
                      self.loss_ensemble['Gabor_l1'] +
                      self.loss_ensemble['Fourier_l1'] +
                      self.loss_ensemble['weight_loss'])
        return total_loss

    # (旧的损失函数 protop_loss, protop_cam_loss, interpret_loss 可以安全地删除或注释掉)

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            # 假设是整体分类任务, labels 应该是 [B]
            labels = labels.view(-1).to(self.device)

            # MODIFIED: 简化模型调用和损失计算
            # 新模型总是返回一个单一的 logits 张量
            outputs = self.model(inputs)
            loss = self.compute_v2_loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            # MODIFIED: 确认 progress_bar 的键名与 loss_ensemble 中的一致
            progress_bar(i, len(self.loader_dict['train']),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | cls: %.3f |dist: %.3f |gabor_f: %.3f |Fourier_f: %.3f |G_l1: %.3f |F_l1: %.3f |id: %.3f |mix_l1: %.3f '
                         % (train_loss / (i + 1), 100. * correct / total, correct, total,
                            self.loss_ensemble['cross_entropy'],
                            self.loss_ensemble['dist_loss'],
                            self.loss_ensemble['gabor_loss'],
                            self.loss_ensemble['Fourier_loss'],
                            self.loss_ensemble['Gabor_l1'],
                            self.loss_ensemble['Fourier_l1'],
                            self.loss_ensemble['identity_loss'],
                            self.loss_ensemble['weight_loss']))  # 'weight_loss'现在是混合权重的L1

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss, val_mf1 = self.evaluate(mode='val')
                self.early_stopping(val_mf1, val_loss, self.model)
                self.activate_train_mode()  # 重新激活训练模式
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

            # MODIFIED: 简化模型调用和损失计算
            outputs = self.model(inputs)
            loss = self.compute_v2_loss(outputs, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            y_pred_argmax = np.argmax(y_pred, 1)
            result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True,
                                                      zero_division=0)
            mf1 = round(result_dict['macro avg']['f1-score'] * 100, 1)

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | MF1: %.3f'
                         % (eval_loss / (i + 1), 100. * correct / total, correct, total, mf1))

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
                break

        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred, mf1 = self.evaluate(mode='test')
        print('')
        return y_true, y_pred


def main():
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

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

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