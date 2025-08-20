# --- train_mtcl.py (Bug Fixed & Logic Corrected) ---

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
from models.protop_cross import ProtoPNet  # 确保这里的导入路径和类名正确
import torch.nn.functional as F

CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args;
        self.fold = fold;
        self.cfg = config
        self.ds_cfg = config['dataset'];
        self.tp_cfg = config['training_params'];
        self.es_cfg = self.tp_cfg['early_stopping']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))
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
        self.lambdas = {
            'cls': self.cfg['classifier']['class_lambda'], 'dist': 0.1, 'identity': 0.05,
            'gabor_spec': 0.01, 'fourier_spec': 0.01, 'orth': 0.1, 'mix_l1': 1e-3
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

    def compute_comprehensive_loss(self, outputs, labels):
        self.loss_ensemble = {}
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        cross_entropy = self.criterion(outputs, labels)
        self.loss_ensemble['cross_entropy'] = self.lambdas['cls'] * cross_entropy
        min_dist = model_module.min_distance
        num_prototypes_per_class = model_module.num_composite_prototypes // model_module.fc.out_features
        prototype_class_identity = torch.zeros(model_module.num_composite_prototypes, model_module.fc.out_features,
                                               device=self.device)
        for j in range(model_module.fc.out_features):
            prototype_class_identity[j * num_prototypes_per_class: (j + 1) * num_prototypes_per_class, j] = 1

        # ========== 【核心修正】 ==========
        # 1. 使用 labels 为批次中的每个样本动态选择正确的类别掩码
        # prototype_class_identity.T 的形状是 [5, 20]
        # labels 的形状是 [64]
        # class_mask 的形状将是 [64, 20]，这正是我们需要的 per-sample mask
        class_mask = prototype_class_identity.T[labels].to(self.device)

        # 2. 使用新的掩码计算同类和异类距离
        inverted_distances = -min_dist

        # 计算同类距离
        same_class_log_mask = torch.log(class_mask)  # 正确的为0, 错误为-inf
        same_class_distances = inverted_distances + same_class_log_mask
        max_same_class_dist = torch.max(same_class_distances, dim=1).values
        clst_loss = torch.mean(-max_same_class_dist)
        self.loss_ensemble['clst_loss'] = self.lambdas['dist'] * clst_loss

        # 计算异类距离
        diff_class_log_mask = torch.log(1 - class_mask)  # 错误为0, 正确为-inf
        diff_class_distances = inverted_distances + diff_class_log_mask
        max_diff_class_dist = torch.max(diff_class_distances, dim=1).values

        # 3. 修正分离损失的逻辑：我们要最大化最近的异类距离，即最小化其负值
        sep_loss = torch.mean(-max_diff_class_dist)
        self.loss_ensemble['sep_loss'] = self.lambdas['identity'] * sep_loss
        # ========== 【修正结束】 ==========

        gabor_bank = model_module.gabor_basis_bank
        fourier_bank = model_module.fourier_basis_bank
        learnable_kernels = model_module.learnable_basis_bank
        gabor_mu_loss = torch.mean(gabor_bank.mu ** 2)
        gabor_sigma_loss = torch.mean(F.relu(gabor_bank.sigma - 0.5))
        self.loss_ensemble['gabor_spec_loss'] = self.lambdas.get('gabor_spec', 0.01) * (
                    gabor_mu_loss + gabor_sigma_loss)
        fourier_amp_variance_loss = torch.var(fourier_bank.A)
        self.loss_ensemble['fourier_spec_loss'] = self.lambdas.get('fourier_spec', 0.01) * fourier_amp_variance_loss
        gabor_kernels = gabor_bank.get_kernels().detach().flatten(1)  # detach to be safe
        fourier_kernels = fourier_bank.get_kernels().detach().flatten(1)
        learnable_flat = learnable_kernels.flatten(1)
        all_fixed_kernels = torch.cat([gabor_kernels, fourier_kernels], dim=0)
        cos_sim = F.cosine_similarity(learnable_flat.unsqueeze(1), all_fixed_kernels.unsqueeze(0), dim=2)
        orthogonality_loss = torch.mean(cos_sim ** 2)
        self.loss_ensemble['orth_loss'] = self.lambdas.get('orth', 0.1) * orthogonality_loss
        mix_l1_loss = torch.norm(model_module.mixing_weights, p=1)
        self.loss_ensemble['weight_loss'] = self.lambdas.get('mix_l1', 1e-4) * mix_l1_loss
        total_loss = sum(self.loss_ensemble.values())
        return total_loss

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0
        self.model.train()
        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            total += labels.size(0);
            inputs = inputs.to(self.device);
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            loss = self.compute_comprehensive_loss(outputs, labels)
            self.optimizer.zero_grad();
            loss.backward();
            self.optimizer.step()
            train_loss += loss.item();
            predicted = torch.argmax(outputs, 1);
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1
            cls_val = self.loss_ensemble.get('cross_entropy', torch.tensor(0.0)).item()
            clst_val = self.loss_ensemble.get('clst_loss', torch.tensor(0.0)).item()
            sep_val = self.loss_ensemble.get('sep_loss', torch.tensor(0.0)).item()
            g_spec_val = self.loss_ensemble.get('gabor_spec_loss', torch.tensor(0.0)).item()
            f_spec_val = self.loss_ensemble.get('fourier_spec_loss', torch.tensor(0.0)).item()
            orth_val = self.loss_ensemble.get('orth_loss', torch.tensor(0.0)).item()
            l1_val = self.loss_ensemble.get('weight_loss', torch.tensor(0.0)).item()
            progress_bar(i, len(self.loader_dict['train']),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) | cls: %.3f | clst: %.3f | sep: %.3f | g_spec: %.3f | f_spec: %.3f | orth: %.3f | L1: %.4f'
                         % (train_loss / (i + 1), 100. * correct / total, correct, total,
                            cls_val, clst_val, sep_val, g_spec_val, f_spec_val, orth_val, l1_val))
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('');
                val_acc, val_loss, val_mf1 = self.evaluate(mode='val')
                self.early_stopping(val_mf1, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop: break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval();
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0);
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))
        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0);
            inputs = inputs.to(self.device);
            labels = labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            loss = self.compute_comprehensive_loss(outputs, labels)
            eval_loss += loss.item();
            predicted = torch.argmax(outputs, 1);
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()]);
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])
            progress_bar(i, len(self.loader_dict[mode]), f'Evaluating {mode} set...')
        y_pred_argmax = np.argmax(y_pred, 1)
        result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True, zero_division=0)
        mf1 = round(result_dict['macro avg']['f1-score'] * 100, 2);
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
            if self.early_stopping.early_stop: print("[INFO] Early stopping triggered."); break
        print("[INFO] Loading best model for final evaluation...")
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
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())
    with open(args.config) as config_file: config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))
    for fold in range(8, config['dataset']['num_splits'] + 1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()