# train_mtcl_merged.py
# Combined trainer: base structure from train_mtcl, ProtoPNet + complex loss + TxtLogger from v4.
# 保留 ProtoPNet、复杂损失项、TxtLogger；其余逻辑尽量沿用 train_mtcl 的结构与流程。

import os
import sys
import json
import argparse
import warnings
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from loader import EEGDataLoader
from models.ProtSleepNet_Fast import ProtoPNet
# from models.ProtSleepNet_Fast_stable import ProtoPNet
# from models.protop import ProtoPNet

warnings.filterwarnings("ignore")
CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class TxtLogger:
    def __init__(self, log_dir, fold, config_name):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.filepath = os.path.join(log_dir, f'train_log_{config_name}_fold{fold}.txt')
        print(f"[INFO] Logging to: {self.filepath}")
        with open(self.filepath, 'a') as f:
            f.write(f"\n{'=' * 20} New Training Session: {time.ctime()} {'=' * 20}\n")

    def log_epoch(self, epoch, metrics):
        line_parts = [f"Epoch: {epoch}"]
        sorted_keys = sorted(metrics.keys())
        for k in sorted_keys:
            v = metrics[k]
            if isinstance(v, float):
                line_parts.append(f"{k}: {v:.5f}")
            else:
                line_parts.append(f"{k}: {v}")
        log_line = " | ".join(line_parts) + "\n"
        with open(self.filepath, 'a') as f:
            f.write(log_line)


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.ds_cfg = config['dataset']
        self.fp_cfg = config.get('feature_pyramid', {})
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('[INFO] Config name: {}'.format(config['name']))

        # Build model, dataloaders, criterion, optimizer, AMP scaler, early stopping, logger
        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        class_weight = torch.tensor(CLASS_WEIGHT).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                    lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # checkpoint dir (follow train_mtcl naming style)
        self.ckpt_path = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)

        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True,
                                            ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name,
                                            mode=self.es_cfg['mode'])

        # TxtLogger from v4
        self.txt_logger = TxtLogger(log_dir='./logs', fold=self.fold, config_name=self.cfg['name'])

        # lambdas for composite loss (from v4)
        self.lambdas = {
            'cls': self.cfg['classifier'].get('class_lambda', 20.0),
            'clst': self.cfg['classifier'].get('clst_lambda', 1.0),
            'sep': self.cfg['classifier'].get('sep_lambda', 0.5),
            'orth': self.cfg['classifier'].get('orth_lambda', 0.1),
            'structure': self.cfg['classifier'].get('structure_lambda', 1.0)
        }

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        # DataParallel if multiple GPUs specified
        if len(self.args.gpu.split(",")) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model

    def build_dataloader(self):
        num_workers = max(1, 4 * len(self.args.gpu.split(",")))
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)
        print('[INFO] Dataloader prepared')
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def activate_train_mode(self):
        self.model.train()

    def compute_comprehensive_loss(self, outputs, labels):
        """
        匹配 ProtSleepNet_Fast 的定制化损失函数
        """
        loss_components = {}
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # ==========================================
        # 1. 交叉熵分类损失 (Classification)
        # ==========================================
        loss_cls = self.criterion(outputs, labels)
        loss_components['loss_cls'] = self.lambdas.get('cls', 1.0) * loss_cls

        # ==========================================
        # 2. 原型聚类与分离损失 (基于 1D Conv 相似度)
        # ==========================================
        # 注意：现在使用的是 max_sim，数值越大代表越匹配
        sim = model_module.max_sim  # shape: [B, Proto_num]
        num_prototypes = model_module.proto_num
        num_classes = model_module.num_classes
        protos_per_class = num_prototypes // num_classes

        # 构建 prototype-class 对齐掩码
        prototype_class_identity = torch.zeros(num_prototypes, num_classes, device=self.device)
        for j in range(num_classes):
            prototype_class_identity[j * protos_per_class: (j + 1) * protos_per_class, j] = 1.0

        class_mask = prototype_class_identity.T[labels]  # shape: [B, Proto_num]

        # 聚类 (Clustering): 属于自己类别的原型，相似度越高越好
        # 最大化 sim 等价于 最小化 -sim
        max_sim_same_class = torch.max(sim + torch.log(class_mask + 1e-9), dim=1).values
        loss_clst = torch.mean(-max_sim_same_class)
        loss_components['loss_clst'] = self.lambdas.get('clst', 0.5) * loss_clst

        # 分离 (Separation): 不属于自己类别的原型，相似度越低越好
        max_sim_diff_class = torch.max(sim + torch.log(1.0 - class_mask + 1e-9), dim=1).values
        loss_sep = torch.mean(max_sim_diff_class)
        loss_components['loss_sep'] = self.lambdas.get('sep', 0.1) * loss_sep

        # ==========================================
        # 3. 先验结构稀疏损失 (L1 Sparsity)
        # ==========================================
        # 强迫模型稀疏地挑选 Gabor/Fourier 基底，防止退化成噪声
        weights = model_module.mixing_weights  # shape: [Proto_num, C_dim, total_bases]
        loss_sparse = torch.mean(torch.abs(weights))
        loss_components['loss_sparse'] = self.lambdas.get('structure', 1.0) * loss_sparse

        # ==========================================
        # 4. 基底正交损失 (Orthogonality)
        # ==========================================
        # 强迫 Learnable 基底学到和固定物理模板正交的未知形态
        l_bank = model_module.learnable_bank.flatten(1)  # [num_l, K]
        fixed_bank = torch.cat([
            model_module.gabor_bank.get_kernels().flatten(1).detach(),
            model_module.fourier_bank.get_kernels().flatten(1).detach()
        ], dim=0)  # [num_g + num_f, K]

        # 计算余弦相似度并惩罚
        l_norm = F.normalize(l_bank, p=2, dim=1)
        f_norm = F.normalize(fixed_bank, p=2, dim=1)
        orth_sim = torch.mm(l_norm, f_norm.t())
        loss_orth = torch.mean(orth_sim ** 2)
        loss_components['loss_orth'] = self.lambdas.get('orth', 0.1) * loss_orth

        # 汇总总损失
        total_loss = sum(loss_components.values())
        return total_loss, loss_components

    def train_one_epoch(self, epoch):
        self.model.train()
        metrics_sum = {}
        total_samples = 0
        correct = 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            bs = inputs.size(0)

            self.optimizer.zero_grad()

            # AMP autocast
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)  # assume model returns logits for classification
                loss, loss_dict = self.compute_comprehensive_loss(outputs, labels)

            # backward with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_samples += bs
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            # accumulate metrics (weighted by batch size)
            if len(metrics_sum) == 0:
                metrics_sum['train_loss'] = 0.0
                for k in loss_dict: metrics_sum[k] = 0.0

            metrics_sum['train_loss'] += loss.item() * bs
            for k, v in loss_dict.items():
                metrics_sum[k] += v.item() * bs

            if i % 20 == 0:
                print(f"\rEpoch {epoch} [{i}/{len(self.loader_dict['train'])}] Loss: {loss.item():.4f}", end="")

            self.train_iter += 1
            # periodic validation as in train_mtcl
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_acc, val_loss = self.evaluate(mode='val')
                # EarlyStopping expects (metric, loss, model)
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

        print("")
        avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
        avg_metrics['train_acc'] = 100. * correct / total_samples
        # log epoch metrics
        self.txt_logger.log_epoch(epoch, avg_metrics)
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            # In case model returns tuple/list (like multiple heads), handle both possibilities
            if isinstance(outputs, (list, tuple)):
                outputs_sum = torch.zeros_like(outputs[0])
                for j in range(len(outputs)):
                    loss += self.criterion(outputs[j], labels)
                    outputs_sum += outputs[j]
            else:
                outputs_sum = outputs
                loss += self.criterion(outputs_sum, labels)

            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()

            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return 100. * correct / total, eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError

    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break

        # load best checkpoint saved by EarlyStopping
        ckpt_full_path = os.path.join(self.ckpt_path, self.ckpt_name)
        if os.path.exists(ckpt_full_path):
            self.model.load_state_dict(torch.load(ckpt_full_path))
        else:
            print(f"[WARN] Checkpoint {ckpt_full_path} not found. Using current model weights.")

        y_true, y_pred = self.evaluate(mode='test')
        print('')
        return y_true, y_pred


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str,
                        # default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json',
                        # default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_wavesensing.json',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_SHHS_wavesensing.json',
                        help='config file path')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')

    # Ensure mode normal
    config['mode'] = config.get('mode', 'normal')

    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    # follow train_mtcl default: iterate folds (here using 1..2 for quick runs or use config value)
    num_folds = config['dataset'].get('num_splits', 2)
    # If user wants full cross-validation, they can set num_splits in config.
    for fold in range(1, num_folds+1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()
