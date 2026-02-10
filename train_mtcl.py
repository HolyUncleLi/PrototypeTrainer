# --- train_mtcl_v4.py ---

import os
import sys
import json
import argparse
import warnings
import numpy as np
import sklearn.metrics as skmet
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import *
from loader import EEGDataLoader

# [IMPORTANT] 导入 V4 模型
from models.protop_gabor import ProtoPNet

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
        """
        metrics: 包含所有指标的字典
        """
        line_parts = [f"Epoch: {epoch}"]
        # 排序 key 保证输出顺序一致
        sorted_keys = sorted(metrics.keys())
        for k in sorted_keys:
            v = metrics[k]
            if isinstance(v, float):
                line_parts.append(f"{k}: {v:.5f}")
            elif isinstance(v, int):
                line_parts.append(f"{k}: {v}")
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
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        class_weight = torch.tensor(CLASS_WEIGHT).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                    lr=self.tp_cfg['lr'],
                                    weight_decay=self.tp_cfg['weight_decay'])

        self.ckpt_dir = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.ckpt_name = f'ckpt_fold-{self.fold:02d}.pth'

        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True,
                                            ckpt_path=self.ckpt_dir, ckpt_name=self.ckpt_name,
                                            mode=self.es_cfg['mode'])

        self.txt_logger = TxtLogger(log_dir='./logs', fold=self.fold, config_name=self.cfg['name'])

        self.lambdas = {
            'cls': self.cfg['classifier'].get('class_lambda', 20.0),
            'clst': self.cfg['classifier'].get('clst_lambda', 1.0),
            'sep': self.cfg['classifier'].get('sep_lambda', 0.5),
            'orth': self.cfg['classifier'].get('orth_lambda', 0.1),
            'structure': self.cfg['classifier'].get('structure_lambda', 1.0)
        }

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print(f'[INFO] Model Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        if len(self.args.gpu.split(",")) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        return model

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4,
                                pin_memory=True)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False,
                                 num_workers=4, pin_memory=True)
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def compute_comprehensive_loss(self, outputs, labels):
        loss_components = {}
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # 1. Cls
        loss_cls = self.criterion(outputs, labels)
        loss_components['loss_cls'] = self.lambdas['cls'] * loss_cls

        # 2. Proto Related
        min_dist = model_module.min_distance
        num_prototypes = model_module.num_composite_prototypes
        num_classes = model_module.fc.out_features
        protos_per_class = num_prototypes // num_classes

        prototype_class_identity = torch.zeros(num_prototypes, num_classes, device=self.device)
        for j in range(num_classes):
            prototype_class_identity[j * protos_per_class: (j + 1) * protos_per_class, j] = 1
        class_mask = prototype_class_identity.T[labels]
        inverted_dist = -min_dist

        # Clst
        max_dist_same_class = torch.max(inverted_dist + torch.log(class_mask + 1e-9), dim=1).values
        loss_clst = torch.mean(-max_dist_same_class)
        loss_components['loss_clst'] = self.lambdas['clst'] * loss_clst

        # Sep
        max_dist_diff_class = torch.max(inverted_dist + torch.log(1 - class_mask + 1e-9), dim=1).values
        loss_sep = torch.mean(max_dist_diff_class)
        loss_components['loss_sep'] = self.lambdas['sep'] * loss_sep

        # 3. Regularization
        weights = model_module.mixing_weights
        splits = model_module.proto_splits
        basis_counts = [model_module.num_gabor_basis, model_module.num_fourier_basis, model_module.num_learnable_basis]

        struc_mask = torch.ones_like(weights)
        row_s, col_s = 0, 0
        for r_c, c_c in zip(splits, basis_counts):
            struc_mask[row_s:row_s + r_c, col_s:col_s + c_c] = 0.0
            row_s += r_c
            col_s += c_c
        loss_struc = torch.mean(torch.abs(weights) * struc_mask)
        loss_components['loss_struc'] = self.lambdas['structure'] * loss_struc

        learnable_k = model_module.learnable_basis_bank.flatten(1)
        fixed_k = torch.cat([model_module.gabor_basis_bank.get_kernels().flatten(1).detach(),
                             model_module.fourier_basis_bank.get_kernels().flatten(1).detach()], dim=0)
        l_norm = F.normalize(learnable_k, p=2, dim=1)
        f_norm = F.normalize(fixed_k, p=2, dim=1)
        similarity = torch.mm(l_norm, f_norm.t())
        loss_orth = torch.mean(similarity ** 2)
        loss_components['loss_orth'] = self.lambdas['orth'] * loss_orth

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

            outputs = self.model(inputs)
            loss, loss_dict = self.compute_comprehensive_loss(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_samples += bs
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()

            if len(metrics_sum) == 0:
                metrics_sum['train_loss'] = 0.0
                for k in loss_dict: metrics_sum[k] = 0.0

            metrics_sum['train_loss'] += loss.item() * bs
            for k, v in loss_dict.items():
                metrics_sum[k] += v.item() * bs

            if i % 20 == 0:
                print(f"\rEpoch {epoch} [{i}/{len(self.loader_dict['train'])}] Loss: {loss.item():.4f}", end="")

        print("")
        avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
        avg_metrics['train_acc'] = 100. * correct / total_samples
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, mode='val'):
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        y_true, y_pred = [], []

        for inputs, labels in self.loader_dict[mode]:
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        acc = 100. * correct / total
        mf1 = skmet.f1_score(y_true, y_pred, average='macro') * 100
        avg_loss = total_loss / total
        return {f'{mode}_acc': acc, f'{mode}_mf1': mf1, f'{mode}_loss': avg_loss}

    def run(self):
        print(f"\n[INFO] Start Training...")
        for epoch in range(1, self.tp_cfg['max_epochs'] + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate('val')
            test_metrics = self.evaluate('test')

            print(
                f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f} | Val MF1: {val_metrics['val_mf1']:.2f}")

            full_log = {**train_metrics, **val_metrics, **test_metrics}
            self.txt_logger.log_epoch(epoch, full_log)

            self.early_stopping(val_metrics['val_mf1'], val_metrics['val_loss'], self.model)
            if self.early_stopping.early_stop:
                print(f"[INFO] Early stopping triggered.")
                break


# ====================================================================
# 3. 主函数
# ====================================================================
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
    # for fold in range(1, config['dataset']['num_splits'] + 1):
    for fold in range(1, 2):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()