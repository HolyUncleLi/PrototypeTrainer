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
        Reuse the comprehensive loss logic from train_mtcl_v4.
        outputs: model logits (assumed shape [B, C])
        labels: long tensor [B]
        Returns: total_loss (scalar tensor), loss_components (dict of tensors)
        """
        loss_components = {}
        model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # 1. Cls
        loss_cls = self.criterion(outputs, labels)
        loss_components['loss_cls'] = self.lambdas['cls'] * loss_cls

        # 2. Proto Related
        # Expect model_module to expose min_distance, num_composite_prototypes, fc, mixing_weights, proto_splits,
        # num_gabor_basis, num_fourier_basis, num_learnable_basis, learnable_basis_bank, current_gabor_k, current_fourier_k
        min_dist = model_module.min_distance  # shape [B, num_prototypes] or [num_prototypes] depending on implementation
        # If min_dist is per-sample, ensure shape [B, num_prototypes]; if global, expand
        if min_dist.dim() == 1:
            # expand to batch size
            min_dist = min_dist.unsqueeze(0).repeat(outputs.size(0), 1)

        num_prototypes = model_module.num_composite_prototypes
        num_classes = model_module.fc.out_features
        protos_per_class = num_prototypes // num_classes

        # Build prototype-class identity matrix
        prototype_class_identity = torch.zeros(num_prototypes, num_classes, device=self.device)
        for j in range(num_classes):
            prototype_class_identity[j * protos_per_class: (j + 1) * protos_per_class, j] = 1.0

        # class_mask: for each sample, which prototypes belong to its class -> shape [B, num_prototypes]
        class_mask = prototype_class_identity.T[labels]  # shape [B, num_prototypes]

        # inverted distances
        inverted_dist = -min_dist  # higher means closer

        # For same-class clustering: pick the max inverted distance among prototypes of same class
        # add tiny log(class_mask) to mask out others
        max_dist_same_class = torch.max(inverted_dist + torch.log(class_mask + 1e-9), dim=1).values
        loss_clst = torch.mean(-max_dist_same_class)
        loss_components['loss_clst'] = self.lambdas['clst'] * loss_clst

        # For separation: pick max inverted distance among prototypes of different classes
        max_dist_diff_class = torch.max(inverted_dist + torch.log(1.0 - class_mask + 1e-9), dim=1).values
        loss_sep = torch.mean(max_dist_diff_class)
        loss_components['loss_sep'] = self.lambdas['sep'] * loss_sep

        # 3. Regularization: structure mask on mixing weights
        weights = model_module.mixing_weights  # expected shape [num_prototypes, total_basis]
        splits = model_module.proto_splits  # list of row splits
        basis_counts = [model_module.num_gabor_basis, model_module.num_fourier_basis, model_module.num_learnable_basis]

        struc_mask = torch.ones_like(weights)
        row_s, col_s = 0, 0
        for r_c, c_c in zip(splits, basis_counts):
            struc_mask[row_s:row_s + r_c, col_s:col_s + c_c] = 0.0
            row_s += r_c
            col_s += c_c
        loss_struc = torch.mean(torch.abs(weights) * struc_mask)
        loss_components['loss_struc'] = self.lambdas['structure'] * loss_struc

        # Orthogonality between learnable basis and fixed kernels
        learnable_k = model_module.learnable_basis_bank.flatten(1)  # [num_learnable, kernel_len]
        # reuse cached kernels if available
        fixed_k = torch.cat([model_module.current_gabor_k.flatten(1).detach(),
                             model_module.current_fourier_k.flatten(1).detach()], dim=0)

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
    for fold in range(12, num_folds+1):
        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()
