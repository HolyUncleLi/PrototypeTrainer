# --- train_rep.py ---

import os
import json
import argparse
import warnings
import numpy as np
import sklearn.metrics as skmet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import EarlyStopping, set_random_seed
from loader import EEGDataLoader

from models.ProtSleepNet_Fast import ProtoPNet
from models.RepSleepNet import RepSleepNet

warnings.filterwarnings("ignore")
CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class ExplainableKDLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.5, beta=0.1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHT).float())

    def forward(self, student_logits, labels):
        # 1. 分类损失
        loss_ce = self.ce_loss(student_logits, labels)

        total_loss = loss_ce
        return total_loss


class EKDTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 实例化教师网络
        self.teacher_model = ProtoPNet(config).to(self.device)
        self._load_teacher_weights()
        self.teacher_model.eval()

        # 使用 forward_hook 隐式截获 Teacher 的隐层特征
        self.teacher_features = None

        def hook(module, input, output):
            self.teacher_features = output

        self.teacher_model.feature_extractor.register_forward_hook(hook)

        # 2. 实例化学生 (未重参数化的多分支原始态)
        self.student_model = RepSleepNet(num_classes=config['classifier']['num_classes'],
                                         seq_len=config['dataset'].get('seq_len', 10)).to(self.device)

        self.loader_dict = self.build_dataloader()

        self.criterion = ExplainableKDLoss(T=3.0, alpha=0.5, beta=0.1).to(self.device)
        # 学生模型优化器
        self.optimizer = optim.Adam(self.student_model.parameters(),
                                    lr=config['training_params'].get('lr', 1e-3),
                                    weight_decay=config['training_params'].get('weight_decay', 1e-4))

        self.ckpt_dir = os.path.join('checkpoints', 'RepSleepNet_' + str(args.seed))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = f'repsleep_fold-{self.fold:02d}.pth'

        self.early_stopping = EarlyStopping(patience=config['training_params']['early_stopping']['patience'],
                                            verbose=True,
                                            ckpt_path=self.ckpt_dir,
                                            ckpt_name=self.ckpt_name)

    def _load_teacher_weights(self):
        print(self.cfg)
        teacher_ckpt = os.path.join('checkpoints', 'Teacher_ckpts',
                                    f'ckpt_fold-{self.fold:02d}.pth')
        if os.path.exists(teacher_ckpt):
            self.teacher_model.load_state_dict(torch.load(teacher_ckpt), strict=False)
            print("[INFO] Teacher 模型 (ProtoPNet) 权重加载成功！")
        else:
            print(f"[WARN] 未找到Teacher权重 {teacher_ckpt}，将使用随机权重进行代码测试。")

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.cfg['training_params']['batch_size'],
                                  shuffle=True, num_workers=4)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.cfg['training_params']['batch_size'],
                                shuffle=False, num_workers=4)
        return {'train': train_loader, 'val': val_loader}  # 训练脚本只需 train 和 val

    def train_one_epoch(self, epoch):
        self.student_model.train()
        total_loss, correct, total = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            B = labels.size(0)

            # --- Student 前向传播 ---
            s_logits, s_feat = self.student_model(inputs)

            loss = self.criterion(s_logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(s_logits, 1)
            correct += pred.eq(labels).sum().item()
            total += B

            if i % 20 == 0:
                print(f"\rEpoch {epoch}[{i}/{len(self.loader_dict['train'])}] "
                      f"Loss: {loss.item():.4f}", end="")
        print("")

    @torch.no_grad()
    def evaluate(self, mode='val'):
        self.student_model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true, y_pred_labels = [], []

        for inputs, labels in self.loader_dict[mode]:
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            s_logits, _ = self.student_model(inputs)
            loss = self.criterion.ce_loss(s_logits, labels)

            eval_loss += loss.item() * inputs.size(0)
            pred = torch.argmax(s_logits, 1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred_labels.extend(pred.cpu().numpy())

        acc = 100. * correct / total
        mf1 = skmet.f1_score(y_true, y_pred_labels, average='macro') * 100
        return acc, mf1, eval_loss / total

    def run(self):
        print(f"\n[INFO] 开始 Fold {self.fold} 训练...")
        max_epochs = self.cfg['training_params'].get('max_epochs', 30)

        for epoch in range(1, max_epochs + 1):
            self.train_one_epoch(epoch)
            val_acc, val_mf1, val_loss = self.evaluate('val')
            print(f"Epoch {epoch} | Val ACC: {val_acc:.2f}% | Val MF1: {val_mf1:.2f} | Val Loss: {val_loss:.4f}")

            # 根据验证集的 MF1 进行早停并保存当前最佳的【未重参数化多分支模型】
            self.early_stopping(val_mf1, val_loss, self.student_model)
            if self.early_stopping.early_stop:
                print("[INFO] 本折训练早停结束。")
                break

        print(f"[INFO] Fold {self.fold} 最佳模型已保存至: {os.path.join(self.ckpt_dir, self.ckpt_name)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--config', type=str, help='config file path',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json',
                        # default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_wavesensing.json',
                        # default='./configs/SleePyCo-Transformer_SL-10_numScales-3_SHHS_wavesensing.json',
                        )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    num_splits = config['dataset'].get('num_splits', 20)

    for fold in range(1, num_splits + 1):
    # for fold in range(1, 5):
        trainer = EKDTrainer(args, fold, config)
        trainer.run()

    print("\n=======================================================")
    print("所有折知识蒸馏训练与验证均已完成")
    print("=======================================================")


if __name__ == "__main__":
    main()