# --- test_rep.py ---

import os
import json
import argparse
import warnings
import time
import numpy as np
import pandas as pd
import sklearn.metrics as skmet
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_random_seed, progress_bar
from loader import EEGDataLoader


# 导入你的轻量化学生模型
from models.RepSleepNet import RepSleepNet
warnings.filterwarnings("ignore")


def save_cm_csv(cm, savepath):
    cm_new = np.zeros((5, 5))
    for x in range(5):
        t = cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y] / t * 100, 2) if t != 0 else 0

    categories = ["W", "N1", "N2", "N3", "REM"]
    df = pd.DataFrame(index=categories, columns=categories)

    for x in range(5):
        for y in range(5):
            df.iloc[x, y] = f"{cm_new[x][y]}% ({int(cm[x][y])})"

    df.to_csv(savepath, index=True)
    print(f"[SUCCESS] 百分比+数据量混淆矩阵已保存至: {savepath}")


def save_cm_counts_csv(cm, savepath):
    """只保存原始数据量的混淆矩阵"""
    categories = ["W", "N1", "N2", "N3", "REM"]
    df = pd.DataFrame(cm.astype(int), index=categories, columns=categories)
    df.to_csv(savepath, index=True)
    print(f"[SUCCESS] 纯数据量混淆矩阵已保存至: {savepath}")


class RepSleepEvaluator:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 初始化未重参数化的基线模型
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        # 2. 定位训练好的 Checkpoint
        self.ckpt_path = os.path.join('checkpoints', 'RepSleepNet_' + str(args.seed))
        self.ckpt_name = f'repsleep_fold-{self.fold:02d}.pth'

    def build_model(self):
        model = RepSleepNet(num_classes=self.cfg['classifier']['num_classes'],
                            seq_len=self.cfg['dataset'].get('seq_len', 10))
        model.to(self.device)
        return model

    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.tp_cfg['batch_size'],
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=False)
        return {'test': test_loader}

    def load_checkpoint(self):
        model_path = os.path.join(self.ckpt_path, self.ckpt_name)
        if not os.path.exists(model_path):
            print(f"[ERROR] Checkpoint未找到: {model_path}")
            return False

        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        return True

    def count_flops_and_params(self, model, input_size):
        """
        手动统计 FLOPs 和参数量，不依赖第三方库
        input_size: tuple, e.g. (batch_size, channels, length)
        """
        flops = 0
        params = 0

        def conv1d_flops(layer, x_shape):
            # x_shape: (batch, in_channels, length)
            batch, in_c, L = x_shape
            out_c = layer.out_channels
            k = layer.kernel_size[0]
            out_L = (L + 2 * layer.padding[0] - k) // layer.stride[0] + 1
            # 每个输出元素需要 in_c * k 次乘加
            return batch * out_c * out_L * (in_c * k)

        def linear_flops(layer, x_shape):
            # x_shape: (batch, in_features)
            batch, in_f = x_shape
            out_f = layer.out_features
            return batch * in_f * out_f

        def bn_flops(layer, x_shape):
            # BN: 每个元素一次加减乘除，近似为 4 FLOPs
            return np.prod(x_shape) * 4

        def relu_flops(x_shape):
            # ReLU: 每个元素一次比较
            return np.prod(x_shape)

        # 遍历模型
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv1d):
                flops += conv1d_flops(module, input_size)
                params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                # 假设输入是 (batch, in_features)
                flops += linear_flops(module, (input_size[0], module.in_features))
                params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.BatchNorm1d):
                flops += bn_flops(module, input_size)
                params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.ReLU):
                flops += relu_flops(input_size)
            else:
                # 其他层可按需扩展
                params += sum(p.numel() for p in module.parameters())

        return flops / 1e6, params / 1e6  # 返回 M 单位

    @torch.no_grad()
    def evaluate_metrics(self, mode='test'):
        """计算模型预测指标：Acc, MF1, Kappa, Per-Class F1"""
        self.model.eval()
        correct, total = 0, 0
        y_true, y_pred_labels = [], []

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            logits, _ = self.model(inputs)

            predicted = torch.argmax(logits, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred_labels.extend(predicted.cpu().numpy())

        acc = 100. * correct / total
        mf1 = skmet.f1_score(y_true, y_pred_labels, average='macro') * 100
        kappa = skmet.cohen_kappa_score(y_true, y_pred_labels)
        class_f1s = skmet.f1_score(y_true, y_pred_labels, average=None) * 100

        return acc, mf1, kappa, class_f1s, y_true, y_pred_labels

    @torch.no_grad()
    def measure_latency_and_flops(self):
        """测量模型的推理速度 (ms/batch) 和 FLOPs (M)"""
        self.model.eval()
        dummy_input = torch.randn(10, 1, 30000).to(self.device)  # BatchSize=10, 时长30s

        # 1. 测算 FLOPs 和 Params
        flops_m, params_m = self.count_flops_and_params(self.model, dummy_input.shape)

        # 2. 测算 Latency
        # 预热
        for _ in range(10): self.model(dummy_input)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): self.model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.time()

        latency = (t1 - t0) / 50 * 1000  # ms per batch
        return latency, flops_m, params_m

    def run_full_test(self, prune_ratio=0.25):
        """执行完整生命周期的评估"""
        if not self.load_checkpoint():
            return {}, [], []

        results = {"Fold": self.fold}
        class_names = ['W', 'N1', 'N2', 'N3', 'REM']

        print(f"\n{'=' * 40}\n[Fold {self.fold}] 开始多维度评估\n{'=' * 40}")

        # ==========================================
        # 阶段 A：重参数化前 (多分支训练态)
        # ==========================================
        acc, mf1, kappa, c_f1,_,_ = self.evaluate_metrics()
        lat, flops, params = self.measure_latency_and_flops()
        results.update({
            "Pre_Acc": acc, "Pre_MF1": mf1, "Pre_Kappa": kappa,
            "Pre_Lat(ms)": lat, "Pre_FLOPs(M)": flops, "Pre_Params(M)": params
        })
        print(f"[阶段A - 重参数化前] Acc: {acc:.2f}%, MF1: {mf1:.2f}, Latency: {lat:.2f}ms, FLOPs: {flops:.2f}M")

        # ==========================================
        # 阶段 B：重参数化后 (单分支部署态 - 无剪枝)
        # ==========================================
        # 强制将 threshold 设为极其小的负数，即只做结构折叠，不剪枝
        self.model.deploy_and_prune(prune_ratio=0.0)

        acc, mf1, kappa, c_f1, _,_ = self.evaluate_metrics()
        lat, flops, params = self.measure_latency_and_flops()
        results.update({
            "Rep_Acc": acc, "Rep_MF1": mf1, "Rep_Kappa": kappa,
            "Rep_Lat(ms)": lat, "Rep_FLOPs(M)": flops, "Rep_Params(M)": params
        })
        print(f"[阶段B - 重参数化后] Acc: {acc:.2f}%, MF1: {mf1:.2f}, Latency: {lat:.2f}ms, FLOPs: {flops:.2f}M")

        # ==========================================
        # 阶段 C：重参数化 + 物理剪枝后
        # ==========================================
        # 加载干净的权重重新开始，以免上面的折叠污染
        self.model = self.build_model()
        self.load_checkpoint()
        self.model.deploy_and_prune(prune_ratio=prune_ratio)

        acc, mf1, kappa, c_f1, y_true, y_pred_labels = self.evaluate_metrics()
        lat, flops, params = self.measure_latency_and_flops()

        # 理论 FLOPs 扣减 (因为 Mask 是软剪枝，真实硬件测例需要手算下降率，这里模拟理论值)
        theoretical_flops = flops * (1 - prune_ratio)
        theoretical_params = params * (1 - prune_ratio)

        results.update({
            "Prune_Acc": acc, "Prune_MF1": mf1, "Prune_Kappa": kappa,
            "Prune_Lat(ms)": lat, "Prune_FLOPs(M)": theoretical_flops, "Prune_Params(M)": theoretical_params
        })
        # 记录每个类的 F1
        for i, name in enumerate(class_names):
            results[f"Prune_F1_{name}"] = c_f1[i]

        print(
            f"[阶段C - 剪枝 {prune_ratio * 100}%] Acc: {acc:.2f}%, MF1: {mf1:.2f}, Latency: {lat:.2f}ms, 理论FLOPs: {theoretical_flops:.2f}M")
        return results, y_true, y_pred_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--config', type=str, help='config file path',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    parser.add_argument('--prune_ratio', type=float, default=0.2, help='剪枝比例')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    all_fold_results = []
    y_true_all, y_pred_all = [], []  # 新增：全局收集所有折的预测结果

    num_splits = config['dataset'].get('num_splits', 20)
    for fold in range(1, num_splits + 1):
        evaluator = RepSleepEvaluator(args, fold, config)
        fold_res, y_true, y_pred_labels = evaluator.run_full_test(prune_ratio=args.prune_ratio)
        if fold_res:
            all_fold_results.append(fold_res)
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred_labels)

    # ==========================================================
    # 数据汇总与本地保存
    # ==========================================================
    # 统一计算全量混淆矩阵
    if len(y_true_all) > 0:
        full_cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3, 4])
        os.makedirs('results', exist_ok=True)

        # 保存两种版本
        save_cm_csv(full_cm, "results/RepSleepNet_FullConfusionMatrix.csv")
        save_cm_counts_csv(full_cm, "results/RepSleepNet_FullConfusionMatrix_Counts.csv")

    if len(all_fold_results) > 0:
        df = pd.DataFrame(all_fold_results)

        mean_row = df.mean().to_dict()
        mean_row['Fold'] = 'MEAN'
        std_row = df.std().to_dict()
        std_row['Fold'] = 'STD'

        df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

        save_path = f"results/RepSleepNet_Full_Eval_{args.prune_ratio * 100}Pruned.csv"
        df.to_csv(save_path, index=False, float_format='%.3f')

        print(f"\n\n[SUCCESS] 保存至: {save_path}")
        print("======== 平均性能汇总 ========")
        print(f"原始精度 (Acc/MF1): {mean_row['Pre_Acc']:.2f}% / {mean_row['Pre_MF1']:.2f}")
        print(f"剪枝精度 (Acc/MF1): {mean_row['Prune_Acc']:.2f}% / {mean_row['Prune_MF1']:.2f}")
        print(f"推理速度 (重参前 -> 重参后): {mean_row['Pre_Lat(ms)']:.2f} ms -> {mean_row['Rep_Lat(ms)']:.2f} ms")
        print(f"理论计算量 (原模型 -> 剪枝后): {mean_row['Rep_FLOPs(M)']:.2f} M -> {mean_row['Prune_FLOPs(M)']:.2f} M")
        print("==================================")


if __name__ == "__main__":
    main()