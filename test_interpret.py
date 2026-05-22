# --- test.py ---

import os
import json
import argparse
import warnings
import numpy as np
import sklearn.metrics as skmet
from collections import OrderedDict  # *** 步骤 1: 在这里添加导入 ***
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from loader import EEGDataLoader
# from models.protop_gabor import ProtoPNet
# from models.ProtSleepNet_Fast import ProtoPNet
from models.ProtSleepNet_Fast_stable import ProtoPNet
warnings.filterwarnings("ignore")


class OneFoldEvaluator:
    """
    一个独立的评估器类，专门用于加载已训练好的模型并在测试集上进行评估。
    它不再继承自 OneFoldTrainer，以实现训练和评估逻辑的解耦。
    """

    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.cfg = config
        self.ds_cfg = config['dataset']
        self.tp_cfg = config['training_params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Config name: {config['name']}")

        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        # 定义检查点路径
        self.ckpt_path = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
        self.ckpt_name = f'ckpt_fold-{self.fold:02d}.pth'

    def build_model(self):
        model = ProtoPNet(self.cfg)
        print(f"[INFO] Number of params of model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # 处理多GPU情况
        if len(self.args.gpu.split(",")) > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))

        model.to(self.device)
        print(f"[INFO] Model prepared, Device used: {self.device} GPU:{self.args.gpu}")
        return model

    def build_dataloader(self):
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.tp_cfg['batch_size'],
                                 shuffle=False,
                                 num_workers=4 * len(self.args.gpu.split(",")),
                                 pin_memory=True,
                                 drop_last=False)  # 评估时绝不应丢弃任何数据
        print('[INFO] Dataloader prepared')
        return {'test': test_loader}

    @torch.no_grad()
    def evaluate(self, mode='test'):
        """
        专用于评估的简化版 evaluate 方法。
        只计算模型输出和性能指标，不计算任何损失函数。
        """
        self.model.eval()
        correct, total = 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            # 核心：只进行前向传播
            outputs = self.model(inputs)

            predicted = torch.argmax(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), f'Evaluating {mode} set...')

        # 计算最终指标
        y_pred_argmax = np.argmax(y_pred, 1)
        result_dict = skmet.classification_report(y_true, y_pred_argmax, digits=3, output_dict=True, zero_division=0)
        mf1 = round(result_dict['macro avg']['f1-score'] * 100, 2)
        accuracy = round(100. * correct / total, 2)

        print(f'\nTest Results | Acc: {accuracy}% ({correct}/{total}) | MF1: {mf1}')
        return y_true, y_pred, mf1

    def load_checkpoint(self):
        """将权重加载逻辑分离出来，支持外部直接调用"""
        model_path = os.path.join(self.ckpt_path, self.ckpt_name)
        if not os.path.exists(model_path):
            print(f"[ERROR] Checkpoint not found at: {model_path}")
            return False

        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        return True

    def run(self):
        print(f'\n[INFO] Evaluating Fold: {self.fold}')
        # 调用加载权重的逻辑
        if not self.load_checkpoint():
            return np.array([]), np.array([]), 0.0

        y_true, y_pred, mf1 = self.evaluate(mode='test')
        print('')
        return y_true, y_pred, mf1


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
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
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    cm = []

    for fold in range(1, config['dataset']['num_splits'] + 1):
        evaluator = OneFoldEvaluator(args, fold, config)

        # ==================== 核心控制开关 ====================
        FAST_PLOT_MODE = True  # 开启后：不进行模型评估指标计算，直接绘制

        ONLY_CORRECT = True  # 只绘制预测正确的样本
        GROUP_BY_TYPE = True  # 分别从 Gabor, Fourier, Learnable 各提取1个最大贡献模板
        PATCH_WINDOW_SEC = 3.0  # 截取片段长度（秒）
        # =====================================================

        if FAST_PLOT_MODE:
            if not evaluator.load_checkpoint():
                continue

            print(f"\n[INFO] 极速绘图模式开启 - 正在搜索 Fold {fold} 的匹配样本...")
            from visualize_prototype import explain_single_sample_comprehensive
            class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

            found_classes = set()
            dataset = evaluator.loader_dict['test'].dataset
            evaluator.model.eval()

            for idx in range(len(dataset)):
                sample_tuple = dataset[idx]
                x, y = sample_tuple
                true_class = int(y.item() if isinstance(y, torch.Tensor) else y)

                # 如果这个类别已经画过了，跳过继续找下一个类别
                if true_class in found_classes:
                    continue

                # 极速单样本前向传播 (耗时极短)
                with torch.no_grad():
                    x_tensor = torch.as_tensor(x).clone().detach().float().to(evaluator.device)
                    if x_tensor.dim() == 2:
                        x_tensor = x_tensor.unsqueeze(0)
                    elif x_tensor.dim() == 1:
                        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)
                    x_tensor = x_tensor.repeat(2, 1, 1)  # 绕过squeeze Bug

                    logits = evaluator.model(x_tensor)
                    pred_class = torch.argmax(logits[0:1], dim=1).item()

                # 如果开启了 ONLY_CORRECT，预测错的不要
                if ONLY_CORRECT and pred_class != true_class:
                    continue

                # 找到了该类别的优良样本，开始画图！
                found_classes.add(true_class)
                true_label_name = class_names[true_class]
                print(f"[Plotting] 发现目标: [{true_label_name}] (Dataset Index: {idx}) -> 绘制中...")

                explain_single_sample_comprehensive(
                    model=evaluator.model,
                    sample_tuple=sample_tuple,
                    device=evaluator.device,
                    class_names=class_names,
                    sample_rate=100,
                    group_by_type=GROUP_BY_TYPE,
                    patch_window_sec=PATCH_WINDOW_SEC,
                    save_name=f'single_sample_fold{fold}_{true_label_name}.svg'
                )

                # 集齐 5 个类别的召唤神龙，直接跳出循环！
                if len(found_classes) == len(class_names):
                    print(f"[INFO] Fold {fold} 的 5 种睡眠期单样本图全部绘制完毕！")
                    break

            # 极速绘图模式下，画完直接结束本 Fold，跳过耗时的 evaluator.run()
            continue

            # ====================================================================
        # 常规指标评估逻辑 (只有当 FAST_PLOT_MODE = False 时才会执行到这里)
        # ====================================================================
        y_true, y_pred, mf1 = evaluator.run()
        if y_true.size == 0:
            continue
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
        summarize_result(config, fold, Y_true, Y_pred)

        '''绘制混淆矩阵'''
        cm.append(confusion_matrix(Y_true.astype(int), Y_pred.argmax(axis=1)))

        '''绘制原型模板图像 & 混合矩阵热力图'''
        from visualize_prototype import generate_publication_figure, plot_mixing_weights_heatmap
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        generate_publication_figure(evaluator.model, evaluator.loader_dict['test'], evaluator.device, class_names)
        plot_mixing_weights_heatmap(evaluator.model, evaluator.device)

    mean_cm = np.mean(cm, axis=0)
    cm_plot(mean_cm, './results/cm.svg')


if __name__ == "__main__":
    main()