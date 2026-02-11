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
from models.protop_gabor import ProtoPNet

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
        # 确保实例化的是正确的 V2 模型
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

    def run(self):
        print(f'\n[INFO] Evaluating Fold: {self.fold}')
        model_path = os.path.join(self.ckpt_path, self.ckpt_name)

        if not os.path.exists(model_path):
            print(f"[ERROR] Checkpoint not found at: {model_path}")
            return np.array([]), np.array([]), 0.0

        # ====================================================================
        # *** 步骤 2: 在这里替换加载逻辑 ***
        # ====================================================================
        # 原始的加载行 (将被替换)
        # self.model.load_state_dict(torch.load(model_path))

        # 修复后的加载逻辑
        state_dict = torch.load(model_path, map_location=self.device)
        # 创建一个新的、空的 state_dict
        new_state_dict = OrderedDict()
        # 遍历加载的 state_dict，移除 'module.' 前缀
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_state_dict[name] = v
        # 将修复后的 state_dict 加载到模型中
        self.model.load_state_dict(new_state_dict)
        # ====================================================================

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

    cm = []

    for fold in range(1, config['dataset']['num_splits'] + 1):
        evaluator = OneFoldEvaluator(args, fold, config)
        y_true, y_pred, mf1 = evaluator.run()
        if y_true.size == 0:
            continue
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
        summarize_result(config, fold, Y_true, Y_pred)

        '''绘制混淆矩阵'''
        # cm.append(confusion_matrix(Y_true.astype(int), Y_pred.argmax(axis=1)))

        '''绘制原型模板图像 & 混合矩阵热力图'''
        '''
        from visualize_prototype import generate_publication_figure, plot_mixing_weights_heatmap
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        # generate_publication_figure(evaluator.model, evaluator.loader_dict['test'], evaluator.device, class_names)
        plot_mixing_weights_heatmap(evaluator.model, evaluator.device)
        '''


        '''
        # 原生版本，弃用
        from visualize_single_prototype import visualize_prototypes_final, explain_single_sample_final
        class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        visualize_prototypes_final(evaluator.model, evaluator.loader_dict['test'], evaluator.device, class_names)
        for i in range(10):
            single_sample, _ = evaluator.loader_dict['test'].dataset[i]
            explain_single_sample_final(evaluator.model, single_sample, evaluator.device, class_names)
        '''

    mean_cm = np.mean(cm, axis=0)
    cm_plot(mean_cm, './results/cm.svg')


if __name__ == "__main__":
    main()