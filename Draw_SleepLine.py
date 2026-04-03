import os
import json
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.special import softmax

import torch
from utils import set_random_seed, progress_bar
from loader import EEGDataLoader
from models.ProtSleepNet_Fast import ProtoPNet

warnings.filterwarnings("ignore")


# ==========================================
# 绘图函数区
# ==========================================

def plot_hypnogram(y_true, y_pred, save_path=None):
    """
    绘制真实的和预测的睡眠阶梯图 (对应图1)
    """
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    epochs = np.arange(len(y_true))

    # 创建上下两个子图，共享X轴
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- 上半部分: 真实标签 ---
    ax1.step(epochs, y_true, color='blue', linewidth=1.5, where='post')
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(stages)
    ax1.set_ylabel('Stages')
    ax1.set_xlim(0, len(epochs))
    # 调整刻度线显示频率
    ax1.set_xticks(np.arange(0, len(epochs) + 1, 100))
    ax1.tick_params(axis='both', direction='out')

    # --- 下半部分: 预测标签及错误点 ---
    ax2.step(epochs, y_pred, color='green', linewidth=1.5, where='post')

    # 找出预测错误的点，画红色的 'x'
    error_indices = np.where(y_true != y_pred)[0]
    ax2.scatter(error_indices, y_pred[error_indices], color='red', marker='x', s=20, zorder=3)

    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(stages)
    ax2.set_ylabel('Stages')
    ax2.set_xlabel('Epoch')
    ax2.set_xlim(0, len(epochs))
    ax2.set_xticks(np.arange(0, len(epochs) + 1, 100))
    ax2.tick_params(axis='both', direction='out')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 睡眠阶梯图已保存至: {save_path}")
    plt.close()


def plot_sleep_probability(y_logits, save_path=None):
    """
    绘制睡眠阶段概率变化图 (对应图2)
    """
    # 将模型输出的 logits 转换为 [0, 1] 之间的概率
    y_probs = softmax(y_logits, axis=1)
    epochs = np.arange(y_probs.shape[0])

    # 提取各个阶段的概率序列 (假设顺序为 W, N1, N2, N3, REM)
    prob_W = y_probs[:, 0]
    prob_N1 = y_probs[:, 1]
    prob_N2 = y_probs[:, 2]
    prob_N3 = y_probs[:, 3]
    prob_REM = y_probs[:, 4]

    # 尽量还原原图的 Viridis 渐变配色方案
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # 设置支持中文的字体 (Windows常用SimHei，Mac常用Arial Unicode MS)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(15, 4))

    # 堆积面积图
    ax.stackplot(epochs, prob_W, prob_N1, prob_N2, prob_N3, prob_REM,
                 colors=colors, labels=labels, alpha=1.0)

    ax.set_title('睡眠阶段概率变化图', fontsize=14)
    ax.set_ylabel('概率 (Probability)', fontsize=12)
    ax.set_xlabel('时间片段 (Epoch)', fontsize=12)
    ax.set_xlim(0, len(epochs))
    ax.set_ylim(0, 1.0)

    # 图例设置到底部中央
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=False, ncol=5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 概率分布图已保存至: {save_path}")
    plt.close()


# ==========================================
# 数据提取与推理逻辑
# ==========================================

def evaluate_single_night(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 1. 实例化模型
    model = ProtoPNet(config).to(device)

    # 2. 加载权重 (复用你 test.py 修复过的加载逻辑)
    ckpt_path = os.path.join('checkpoints', config['name'] + '_' + str(args.seed))
    ckpt_name = f'ckpt_fold-{args.fold:02d}.pth'
    model_path = os.path.join(ckpt_path, ckpt_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到权重文件: {model_path}")

    print(f"[INFO] 正在加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. 加载数据集
    # 这里我们直接拿到 Dataset 对象，而不是 DataLoader，
    # 这样可以方便地通过索引严格提取连续的 N 个 Epoch，不被打乱
    print(f"[INFO] 正在加载 Fold {args.fold} 的测试集...")
    test_dataset = EEGDataLoader(config, args.fold, set='test')
    total_len = len(test_dataset)
    print(f"[INFO] 测试集总长度: {total_len} 个 Epochs")

    # 4. 确定要提取的区间
    start_idx = args.start_epoch
    end_idx = min(start_idx + args.num_epochs, total_len)
    actual_epochs = end_idx - start_idx

    if start_idx >= total_len:
        raise ValueError(f"起始索引 {start_idx} 超出数据集总长度 {total_len}")

    print(f"[INFO] 正在提取受试者数据 (Epochs: {start_idx} 到 {end_idx - 1}, 共 {actual_epochs} 个片段)")

    y_true_list = []
    y_logits_list = []

    # 5. 逐个样本推理
    with torch.no_grad():
        for i in range(start_idx, end_idx):
            # 获取单条数据: (channels, length), label
            inputs, label = test_dataset[i]

            # 增加 batch 维度: (1, channels, length)
            inputs = inputs.unsqueeze(0)

            # 【核心修复】：为了绕过模型内部在 BatchSize=1 时的维度坍缩 Bug (squeeze)
            # 我们将数据复制一份，伪装成 BatchSize=2 送入模型
            inputs = torch.cat([inputs, inputs], dim=0).to(device)

            # 模型前向传播
            outputs = model(inputs)

            y_true_list.append(label.item())

            # 此时 outputs 的 shape 是 [2, num_classes]，我们只取第 0 个的输出即可
            y_logits_list.append(outputs.cpu().numpy()[0])

            if (i - start_idx) % 100 == 0 or (i == end_idx - 1):
                progress_bar(i - start_idx, actual_epochs, "推理中...")

    y_true = np.array(y_true_list)
    y_logits = np.array(y_logits_list)
    y_pred = np.argmax(y_logits, axis=1)

    # 简单计算该受试者准确率
    acc = np.mean(y_true == y_pred) * 100
    print(f"\n[INFO] 该片段预测准确率: {acc:.2f}%")

    return y_true, y_pred, y_logits


# ==========================================
# 主函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="绘制单人整晚睡眠分期图")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str,
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json',
                        help='config file path')

    # --- 新增的参数 ---
    parser.add_argument('--fold', type=int, default=1,
                        help='要读取哪个 Fold 的测试集模型和数据')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='该受试者在测试集中的起始 Epoch 索引 (默认从第0个开始)')
    parser.add_argument('--num_epochs', type=int, default=1050,
                        help='该受试者一整晚的 Epoch 数量 (Sleep-EDF通常在 800-1200 之间)')
    parser.add_argument('--save_dir', type=str, default='./Test/results/LineChart',
                        help='图片保存路径')

    args = parser.parse_args()

    # 显卡设置
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    # 读取配置文件
    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    # 1. 提取数据并推理
    y_true, y_pred, y_logits = evaluate_single_night(args, config)

    # 2. 绘制并保存图片
    hypno_path = os.path.join(args.save_dir, f'hypnogram_fold{args.fold}_start{args.start_epoch}.png')
    prob_path = os.path.join(args.save_dir, f'probability_fold{args.fold}_start{args.start_epoch}.png')

    print("[INFO] 正在生成图片...")
    plot_hypnogram(y_true, y_pred, save_path=hypno_path)
    plot_sleep_probability(y_logits, save_path=prob_path)
    print("[INFO] 所有任务完成！")


if __name__ == "__main__":
    main()