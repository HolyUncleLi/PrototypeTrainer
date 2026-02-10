# --- plot_logs.py ---

import matplotlib.pyplot as plt
import os
import argparse


def parse_txt_log(filepath):
    """
    解析格式为: Epoch:1 | train_loss:0.5 | val_acc:80.2 ... 的文本文件
    """
    data = {
        'epoch': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'train_loss': [],
        'sub_losses': {}  # 存放 loss_cls, loss_clst 等
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if not line.startswith("Epoch"): continue

        parts = line.strip().split(' | ')
        metrics = {}
        for part in parts:
            if ':' in part:
                key, val = part.split(':')
                metrics[key.strip()] = float(val)

        # 填充数据
        data['epoch'].append(metrics['Epoch'])
        data['train_acc'].append(metrics.get('train_acc', 0))
        data['val_acc'].append(metrics.get('val_acc', 0))
        data['test_acc'].append(metrics.get('test_acc', 0))
        data['train_loss'].append(metrics.get('train_loss', 0))

        # 自动发现子损失项
        for k, v in metrics.items():
            if k.startswith('loss_') and k != 'train_loss':  # 子 loss
                if k not in data['sub_losses']:
                    data['sub_losses'][k] = []
                data['sub_losses'][k].append(v)

    return data


def plot_curves(log_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = parse_txt_log(log_path)
    epochs = data['epoch']

    # 1. 精度曲线 (Acc)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_acc'], label='Train Acc', marker='.')
    plt.plot(epochs, data['val_acc'], label='Val Acc', marker='.')
    plt.plot(epochs, data['test_acc'], label='Test Acc', marker='.')
    plt.title('Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.show()

    # 2. 总 Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Total Train Loss', color='red')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'total_loss_curve.png'))
    plt.show()

    # 3. 子 Loss 变化 (放在一张图里，可能需要双y轴或log scale)
    plt.figure(figsize=(12, 8))
    for loss_name, loss_values in data['sub_losses'].items():
        plt.plot(epochs, loss_values, label=loss_name)

    plt.title('Detailed Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')  # 使用对数坐标，因为 loss 大小差异可能很大
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'detailed_losses.png'))
    plt.show()


if __name__ == "__main__":
    # 使用示例
    # 请将 log_file 替换为你实际生成的 txt 路径
    log_file = './logs/train_log_SleePyCo-Transformer_fold1.txt'
    if os.path.exists(log_file):
        plot_curves(log_file)
    else:
        print(f"File not found: {log_file}")