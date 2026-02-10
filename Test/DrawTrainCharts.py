# --- plot_logs.py ---

import matplotlib.pyplot as plt
import os
import argparse
import glob


def parse_txt_log(filepath):
    data = {
        'epoch': [],
        'train_loss': [],
        'sub_losses': {}
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

        data['epoch'].append(metrics['Epoch'])
        data['train_loss'].append(metrics.get('train_loss', 0))

        for k, v in metrics.items():
            if k.startswith('loss_') and k != 'train_loss':
                if k not in data['sub_losses']:
                    data['sub_losses'][k] = []
                data['sub_losses'][k].append(v)

    return data


def plot_curves(log_path, output_dir='./results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = parse_txt_log(log_path)
    epochs = data['epoch']

    # ------------------------------------------------
    # Plot 1: Total Loss Only
    # ------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Total Train Loss', color='black', linewidth=2)
    plt.title('Total Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path1 = os.path.join(output_dir, 'total_loss_curve.png')
    plt.savefig(save_path1)
    print(f"Saved Total Loss plot to {save_path1}")
    plt.close()

    # ------------------------------------------------
    # Plot 2: All Individual Loss Components
    # ------------------------------------------------
    plt.figure(figsize=(12, 8))

    # 定义不同 loss 的样式，方便区分
    styles = ['-', '--', '-.', ':']

    for i, (loss_name, loss_values) in enumerate(data['sub_losses'].items()):
        if len(loss_values) == len(epochs):
            plt.plot(epochs, loss_values, label=loss_name, linestyle=styles[i % len(styles)], linewidth=2)

    plt.title('Detailed Loss Components (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value (Log Scale)')
    plt.yscale('log')  # 使用对数坐标，因为不同 Loss 量级差异很大
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    save_path2 = os.path.join(output_dir, 'detailed_loss_components.png')
    plt.savefig(save_path2)
    print(f"Saved Components plot to {save_path2}")
    plt.close()


if __name__ == "__main__":
    # 自动寻找最新的 log 文件
    log_files = glob.glob('./logs/*.txt')
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"Found latest log: {latest_log}")
        plot_curves(latest_log)
    else:
        print("No log files found in ./logs/")