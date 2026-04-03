# --- plot_logs.py ---

import matplotlib.pyplot as plt
import os
import glob


def parse_txt_log(filepath):
    data = {
        'epoch': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
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
        data['train_acc'].append(metrics.get('train_acc', 0))
        data['val_acc'].append(metrics.get('val_acc', 0))
        data['test_acc'].append(metrics.get('test_acc', 0))
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

    # --- Plot 1: Accuracy (Train vs Val vs Test) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_acc'], label='Train Acc', linestyle='-', marker='.')
    plt.plot(epochs, data['val_acc'], label='Val Acc', linestyle='-', marker='.')
    # plt.plot(epochs, data['test_acc'], label='Test Acc', linestyle='--', alpha=0.5)
    plt.title('Accuracy Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    # --- Plot 2: Total Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], label='Total Train Loss', color='red')
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'total_loss_curve.png'))
    plt.close()

    # --- Plot 3: Sub Losses ---
    plt.figure(figsize=(12, 8))
    styles = ['-', '--', '-.', ':']
    for i, (k, v) in enumerate(data['sub_losses'].items()):
        plt.plot(epochs, v, label=k, linestyle=styles[i % len(styles)])
    plt.title('Detailed Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.savefig(os.path.join(output_dir, 'detailed_loss_components.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    log_files = glob.glob('../logs/*.txt')
    if log_files:
        latest = max(log_files, key=os.path.getmtime)
        print(f"Plotting log: {latest}")
        plot_curves(latest)