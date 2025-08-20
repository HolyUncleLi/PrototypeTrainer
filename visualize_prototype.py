import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.gridspec as gridspec
import json
import os
import sys


save_path = ""

# =============================================================================
# 1. 核心绘图与辅助函数 (Final Corrected Version)
# =============================================================================

def get_key_waveform_from_indices(signal_epoch, activation_idx, model_stem,
                                  proto_kernel_size_in_feature_space, sample_rate):
    """
    【最终版】恢复的、无硬性长度限制的波形提取函数。
    """
    total_stride = 1
    for layer in model_stem:
        if hasattr(layer, 'stride'):
            stride_val = layer.stride
            if isinstance(stride_val, tuple):
                total_stride *= stride_val[0]
            else:
                total_stride *= stride_val

    start_idx_in_signal = int(activation_idx * total_stride)
    proto_len_in_signal = int(proto_kernel_size_in_feature_space * total_stride)
    end_idx_in_signal = start_idx_in_signal + proto_len_in_signal

    signal_np = signal_epoch.squeeze().cpu().numpy()
    end_idx_in_signal = min(end_idx_in_signal, len(signal_np))

    wavelet = signal_np[start_idx_in_signal: end_idx_in_signal]
    return wavelet


def generate_publication_figure(model, data_loader, device, class_names, sample_rate=100):
    """
    【最终版】生成您指定的、布局精确、内容正确的最终图表。
    """
    print("--- Generating Final Publication Figure ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model
    model.eval()
    model.to(device)

    num_prototypes = model_to_access.num_composite_prototypes

    print("Step 1: Searching for best matching signal for each prototype...")
    best_matches = {i: {'min_dist': float('inf')} for i in range(num_prototypes)}
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            batch_min_dists = model_to_access.min_distance
            batch_indices = model_to_access.min_indices
            for p_idx in range(num_prototypes):
                min_val, min_batch_idx = torch.min(batch_min_dists[:, p_idx], dim=0)
                if min_val.item() < best_matches[p_idx]['min_dist']:
                    best_matches[p_idx].update({
                        'min_dist': min_val.item(),
                        'signal_epoch': inputs[min_batch_idx].cpu(),
                        'activation_idx': batch_indices[min_batch_idx, p_idx].item()
                    })

    print("Step 2: Pre-calculating all necessary data...")
    gabor_kernels = model_to_access.gabor_basis_bank.get_kernels().squeeze(1)
    fourier_kernels = model_to_access.fourier_basis_bank.get_kernels().squeeze(1)
    learnable_kernels = model_to_access.learnable_basis_bank.data.squeeze(1)
    all_basis_kernels = torch.cat([gabor_kernels, fourier_kernels, learnable_kernels], dim=0)
    reconstructed_prototypes = torch.matmul(model_to_access.mixing_weights.data,
                                            all_basis_kernels).cpu().detach().numpy()

    fc_weights = model_to_access.fc.weight.data.cpu().detach().numpy()
    vmin, vmax = 0, max(1, fc_weights.max())

    print("Step 3: Creating and laying out the figure...")

    # --- 核心布局修改 ---
    # 减小figsize以使整体更紧凑
    fig = plt.figure(figsize=(15, 0.7 * num_prototypes))

    # 调整GridSpec以压缩波形图宽度
    gs = gridspec.GridSpec(
        num_prototypes + 1, 4,
        figure=fig,
        height_ratios=[0.5] + [1] * num_prototypes,
        width_ratios=[1.2, 2, 2, 4],  # Y轴标签, 波形1, 波形2, 热力图
        hspace=0.2, wspace=0.15
    )

    # --- 顶部列标签 ---
    ax_left_title = fig.add_subplot(gs[0, 1:3])
    ax_left_title.set_title("Nearest Patch of Prototype", fontsize=12, weight='bold', y=0.8)
    ax_left_title.axis('off')

    ax_right_title_container = fig.add_subplot(gs[0, 3])
    for i, name in enumerate(class_names):
        ax_right_title_container.text((i + 0.5) / len(class_names), 0.5, name,
                                      ha='center', va='center', fontsize=10, weight='bold')
    ax_right_title_container.axis('off')

    axes_to_hide = fig.add_subplot(gs[0, 0])
    axes_to_hide.axis('off')

    # --- 填充每一行数据 ---
    for p_idx in range(num_prototypes):
        row_idx = p_idx + 1

        # Y轴标签
        ax_ylabel = fig.add_subplot(gs[row_idx, 0])
        prototype_name = f"W{p_idx}:"
        ax_ylabel.text(0.95, 0.5, prototype_name, ha='right', va='center', fontsize=10, transform=ax_ylabel.transAxes)
        ax_ylabel.axis('off')

        # 左侧图 1: 激活波形 (内容已恢复)
        ax_wave1 = fig.add_subplot(gs[row_idx, 1])
        if 'signal_epoch' in best_matches[p_idx]:
            wavelet = get_key_waveform_from_indices(
                best_matches[p_idx]['signal_epoch'],
                best_matches[p_idx]['activation_idx'],
                model_to_access.stem,
                model_to_access.prototype_kernel_size,
                sample_rate
            )
            time_axis = np.arange(len(wavelet)) / sample_rate
            ax_wave1.plot(time_axis, wavelet, color='black', linewidth=0.7)
        ax_wave1.axis('off')

        # 左侧图 2: 重构原型 (内容已恢复)
        ax_wave2 = fig.add_subplot(gs[row_idx, 2])
        reconstructed_composite = reconstructed_prototypes[p_idx]
        time_axis_proto = np.arange(len(reconstructed_composite)) / sample_rate
        ax_wave2.plot(time_axis_proto, reconstructed_composite, color='crimson', linewidth=0.9)
        ax_wave2.axis('off')

        # 右侧: 矩形热力图 (布局已修正)
        ax_right = fig.add_subplot(gs[row_idx, 3])
        support_scores = fc_weights[:, p_idx].reshape(1, -1)
        display_scores = np.maximum(0, support_scores)

        im = ax_right.imshow(display_scores, cmap='Reds', vmin=vmin, vmax=vmax, aspect='auto')

        for i in range(len(class_names)):
            val = display_scores[0, i]
            norm_val = (val - vmin) / (vmax - vmin + 1e-6)
            text_color = "white" if norm_val > 0.6 else "black"
            ax_right.text(i, 0, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

        ax_right.set_xticks([]);
        ax_right.set_yticks([])

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Reds'), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)

    # --- 【核心修改】: 手动调整子图间距和边距以消除空白 ---
    fig.subplots_adjust(left=0.05, right=0.89, top=0.92, bottom=0.05)

    plt.show()


def plot_mixing_weights_heatmap(model, device):
    """
    【FIGURE 2】Generates the heatmap of the mixing_weights matrix.
    """
    print("\n--- Generating Figure 2: Basis Prototype Mixing Matrix ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model
    model.eval()
    model.to(device)

    weights = model_to_access.mixing_weights.data.cpu().numpy()  # Shape: [20, 50]
    num_composite, num_basis = weights.shape

    num_g = model_to_access.num_gabor_basis
    num_f = model_to_access.num_fourier_basis
    num_l = model_to_access.num_learnable_basis

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(weights, cmap='viridis', aspect='auto', interpolation='nearest')

    # Configure axes and labels
    ax.set_yticks(np.arange(num_composite))
    ax.set_yticklabels([f"Composite P{i}" for i in range(num_composite)])

    x_labels = [f"G{i}" for i in range(num_g)] + \
               [f"F{i}" for i in range(num_f)] + \
               [f"L{i}" for i in range(num_l)]
    ax.set_xticks(np.arange(num_basis))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)

    ax.set_xlabel("Basis Prototypes (G: Gabor, F: Fourier, L: Learnable)", fontsize=12)
    ax.set_ylabel("Composite Prototypes", fontsize=12)

    # Add vertical lines to separate basis types
    ax.axvline(x=num_g - 0.5, color='white', linestyle='--', linewidth=2)
    ax.axvline(x=num_g + num_f - 0.5, color='white', linestyle='--', linewidth=2)

    # Add title and colorbar
    ax.set_title("Mixing Weights: How Basis Prototypes form Composite Prototypes", fontsize=16, pad=20)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Mixing Weight Value', fontsize=12)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. 主程序
# =============================================================================

if __name__ == "__main__":
    try:
        from models.protop_cross import ProtoPNet
    except ImportError:
        print("ERROR: Could not import 'ProtoPNet'.")
        sys.exit(1)

    config_path = './configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json'
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: Config file not found. Using a dummy config.")
        config = {'classifier': {'afr_reduced_dim': 128, 'num_classes': 5, 'prototype_num': 20,
                                 'prototype_shape': [20, 128, 15]}}

    model = ProtoPNet(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model created and moved to {device}.")

    dummy_loader = DataLoader(
        TensorDataset(torch.randn(64, 1, 30000), torch.randint(0, 5, (64,))),
        batch_size=32
    )
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    generate_publication_figure(model, dummy_loader, device, class_names)
    single_sample, _ = dummy_loader.dataset[0]
    plot_mixing_weights_heatmap(model, device)