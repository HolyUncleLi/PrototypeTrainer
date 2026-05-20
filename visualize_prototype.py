# --- visualize_prototype.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.gridspec as gridspec
import sys


# =============================================================================
# 辅助函数
# =============================================================================

def get_key_waveform_from_indices(signal_epoch, activation_idx,
                                  proto_kernel_size_in_feature_space, sample_rate):
    """
    【修正点 1】：不再依赖遍历 model.stem 来计算 stride。
    针对 ProtSleepNet_Fast 架构，输入序列长 30000，特征层因 AdaptiveAvgPool1d 被固定为 256。
    因此等效步长 (stride ratio) 就是 input_len / latent_len。
    """
    input_len = 30000
    latent_len = 256
    total_stride = input_len / latent_len

    # 计算在原始 30000 长度信号中的起止索引
    start_idx_in_signal = int(activation_idx * total_stride)
    proto_len_in_signal = int(proto_kernel_size_in_feature_space * total_stride)
    end_idx_in_signal = start_idx_in_signal + proto_len_in_signal

    signal_np = signal_epoch.squeeze().cpu().numpy()
    end_idx_in_signal = min(end_idx_in_signal, len(signal_np))

    wavelet = signal_np[start_idx_in_signal: end_idx_in_signal]
    return wavelet


def generate_publication_figure(model, data_loader, device, class_names, sample_rate=100):
    print("\n--- Generating Final Publication Figure ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model
    model_to_access.eval()
    model_to_access.to(device)

    num_prototypes = model_to_access.num_composite_prototypes

    print("Step 1: Searching for best matching signal for each prototype...")
    best_matches = {i: {'min_dist': float('inf')} for i in range(num_prototypes)}

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            # 强制使用解包后的 model_to_access 进行前向传播，
            # 防止 DataParallel 吞掉 forward 内部保存的 self.min_distance 属性
            _ = model_to_access(inputs)

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

    full_weights = model_to_access.mixing_weights.detach()
    reconstructed_prototypes = torch.matmul(full_weights, all_basis_kernels).cpu().detach().numpy()

    fc_weights = model_to_access.fc.weight.data.cpu().detach().numpy()
    vmin, vmax = 0, max(1, fc_weights.max())

    print("Step 3: Creating and laying out the figure...")
    # 动态调整画图尺寸，避免原型过多导致图片崩溃
    plot_num = min(num_prototypes, 20)  # 建议这里限制最多画前20个，否则内存溢出
    fig = plt.figure(figsize=(15, 0.7 * plot_num))
    gs = gridspec.GridSpec(
        plot_num + 1, 4,
        figure=fig,
        height_ratios=[0.5] + [1] * plot_num,
        width_ratios=[1.2, 2, 2, 4],
        hspace=0.2, wspace=0.15
    )

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

    for p_idx in range(plot_num):
        row_idx = p_idx + 1

        ax_ylabel = fig.add_subplot(gs[row_idx, 0])
        prototype_name = f"W{p_idx}:"
        ax_ylabel.text(0.95, 0.5, prototype_name, ha='right', va='center', fontsize=10, transform=ax_ylabel.transAxes)
        ax_ylabel.axis('off')

        ax_wave1 = fig.add_subplot(gs[row_idx, 1])
        if 'signal_epoch' in best_matches[p_idx]:
            # 移除对 model_stem 的调用参数
            wavelet = get_key_waveform_from_indices(
                best_matches[p_idx]['signal_epoch'],
                best_matches[p_idx]['activation_idx'],
                model_to_access.prototype_kernel_size,
                sample_rate
            )
            time_axis = np.arange(len(wavelet)) / sample_rate
            ax_wave1.plot(time_axis, wavelet, color='black', linewidth=0.7)
        ax_wave1.axis('off')

        ax_wave2 = fig.add_subplot(gs[row_idx, 2])
        reconstructed_composite = reconstructed_prototypes[p_idx]
        time_axis_proto = np.arange(len(reconstructed_composite)) / sample_rate
        ax_wave2.plot(time_axis_proto, reconstructed_composite, color='crimson', linewidth=0.9)
        ax_wave2.axis('off')

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

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('Reds'), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.05, right=0.89, top=0.92, bottom=0.05)
    # 保存结果，防止弹窗阻塞
    plt.savefig('./results/prototype_vis.svg', bbox_inches='tight')
    print("Saved Prototype Visualization to ./results/prototype_vis.svg")
    plt.show() # 如果在服务器上运行，建议注释掉 plt.show()


def plot_mixing_weights_heatmap(model, device):
    print("\n--- Generating Figure 2: Basis Prototype Mixing Matrix ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model
    model_to_access.eval()
    model_to_access.to(device)

    weights = model_to_access.mixing_weights.detach().cpu().numpy()
    num_composite, num_basis = weights.shape

    num_g = model_to_access.num_gabor_basis
    num_f = model_to_access.num_fourier_basis
    num_l = model_to_access.num_learnable_basis

    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(weights, cmap='viridis', aspect='auto', interpolation='nearest')

    ax.set_yticks(np.arange(num_composite))

    # 防止复合原型数量达到几百个时坐标轴标签糊作一团，适当稀疏显示
    if num_composite > 50:
        ax.set_yticks(np.arange(0, num_composite, 10))
        ax.set_yticklabels([f"Composite P{i}" for i in range(0, num_composite, 10)])
    else:
        ax.set_yticklabels([f"Composite P{i}" for i in range(num_composite)])

    x_labels = [f"G{i}" for i in range(num_g)] + \
               [f"F{i}" for i in range(num_f)] + \
               [f"L{i}" for i in range(num_l)]
    ax.set_xticks(np.arange(num_basis))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)

    ax.set_xlabel("Basis Prototypes (G: Gabor, F: Fourier, L: Learnable)", fontsize=12)
    ax.set_ylabel("Composite Prototypes", fontsize=12)

    ax.axvline(x=num_g - 0.5, color='white', linestyle='--', linewidth=2)
    ax.axvline(x=num_g + num_f - 0.5, color='white', linestyle='--', linewidth=2)

    ax.set_title("Mixing Weights: Distinct Prototype Families", fontsize=16, pad=20)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    cbar.set_label('Mixing Weight Value', fontsize=12)

    plt.tight_layout()
    plt.savefig('./results/mixing_weights.svg', bbox_inches='tight')
    print("Saved Mixing Weights to ./results/mixing_weights.svg")
    # plt.show()