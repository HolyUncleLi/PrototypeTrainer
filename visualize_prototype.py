# --- explain_academic.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.gridspec as gridspec
import json
import os  # 引入 os 模块
import sys


# =============================================================================
# 1. 核心绘图函数
# =============================================================================

def plot_prototype_activation(activating_waveform, reconstructed_composite, full_signal, start_idx_in_signal,
                              target_str, p_num_str, sample_rate=100, context_len=3000):
    """
    【最终版】为 visualize_prototypes_final 定制的学术绘图函数。
    绘制三联图：激活波形、重构原型、3秒上下文。
    """
    fs = sample_rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Analysis of Prototype {p_num_str} (Best match for a "{target_str}" sample)', fontsize=16, y=1.02)

    # --- 图 1: 实际激活的EEG波形 ---
    wavelet_time_axis = np.arange(len(activating_waveform)) / fs
    axes[0].plot(wavelet_time_axis, activating_waveform, color='dodgerblue')
    axes[0].set_title("1. Activating EEG Waveform")
    axes[0].set_xlabel("Time (s)");
    axes[0].set_ylabel("Amplitude (µV)");
    axes[0].grid(True)
    axes[0].text(0.95, 0.95, f'Duration: {len(activating_waveform) / fs:.2f}s',
                 ha='right', va='top', transform=axes[0].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    # --- 图 2: 重构的复合模板波形 ---
    proto_time_axis = np.arange(len(reconstructed_composite)) / fs
    axes[1].plot(proto_time_axis, reconstructed_composite, color='coral')
    axes[1].set_title("2. Reconstructed Composite Prototype")
    axes[1].set_xlabel("Time (s)");
    axes[1].set_ylabel("Amplitude (a.u.)");
    axes[1].grid(True)

    # 计算波形相似度
    if len(activating_waveform) == len(reconstructed_composite) and len(activating_waveform) > 1:
        corr = np.corrcoef(activating_waveform, reconstructed_composite)[0, 1]
        axes[1].text(0.95, 0.95, f'Corr w/ Waveform: {corr:.3f}',
                     ha='right', va='top', transform=axes[1].transAxes,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))

    # --- 图 3: 在3秒上下文中的位置 ---
    context_start = max(0, start_idx_in_signal - (context_len - len(activating_waveform)) // 2)
    context_end = min(len(full_signal), context_start + context_len)
    context_signal = full_signal[context_start:context_end]
    context_time_axis = np.arange(context_start, context_end) / fs

    highlight_start_rel = start_idx_in_signal - context_start
    highlight_end_rel = highlight_start_rel + len(activating_waveform)

    axes[2].plot(context_time_axis, context_signal, color='gray', alpha=0.7)
    axes[2].plot(context_time_axis[highlight_start_rel:highlight_end_rel],
                 context_signal[highlight_start_rel:highlight_end_rel], color='crimson', linewidth=2.5)
    axes[2].set_title(f"3. Location in a {context_len / fs:.1f}s Context")
    axes[2].set_xlabel("Time within 30s Epoch (s)");
    axes[2].set_ylabel("Amplitude (µV)");
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_prototype_contribution_matrix(contribution_matrix, class_names, top_prototypes,
                                       pred_class_idx, pred_class_name, pred_confidence):
    """
    【最终版】为 explain_single_sample_final 定制的学术矩阵绘图函数。
    绘制贡献度矩阵热力图。
    """
    num_classes, num_prototypes = contribution_matrix.shape
    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(contribution_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    for i in range(num_classes):
        for j in range(num_prototypes):
            val = contribution_matrix[i, j]
            color = "white" if abs(val) > 0.5 * np.max(abs(contribution_matrix)) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(np.arange(num_prototypes));
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels([f"P{i}" for i in range(num_prototypes)], rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Composite Prototypes", fontsize=12);
    ax.set_ylabel("Sleep Stages", fontsize=12)

    ax.get_yticklabels()[pred_class_idx].set_fontweight('bold')
    ax.get_yticklabels()[pred_class_idx].set_color('blue')
    rect = plt.Rectangle((-0.5, pred_class_idx - 0.5), num_prototypes, 1, fill=False, edgecolor='blue', lw=3)
    ax.add_patch(rect)

    for p_idx in top_prototypes:
        rect_col = plt.Rectangle((p_idx - 0.5, -0.5), 1, num_classes, fill=False, edgecolor='orange', lw=2,
                                 linestyle='--')
        ax.add_patch(rect_col)

    title_str = (f"Prototype Contribution Matrix for a Single Sample\n"
                 f"Final Prediction: '{pred_class_name}' (Confidence: {pred_confidence:.2%})")
    ax.set_title(title_str, fontsize=16, pad=20)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Contribution Score (Support/Opposition)', fontsize=12)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. 核心解释函数
# =============================================================================

def get_key_waveform_from_indices(signal_epoch, activation_idx, model_stem, proto_kernel_size_in_feature_space,
                                  sample_rate):
    """
    【最终版】根据模型输出的激活索引，从原始信号中提取关键波形。
    """
    total_stride = 1
    for layer in model_stem:
        if hasattr(layer, 'stride'):
            stride_val = layer.stride
            if isinstance(stride_val, tuple):
                total_stride *= stride_val[0]
            else:
                total_stride *= stride_val

    start_idx_in_signal = activation_idx * total_stride
    proto_len_in_signal = proto_kernel_size_in_feature_space * total_stride
    end_idx_in_signal = start_idx_in_signal + proto_len_in_signal

    signal_np = signal_epoch.squeeze().cpu().numpy()
    end_idx_in_signal = min(end_idx_in_signal, len(signal_np))

    wavelet = signal_np[int(start_idx_in_signal): int(end_idx_in_signal)]
    return wavelet, int(start_idx_in_signal)


def visualize_prototypes_final(model, data_loader, device, class_names, sample_rate=100):
    """
    【最终版】为每个原型找到最佳匹配信号，并调用新的定制化绘图函数。
    """
    print("--- Visualizing Prototypes (Academic Style) ---")
    is_parallel = isinstance(model, nn.DataParallel);
    model_to_access = model.module if is_parallel else model
    model.eval();
    model.to(device)
    num_prototypes = model_to_access.num_composite_prototypes
    best_matches = {i: {'min_dist': float('inf')} for i in range(num_prototypes)}

    print("Step 1: Searching for best matching signal for each prototype...")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)
            batch_min_dists = model_to_access.min_distance
            for p_idx in range(num_prototypes):
                min_val, min_batch_idx = torch.min(batch_min_dists[:, p_idx], dim=0)
                if min_val.item() < best_matches[p_idx]['min_dist']:
                    best_matches[p_idx].update({
                        'min_dist': min_val.item(), 'signal_epoch': inputs[min_batch_idx].cpu(),
                        'label_idx': labels[min_batch_idx].item(),
                        'activation_idx': model_to_access.min_indices[min_batch_idx, p_idx].item()
                    })

    print("Step 2: Reconstructing, extracting key waveforms, and plotting...")
    proto_kernel_size = model_to_access.prototype_kernel_size
    gabor_kernels = model_to_access.gabor_basis_bank.get_kernels().squeeze(1)
    fourier_kernels = model_to_access.fourier_basis_bank.get_kernels().squeeze(1)
    learnable_kernels = model_to_access.learnable_basis_bank.data.squeeze(1)
    all_basis_kernels = torch.cat([gabor_kernels, fourier_kernels, learnable_kernels], dim=0)

    for p_idx, info in best_matches.items():
        if 'signal_epoch' not in info: continue
        print(f"\nAnalyzing Prototype #{p_idx}...")

        wavelet, start_idx = get_key_waveform_from_indices(
            info['signal_epoch'], info['activation_idx'], model_to_access.stem, proto_kernel_size, sample_rate
        )
        mixing_weights = model_to_access.mixing_weights.data[p_idx]
        reconstructed_composite = torch.matmul(mixing_weights, all_basis_kernels).cpu().detach().numpy()

        plot_prototype_activation(
            activating_waveform=wavelet,
            reconstructed_composite=reconstructed_composite,
            full_signal=info['signal_epoch'].squeeze().numpy(),
            start_idx_in_signal=start_idx,
            target_str=class_names[info['label_idx']],
            p_num_str=str(p_idx)
        )


def explain_single_sample_final(model, eeg_sample, device, class_names, sample_rate=100):
    """
    【最终版】计算完整的贡献度矩阵并调用新的矩阵绘图函数。
    """
    print(f"\n--- Deep Explanation for a Single Sample (Academic Style) ---")
    is_parallel = isinstance(model, nn.DataParallel);
    model_to_access = model.module if is_parallel else model
    model.eval();
    model.to(device)
    if eeg_sample.dim() == 2: eeg_sample = eeg_sample.unsqueeze(0)
    eeg_sample = eeg_sample.to(device)

    with torch.no_grad():
        logits = model(eeg_sample)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

        print(f"Predicted Class: '{class_names[pred_idx]}' (Confidence: {probs[0, pred_idx]:.2%})")

        similarity_scores = \
        model_to_access.bn(torch.log((model_to_access.min_distance + 1) / (model_to_access.min_distance + 1e-4)))[0]
        fc_weights = model_to_access.fc.weight.data

        contribution_matrix = fc_weights * similarity_scores.unsqueeze(0)
        contribution_matrix_np = contribution_matrix.cpu().numpy()

        pred_class_contributions = contribution_matrix_np[pred_idx, :]
        top_prototypes_indices = np.argsort(pred_class_contributions)[-3:]

        plot_prototype_contribution_matrix(
            contribution_matrix=contribution_matrix_np,
            class_names=class_names,
            top_prototypes=top_prototypes_indices,
            pred_class_idx=pred_idx,
            pred_class_name=class_names[pred_idx],
            pred_confidence=probs[0, pred_idx].item()
        )


# =============================================================================
# 3. 主程序演示
# =============================================================================
if __name__ == "__main__":
    # 【重要】确保此文件可以找到您的模型定义文件。
    # 假设您的模型文件名为 'protop_cross_final_attention.py' 且在 'models' 文件夹下
    try:
        from models.protop_cross import ProtoPNet
    except ImportError:
        print("错误: 无法导入模型'ProtoPNet'。请确保：")
        print("1. 您的最终模型文件名是 'protop_cross_final_attention.py'。")
        print("2. 该文件位于一个名为 'models' 的子文件夹中。")
        print("3. 'models' 文件夹下有一个 '__init__.py' 文件。")
        sys.exit(1)

    # 加载您的配置文件
    # 【重要】请将此路径修改为您配置文件的实际路径
    config_path = './configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json'
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"警告: 配置文件 '{config_path}' 未找到。将使用一个最小化的虚拟config。")
        config = {'classifier': {'afr_reduced_dim': 128, 'num_classes': 5, 'prototype_num': 20,
                                 'prototype_shape': [20, 128, 15]}}

    # 实例化您最终的模型
    model = ProtoPNet(config)

    # 【可选】加载您已经训练好的模型权重
    # checkpoint_path = 'path/to/your/best_model.pth'
    # try:
    #     model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    #     print(f"成功从 '{checkpoint_path}' 加载预训练模型权重。")
    # except Exception as e:
    #     print(f"无法加载预训练权重, 将使用随机初始化的模型。错误: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"模型已创建并移至 {device}。")

    # 创建一个虚拟的数据加载器用于演示
    dummy_loader = DataLoader(
        TensorDataset(torch.randn(32, 1, 30000), torch.randint(0, 5, (32,))),
        batch_size=16
    )
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("\n" + "=" * 80)
    print("DEMO 1: VISUALIZING PROTOTYPES (生成学术三联图)")
    print("=" * 80)
    visualize_prototypes_final(model, dummy_loader, device, class_names)

    print("\n" + "=" * 80)
    print("DEMO 2: DEEP EXPLAINING A SINGLE PREDICTION (生成学术贡献度矩阵图)")
    print("=" * 80)
    single_sample, _ = dummy_loader.dataset[0]
    explain_single_sample_final(model, single_sample, device, class_names)