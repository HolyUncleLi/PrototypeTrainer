# --- explain_final_model_deep_FIXED.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.gridspec as gridspec


# =============================================================================
# 1. 辅助工具 (FFT可视化)
# =============================================================================

def fft_visualize_modified(ax, signal, fs, title="Frequency Spectrum"):
    """一个独立的FFT可视化函数。"""
    if signal is None or len(signal) < 2:
        ax.set_title(title + " (No Signal)")
        return
    signal = signal - signal.mean()
    n = len(signal)
    Y = np.fft.fft(signal)
    Y_db = 20 * np.log10(np.abs(Y[:n // 2]) * 2 / n)
    freq = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    ax.plot(freq, Y_db)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, linestyle='--', alpha=0.6)


# =============================================================================
# 2. 核心可视化与解释函数 (已升级)
# =============================================================================

def prototype_plot_final(prototype_wavelet, proto_plot, feature, target_str, p_num_str, sample_rate=100):
    """
    【已升级】多功能绘图函数。
    - 如果 prototype_wavelet 是一个 NumPy 数组, 执行旧的简单绘图。
    - 如果 prototype_wavelet 是一个字典, 执行新的深度解释性绘图。
    """
    if isinstance(prototype_wavelet, dict):
        data = prototype_wavelet;
        fs = sample_rate
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Deep Explanation for Prototype {p_num_str} activating for a "{target_str}" sample', fontsize=20,
                     y=0.98)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0]);
        activating_waveform = data['activating_waveform']
        time_axis_wave = np.arange(len(activating_waveform)) / fs
        ax1.plot(time_axis_wave, activating_waveform);
        ax1.set_title("1. Actual Activating EEG Waveform")
        ax1.set_xlabel("Time (s)");
        ax1.set_ylabel("Amplitude");
        ax1.grid(True)
        ax2 = fig.add_subplot(gs[0, 1]);
        fft_visualize_modified(ax2, activating_waveform, fs, title="2. Waveform's Frequency Spectrum")
        ax3 = fig.add_subplot(gs[0, 2]);
        full_time_axis = np.arange(len(feature)) / fs
        ax3.plot(full_time_axis, np.squeeze(feature), alpha=0.5);
        ax3.plot(full_time_axis, proto_plot, 'r', linewidth=2)
        ax3.set_title("3. Context in 30s Epoch");
        ax3.set_yticks([]);
        ax3.set_xlabel("Time (s)")
        ax4 = fig.add_subplot(gs[1, 0]);
        reconstructed_composite = data['reconstructed_composite']
        time_axis_proto = np.arange(len(reconstructed_composite)) / fs
        ax4.plot(time_axis_proto, reconstructed_composite);
        ax4.set_title(f"4. Reconstructed Composite Prototype")
        ax4.set_xlabel("Time (s)");
        ax4.set_ylabel("Amplitude");
        ax4.grid(True)
        ax5 = fig.add_subplot(gs[1, 1:]);
        ax5.set_title("5. Basis Prototype Composition");
        ax5.axis('off')
        info_text = "This composite prototype is mainly built from:\n\n"
        for i, info in enumerate(data['top_basis_info']):
            info_text += (
                f"{i + 1}. Type: {info['type']:>9s} | Basis Index: #{info['index']:<3} | Mixing Weight: {info['weight']:.3f}\n")
        ax5.text(0.0, 0.95, info_text, fontsize=12, va='top', fontfamily='monospace')
        for i, info in enumerate(data['top_basis_info']):
            if i >= 3: break
            ax_basis = fig.add_subplot(gs[2, i]);
            basis_waveform = data['top_basis_waveforms'][i]
            ax_basis.plot(time_axis_proto, basis_waveform)
            ax_basis.set_title(f"6.{i + 1} Top Basis: {info['type']} #{info['index']} (Weight: {info['weight']:.3f})")
            ax_basis.set_xlabel("Time (s)");
            ax_basis.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]);
        plt.show()
        return

    fs = sample_rate;
    fig, ax = plt.subplots(1, 3, figsize=(24, 4))
    fig.suptitle(f'Prototype {p_num_str} activating for a "{target_str}" sample', fontsize=16)
    fft_visualize_modified(ax[0], np.squeeze(prototype_wavelet), fs=fs)
    ax[1].plot(np.arange(len(prototype_wavelet)) / fs, np.squeeze(prototype_wavelet))
    ax[1].set_title(f"Critical Waveform ({len(prototype_wavelet) / fs:.2f}s)")
    ax[1].set_xlabel("Time (s)");
    ax[1].set_ylabel("Amplitude");
    ax[1].grid(True)
    ax[2].plot(np.arange(len(feature)) / fs, np.squeeze(feature), alpha=0.5)
    ax[2].plot(np.arange(len(proto_plot)) / fs, proto_plot, 'r', linewidth=2)
    ax[2].set_title("Context in 30s Epoch");
    ax[2].set_yticks([]);
    ax[2].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]);
    plt.show()


def get_key_waveform_from_indices(signal_epoch, activation_idx, model_stem, proto_kernel_size_in_feature_space,
                                  sample_rate):
    """根据模型输出的激活索引，直接从原始信号中提取关键波形。"""
    total_stride = 1
    for layer in model_stem:
        # ========== 【核心修正】 ==========
        # 增加一个健壮的检查来处理 stride 属性
        if hasattr(layer, 'stride'):
            stride_val = layer.stride
            # 检查 stride_val 是元组还是整数
            if isinstance(stride_val, tuple):
                total_stride *= stride_val[0]
            else:  # 如果是整数
                total_stride *= stride_val
        # ==================================

    start_idx_in_signal = activation_idx * total_stride
    proto_len_in_signal = proto_kernel_size_in_feature_space * total_stride
    end_idx_in_signal = start_idx_in_signal + proto_len_in_signal
    signal_np = signal_epoch.squeeze().cpu().numpy()

    # 增加边界检查，防止索引越界
    end_idx_in_signal = min(end_idx_in_signal, len(signal_np))

    wavelet = signal_np[int(start_idx_in_signal): int(end_idx_in_signal)]
    return wavelet, int(start_idx_in_signal)


def visualize_prototypes_final(model, data_loader, device, class_names, sample_rate=100):
    """为每个原型找到最佳匹配信号，并可视化其激活片段（调用旧的绘图逻辑）。"""
    print("--- Visualizing Prototypes (Attention-based, No Perturbation) ---")
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
    print("Step 2: Extracting key waveforms and plotting...")
    proto_kernel_size = model_to_access.prototype_kernel_size
    for p_idx, info in best_matches.items():
        if 'signal_epoch' not in info: continue
        print(f"\nAnalyzing Prototype #{p_idx}...")
        wavelet, start_idx = get_key_waveform_from_indices(
            info['signal_epoch'], info['activation_idx'], model_to_access.stem, proto_kernel_size, sample_rate
        )
        full_signal = info['signal_epoch'].squeeze().numpy()
        plot_context = np.full_like(full_signal, np.nan)
        plot_context[start_idx: start_idx + len(wavelet)] = wavelet
        prototype_plot_final(wavelet, plot_context, full_signal, class_names[info['label_idx']], str(p_idx))


def explain_single_sample_final(model, eeg_sample, device, class_names, sample_rate=100):
    """对单个样本进行分类，并生成包含深度解释信息的字典，调用新的绘图逻辑。"""
    print(f"\n--- Deep Explanation for a Single Sample (Attention-based) ---")
    is_parallel = isinstance(model, nn.DataParallel);
    model_to_access = model.module if is_parallel else model
    model.eval();
    model.to(device)
    if eeg_sample.dim() == 3: eeg_sample = eeg_sample.unsqueeze(0)
    eeg_sample = eeg_sample.to(device)
    with torch.no_grad():
        logits = model(eeg_sample)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        print(f"Predicted Class: '{class_names[pred_idx]}' (Confidence: {probs[0, pred_idx]:.2%})")
        similarity_scores = \
        model_to_access.bn(torch.log((model_to_access.min_distance + 1) / (model_to_access.min_distance + 1e-4)))[0]
        fc_weights = model_to_access.fc.weight.data[pred_idx]
        contributions = []
        for p_idx in range(model_to_access.num_composite_prototypes):
            score = fc_weights[p_idx] * similarity_scores[p_idx]
            contributions.append((p_idx, score.item()))
        contributions.sort(key=lambda x: x[1], reverse=True)
        print("\nPrototype Contribution Analysis (ranked by influence):")
        for i, (p_idx, score) in enumerate(contributions):
            print(
                f"{i + 1:<3}. Prototype #{p_idx:<3} | Contribution Score: {score:8.4f} {'<-- Most Influential' if i == 0 else ''}")
        most_influential_p_idx = contributions[0][0]
        print(f"\nDeconstructing the most influential prototype (#{most_influential_p_idx})...")
        mixing_weights = model_to_access.mixing_weights.data[most_influential_p_idx]
        gabor_kernels = model_to_access.gabor_basis_bank.get_kernels().squeeze(1)
        fourier_kernels = model_to_access.fourier_basis_bank.get_kernels().squeeze(1)
        learnable_kernels = model_to_access.learnable_basis_bank.data.squeeze(1)
        all_basis_kernels = torch.cat([gabor_kernels, fourier_kernels, learnable_kernels], dim=0)
        top_k = 3
        top_weights, top_indices = torch.topk(mixing_weights, k=top_k)
        top_basis_info = [];
        top_basis_waveforms = []
        num_g = model_to_access.num_gabor_basis;
        num_f = model_to_access.num_fourier_basis
        for i in range(top_k):
            basis_idx = top_indices[i].item();
            weight = top_weights[i].item()
            waveform = all_basis_kernels[basis_idx].cpu().numpy();
            top_basis_waveforms.append(waveform)
            if basis_idx < num_g:
                b_type = "Gabor"; b_orig_idx = basis_idx
            elif basis_idx < num_g + num_f:
                b_type = "Fourier"; b_orig_idx = basis_idx - num_g
            else:
                b_type = "Learnable"; b_orig_idx = basis_idx - num_g - num_f
            top_basis_info.append({'type': b_type, 'index': b_orig_idx, 'weight': weight})
        reconstructed_composite = torch.matmul(mixing_weights, all_basis_kernels).cpu().numpy()
        activation_idx = model_to_access.min_indices[0, most_influential_p_idx].item()
        proto_kernel_size = model_to_access.prototype_kernel_size
        wavelet, start_idx = get_key_waveform_from_indices(
            eeg_sample, activation_idx, model_to_access.stem, proto_kernel_size, sample_rate)
        full_signal = eeg_sample.squeeze().cpu().numpy()
        plot_context = np.full_like(full_signal, np.nan)
        plot_context[start_idx: start_idx + len(wavelet)] = wavelet
        plot_data = {
            'activating_waveform': wavelet, 'reconstructed_composite': reconstructed_composite,
            'top_basis_waveforms': top_basis_waveforms, 'top_basis_info': top_basis_info
        }
        prototype_plot_final(plot_data, plot_context, full_signal,
                             f"Predicted: {class_names[pred_idx]}",
                             f"#{most_influential_p_idx} (Most Influential)")


# =============================================================================
# 5. 主程序演示
# =============================================================================
if __name__ == "__main__":
    # 确保此文件与您的模型文件在同一目录，或模型文件在Python路径中
    from models.protop_cross import ProtoPNet
    import json

    config_path = './SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json'
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{config_path}' 未找到。请确保路径正确。")
        config = {'classifier': {'afr_reduced_dim': 128, 'num_classes': 5, 'prototype_num': 20,
                                 'prototype_shape': [20, 128, 15]}}

    model = ProtoPNet(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Final model created on {device}.")

    dummy_loader = DataLoader(
        TensorDataset(torch.randn(32, 1, 30000), torch.randint(0, 5, (32,))),
        batch_size=16
    )
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("\n" + "=" * 80)
    print("DEMO 1: VISUALIZING PROTOTYPES (调用旧的绘图逻辑)")
    print("=" * 80)
    visualize_prototypes_final(model, dummy_loader, device, class_names)

    print("\n" + "=" * 80)
    print("DEMO 2: DEEP EXPLAINING A SINGLE PREDICTION (调用新的绘图逻辑)")
    print("=" * 80)
    single_sample, _ = dummy_loader.dataset[0]
    explain_single_sample_final(model, single_sample, device, class_names)