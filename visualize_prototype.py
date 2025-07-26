import torch
import numpy as np
import matplotlib.pyplot as plt


def fft_visualize_modified(ax, signal, fs):
    """一个独立的FFT可视化函数，用于在新函数中调用。"""
    if signal is None or len(signal) == 0:
        return
    # 信号中心化
    signal = signal - signal.mean()
    length = len(signal) // 2
    if length == 0: return  # 防止信号太短

    # 计算FFT并转换为dB
    fft_data = 20 * (np.log10(np.abs(2 * np.fft.fft(signal) / len(signal))))[1:length]
    fft_x = np.fft.fftfreq(len(signal), 1 / fs)[1:length]

    ax.plot(fft_x, fft_data)
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, linestyle='--', alpha=0.6)


def visualize_filters_via_data(model, data_loader, device, sample_rate=100):
    """
    通过在数据中寻找最佳匹配来可视化Gabor和Fourier滤波器。

    Args:
        model: 训练好的ProtoPNet模型实例（未被DataParallel包装的）。
        data_loader: 用于搜索的PyTorch DataLoader（例如，训练集或测试集）。
        device: 'cuda' or 'cpu'。
        sample_rate (int): 信号的采样率。
    """
    print("--- Visualizing Filters via Best Data Matches ---")

    model.eval()
    model.to(device)

    num_gabor_filters = model.module.gabor.num
    num_fourier_filters = model.module.fourier.num
    num_total_prototypes = num_gabor_filters + num_fourier_filters

    # 初始化用于存储最佳匹配信息的字典
    best_matches = {
        i: {'min_dist': float('inf'), 'signal_epoch': None, 'patch_idx': -1, 'patch': None}
        for i in range(num_total_prototypes)
    }

    print("Step 1: Searching for best matching signal patches...")
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)

            # 关键：从模型中获取距离矩阵和原始卷积特征
            # 注意：如果模型被nn.DataParallel包装，需要使用 model.module
            # 这里假设传入的是未包装的模型
            distances = model.module.distance  # Shape: [B, P, K]
            conv_features = model.module.xfeat  # Shape: [B, C, K]

            # 在批次内寻找每个原型的最小距离
            batch_min_dists, batch_argmin_dists = torch.min(distances, dim=2)  # Shape: [B, P]

            for p_idx in range(num_total_prototypes):
                # 找到批次中原型p的最小距离
                min_val_in_batch, min_batch_idx = torch.min(batch_min_dists[:, p_idx], dim=0)

                if min_val_in_batch.item() < best_matches[p_idx]['min_dist']:
                    # 如果找到了一个新的全局最小距离，则更新信息
                    best_matches[p_idx]['min_dist'] = min_val_in_batch.item()

                    # 记录对应的原始30秒EEG信号
                    # .squeeze(0) 是因为我们现在处理单个样本
                    best_matches[p_idx]['signal_epoch'] = inputs[min_batch_idx].squeeze(0).cpu().numpy()

                    # 记录这个最佳匹配在卷积特征图中的位置

                    patch_idx_in_feature_map = batch_argmin_dists[min_batch_idx, p_idx].item()
                    best_matches[p_idx]['patch_idx'] = patch_idx_in_feature_map

    print("Step 2: Extracting and plotting the best patches...")

    # 从卷积特征和原始信号中提取波形
    # 注意：这是一个近似。卷积后的索引到原始信号索引的映射依赖于特征提取网络的下采样率。
    # 为了简化，我们假设可以从原始信号的对应位置提取，这需要根据您的MRCNN结构进行精确计算。
    # 这里的实现是一个通用的近似。

    # 假设特征提取网络总下采样率为 R
    # R = Stride1 * Stride2 * ...
    # 这里我们先用一个估算值，您需要根据MRCNN中的MaxPool和stride计算精确值
    # 从MRCNN代码看，下采样很复杂，我们先用一个近似值
    APPROX_DOWNSAMPLE_RATIO = 30  # 这是一个需要您根据模型精确调整的超参数

    # 绘制 Gabor 滤波器找到的波形
    print("\n--- Gabor Filter Matches ---")
    ks = model.module.gabor.ks  # 滤波器核的大小
    for i in range(num_gabor_filters):
        info = best_matches[i]
        if info['signal_epoch'] is None:
            print(f"Gabor Filter {i}: No match found.")
            continue

        # 计算在原始信号中的起始位置
        start_idx = info['patch_idx'] * APPROX_DOWNSAMPLE_RATIO
        end_idx = start_idx + ks

        # 提取关键波形
        wavelet = info['signal_epoch'][start_idx:end_idx]

        # 准备用于绘图的完整信号和高亮部分
        full_signal = info['signal_epoch']
        highlight_plot = np.full_like(full_signal, np.nan)
        highlight_plot[start_idx:end_idx] = wavelet
        print(start_idx, end_idx, end_idx - start_idx)

        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        fig.suptitle(f"Gabor Filter {i + 1} - Best Match (Distance: {info['min_dist']:.2f})", fontsize=16)

        fft_visualize_modified(axes[0], wavelet, fs=sample_rate)

        axes[1].plot(np.arange(len(wavelet)) / sample_rate, wavelet)
        axes[1].set_title("Time-Domain Waveform")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, linestyle='--', alpha=0.6)

        axes[2].plot(np.arange(len(full_signal)) / sample_rate, full_signal, alpha=0.5, label='Original EEG')
        axes[2].plot(np.arange(len(highlight_plot)) / sample_rate, highlight_plot, color='red', linewidth=2,
                     label=f'Best Match Patch')
        axes[2].set_title("Context in 30s Epoch")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # 绘制 Fourier 滤波器找到的波形
    print("\n--- Fourier Filter Matches ---")
    ks = model.module.fourier.ks
    for i in range(num_fourier_filters):
        p_idx = i + num_gabor_filters  # 傅里叶滤波器在原型列表中的索引
        info = best_matches[p_idx]
        if info['signal_epoch'] is None:
            print(f"Fourier Filter {i}: No match found.")
            continue

        start_idx = info['patch_idx'] * APPROX_DOWNSAMPLE_RATIO
        end_idx = start_idx + ks
        wavelet = info['signal_epoch'][start_idx:end_idx]

        full_signal = info['signal_epoch']
        highlight_plot = np.full_like(full_signal, np.nan)
        highlight_plot[start_idx:end_idx] = wavelet

        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        fig.suptitle(f"Fourier Filter {i + 1} - Best Match (Distance: {info['min_dist']:.2f})", fontsize=16)
        fft_visualize_modified(axes[0], wavelet, fs=sample_rate)
        axes[1].plot(np.arange(len(wavelet)) / sample_rate, wavelet)
        axes[1].set_title("Time-Domain Waveform")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[2].plot(np.arange(len(full_signal)) / sample_rate, full_signal, alpha=0.5, label='Original EEG')
        axes[2].plot(np.arange(len(highlight_plot)) / sample_rate, highlight_plot, color='red', linewidth=2,
                     label=f'Best Match Patch')
        axes[2].set_title("Context in 30s Epoch")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- 如何使用 ---
# 假设您已经有了训练好的模型和数据加载器
#
# with open(args.config) as config_file:
#     config = json.load(config_file)
#
# # 注意：这里传入的模型应该是未被 nn.DataParallel 包装的
# # 如果你加载的 state_dict 是来自 DataParallel 模型，需要先处理
# model = ProtoPNet(config)
# state_dict = torch.load('path/to/your/model.pth')
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # 移除 `module.` 前缀
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)

# # 准备数据加载器
# test_dataset = EEGDataLoader(config, fold=3, set='test')
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# visualize_filters_via_data(model, test_loader, device)