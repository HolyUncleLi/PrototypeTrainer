import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from collections import OrderedDict


# =============================================================================
# 1. 辅助工具与模型桩模块 (与之前相同)
# =============================================================================

def fft_visualize_modified(ax, signal, fs):
    """一个独立的FFT可视化函数。"""
    if signal is None or len(signal) < 2:
        ax.set_title("Frequency Spectrum (No Signal)")
        return
    signal = signal - signal.mean()
    n = len(signal)
    Y = np.fft.fft(signal)
    Y_db = 20 * np.log10(np.abs(Y[:n // 2]) * 2 / n)
    freq = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    ax.plot(freq, Y_db)
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, linestyle='--', alpha=0.6)


class PerturbationSignal(Dataset):
    """
    参考您提供的代码：创建一个数据集，其中每个样本都是原始信号被一个滑动的零窗口扰动后的版本。
    """

    def __init__(self, signal, length):
        if isinstance(signal, torch.Tensor):
            signal = signal.squeeze().cpu().numpy()
        if signal.ndim == 1:
            signal = np.expand_dims(signal, 0)
        self.signal = signal
        self.length = length
        self.num_perturbations = self.signal.shape[1] - self.length + 1

    def __len__(self):
        return self.num_perturbations

    def __getitem__(self, idx):
        perturbed_signal = np.copy(self.signal)
        perturbed_signal[:, idx: idx + self.length] = 0
        return torch.from_numpy(perturbed_signal).float(), torch.tensor(idx).long()


# --- 最小化的桩(Stub)模块 ---
class DummyMRCNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, out_dim, kernel_size=39, stride=39, padding=0)

    def forward(self, x): return self.conv1(x)


class DummyTCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(config['classifier']['afr_reduced_dim'], config['classifier']['afr_reduced_dim'], 1)

    def forward(self, x): return self.conv(x)


class GaborFilterBank(nn.Module):
    def __init__(self, num_filters, kernel_size, sample_rate=100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size

    def get_kernels(self): return torch.randn(self.num, 1, self.ks)


class FourierFilterBank(nn.Module):
    def __init__(self, num_filters, kernel_size, sample_rate=100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size

    def get_kernels(self): return torch.randn(self.num, 1, self.ks)


class ProtoPNet(nn.Module):
    # 与之前相同的简化版ProtoPNet定义
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.protop_num = config['gabor_params']['num_filters'] + config['fourier_params']['num_filters']
        self.mrcnn = DummyMRCNN(config['classifier']['afr_reduced_dim'])
        self.conv_features = DummyTCN(config)
        self.gabor = GaborFilterBank(**config['gabor_params'])
        self.fourier = FourierFilterBank(**config['fourier_params'])
        self.fc = nn.Linear(2 * self.protop_num, config['classifier']['num_classes'], bias=False)
        self.distance = self.proportion = self.similarity = self.xfeat = None

    def forward(self, x):
        self.xfeat = self.conv_features(self.mrcnn(x))
        prototype_vectors = torch.cat((self.gabor.get_kernels().to(x.device), self.fourier.get_kernels().to(x.device)),
                                      dim=0)
        xp = F.conv1d(self.xfeat, prototype_vectors)
        x_feat_sq_sum = F.conv1d(self.xfeat.pow(2),
                                 torch.ones(1, self.xfeat.shape[1], 1).to(x.device).repeat(len(prototype_vectors), 1,
                                                                                           1))
        proto_sq_sum = prototype_vectors.pow(2).sum(dim=(1, 2)).view(-1, 1)
        self.distance = F.relu(x_feat_sq_sum - 2 * xp + proto_sq_sum)
        epsilon = 1e-4
        log_dist = torch.log((self.distance + 1) / (self.distance + epsilon))
        self.proportion = torch.mean(log_dist, dim=2)
        self.similarity = torch.log(
            (torch.min(self.distance, dim=-1).values + 1) / (torch.min(self.distance, dim=-1).values + epsilon))
        similarity_sum = torch.cat([self.proportion, self.similarity], dim=1)
        logits = self.fc(similarity_sum)
        return F.log_softmax(logits, dim=-1)


# =============================================================================
# 2. 最终的可视化与解释函数 (已修复内存问题)
# =============================================================================

def find_critical_segment_with_perturbation(model, signal_epoch, p_idx, feature_map_idx, perturb_len, device):
    """
    一个核心辅助函数，对单个信号应用扰动分析来找到关键片段。
    【已修复内存问题】
    """
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model

    # --- 核心修改 1: 大幅降低扰动分析的批次大小 ---
    # 这个值可以根据您的GPU VRAM大小进行调整 (例如 16, 32, 64)
    perturb_batch_size = 32

    pert_dataset = PerturbationSignal(signal_epoch, perturb_len)
    pert_loader = DataLoader(pert_dataset, batch_size=perturb_batch_size, shuffle=False, num_workers=0)

    all_distances = []
    with torch.no_grad():  # 确保在no_grad上下文中
        for perturbed_batch, _ in pert_loader:
            perturbed_batch = perturbed_batch.to(device)
            _ = model(perturbed_batch)

            dist_slice = model_to_access.distance[:, p_idx, feature_map_idx].cpu().numpy()
            all_distances.extend(dist_slice)

            # --- 核心修改 2: 主动清理GPU缓存 ---
            # 删除不再需要的张量，并清空缓存，为下一次迭代释放内存
            del perturbed_batch, dist_slice
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    critical_start_idx = np.argmax(all_distances)
    wavelet = signal_epoch.squeeze().cpu().numpy()[critical_start_idx: critical_start_idx + perturb_len]
    return wavelet, critical_start_idx


def prototype_plot(prototype_wavelet, proto_plot, feature, target_str, p_num_str):
    """参考您提供的绘图函数，进行适配。"""
    fs = 100
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


def visualize_prototypes_with_perturbation(model, data_loader, device, class_names, perturb_sec=3.0, sample_rate=100):
    """为每个原型找到最佳匹配信号，然后用扰动法找到并可视化其关键激活片段。"""
    print("--- Visualizing Prototypes with Perturbation Analysis ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model

    model.eval();
    model.to(device)

    num_prototypes = model_to_access.protop_num
    best_matches = {i: {'min_dist': float('inf')} for i in range(num_prototypes)}

    print("Step 1: Searching for best matching signal for each prototype...")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)
            distances = model_to_access.distance
            batch_min_dists, batch_argmin_dists = torch.min(distances, dim=2)
            for p_idx in range(num_prototypes):
                min_val, min_batch_idx = torch.min(batch_min_dists[:, p_idx], dim=0)
                if min_val.item() < best_matches[p_idx]['min_dist']:
                    best_matches[p_idx].update({
                        'min_dist': min_val.item(), 'signal_epoch': inputs[min_batch_idx].cpu(),
                        'label_idx': labels[min_batch_idx].item(),
                        'feature_map_idx': batch_argmin_dists[min_batch_idx, p_idx].item()
                    })

    print("Step 2: Applying perturbation analysis and plotting...")
    perturb_len = int(perturb_sec * sample_rate)
    for p_idx, info in best_matches.items():
        if 'signal_epoch' not in info: continue
        print(f"\nAnalyzing Prototype #{p_idx}...")
        wavelet, start_idx = find_critical_segment_with_perturbation(
            model, info['signal_epoch'], p_idx, info['feature_map_idx'], perturb_len, device
        )
        full_signal = info['signal_epoch'].squeeze().numpy()
        plot_context = np.full_like(full_signal, np.nan)
        plot_context[start_idx: start_idx + len(wavelet)] = wavelet
        prototype_plot(wavelet, plot_context, full_signal, class_names[info['label_idx']], str(p_idx))


def explain_single_sample_with_perturbation(model, eeg_sample, device, class_names, perturb_sec=3.0, sample_rate=100):
    """对单个样本进行分类，并用扰动法详细解释其分类依据。"""
    print(f"\n--- Explanation for a Single Sample with Perturbation Analysis ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model

    model.eval();
    model.to(device)
    eeg_sample = eeg_sample.to(device)

    with torch.no_grad():
        log_probs = model(eeg_sample)
        probs = torch.exp(log_probs)
        pred_idx = torch.argmax(probs, dim=1).item()
        print(f"Predicted Class: '{class_names[pred_idx]}' (Confidence: {probs[0, pred_idx]:.2%})")
        prop_sim = model_to_access.proportion[0];
        wave_sim = model_to_access.similarity[0]
        fc_weights = model_to_access.fc.weight.data
        contributions = []
        prop_weights = fc_weights[pred_idx, :model_to_access.protop_num]
        wave_weights = fc_weights[pred_idx, model_to_access.protop_num:]
        for p_idx in range(model_to_access.protop_num):
            score = (prop_weights[p_idx] * prop_sim[p_idx]) + (wave_weights[p_idx] * wave_sim[p_idx])
            contributions.append((p_idx, score.item()))
        contributions.sort(key=lambda x: x[1], reverse=True)
        print("\nPrototype Contribution Analysis (ranked by influence):")
        for i, (p_idx, score) in enumerate(contributions):
            print(
                f"{i + 1:<3}. Prototype #{p_idx:<3} | Contribution Score: {score:8.2f} {'<-- Most Influential' if i == 0 else ''}")

        most_influential_p_idx = contributions[0][0]
        distances_for_sample = model_to_access.distance[0]
        _, feature_map_idx = torch.min(distances_for_sample[most_influential_p_idx], dim=0)

        perturb_len = int(perturb_sec * sample_rate)
        wavelet, start_idx = find_critical_segment_with_perturbation(
            model, eeg_sample, most_influential_p_idx, feature_map_idx.item(), perturb_len, device
        )
        full_signal = eeg_sample.squeeze().cpu().numpy()
        plot_context = np.full_like(full_signal, np.nan)
        plot_context[start_idx: start_idx + len(wavelet)] = wavelet
        prototype_plot(wavelet, plot_context, full_signal, f"Predicted: {class_names[pred_idx]}",
                       f"#{most_influential_p_idx} (Most Influential)")


'''
# =============================================================================
# 3. 主程序演示
# =============================================================================
if __name__ == "__main__":
    dummy_config = {
        'classifier': {'afr_reduced_dim': 32, 'num_classes': 5},
        'gabor_params': {'num_filters': 5, 'kernel_size': 27},
        'fourier_params': {'num_filters': 5, 'kernel_size': 27},
    }
    model = ProtoPNet(dummy_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Dummy model created on {device}.")

    dummy_loader = DataLoader(
        TensorDataset(torch.randn(16, 1, 30000), torch.randint(0, 5, (16,))),
        batch_size=8
    )
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    print("\n" + "=" * 80)
    print("DEMO 1: VISUALIZING PROTOTYPES (now memory-safe)")
    print("=" * 80)
    visualize_prototypes_with_perturbation(model, dummy_loader, device, class_names)

    print("\n" + "=" * 80)
    print("DEMO 2: EXPLAINING A SINGLE PREDICTION (now memory-safe)")
    print("=" * 80)
    single_sample, _ = dummy_loader.dataset[0]
    explain_single_sample_with_perturbation(model, single_sample.unsqueeze(0), device, class_names)
'''