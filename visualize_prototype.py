# --- visualize_prototype.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.gridspec as gridspec
import sys
import os


# 绘制原型模板支持程度图
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


# 绘制交叉注意力权重图
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


# 解释模型的推理结果
def explain_single_sample_comprehensive(model, sample_tuple, device, class_names, sample_rate=100,
                                        top_k=3, group_by_type=True, patch_window_sec=3.0,
                                        save_name='single_sample_explanation.svg'):
    """
    单样本综合解释分析可视化：
    修正了激活点定位公式，引入边界平移保护，杜绝片段截取空白问题。
    """
    print(f"\n--- Generating Comprehensive Explanation Figure: {save_name} ---")
    is_parallel = isinstance(model, nn.DataParallel)
    model_to_access = model.module if is_parallel else model
    model_to_access.eval()
    model_to_access.to(device)

    # 1. 解析单样本数据
    eeg_signal, label = sample_tuple
    eeg_tensor = torch.as_tensor(eeg_signal).clone().detach().float().to(device)
    if eeg_tensor.dim() == 2:
        eeg_tensor = eeg_tensor.unsqueeze(0)
    elif eeg_tensor.dim() == 1:
        eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(0)

    # 绕过 batch_size=1 导致的无参数 squeeze() 降维崩溃 Bug
    eeg_tensor = eeg_tensor.repeat(2, 1, 1)

    # 2. 前向传播：获取特征和最近距离索引
    with torch.no_grad():
        logits, min_indices = model_to_access(eeg_tensor, return_indices=True)
        logits = logits[0:1]
        min_indices = min_indices[0:1]

        min_dist = model_to_access.min_distance[0:1]
        similarity = torch.log((min_dist + 1) / (min_dist + 1e-4))
        bn_similarity = model_to_access.bn(similarity)

    pred_class = torch.argmax(logits, dim=1).item()
    true_class = label.item() if isinstance(label, torch.Tensor) else int(label)

    # 3. 计算各个原型对预测结果的贡献度
    fc_weights = model_to_access.fc.weight.data
    class_weights = fc_weights[pred_class]
    contributions = (class_weights * bn_similarity[0]).cpu().numpy()

    # --- 依据原型类别分组提取 ---
    if group_by_type and hasattr(model_to_access, 'proto_splits'):
        n_g, n_f, n_l = model_to_access.proto_splits
        idx_g = np.argmax(contributions[0:n_g])
        idx_f = np.argmax(contributions[n_g:n_g + n_f]) + n_g
        idx_l = np.argmax(contributions[n_g + n_f:]) + n_g + n_f

        top_k_p_indices = [idx_g, idx_f, idx_l]
        group_titles = ['[Gabor]', '[Fourier]', '[Learnable]']
        top_k = 3  # 强制设为3组
    else:
        top_k_p_indices = np.argsort(contributions)[::-1][:top_k]
        group_titles = [f"[Rank {j + 1}]" for j in range(top_k)]

    # 4. 参数准备
    eeg_np = eeg_tensor[0].squeeze().cpu().numpy()
    time_axis_full = np.arange(len(eeg_np)) / sample_rate

    input_len = len(eeg_np)
    latent_len = 256  # 网络压缩后的潜空间步数

    # 重构模型中学习到的复合原型波形
    gabor_k = model_to_access.gabor_basis_bank.get_kernels().squeeze(1)
    fourier_k = model_to_access.fourier_basis_bank.get_kernels().squeeze(1)
    learn_k = model_to_access.learnable_basis_bank.data.squeeze(1)
    all_basis = torch.cat([gabor_k, fourier_k, learn_k], dim=0)
    reconstructed_protos = torch.matmul(model_to_access.mixing_weights.detach(), all_basis).cpu().detach().numpy()

    # 5. 开始综合绘图
    fig = plt.figure(figsize=(16, 4 + 2.5 * top_k))
    gs = gridspec.GridSpec(top_k + 1, 2, height_ratios=[2] + [1] * top_k, hspace=0.6)

    # --- 顶栏：完整的 EEG 输入与标签对照 ---
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(time_axis_full, eeg_np, color='dimgray', linewidth=0.8, label='Original Input EEG')
    title_color = 'darkgreen' if true_class == pred_class else 'darkred'
    title_str = f"Single Sample Evaluation | True Label: [{class_names[true_class]}] | Predicted: [{class_names[pred_class]}]"
    ax_full.set_title(title_str, fontsize=16, weight='bold', color=title_color)
    ax_full.set_xlabel("Time (Seconds)", fontsize=12)
    ax_full.set_ylabel("Amplitude", fontsize=12)
    ax_full.set_xlim(0, input_len / sample_rate)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # --- 下栏：Top K 原型分解 ---
    for i, p_idx in enumerate(top_k_p_indices):
        color = colors[i % len(colors)]

        # ================== 核心修正：避免空数组的边界保护算法 ==================
        act_idx = min_indices[0, p_idx].item()
        # 1. 直接等比例映射定位原信号中心点（不额外加半长避免溢出）
        center_idx = int((act_idx / latent_len) * input_len)

        # 2. 根据指定的窗口秒数计算所需的总点数
        window_pts = int(patch_window_sec * sample_rate)

        # 3. 计算初步的起始和结束点
        start_idx = center_idx - window_pts // 2
        end_idx = start_idx + window_pts

        # 4. 边界滑窗保护：如果撞墙了，向内推挤，保证片段长度始终饱满
        if start_idx < 0:
            start_idx = 0
            end_idx = window_pts
        elif end_idx > input_len:
            end_idx = input_len
            start_idx = input_len - window_pts
        # ====================================================================

        # 在原图中用色块高亮标出激发的片段
        start_time, end_time = start_idx / sample_rate, end_idx / sample_rate
        ax_full.axvspan(start_time, end_time, color=color, alpha=0.35, label=f'{group_titles[i]} Target Region')

        # 子图左侧：被激发的实际脑电波短片段（Patch）
        ax_patch = fig.add_subplot(gs[i + 1, 0])
        patch_wave = eeg_np[start_idx:end_idx]
        time_patch = np.arange(len(patch_wave)) / sample_rate
        ax_patch.plot(time_patch, patch_wave, color=color, linewidth=1.5)
        ax_patch.set_title(
            f"{group_titles[i]} Extracted EEG Patch -> Proto {p_idx} (Contribution: {contributions[p_idx]:.3f})",
            fontsize=11)
        ax_patch.set_xlabel("Time (s)")

        # 子图右侧：学习到的原型模板波形 (Template)
        ax_proto = fig.add_subplot(gs[i + 1, 1])
        proto_wave = reconstructed_protos[p_idx]
        time_proto = np.arange(len(proto_wave)) / sample_rate
        ax_proto.plot(time_proto, proto_wave, color='crimson', linewidth=1.5)
        ax_proto.set_title(f"{group_titles[i]} Learned Template {p_idx}", fontsize=11)
        ax_proto.set_xlabel("Time (s)")

    ax_full.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    os.makedirs('./results', exist_ok=True)
    save_path = os.path.join('./results', save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {save_path}")
    plt.show()
    plt.close(fig)