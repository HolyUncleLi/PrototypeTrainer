# --- models/protop_cross_v4.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# 1. 基础组件: 物理 Gabor 层 & 无损折叠
# ====================================================================

class LearnableGaborConv1d(nn.Module):
    """
    物理层: 加入 stride 参数。
    对于睡眠脑电，100Hz 采样率下，使用 stride=2 或 4 仍然能完美保留 Delta/Theta/Alpha/Beta/Spindle 特征。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, sample_rate=100.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.mu_f = nn.Parameter(torch.rand(out_channels) * 30.0 + 0.5)
        self.sigma = nn.Parameter(torch.ones(out_channels) * 10.0)

        t = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) / sample_rate
        self.register_buffer('t', t)

    def get_filter(self):
        t = self.t.view(1, 1, -1)
        mu_f = self.mu_f.view(-1, 1, 1)
        sigma = self.sigma.view(-1, 1, 1)
        envelope = torch.exp(-0.5 * (t ** 2) / (sigma ** 2))
        carrier_cos = torch.cos(2 * math.pi * mu_f * t)
        carrier_sin = torch.sin(2 * math.pi * mu_f * t)
        return envelope * carrier_cos, envelope * carrier_sin

    def forward(self, x):
        w_real, w_imag = self.get_filter()
        out_real = F.conv1d(x, w_real, padding=self.padding, stride=self.stride)
        out_imag = F.conv1d(x, w_imag, padding=self.padding, stride=self.stride)
        magnitude = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)
        return magnitude, out_real


class TemporalFolder(nn.Module):
    """
    无损时间折叠 (类似 PixelUnshuffle)。
    将 [B, C, L] -> [B, C*r, L/r]
    既减小了显存压力 (L变小)，又增加了特征维度 (C变大)，为增加参数量提供了空间。
    """

    def __init__(self, downscale_factor):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x):
        b, c, l = x.size()
        # 补齐长度以防无法整除
        if l % self.r != 0:
            pad = self.r - (l % self.r)
            x = F.pad(x, (0, pad))
            l = x.size(2)

        # [B, C, L/r, r] -> [B, C, r, L/r] -> [B, C*r, L/r]
        x = x.view(b, c, l // self.r, self.r)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(b, c * self.r, l // self.r)
        return x


# ====================================================================
# 2. 高级特征流 (High-Capacity Streams)
# ====================================================================

class HighCapSemanticStream(nn.Module):
    """
    高容量语义流。
    由于输入长度被折叠变短了，我们可以大幅增加通道数 (channels)，
    从而显著增加模型的参数量 (Params)，提高拟合能力，同时显存占用可控。
    """

    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Expand
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.MaxPool1d(2),

            # Layer 2: Deep Processing (High Parameters here!)
            # 256 input * 512 output * 5 kernel = ~650k Params (这里参数量就上来了)
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),

            # Layer 3: Bottleneck
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class FoldedMorphologicalStream(nn.Module):
    """
    折叠形态流。
    在折叠后的特征空间上操作，保留细节但计算高效。
    """

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        # in_channels 已经是折叠后的 (e.g., 64*4 = 256)
        self.net = nn.Sequential(
            # Dilated Conv 1
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            # Dilated Conv 2
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HolographicFusion(nn.Module):
    """
    全息融合：将压缩的语义信息注入到高分辨率的形态特征中。
    解决“既要细节又要分类精度”的矛盾。
    """

    def __init__(self, high_res_dim, low_res_dim):
        super().__init__()
        self.scale_factor = 16  # 对应 SemanticStream 的两次 Pool(4) -> 16倍
        self.project = nn.Conv1d(low_res_dim, high_res_dim, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(high_res_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, high_res, low_res):
        # high_res: [B, C1, L]
        # low_res:  [B, C2, L/16]

        # 1. 上采样语义特征对齐时间轴
        low_res_upsampled = F.interpolate(low_res, size=high_res.shape[-1], mode='linear', align_corners=True)
        low_res_proj = self.project(low_res_upsampled)

        # 2. 门控融合
        concat = torch.cat([high_res, low_res_proj], dim=1)
        mask = self.gate(concat)

        # 输出 = 形态特征 * (1 + 语义加权)
        # 这样保证了基础特征仍然是形态学的，适合原型匹配
        out = high_res * (1 + mask * low_res_proj)
        return out


# ====================================================================
# 3. 骨干网络 V4
# ====================================================================

class LGWDS_Net_V4(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        # 1. 物理层: 增加 Stride=2，直接减少 50% 显存
        self.gabor_layer = LearnableGaborConv1d(1, 64, kernel_size=63, stride=2)
        # Output: [B, 64, 15000]

        # 2. 折叠层: Factor=4
        # 将 15000 长度折叠为 3750。通道数变为 64*4 = 256
        self.folder = TemporalFolder(downscale_factor=4)

        # 3. 双流 (输入通道均为 256)
        # 语义流: 增加大量参数
        self.semantic_stream = HighCapSemanticStream(in_channels=256, hidden_dim=256)
        # 形态流: 轻量级保真
        self.morph_stream = FoldedMorphologicalStream(in_channels=256, hidden_dim=128)

        # 4. 融合与输出
        # 语义流输出: [B, 256, 1875] (经过了一次 pool2)
        # 形态流输出: [B, 128, 3750]
        # 我们将语义流上采样对齐
        self.final_fusion = nn.Conv1d(128 + 256, out_dim, kernel_size=1)

        # 激进的池化: 3750 -> ~200
        # 这一步非常重要，它确保进入 Attention 的序列长度 L 很小
        # 从而避免 Attention 计算时的显存爆炸
        self.final_pool = nn.AdaptiveAvgPool1d(256)

    def forward(self, x):
        # x: [B, 1, 30000]

        # 1. Gabor [B, 64, 15000]
        mag, raw_real = self.gabor_layer(x)

        # 2. Folding [B, 256, 3750]
        mag_fold = self.folder(mag)
        raw_fold = self.folder(raw_real)

        # 3. Streams
        sem = self.semantic_stream(mag_fold)  # [B, 256, 1875]
        morph = self.morph_stream(raw_fold)  # [B, 128, 3750]

        # 4. Align & Fuse
        sem_up = F.interpolate(sem, size=morph.shape[-1], mode='nearest')
        cat = torch.cat([morph, sem_up], dim=1)  # [B, 384, 3750]

        out = self.final_fusion(cat)  # [B, out_dim, 3750]
        out = self.final_pool(out)  # [B, out_dim, 256] -> 显存安全区

        return out


# ====================================================================
# 4. 辅助模块 (包含内存优化的 Attention)
# ====================================================================

# GaborFilterBank, FourierFilterBank 保持不变 (请复制之前的代码)
class GaborFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A, self.mu, self.sigma = [nn.Parameter(p) for p in
                                       [torch.ones(self.num), torch.zeros(self.num), torch.ones(self.num) * 0.1]]
        self.f = nn.Parameter(torch.linspace(1.0, 40.0, num_filters) + torch.randn(num_filters) * 0.1)
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        A, mu, sigma, f, phi = [p.view(-1, 1, 1) for p in
                                [self.A, self.mu, self.sigma.abs() + 1e-4, self.f.clamp(0.1, 50.0), self.phi]]
        gauss = torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        sinus = torch.cos(2 * torch.pi * f * t + phi)
        return A * gauss * sinus


class FourierFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A = nn.Parameter(torch.ones(self.num));
        self.f = nn.Parameter(torch.linspace(1.0, 40.0, num_filters) + torch.randn(num_filters) * 0.5)
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        A, f, phi = [p.view(-1, 1, 1) for p in [self.A, self.f.clamp(0.1, 50.0), self.phi]]
        return A * torch.cos(2 * torch.pi * f * t + phi)


class MultiLatentSpaceSimilarity(nn.Module):
    """ 内存优化版 + 维度修复版 """

    def __init__(self, dim, splits, heads=4, dim_head=32):
        super().__init__()
        self.splits = splits
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.q_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])

    def forward(self, x, prototypes):
        batch_size, C, seq_len = x.shape
        _, _, proto_len = prototypes.shape
        x_perm = x.permute(0, 2, 1)
        proto_groups = torch.split(prototypes, self.splits, dim=0)
        all_distances = []
        all_indices = []

        for i, p_group in enumerate(proto_groups):
            num_p_group = p_group.shape[0]
            if num_p_group == 0: continue

            p_perm = p_group.permute(0, 2, 1)
            replicated_p = p_perm.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Query Projection & Reduction
            q = self.q_projs[i](replicated_p)
            q = q.view(batch_size, num_p_group, proto_len, self.heads, self.dim_head)
            q = q.mean(dim=2)  # Aggregate kernel time dim
            q = q.permute(0, 2, 1, 3)  # [B, H, P, D]

            # Key/Value Projection
            k = self.k_projs[i](x_perm)
            v = self.v_projs[i](x_perm)
            k = k.view(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)

            # Attention (Matrix Mul - Low Memory)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

            # Distance
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_p_group, -1)
            original_q_projected = self.q_projs[i](replicated_p).mean(dim=2)
            dist = F.mse_loss(original_q_projected, out, reduction='none').mean(dim=-1)

            # Indices
            heatmap = attn.mean(dim=1)
            indices = heatmap.argmax(dim=-1)
            all_distances.append(dist)
            all_indices.append(indices)

        final_distances = torch.cat(all_distances, dim=1)
        final_indices = torch.cat(all_indices, dim=1)
        return final_distances, final_indices


# ====================================================================
# 5. ProtoPNet V4 (Main Model)
# ====================================================================

class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.prototype_kernel_size = self.cfg['classifier']['prototype_shape'][2]

        total_prototypes = self.cfg['classifier']['prototype_num']
        n_g = total_prototypes // 3
        n_f = total_prototypes // 3
        n_l = total_prototypes - n_g - n_f
        self.proto_splits = [n_g, n_f, n_l]
        self.num_composite_prototypes = total_prototypes
        num_classes = self.cfg['classifier']['num_classes']

        # [NEW] V4 Feature Extractor
        self.feature_extractor = LGWDS_Net_V4(out_dim=afr_reduced_cnn_size)

        self.similarity_calculator = MultiLatentSpaceSimilarity(
            dim=afr_reduced_cnn_size,
            splits=self.proto_splits,
            heads=4,
            dim_head=32
        )

        # Basis Banks
        self.num_gabor_basis, self.num_fourier_basis = 20, 20
        self.gabor_basis_bank = GaborFilterBank(self.num_gabor_basis, self.prototype_kernel_size, sample_rate=100.0)
        self.fourier_basis_bank = FourierFilterBank(self.num_fourier_basis, self.prototype_kernel_size,
                                                    sample_rate=100.0)
        self.num_learnable_basis = 10
        self.learnable_basis_bank = nn.Parameter(torch.randn(self.num_learnable_basis, 1, self.prototype_kernel_size))
        nn.init.xavier_uniform_(self.learnable_basis_bank)

        num_total_basis = self.num_gabor_basis + self.num_fourier_basis + self.num_learnable_basis
        self.mixing_weights = nn.Parameter(torch.randn(self.num_composite_prototypes, num_total_basis) * 0.01)

        # Init weights
        with torch.no_grad():
            self.mixing_weights[0:n_g, 0:self.num_gabor_basis].add_(0.1)
            self.mixing_weights[n_g:n_g + n_f, self.num_gabor_basis:self.num_gabor_basis + self.num_fourier_basis].add_(
                0.1)
            self.mixing_weights[n_g + n_f:, self.num_gabor_basis + self.num_fourier_basis:].add_(0.1)

        self.bn = nn.BatchNorm1d(self.num_composite_prototypes)
        self.fc = nn.Linear(self.num_composite_prototypes, num_classes)
        self.min_distance, self.min_indices = None, None

    def forward(self, x, return_indices=False):
        # x: [B, 1, 30000]

        # 1. Extract Features (Low Memory, High Param)
        features = self.feature_extractor(x)  # [B, C, 256]
        C = features.shape[1]

        # 2. Generate Prototypes
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learn_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learn_kernels), dim=0)

        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 3. Calculate Similarity (Optimized Attention)
        min_distance, min_indices = self.similarity_calculator(features, composite_prototypes)
        self.min_distance, self.min_indices = min_distance, min_indices

        similarity = torch.log((self.min_distance + 1) / (self.min_distance + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        return (logits, self.min_indices) if return_indices else logits



import math
import warnings
import argparse
import os
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=49, help='random seed')
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--config', type=str, help='config file path',
                    default='./SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

with open(args.config) as config_file:
    config = json.load(config_file)
config['name'] = os.path.basename(args.config).replace('.json', '')
config['mode'] = 'normal'

model = ProtoPNet(config).cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量 (Total Trainable Params): {total_params} M")

x = torch.rand([64, 1, 30000]).cuda()
out = model(x)
print(out, out.shape)