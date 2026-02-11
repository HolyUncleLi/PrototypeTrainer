# --- models/protop_cross_v5.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# 1. 物理层 (保留创新点)
# ====================================================================

class LearnableGaborConv1d(nn.Module):
    """
    第一层保持 Gabor 卷积，赋予模型物理可解释的初始化。
    使用 stride=2 进行初步降采样。
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
        # 输出模值，保留能量特征
        return torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)


# ====================================================================
# 2. 核心骨干: 多级残差金字塔 (Pyramidal ResNet)
# ====================================================================

class ResBlock(nn.Module):
    """
    标准的残差块，用于提取深层特征。
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride,
                               padding=3 + (dilation - 1) * 3, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.gelu(out)
        return out


class HierarchicalFeatureExtractor(nn.Module):
    """
    层级特征提取器：替代之前的折叠流。
    结构类似 ResNet，逐层下采样，逐层增加通道。
    """

    def __init__(self, input_dim=64, out_dim=128):
        super().__init__()

        # Stage 1: 保持 64 通道，降采样 4倍 (30000/2 -> 3750)
        self.layer1 = nn.Sequential(
            ResBlock(input_dim, 64, stride=2),
            ResBlock(64, 64, stride=2)
        )

        # Stage 2: 升维到 128，降采样 4倍 (3750 -> 937)
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=2)
        )

        # Stage 3: 升维到 256，降采样 2倍 (937 -> 468)
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1, dilation=2)  # 使用空洞卷积增加感受野
        )

        # TCN Context Layer: 最后的时序整合
        self.tcn = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # 最终映射到原型维度
        self.final_proj = nn.Conv1d(256, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tcn(x)
        x = self.final_proj(x)
        return x


# ====================================================================
# 3. 辅助模块 (Memory Efficient Attention - 保持 V4 的修复版)
# ====================================================================

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
    """
    [最终稳定版]
    使用 Broadcasting + Mean Reduction，显存友好且支持任意 kernel size
    """

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
            # Query: [B, P, K, C]
            replicated_p = p_perm.unsqueeze(0).expand(batch_size, -1, -1, -1)
            q = self.q_projs[i](replicated_p)
            q = q.view(batch_size, num_p_group, proto_len, self.heads, self.dim_head)
            q = q.mean(dim=2)  # 聚合 kernel 时间维 [B, P, H, D]
            q = q.permute(0, 2, 1, 3)  # [B, H, P, D]

            # Key, Value: [B, L, C]
            k = self.k_projs[i](x_perm)
            v = self.v_projs[i](x_perm)
            k = k.view(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.heads, self.dim_head).permute(0, 2, 1, 3)

            # Attention
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

            # Recon & Distance
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_p_group, -1)
            original_q_projected = self.q_projs[i](replicated_p).mean(dim=2)
            dist = F.mse_loss(original_q_projected, out, reduction='none').mean(dim=-1)

            heatmap = attn.mean(dim=1)
            indices = heatmap.argmax(dim=-1)
            all_distances.append(dist)
            all_indices.append(indices)

        final_distances = torch.cat(all_distances, dim=1)
        final_indices = torch.cat(all_indices, dim=1)
        return final_distances, final_indices


# ====================================================================
# 4. ProtoPNet V5 (主模型)
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

        # 1. 物理层 Gabor Stem (Stride=2)
        self.gabor_stem = LearnableGaborConv1d(1, 64, kernel_size=63, stride=2)

        # 2. 层级特征提取 (Pyramidal)
        self.feature_extractor = HierarchicalFeatureExtractor(input_dim=64, out_dim=afr_reduced_cnn_size)

        # 3. 相似度模块
        self.similarity_calculator = MultiLatentSpaceSimilarity(
            dim=afr_reduced_cnn_size,
            splits=self.proto_splits,
            heads=4,
            dim_head=32
        )

        # 4. 原型库
        self.num_gabor_basis, self.num_fourier_basis = 20, 20
        self.gabor_basis_bank = GaborFilterBank(self.num_gabor_basis, self.prototype_kernel_size, sample_rate=100.0)
        self.fourier_basis_bank = FourierFilterBank(self.num_fourier_basis, self.prototype_kernel_size,
                                                    sample_rate=100.0)
        self.num_learnable_basis = 10
        self.learnable_basis_bank = nn.Parameter(torch.randn(self.num_learnable_basis, 1, self.prototype_kernel_size))
        nn.init.xavier_uniform_(self.learnable_basis_bank)

        num_total_basis = self.num_gabor_basis + self.num_fourier_basis + self.num_learnable_basis
        self.mixing_weights = nn.Parameter(torch.randn(self.num_composite_prototypes, num_total_basis) * 0.01)

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

        # 1. Gabor Stem -> [B, 64, 15000]
        x_gabor = self.gabor_stem(x)

        # 2. Hierarchical Extraction -> [B, 128, 468]
        features = self.feature_extractor(x_gabor)
        C = features.shape[1]

        # 3. Generate Prototypes
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learn_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learn_kernels), dim=0)

        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 4. Similarity
        min_distance, min_indices = self.similarity_calculator(features, composite_prototypes)
        self.min_distance, self.min_indices = min_distance, min_indices

        similarity = torch.log((self.min_distance + 1) / (self.min_distance + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        return (logits, self.min_indices) if return_indices else logits


'''
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
'''