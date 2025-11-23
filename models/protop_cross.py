# --- protop_cross.py ---

import math
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import warnings
import argparse
import os
import json


# ====================================================================
# 1. 基础模块 (保持不变)
# ====================================================================
# ... (ResidualBlock, EEGNetProto_Slim, GaborFilterBank, FourierFilterBank, TCNBlock, EnhancedTCN 保持不变，此处省略以节省篇幅，请保留原有的这些类) ...

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
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


class EEGNetProto_Slim(nn.Module):
    def __init__(self, input_channels, afr_reduced_cnn_size, block, num_blocks, fixed_output_size=256):
        super(EEGNetProto_Slim, self).__init__()
        self.in_channels = input_channels
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=1)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=fixed_output_size)
        self.final_conv = nn.Conv1d(128, afr_reduced_cnn_size, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)
        out = self.dropout(out)
        out = self.final_conv(out)
        return out


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


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.dropout(self.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(self.gelu(self.bn2(self.conv2(out))))
        res = x if self.shortcut is None else self.shortcut(x)
        return out + res


class EnhancedTCN(nn.Module):
    def __init__(self, input_dim, num_levels=4, kernel_size=7):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers.append(TCNBlock(input_dim, input_dim, kernel_size=kernel_size, dilation=dilation_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ====================================================================
# 2. 核心高级模块
# ====================================================================

class MultiLatentSpaceSimilarity(nn.Module):
    """
    多潜空间相似度模块。
    该模块为 Gabor, Fourier, Learnable 族群分别维护独立的 Query/Key/Value 投影。
    这样做的目的是让模型能够针对不同的特征类型（时域 vs 频域）学习不同的匹配空间。
    """

    def __init__(self, dim, splits, heads=4, dim_head=32):
        super().__init__()
        self.splits = splits  # e.g., [7, 7, 6]
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        # 为三个族群定义独立的三组投影矩阵
        # 索引 0: Gabor Space, 1: Fourier Space, 2: Learnable Space
        self.q_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])

    def forward(self, x, prototypes):
        # x: [B, C, L_seq]
        # prototypes: [Total_Proto, C, L_proto]
        batch_size, C, seq_len = x.shape
        _, _, proto_len = prototypes.shape

        # 信号转置
        x_perm = x.permute(0, 2, 1)

        # 【关键步骤】
        # 这里我们将所有复合原型根据 splits 切分成三组。
        # 第一组会被强制送入 Gabor 潜空间 (q_projs[0], k_projs[0]) 进行匹配
        # 这就是“引导”模型学习不同空间的物理机制。
        proto_groups = torch.split(prototypes, self.splits, dim=0)

        all_distances = []
        all_indices = []

        for i, p_group in enumerate(proto_groups):
            num_p_group = p_group.shape[0]
            if num_p_group == 0: continue

            # 1. 投影 Query (原型) -> 使用族群专有的 Q 投影
            p_perm = p_group.permute(0, 2, 1)
            replicated_p = p_perm.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            q = self.q_projs[i](replicated_p)

            # 2. 投影 Key, Value (信号) -> 使用族群专有的 K, V 投影
            # 意味着同一个信号 x 在面对不同原型时，会展现出不同的特征面貌
            k = self.k_projs[i](x_perm)
            v = self.v_projs[i](x_perm)

            # --- 标准的 Attention 计算 ---
            q = q.view(batch_size, num_p_group, proto_len, self.heads, -1).permute(0, 3, 1, 2, 4)
            q_reshaped = q.reshape(batch_size * self.heads * num_p_group, proto_len, -1)

            k = k.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
            v = v.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

            k_reshaped = k.unsqueeze(2).repeat(1, 1, num_p_group, 1, 1).reshape(batch_size * self.heads * num_p_group,
                                                                                seq_len, -1)
            v_reshaped = v.unsqueeze(2).repeat(1, 1, num_p_group, 1, 1).reshape(batch_size * self.heads * num_p_group,
                                                                                seq_len, -1)

            dots = torch.bmm(q_reshaped, k_reshaped.transpose(-1, -2)) * self.scale
            attn = dots.softmax(dim=-1)
            out = torch.bmm(attn, v_reshaped)

            # 重构与距离计算
            out = out.view(batch_size, self.heads, num_p_group, proto_len, -1).permute(0, 2, 3, 1, 4).reshape(
                batch_size, num_p_group, proto_len, -1)
            original_q_projected = self.q_projs[i](replicated_p)
            dist = F.mse_loss(original_q_projected, out, reduction='none').mean(dim=[2, 3])

            attn_map = attn.view(batch_size, self.heads, num_p_group, proto_len, seq_len)
            heatmap = attn_map.mean(dim=[1, 3])
            indices = heatmap.argmax(dim=-1)

            all_distances.append(dist)
            all_indices.append(indices)

        final_distances = torch.cat(all_distances, dim=1)
        final_indices = torch.cat(all_indices, dim=1)

        return final_distances, final_indices


# ====================================================================
# 3. 最终模型 (软约束版本)
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.prototype_kernel_size = self.cfg['classifier']['prototype_shape'][2]

        # 计算每组原型数量
        total_prototypes = self.cfg['classifier']['prototype_num']
        n_g = total_prototypes // 3
        n_f = total_prototypes // 3
        n_l = total_prototypes - n_g - n_f
        self.proto_splits = [n_g, n_f, n_l]  # 例如 [6, 6, 8]
        self.num_composite_prototypes = total_prototypes

        num_classes = self.cfg['classifier']['num_classes']

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )

        self.feature_extractor = EEGNetProto_Slim(
            input_channels=64, afr_reduced_cnn_size=afr_reduced_cnn_size,
            block=ResidualBlock, num_blocks=[2, 2, 2, 2], fixed_output_size=256
        )

        self.tcn_layer = EnhancedTCN(input_dim=afr_reduced_cnn_size, num_levels=4)

        # 多潜空间模块：负责将不同组的原型映射到不同空间
        self.similarity_calculator = MultiLatentSpaceSimilarity(
            dim=afr_reduced_cnn_size,
            splits=self.proto_splits,
            heads=4,
            dim_head=32
        )

        # 原型库
        self.num_gabor_basis, self.num_fourier_basis = 20, 20
        self.gabor_basis_bank = GaborFilterBank(self.num_gabor_basis, self.prototype_kernel_size, sample_rate=100.0)
        self.fourier_basis_bank = FourierFilterBank(self.num_fourier_basis, self.prototype_kernel_size,
                                                    sample_rate=100.0)
        self.num_learnable_basis = 10
        self.learnable_basis_bank = nn.Parameter(torch.randn(self.num_learnable_basis, 1, self.prototype_kernel_size))
        nn.init.xavier_uniform_(self.learnable_basis_bank)

        # --- 核心修改：恢复全连接混合权重 ---
        # 我们不再拆分矩阵，而是保留一个完整的 learnable 矩阵。
        # 结构化（Structure）将由 Loss 函数在训练中诱导产生，而不是强制为 0。
        num_total_basis = self.num_gabor_basis + self.num_fourier_basis + self.num_learnable_basis
        self.mixing_weights = nn.Parameter(torch.randn(self.num_composite_prototypes, num_total_basis) * 0.01)

        # 初始化技巧：为了帮助模型更快收敛，我们可以稍微增大“期望对角块”的初始值
        # 这不是强制约束，只是一个好的初始点
        with torch.no_grad():
            # Gabor Block
            self.mixing_weights[0:n_g, 0:self.num_gabor_basis].add_(0.1)
            # Fourier Block
            self.mixing_weights[n_g:n_g + n_f, self.num_gabor_basis:self.num_gabor_basis + self.num_fourier_basis].add_(
                0.1)
            # Learnable Block
            self.mixing_weights[n_g + n_f:, self.num_gabor_basis + self.num_fourier_basis:].add_(0.1)

        self.bn = nn.BatchNorm1d(self.num_composite_prototypes)
        self.fc = nn.Linear(self.num_composite_prototypes, num_classes)
        self.min_distance, self.min_indices = None, None

    def forward(self, x, return_indices=False):
        stem_features = self.stem(x)
        conv_features = self.feature_extractor(stem_features)
        temporal_features = self.tcn_layer(conv_features)

        C = temporal_features.shape[1]

        # 1. 获取所有 Basis Kernel
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learn_kernels = self.learnable_basis_bank.repeat(1, C, 1)

        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learn_kernels), dim=0)

        # 2. 生成所有复合原型 (全连接混合)
        # 此时，第 0 个原型可能包含 Fourier 成分，这是允许的，但我们会通过 Loss 惩罚它
        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 3. 多潜空间匹配
        # 在这里，我们将 composite_prototypes 切分。
        # 只有前 n_g 个原型会被送入 "Gabor 潜空间" (Similarity Calculator 内部逻辑)。
        # 如果训练得当，mixing_weights 会让前 n_g 个原型主要由 Gabor Basis 组成。
        min_distance, min_indices = self.similarity_calculator(temporal_features, composite_prototypes)
        self.min_distance, self.min_indices = min_distance, min_indices

        similarity = torch.log((self.min_distance + 1) / (self.min_distance + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        return (logits, self.min_indices) if return_indices else logits


'''
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