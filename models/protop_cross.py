# --- protop_cross_final_attention.py ---

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
# 1. 基础模块
# ====================================================================

class ResidualBlock(nn.Module):
    """
    一个标准的残差块。
    """

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
    """
    一个参数量优化后的轻量级特征提取器主干，用于处理由Stem输出的短序列。
    """

    def __init__(self, input_channels, afr_reduced_cnn_size, block, num_blocks, fixed_output_size=256):
        super(EEGNetProto_Slim, self).__init__()
        self.in_channels = input_channels

        # 构建残差层 (32 -> 32 -> 64 -> 128)
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
    """
    Gabor 基础原型库。
    """

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
    """
    Fourier 基础原型库。
    """

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


# ====================================================================
# 2. 核心高级模块
# ====================================================================

class TCNBlock(nn.Module):
    """
    一个带残差连接的TCN基本块。
    """

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
    """
    由多个TCNBlock堆叠而成的深度时序卷积网络。
    """

    def __init__(self, input_dim, num_levels=4, kernel_size=7):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers.append(TCNBlock(input_dim, input_dim, kernel_size=kernel_size, dilation=dilation_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CrossAttentionSimilarity(nn.Module):
    """
    【全新】基于交叉注意力的相似度计算模块。
    它将原型作为Query，输入信号作为Key和Value，计算注意力距离。
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x, prototypes):
        batch_size, C, seq_len = x.shape
        num_prototypes, _, proto_len = prototypes.shape

        x = x.permute(0, 2, 1)
        prototypes = prototypes.permute(0, 2, 1)
        replicated_prototypes = prototypes.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        q = self.to_q(replicated_prototypes)
        k = self.to_k(x)
        v = self.to_v(x)

        q = q.view(batch_size, num_prototypes, proto_len, self.heads, -1).permute(0, 3, 1, 2, 4)
        k = k.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        q = q.reshape(batch_size * self.heads * num_prototypes, proto_len, -1)
        k = k.unsqueeze(2).repeat(1, 1, num_prototypes, 1, 1).reshape(batch_size * self.heads * num_prototypes, seq_len,
                                                                      -1)
        v = v.unsqueeze(2).repeat(1, 1, num_prototypes, 1, 1).reshape(batch_size * self.heads * num_prototypes, seq_len,
                                                                      -1)

        dots = torch.bmm(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.bmm(attn, v)

        out = out.view(batch_size, self.heads, num_prototypes, proto_len, -1).permute(0, 2, 3, 1, 4).reshape(batch_size,
                                                                                                             num_prototypes,
                                                                                                             proto_len,
                                                                                                             -1)

        original_q_projected = self.to_q(replicated_prototypes)
        distance = F.mse_loss(original_q_projected, out, reduction='none').mean(dim=[2, 3])

        attn_map = attn.view(batch_size, self.heads, num_prototypes, proto_len, seq_len)
        heatmap = attn_map.mean(dim=[1, 3])
        activation_indices = heatmap.argmax(dim=-1)

        return distance, activation_indices


# ====================================================================
# 3. 最终的、完全优化的模型
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.prototype_kernel_size = self.cfg['classifier']['prototype_shape'][2]
        self.num_composite_prototypes = self.cfg['classifier']['prototype_num']
        num_classes = self.cfg['classifier']['num_classes']

        # 速度优化核心: 高效的 Stem 模块
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )

        # 参数优化核心: Slim 主干网络
        self.feature_extractor = EEGNetProto_Slim(
            input_channels=64, afr_reduced_cnn_size=afr_reduced_cnn_size,
            block=ResidualBlock, num_blocks=[2, 2, 2, 2], fixed_output_size=256
        )

        # 深度时序建模模块
        self.tcn_layer = EnhancedTCN(input_dim=afr_reduced_cnn_size, num_levels=4)

        # 性能提升核心: 交叉注意力相似度模块
        self.similarity_calculator = CrossAttentionSimilarity(
            dim=afr_reduced_cnn_size, heads=4, dim_head=32
        )

        # 原型库、混合权重、分类器的定义
        self.num_gabor_basis, self.num_fourier_basis = 20, 20
        self.gabor_basis_bank = GaborFilterBank(self.num_gabor_basis, self.prototype_kernel_size, sample_rate=100.0)
        self.fourier_basis_bank = FourierFilterBank(self.num_fourier_basis, self.prototype_kernel_size,
                                                    sample_rate=100.0)
        self.num_learnable_basis = 10
        self.learnable_basis_bank = nn.Parameter(torch.randn(self.num_learnable_basis, 1, self.prototype_kernel_size))
        nn.init.xavier_uniform_(self.learnable_basis_bank)
        num_total_basis = self.num_gabor_basis + self.num_fourier_basis + self.num_learnable_basis
        self.mixing_weights = nn.Parameter(torch.rand(self.num_composite_prototypes, num_total_basis))
        nn.init.xavier_uniform_(self.mixing_weights)
        self.bn = nn.BatchNorm1d(self.num_composite_prototypes)
        self.fc = nn.Linear(self.num_composite_prototypes, num_classes)
        self.min_distance, self.min_indices = None, None

    def forward(self, x, return_indices=False):
        # 1. 通过 Stem 快速压缩序列
        stem_features = self.stem(x)

        # 2. 在短序列上进行深度特征提取
        conv_features = self.feature_extractor(stem_features)

        # 3. 进行时序建模
        temporal_features = self.tcn_layer(conv_features)

        C = temporal_features.shape[1]

        # 4. 构建复合原型
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learnable_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learnable_kernels), dim=0)
        # composite_prototypes = torch.matmul(F.relu(self.mixing_weights), base_prototypes.flatten(1))
        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 5. 使用交叉注意力计算距离和激活位置
        min_distance, min_indices = self.similarity_calculator(temporal_features, composite_prototypes)
        self.min_distance, self.min_indices = min_distance, min_indices

        # 6. 将距离转换为相似度并分类
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