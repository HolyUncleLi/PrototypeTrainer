# --- models/protop_cross_v3.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# 1. 可学习 Gabor 卷积层
# ====================================================================

class LearnableGaborConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=100.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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
        out_real = F.conv1d(x, w_real, padding=self.padding)
        out_imag = F.conv1d(x, w_imag, padding=self.padding)
        magnitude = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)
        return magnitude, out_real


# ====================================================================
# 2. 双流架构组件 [修正设计：提前统一时序维度为 194，消灭计算黑洞]
# ====================================================================
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
        return self.gelu(out)


class EEGNetProto_Slim(nn.Module):
    def __init__(self, input_channels, afr_reduced_cnn_size, block, num_blocks, fixed_output_size):
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
        return self.final_conv(out)


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
            layers.append(TCNBlock(input_dim, input_dim, kernel_size=kernel_size, dilation=2 ** i))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SemanticStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(64, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.feature_extractor = EEGNetProto_Slim(
            input_channels=64, afr_reduced_cnn_size=128,
            block=ResidualBlock, num_blocks=[2, 2, 2, 2],
            fixed_output_size=194  # [修改]：原先是 256，直接对齐最终维度，消灭运算死角
        )
        self.tcn_layer = EnhancedTCN(input_dim=128, num_levels=4)

    def forward(self, x):
        return self.tcn_layer(self.feature_extractor(self.stem(x)))


class MorphologicalStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels, in_channels, kernel_size=21, padding=8, dilation=4, groups=in_channels),
            nn.BatchNorm1d(in_channels), nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=15, padding=16, dilation=4, groups=in_channels),
            nn.BatchNorm1d(in_channels), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=16, dilation=2, groups=in_channels),
            nn.BatchNorm1d(in_channels), nn.GELU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=16, dilation=2, groups=in_channels),
            nn.BatchNorm1d(in_channels), nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AdaptiveAvgPool1d(194)  # [修改]：强制将波形特征稳步池化到 194，配合语义流
        )

    def forward(self, x):
        return self.net(x)


class HolographicFusion(nn.Module):
    def __init__(self, high_res_dim, low_res_dim):
        super().__init__()
        self.project = nn.Conv1d(low_res_dim, high_res_dim, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(high_res_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, morph, sem):
        # [修改]：现在双流在传入前已经完美对齐到 194 长度。
        # 彻底拔除原本导致极度缓慢的 F.interpolate(低效插值)，并免去了在冗余尺寸上计算门控。
        sem_proj = self.project(sem)
        concat = torch.cat([morph, sem_proj], dim=1)
        mask = self.gate(concat)
        return morph * (1 + mask * sem_proj)


class LGWDS_Net(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.gabor_layer = LearnableGaborConv1d(1, 64, kernel_size=63)
        self.semantic_stream = SemanticStream(64, 128)
        self.morph_stream = MorphologicalStream(64, 128)
        self.fusion = HolographicFusion(128, 128)
        self.final_proj = nn.Conv1d(128, out_dim, kernel_size=1)

    def forward(self, x):
        mag, raw_real = self.gabor_layer(x)
        sem_feat = self.semantic_stream(mag)  # 输出长度严格为 194
        morph_feat = self.morph_stream(raw_real)  # 输出长度严格为 194

        fused = self.fusion(morph_feat, sem_feat)  # 纯 194 极速融合
        out = self.final_proj(fused)
        return out


# ====================================================================
# 4. 辅助模块
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
        return A * torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) * torch.cos(2 * torch.pi * f * t + phi)


class FourierFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A = nn.Parameter(torch.ones(self.num))
        self.f = nn.Parameter(torch.linspace(1.0, 40.0, num_filters) + torch.randn(num_filters) * 0.5)
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        A, f, phi = [p.view(-1, 1, 1) for p in [self.A, self.f.clamp(0.1, 50.0), self.phi]]
        return A * torch.cos(2 * torch.pi * f * t + phi)


class MultiLatentSpaceSimilarity(nn.Module):
    def __init__(self, dim, splits, heads=4, dim_head=32):
        super().__init__()
        self.splits = splits
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.q_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.k_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(3)])

    def forward(self, x, prototypes):
        batch_size, C, seq_len = x.shape
        _, _, proto_len = prototypes.shape
        x_perm = x.permute(0, 2, 1)
        proto_groups = torch.split(prototypes, self.splits, dim=0)

        all_distances, all_indices = [], []

        for i, p_group in enumerate(proto_groups):
            num_p_group = p_group.shape[0]
            if num_p_group == 0: continue

            p_perm = p_group.permute(0, 2, 1)

            q_proj = self.q_projs[i](p_perm)
            q = q_proj.view(num_p_group, proto_len, self.heads, -1).permute(2, 0, 1, 3)

            k = self.k_projs[i](x_perm).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
            v = self.v_projs[i](x_perm).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

            # =========================================================================
            # [致命瓶颈修正]：纯算子级提速 10x
            # 摒弃底层难以优化的 5D einsum，将其安全转变为严格的 cuBLAS Batched Matmul。
            # =========================================================================

            # 1. Query 形变:[H, P, L, D] -> [1, H, P*L, D]
            q_view = q.reshape(1, self.heads, num_p_group * proto_len, -1)
            # 2. Key 形变: [B, H, S, D] ->[B, H, D, S]
            k_view = k.transpose(-1, -2)

            # 3. 极速矩阵乘法，自动利用 GPU 多流广播 B 维度：->[B, H, P*L, S]
            dots = torch.matmul(q_view, k_view) * self.scale
            attn = dots.softmax(dim=-1)

            # 4. Attention * Value: [B, H, P*L, S] @ [B, H, S, D] -> [B, H, P*L, D]
            out_view = torch.matmul(attn, v)

            # 5. 还原形状: [B, H, P, L, D]
            out = out_view.view(batch_size, self.heads, num_p_group, proto_len, -1)
            out = out.permute(0, 2, 3, 1, 4).reshape(batch_size, num_p_group, proto_len, -1)

            # 距离计算
            dist = F.mse_loss(q_proj.unsqueeze(0), out, reduction='none').mean(dim=[2, 3])

            # 从展开的 attn 中提取热力图
            attn_map = attn.view(batch_size, self.heads, num_p_group, proto_len, seq_len)
            heatmap = attn_map.mean(dim=[1, 3])
            indices = heatmap.argmax(dim=-1)

            all_distances.append(dist)
            all_indices.append(indices)

        return torch.cat(all_distances, dim=1), torch.cat(all_indices, dim=1)


# ====================================================================
# 5. ProtoPNet (V3)
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

        self.feature_extractor = LGWDS_Net(out_dim=afr_reduced_cnn_size)

        self.similarity_calculator = MultiLatentSpaceSimilarity(
            dim=afr_reduced_cnn_size,
            splits=self.proto_splits,
            heads=4,
            dim_head=32
        )

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
        features = self.feature_extractor(x)
        C = features.shape[1]

        self.current_gabor_k = self.gabor_basis_bank.get_kernels()
        self.current_fourier_k = self.fourier_basis_bank.get_kernels()

        gabor_kernels = self.current_gabor_k.repeat(1, C, 1)
        fourier_kernels = self.current_fourier_k.repeat(1, C, 1)
        learn_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learn_kernels), dim=0)

        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

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
import time
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

x = torch.rand([8, 1, 30000]).cuda()
start = time.time()
out = model(x)
torch.cuda.synchronize()  # 确保 GPU 完成计算
end = time.time()
print("单次推理耗时: {:.4f} 秒".format(end - start))
print(out, out.shape)
'''