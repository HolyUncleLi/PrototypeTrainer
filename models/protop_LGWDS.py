# --- models/protop_cross_v3.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================================================================
# 1. 颠覆性创新: 可学习 Gabor 卷积层 (Learnable Gabor Convolution)
# ====================================================================

class LearnableGaborConv1d(nn.Module):
    """
    这不是普通的卷积。它的权重是由 Gabor 函数动态生成的。
    参数是物理意义明确的：频率、带宽、中心位置。
    输出包括：幅值(Magnitude) 和 相位(Phase)，保留了完整的信号信息。
    """

    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=100.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 物理参数初始化
        # 频率分布在 0.5Hz 到 30Hz 之间 (覆盖 Delta 到 Beta)
        self.mu_f = nn.Parameter(torch.rand(out_channels) * 30.0 + 0.5)
        # 带宽 (控制时频分辨率的平衡)
        self.sigma = nn.Parameter(torch.ones(out_channels) * 10.0)

        # 时间轴
        t = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) / sample_rate
        self.register_buffer('t', t)

    def get_filter(self):
        # 动态生成滤波器权重
        t = self.t.view(1, 1, -1)
        mu_f = self.mu_f.view(-1, 1, 1)
        sigma = self.sigma.view(-1, 1, 1)

        # Gabor 包络
        envelope = torch.exp(-0.5 * (t ** 2) / (sigma ** 2))
        # 复数载波
        carrier_cos = torch.cos(2 * math.pi * mu_f * t)
        carrier_sin = torch.sin(2 * math.pi * mu_f * t)

        # 生成实部和虚部滤波器
        filter_real = envelope * carrier_cos
        filter_imag = envelope * carrier_sin

        return filter_real, filter_imag

    def forward(self, x):
        # x: [B, 1, L]
        # 动态获取权重
        w_real, w_imag = self.get_filter()  # [Out, 1, K]

        # 分别卷积
        out_real = F.conv1d(x, w_real, padding=self.padding)
        out_imag = F.conv1d(x, w_imag, padding=self.padding)

        # 计算幅值 (用于语义流 - 关注能量)
        magnitude = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)

        # 保留原始复数特征 (用于形态流 - 关注波形细节)
        return magnitude, out_real


# ====================================================================
# 2. 双流架构组件
# ====================================================================

class SemanticStream(nn.Module):
    """
    语义流：负责极度压缩信息，提取为了分类所需的抽象特征。
    使用大步长池化。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(in_channels * 2),
            nn.GELU(),
            nn.MaxPool1d(4),  # 强降采样
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(4)  # 再次强降采样
        )

    def forward(self, x):
        return self.net(x)


class MorphologicalStream(nn.Module):
    """
    形态流：负责保留波形形状。
    不使用 Pooling，而是使用空洞卷积 (Dilated Conv) 增加感受野。
    这里提取的特征与原始波形在时间轴上是对齐的。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool1d(4),
            # Dilation=2
            nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=8, dilation=2, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            # Dilation=4
            nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=16, dilation=4, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            # 1x1 混合
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
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
# 3. 核心骨干网络
# ====================================================================

class LGWDS_Net(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        # 第一层：可学习 Gabor (64个滤波器)
        self.gabor_layer = LearnableGaborConv1d(1, 64, kernel_size=63)

        # 双流
        self.semantic_stream = SemanticStream(64, 128)
        self.morph_stream = MorphologicalStream(64, 64)

        # 融合
        self.fusion = HolographicFusion(64, 128)

        self.maxpool = nn.AdaptiveAvgPool1d(777)

        # 最终映射到原型维度
        self.final_proj = nn.Conv1d(64, out_dim, kernel_size=1)

        # 为了降低计算量，我们在融合后做一个温和的 Average Pooling
        # 这样既保留了波形大致轮廓，又减少了原型计算开销
        self.final_pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # x: [B, 1, 3000]

        # 1. 物理层
        mag, raw_real = self.gabor_layer(x)  # [B, 64, 3000]

        # 2. 双流处理
        sem_feat = self.semantic_stream(mag)  # [B, 128, 187] (抽象)
        morph_feat = self.morph_stream(raw_real)  # [B, 64, 3000] (具体)
        # morph_feat = self.maxpool(morph_feat)
        print(sem_feat.shape, morph_feat.shape)

        # 3. 全息融合
        fused = self.fusion(morph_feat, sem_feat)  # [B, 64, 3000]
        print(fused.shape)
        fused = self.maxpool(fused)
        print(fused.shape)
        # 4. 最终输出
        out = self.final_proj(fused)  # [B, 128, 3000]
        out = self.final_pool(out)  # [B, 128, 750] -> 仍保留了足够的时间分辨率
        return out


# ====================================================================
# 4. 辅助模块 (保留原有的 MultiLatentSpaceSimilarity 等)
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
        all_distances = []
        all_indices = []
        for i, p_group in enumerate(proto_groups):
            num_p_group = p_group.shape[0]
            if num_p_group == 0: continue
            p_perm = p_group.permute(0, 2, 1)
            replicated_p = p_perm.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            q = self.q_projs[i](replicated_p)
            k = self.k_projs[i](x_perm)
            v = self.v_projs[i](x_perm)
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

        # [NEW] 核心替换：使用 LG-WDS 特征提取器
        self.feature_extractor = LGWDS_Net(out_dim=afr_reduced_cnn_size)

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
        # x: [B, 1, 3000]

        # 1. 提取特征 (全息特征)
        features = self.feature_extractor(x)  # [B, C, L_reduced]
        C = features.shape[1]

        # 2. 原型生成
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learn_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learn_kernels), dim=0)

        composite_prototypes = torch.matmul(self.mixing_weights, base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 3. 相似度
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
