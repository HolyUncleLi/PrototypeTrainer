# ====================================================================
# 导入区
# ====================================================================
import math
import warnings
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x)
        weights = weights.unsqueeze(-1)
        return x * weights.expand_as(x)


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False, isFTConv=True):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,
                                          bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left).cuda()
            pad_right = torch.zeros(D_out, D_in, pad_length_right).cuda()
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left).cuda() * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right).cuda() * pad_values

        x = torch.cat((pad_left, x), dim=-1)
        x = torch.cat((x, pad_right), dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.05):
        super(Block, self).__init__()

        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)
        self.se = SEBlock(in_dim=dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act1 = nn.PReLU()
        self.ffn1norm1 = nn.BatchNorm1d(nvars * dff)
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1norm2 = nn.BatchNorm1d(nvars * dmodel)
        self.ffn1act2 = nn.PReLU()
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel
        self.shortcut = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                  padding=0, dilation=1)

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)

        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)
        x = self.se(x)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act1(x)
        x = self.ffn1drop2(self.ffn1pw2(x))

        x = x.reshape(B, M, D, N)
        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCN(nn.Module):
    def __init__(self, ):

        super(ModernTCN, self).__init__()

        self.batchsize = 64
        self.seq_len = 10
        self.channeldim = 128
        self.featuredim = 80  # seq len * 8
        self.embeddim = 80
        self.patch_size = 16
        self.patch_stride = 8
        self.downsample_ratio = 4
        self.class_num = 5
        self.num_stage = 2

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=16, stride=8),
            nn.BatchNorm1d(64)
        )
        self.downsample_layers.append(stem)
        downsample_layer = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=self.downsample_ratio * 2, stride=self.downsample_ratio),
        )
        self.downsample_layers.append(downsample_layer)

        self.num_stage = 2
        self.stages = nn.ModuleList()
        layer = Stage(4, 1, 51, 5, dmodel=64, dw_model=64, nvars=1, small_kernel_merged=False, drop=0.1)
        self.stages.append(layer)
        layer = Stage(4, 1, 31, 5, dmodel=128, dw_model=128, nvars=1, small_kernel_merged=False, drop=0.1)
        self.stages.append(layer)

        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.flatten = nn.Flatten()

    def forward_feature(self, x):
        B, C, N = x.shape

        x = x.unsqueeze(2)
        x = x.reshape(B, 1, C, N)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)

            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)

            x = self.stages[i](x)
        return x

    def classification2(self, x, tags=None):
        x = self.forward_feature(x).squeeze()
        x = self.avgpool(x)
        return x

    def forward(self, x, tags=None, pre_stage=2):
        x = self.classification2(x, tags=tags)
        return x


# ====================================================================
# 1. 基础模块 (保持不变)
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
# 1. Gabor 分支特征提取
# ====================================================================
class LearnableGaborConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=63, sample_rate=1.0):
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
        out_real = F.conv1d(x, w_real, stride=1, padding=self.padding)
        out_imag = F.conv1d(x, w_imag, stride=1, padding=self.padding)
        magnitude = torch.sqrt(out_real.pow(2) + out_imag.pow(2) + 1e-8)
        return magnitude, out_real


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
            block=ResidualBlock, num_blocks=[2, 2, 2, 2], fixed_output_size=256
        )
        self.pool = nn.AdaptiveAvgPool1d(256)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_extractor(x)
        return x


class MorphologicalStream(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()
        self.tcn = ModernTCN()

    def forward(self, x):
        return self.tcn(x)


class HolographicFusion(nn.Module):
    def __init__(self, high_res_dim, low_res_dim):
        super().__init__()
        self.project = nn.Conv1d(low_res_dim, high_res_dim, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(high_res_dim * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, morph_feat, sem_feat):
        sem_proj = self.project(sem_feat)
        concat = torch.cat([morph_feat, sem_proj], dim=1)
        mask = self.gate(concat)
        return morph_feat * (1 + mask * sem_proj)


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
        sem_feat = self.semantic_stream(mag)
        morph_feat = self.morph_stream(raw_real)

        fused = self.fusion(morph_feat, sem_feat)
        out = self.final_proj(fused)
        return out


# ====================================================================
# 【保留的辅助模块】虽然前向传播不再计算，但保留原始定义不删
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


# ====================================================================
# 【完全保留的】原版交叉注意力特征匹配层
# ====================================================================
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

            dots = torch.einsum('hpld,bhsd->bhpls', q, k) * self.scale
            attn = dots.softmax(dim=-1)

            out = torch.einsum('bhpls,bhsd->bhpld', attn, v)
            out = out.permute(0, 2, 3, 1, 4).reshape(batch_size, num_p_group, proto_len, -1)

            dist = F.mse_loss(q_proj.unsqueeze(0), out, reduction='none').mean(dim=[2, 3])

            heatmap = attn.mean(dim=[1, 3])
            indices = heatmap.argmax(dim=-1)

            all_distances.append(dist)
            all_indices.append(indices)

        return torch.cat(all_distances, dim=1), torch.cat(all_indices, dim=1)


# ====================================================================
# 【重构的】ProtoPNet：标准参数与简单的物理初始化
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']

        self.prototype_shape = self.cfg['classifier']['prototype_shape']
        self.protop_num = self.prototype_shape[0]

        # 为了兼容原始计算逻辑，将数量拆分为 3 份
        n_g = self.protop_num // 3
        n_f = self.protop_num // 3
        n_l = self.protop_num - n_g - n_f
        self.proto_splits = [n_g, n_f, n_l]

        # 【兼容】供外部训练脚本读取的常规变量，只使用最基本的变量赋值
        self.num_composite_prototypes = self.protop_num
        self.num_gabor_basis = n_g
        self.num_fourier_basis = n_f
        self.num_learnable_basis = n_l

        num_classes = self.cfg['classifier']['num_classes']

        self.feature_extractor = LGWDS_Net(out_dim=afr_reduced_cnn_size)
        self.tcn_layer = EnhancedTCN(input_dim=afr_reduced_cnn_size, num_levels=4)

        self.similarity_calculator = MultiLatentSpaceSimilarity(
            dim=afr_reduced_cnn_size,
            splits=self.proto_splits,
            heads=4,
            dim_head=32
        )

        # -------------------------------------------------------------
        # 1. 直接创建供网络优化用的 [50, 128, 15] 的完整 nn.Parameter
        # -------------------------------------------------------------
        self.prototype_vectors = nn.Parameter(torch.empty(self.prototype_shape), requires_grad=True)
        self._initialize_constrained_prototypes(n_g, n_f, n_l)

        # -------------------------------------------------------------
        # 2. 【兼容】定义一个简单的 dummy 参数替代原来的 mixing_weights
        #    这样 `loss_struc = torch.mean(torch.abs(weights) * struc_mask)`
        #    在外部就能被正常执行，并且会自动把无用的 dummy weights 惩罚到 0，不会报错。
        # -------------------------------------------------------------
        num_total_basis = n_g + n_f + n_l
        self.mixing_weights = nn.Parameter(torch.zeros(self.protop_num, num_total_basis), requires_grad=True)

        self.bn = nn.BatchNorm1d(self.protop_num)
        self.fc = nn.Linear(self.protop_num, num_classes)
        self.min_distance, self.min_indices = None, None

    def _initialize_constrained_prototypes(self, n_g, n_f, n_l):
        """一次性以 Gabor/Fourier 为约束随机初始化 128通道的原型矩阵"""
        P, C, L = self.prototype_shape
        with torch.no_grad():
            t = torch.linspace(-1, 1, steps=L).view(1, 1, L)

            # Gabor 约束：直接利用通道维度 C 进行广播，确保128通道各有不同
            f_g = torch.empty(n_g, C, 1).uniform_(1.0, 10.0)
            phi_g = torch.empty(n_g, C, 1).uniform_(0, 2 * math.pi)
            sigma_g = torch.empty(n_g, C, 1).uniform_(0.2, 0.8)
            gabor = torch.exp(-(t ** 2) / (2 * sigma_g ** 2)) * torch.cos(2 * math.pi * f_g * t + phi_g)

            # Fourier 约束
            f_f = torch.empty(n_f, C, 1).uniform_(1.0, 10.0)
            phi_f = torch.empty(n_f, C, 1).uniform_(0, 2 * math.pi)
            fourier = torch.cos(2 * math.pi * f_f * t + phi_f)

            # Random 约束
            random_proto = torch.randn(n_l, C, L) * 0.1

            self.prototype_vectors[:n_g] = gabor
            self.prototype_vectors[n_g:n_g + n_f] = fourier
            self.prototype_vectors[n_g + n_f:] = random_proto

    # -----------------------------------------------------------------
    # 【兼容】使用标准的 Python @property 为外部脚本提供计算 loss_orth 所需的张量切片
    #  外部脚本通过 F.normalize 对它们进行处理后计算出的余弦相似度，
    #  将直接天然起到推开原型距离、保持原型的多样性（Diversity）的作用！
    # -----------------------------------------------------------------
    @property
    def current_gabor_k(self):
        return self.prototype_vectors[:self.num_gabor_basis]

    @property
    def current_fourier_k(self):
        return self.prototype_vectors[self.num_gabor_basis: self.num_gabor_basis + self.num_fourier_basis]

    @property
    def learnable_basis_bank(self):
        return self.prototype_vectors[self.num_gabor_basis + self.num_fourier_basis:]

    def forward(self, x, return_indices=False):
        features = self.feature_extractor(x)
        features = self.tcn_layer(features)

        # 极简前向传播：直接送入参数模板矩阵
        min_distance, min_indices = self.similarity_calculator(features, self.prototype_vectors)
        self.min_distance, self.min_indices = min_distance, min_indices

        similarity = torch.log((self.min_distance + 1) / (self.min_distance + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        return (logits, self.min_indices) if return_indices else logits


# ====================================================================
# 2. 纯内置耗时分析工具 (Profiler)
# ====================================================================
class BuiltInProfiler:
    def __init__(self, model):
        self.model = model
        self.fwd_events = {}
        self.bwd_events = {}
        self.fwd_times = {}
        self.bwd_times = {}
        self.hooks = []

        target_modules = {
            '1. GaborConv_Layer': self.model.feature_extractor.gabor_layer,
            '2. Semantic_Stream': self.model.feature_extractor.semantic_stream,
            '3. Morph_Stream': self.model.feature_extractor.morph_stream,
            '4. Fusion_Pool': self.model.feature_extractor.fusion,
            '5. Similarity_Attention': self.model.similarity_calculator,
            '6. Tcn model': self.model.tcn_layer,
            '7. Entire_ProtoPNet': self.model
        }

        for name, mod in target_modules.items():
            self._register_hooks(name, mod)

    def _register_hooks(self, name, mod):
        def fwd_pre(m, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.fwd_events[name] = {'start': start}

        def fwd_post(m, input, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.fwd_events[name]['end'] = end

        self.hooks.append(mod.register_forward_pre_hook(fwd_pre))
        self.hooks.append(mod.register_forward_hook(fwd_post))

        def bwd_pre(m, grad_output):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.bwd_events[name] = {'start': start}

        def bwd_post_full(m, grad_input, grad_output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.bwd_events[name]['end'] = end

        try:
            self.hooks.append(mod.register_full_backward_pre_hook(bwd_pre))
            self.hooks.append(mod.register_full_backward_hook(bwd_post_full))
        except AttributeError:
            pass

    def calculate_times(self):
        torch.cuda.synchronize()
        for name, events in self.fwd_events.items():
            if 'start' in events and 'end' in events:
                self.fwd_times[name] = events['start'].elapsed_time(events['end'])

        for name, events in self.bwd_events.items():
            if 'start' in events and 'end' in events:
                self.bwd_times[name] = events['start'].elapsed_time(events['end'])

    def print_report(self):
        self.calculate_times()
        print("\n" + "=" * 80)
        print(f" 🚀 [极限 1ms 版本] 模型各模块耗时分析报告 (GPU Time, 单位: ms)")
        print("=" * 80)
        print(f"| {'Module Name':<25} | {'Forward (ms)':<15} | {'Backward (ms)':<15} | {'Total (ms)':<12} |")
        print("-" * 80)

        for name in sorted(self.fwd_times.keys()):
            fwd_t = self.fwd_times.get(name, 0.0)
            bwd_t = self.bwd_times.get(name, 0.0)
            total_t = fwd_t + bwd_t
            print(f"| {name:<25} | {fwd_t:<15.2f} | {bwd_t:<15.2f} | {total_t:<12.2f} |")

        print("=" * 80 + "\n")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ====================================================================
# 3. 执行入口区
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=49, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    parser.add_argument('--config', type=str, help='config file path',
                        default='./SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if os.path.exists(args.config):
        with open(args.config) as config_file:
            config = json.load(config_file)
        config['name'] = os.path.basename(args.config).replace('.json', '')
    else:
        config = {
            'name': 'test_config',
            'classifier': {
                'afr_reduced_dim': 128,
                'prototype_shape': [50, 128, 15],
                'prototype_num': 50,
                'num_classes': 5
            }
        }

    config['mode'] = 'normal'

    model = ProtoPNet(config).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n==============================================")
    print(f"✅ 模型总参数量 (Total Trainable Params): {total_params / 1e6:.4f} M")
    print(f"==============================================\n")

    profiler = BuiltInProfiler(model)
    model.train()

    print("[INFO] 正在执行 Warmup 预热...")
    x_warm = torch.rand([8, 1, 30000]).cuda()
    out_warm = model(x_warm)
    out_warm.sum().backward()

    profiler.fwd_events.clear()
    profiler.bwd_events.clear()

    print("[INFO] 正在进行真实耗时测试...")
    x = torch.rand([8, 1, 30000]).cuda()
    out = model(x)
    loss = out.sum()
    loss.backward()

    profiler.print_report()
    profiler.remove_hooks()

    print("\n[Your Output]:")
    print(out)
    print("Output Shape:", out.shape)