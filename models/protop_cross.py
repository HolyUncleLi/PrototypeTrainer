# --- protop_cross_v2.py (Bug Fixed Version) ---

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels);
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
        out = self.gelu(self.bn1(self.conv1(x)));
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x);
        out = self.gelu(out)
        return out


class EEGNetProto_Light(nn.Module):
    def __init__(self, input_channels, afr_reduced_cnn_size, block, num_blocks, fixed_output_size=256):
        super(EEGNetProto_Light, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=100, stride=2, padding=49, bias=False)
        self.bn1 = nn.BatchNorm1d(32);
        self.gelu = nn.GELU()
        self.pool1 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=fixed_output_size)
        self.final_conv = nn.Conv1d(256, afr_reduced_cnn_size, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1);
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.gelu(self.bn1(self.conv1(x)))
        out = self.pool1(out);
        out = self.layer1(out);
        out = self.layer2(out)
        out = self.layer3(out);
        out = self.layer4(out)
        out = self.adaptive_pool(out);
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
        # ========== BUG FIX START ==========
        # 将 t (时间轴) 的塑形与其它参数 (滤波器轴) 的塑形分开
        t = self.t.view(1, 1, -1)  # Shape: [1, 1, 15]

        # 将所有滤波器参数塑形为 [20, 1, 1]
        A, mu, sigma, f, phi = [p.view(-1, 1, 1) for p in
                                [self.A, self.mu, self.sigma.abs() + 1e-4,
                                 self.f.clamp(0.1, 50.0), self.phi]]
        # ========== BUG FIX END ==========

        gauss = torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2));
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
        # ========== BUG FIX START ==========
        # 将 t (时间轴) 的塑形与其它参数 (滤波器轴) 的塑形分开
        t = self.t.view(1, 1, -1)  # Shape: [1, 1, 15]

        # 将所有滤波器参数塑形为 [20, 1, 1]
        A, f, phi = [p.view(-1, 1, 1) for p in [self.A, self.f.clamp(0.1, 50.0), self.phi]]
        # ========== BUG FIX END ==========

        return A * torch.cos(2 * torch.pi * f * t + phi)


# ====================================================================
# 2. 改进的核心模块 (TCN)
# ====================================================================

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation // 2,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels);
        self.gelu = nn.GELU();
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
# 3. 最终的改进模型: ProtoPNet (与您的训练脚本兼容)
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config

        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.prototype_kernel_size = self.cfg['classifier']['prototype_shape'][2]
        self.num_composite_prototypes = self.cfg['classifier']['prototype_num']
        num_classes = self.cfg['classifier']['num_classes']

        self.feature_extractor = EEGNetProto_Light(
            input_channels=1, afr_reduced_cnn_size=afr_reduced_cnn_size,
            block=ResidualBlock, num_blocks=[2, 2, 2, 2], fixed_output_size=256
        )
        self.tcn_layer = EnhancedTCN(input_dim=afr_reduced_cnn_size, num_levels=4)

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

    def _l2_convolution(self, x, prototypes):
        C, W = x.shape[1], prototypes.shape[2]
        pad = (W - 1) // 2
        x2 = x ** 2;
        ones_weight = torch.ones(1, C, W, device=x.device)
        x2_patch_sum = F.conv1d(x2, ones_weight, padding=pad)
        p2_sum = torch.sum(prototypes ** 2, dim=(1, 2)).view(-1, 1)
        xp = F.conv1d(x, prototypes, padding=pad)
        return F.relu(x2_patch_sum - 2 * xp + p2_sum)

    def forward(self, x, return_indices=False):
        conv_features = self.feature_extractor(x)
        temporal_features = self.tcn_layer(conv_features)
        C = temporal_features.shape[1]

        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        learnable_kernels = self.learnable_basis_bank.repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels, learnable_kernels), dim=0)

        composite_prototypes = torch.matmul(F.relu(self.mixing_weights), base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        distance = self._l2_convolution(temporal_features, composite_prototypes)
        pooled_output = F.max_pool1d(-distance, kernel_size=distance.shape[2], return_indices=True)
        self.min_distance = -pooled_output[0].squeeze(2)
        self.min_indices = pooled_output[1].squeeze(2)

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