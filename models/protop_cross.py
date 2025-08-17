import math
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


# ====================================================================
# 1. Feature Extraction: MRCNN
# ====================================================================
class GELU(nn.Module):
    def __init__(self): super(GELU, self).__init__()

    def forward(self, x): return torch.nn.functional.gelu(x)


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24), nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4), nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        )
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=25, bias=False, padding=200), nn.BatchNorm1d(64), self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2), nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3), nn.BatchNorm1d(128), self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3), nn.BatchNorm1d(128), self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.AFR = nn.Conv1d(128, afr_reduced_cnn_size, kernel_size=1)

    def forward(self, x):
        x1 = self.features1(x);
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat);
        x_concat = self.AFR(x_concat)
        return x_concat


class TCN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Conv1d(input_dim, input_dim, kernel_size=7, padding='same')

    def forward(self, x): return self.layer(x)


# ====================================================================
# 2. Base Prototype Libraries
# ====================================================================
class GaborFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A, self.mu, self.sigma = [nn.Parameter(p) for p in
                                       [torch.ones(self.num), torch.zeros(self.num), torch.ones(self.num) * 0.1]]
        self.f = nn.Parameter(torch.linspace(1.0, 30.0, num_filters) + torch.randn(num_filters) * 0.1)
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t, A, mu, sigma, f, phi = [p.view(-1, 1, 1) for p in
                                   [self.t.view(1, 1, -1), self.A, self.mu, self.sigma.abs() + 1e-4,
                                    self.f.clamp(0.1, 50.0), self.phi]]
        gauss = torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2));
        sinus = torch.cos(2 * torch.pi * f * t + phi)
        return A * gauss * sinus


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
        t, A, f, phi = [p.view(-1, 1, 1) for p in [self.t.view(1, 1, -1), self.A, self.f.clamp(0.1, 50.0), self.phi]]
        return A * torch.cos(2 * torch.pi * f * t + phi)


# ====================================================================
# 3. CORRECTED Core Model: ProtoPNet
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config

        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.mrcnn = MRCNN(afr_reduced_cnn_size)
        self.conv_features = TCN(afr_reduced_cnn_size)

        self.num_gabor_basis, self.num_fourier_basis = 20, 20
        self.prototype_kernel_size = self.cfg['classifier']['prototype_shape'][2]

        self.gabor_basis_bank = GaborFilterBank(self.num_gabor_basis, self.prototype_kernel_size)
        self.fourier_basis_bank = FourierFilterBank(self.num_fourier_basis, self.prototype_kernel_size)

        self.num_composite_prototypes = self.cfg['classifier']['prototype_num']
        num_total_basis = self.num_gabor_basis + self.num_fourier_basis
        self.mixing_weights = nn.Parameter(torch.rand(self.num_composite_prototypes, num_total_basis))
        nn.init.xavier_uniform_(self.mixing_weights)

        num_classes = self.cfg['classifier']['num_classes']
        self.bn = nn.BatchNorm1d(self.num_composite_prototypes)
        self.fc = nn.Linear(self.num_composite_prototypes, num_classes)

        # 初始化 min_distance 属性，这是一个好习惯
        self.min_distance = None

    def _l2_convolution(self, x, prototypes):
        C, W = x.shape[1], prototypes.shape[2]
        pad = (W - 1) // 2
        x2 = x ** 2
        ones_weight = torch.ones(1, C, W, device=x.device)
        x2_patch_sum = F.conv1d(x2, ones_weight, padding=pad)
        p2_sum = torch.sum(prototypes ** 2, dim=(1, 2)).view(-1, 1)
        xp = F.conv1d(x, prototypes, padding=pad)
        return F.relu(x2_patch_sum - 2 * xp + p2_sum)

    def forward(self, x):
        # 1. 提取特征
        conv_features = self.conv_features(self.mrcnn(x))
        C = conv_features.shape[1]

        # 2. 构建原型
        gabor_kernels = self.gabor_basis_bank.get_kernels().repeat(1, C, 1)
        fourier_kernels = self.fourier_basis_bank.get_kernels().repeat(1, C, 1)
        base_prototypes = torch.cat((gabor_kernels, fourier_kernels), dim=0)
        composite_prototypes = torch.matmul(F.relu(self.mixing_weights), base_prototypes.flatten(1))
        composite_prototypes = composite_prototypes.view(self.num_composite_prototypes, C, self.prototype_kernel_size)

        # 3. 计算距离图
        distance = self._l2_convolution(conv_features, composite_prototypes)

        # 4. 全局池化
        min_distance = -F.max_pool1d(-distance, kernel_size=distance.shape[2]).squeeze(2)

        # 5. *** 关键修复 ***
        # 将 min_distance 保存为模型属性
        self.min_distance = min_distance

        # 6. 计算相似度并分类
        similarity = torch.log((min_distance + 1) / (min_distance + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        return logits

'''
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
x = torch.rand([64, 1, 30000]).cuda()
out = model(x)
print(out, out.shape)
'''