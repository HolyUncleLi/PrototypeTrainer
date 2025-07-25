import math
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from .FGN import getFGN


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def layer_norm(x):
    eps = 1e-6
    mean = torch.mean(x, 2, keepdims=True)  ## [?, C, 1]
    std = torch.std(x, 2, keepdims=True)  ## [?, C, 1]
    return (x - mean) / (std + eps)


class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        self.protop_num = self.cfg['classifier']['prototype_num']
        # self.feature_dim = self.cfg['classifier']['feature_dim']
        afr_reduced_cnn_size = self.cfg['classifier']['afr_reduced_dim']
        self.feature_dim = afr_reduced_cnn_size
        # self.xfeat = torch.rand([64,27,777])

        # 特征提取
        self.inplanes = 128
        self.mrcnn = MRCNN(afr_reduced_cnn_size)  # use MRCNN_SHHS for SHHS dataset
        # self.arf = self._make_layer(SEBasicBlock, 128, 1)
        self.fgn = getFGN()
        self.gate_conv = nn.Conv1d(afr_reduced_cnn_size, 2, kernel_size=9, padding='same', stride=1)
        self.prototype_shape = self.cfg['classifier']['prototype_shape']
        self.bn0 = nn.BatchNorm1d(self.protop_num)
        self.bn1 = nn.BatchNorm1d(self.protop_num)
        self.fc = nn.Linear(int(2 * self.protop_num), self.cfg['classifier']['num_classes'], bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)


        # 模板
        self.gabor = GaborFilterBank(num_filters=5,
                                    kernel_size=128,
                                    sample_rate=100)
        self.fourier = FourierFilterBank(num_filters=5,
                                         kernel_size=128,
                                         sample_rate=100)

        # self.prototype_base = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        # self.prototype_vectors = torch.rand(self.prototype_shape, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.distance = None
        self.min_distance = None
        self.gate = None
        self.similarity = None
        self.proportion = None

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _l2_convolution(self, x):
        g_proto = self.gabor(x)
        f_proto = self.fourier(x)
        prototype_vectors = torch.cat((g_proto, f_proto), dim=1)

        x2 = x ** 2
        x2_patch_sum = F.conv1d(input=x2, weight=self.ones)
        p2 = prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        p2_reshape = p2.view(-1, 1)

        # x 和 prototype 进行匹配
        # xp = F.conv1d(input=x, weight=self.prototype_vectors)
        # distance = -2 * prototype_vectors + p2_reshape + x2_patch_sum
        # print(prototype_vectors.shape, p2_reshape.shape, x2_patch_sum.shape)
        distance = -2 * prototype_vectors + x2_patch_sum
        return distance

    def prototype_distance(self, x):
        x_feat = self.mrcnn(x)
        # conv_features = self.arf(x_feat)
        conv_features = self.fgn(x_feat)

        self.xfeat = conv_features
        distance = self._l2_convolution(conv_features)

        ## Gate
        gate = self.gate_conv(conv_features)  ##[?, 2, K]
        gate = gate.max(2).values  ##[?, 2]
        #    gate = torch.mean(gate, 2, keepdims=False)                ##[?, 2]
        gate = gate.view(-1, 1, 2)
        gate = torch.softmax(gate, dim=-1)
        return distance, gate

    def _layer_norm(self, x):
        eps = 1e-6
        mean = torch.mean(x, 2, keepdims=True)  ## [?, C, 1]
        std = torch.std(x, 2, keepdims=True)  ## [?, C, 1]
        # return (x - mean) / (std + eps)
        return x / (std + eps)

    def _min_max_scaling(self, feature):
        ## feature shape [B, P]
        min_feature = torch.min(feature, -1).values  ##[B]
        max_feature = torch.max(feature, -1).values  ##[B]
        return (feature - min_feature.unsqueeze(1)) / (max_feature.unsqueeze(1) - min_feature.unsqueeze(1))

    def forward(self, x):
        epsilon = 1e-4
        self.distance, self.gate = self.prototype_distance(x)
        distance = self._layer_norm(self.distance)  # [?, P, K]
        distance = distance.clamp(min=epsilon)

        ## version 2
        similarity = torch.log((distance + 1) / (distance + epsilon))  # [B, P, K]
        # similarity = torch.exp(-distance)
        proportion_similarity = torch.mean(similarity, dim=2)  # [B, P]
        proportion_similarity = self.bn0(proportion_similarity)
        proportion_similarity = self.relu1(proportion_similarity)
        # proportion_similarity = self._min_max_scaling(proportion_similarity)
        self.proportion = proportion_similarity

        min_distance_1 = -torch.max(-distance, dim=-1).values  # [?, P]
        self.min_distance = min_distance_1.clone()
        wave_similarity = torch.log((min_distance_1 + 1) / (min_distance_1 + epsilon))  # [B, P]
        # wave_similarity = torch.exp(-min_distance_1)
        wave_similarity = self.bn1(wave_similarity)
        wave_similarity = self.relu2(wave_similarity)
        # wave_similarity = self._min_max_scaling(wave_similarity)
        self.similarity = wave_similarity

        similarity_sum = torch.stack([proportion_similarity, wave_similarity], 2).view(-1, int(2 * self.protop_num))
        logits_sum = self.fc(similarity_sum)
        probability = torch.softmax(logits_sum, -1)
        return torch.log(probability)


class Feature_Extraction(nn.Module):
    def __init__(self, config):
        super(Feature_Extraction, self).__init__()
        self.cfg = config
        self.in_features = config['feature_pyramid']['dim']
        self.feature_dim = self.cfg['classifier']['feature_dim']

        self.conv_batch1 = nn.Sequential(
            nn.Conv1d(self.in_features, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.conv_batch2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.conv_batch3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.conv_batch4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))
        self.conv_batch5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.conv_batch6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.conv_batch7 = nn.Sequential(
            nn.Conv1d(256, self.feature_dim, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

        self.basic_blocks_1 = clones(DarkResidualBlock(32), 4)
        self.basic_blocks_2 = clones(DarkResidualBlock(64), 4)
        self.basic_blocks_3 = clones(DarkResidualBlock(128), 8)
        self.basic_blocks_4 = clones(DarkResidualBlock(256), 8)
        self.basic_blocks_5 = clones(DarkResidualBlock(512), 4)

    def _residual_blocks(self, blocks, out):
        residual = [out]
        for i, bb in enumerate(blocks):
            out = bb(out)
            if len(residual) > 1:
                for residual_tensor in residual[:-1]:
                    out += residual_tensor
            residual.append(out)
        return out

    def forward(self, x):
        # x = x.transpose(1, 2)
        out = self.conv_batch1(x)
        out = self.conv_batch2(out)
        out = self._residual_blocks(self.basic_blocks_1, out)
        out = self.conv_batch3(out)
        out = self._residual_blocks(self.basic_blocks_2, out)
        out = self.conv_batch4(out)
        out = self._residual_blocks(self.basic_blocks_3, out)
        out = self.conv_batch5(out)
        out = self._residual_blocks(self.basic_blocks_4, out)
        # out = self.conv_batch6(out)
        # out = self._residual_blocks(self.basic_blocks_5, out)
        out = self.conv_batch7(out)
        return out


class DarkResidualBlock(nn.Module):
    def __init__(self, channel_in):
        super(DarkResidualBlock, self).__init__()

        self.reduced_channel = channel_in // 2
        self.channel_in = channel_in

        self.conv_batch1 = nn.Sequential(
            nn.Conv1d(self.channel_in, self.reduced_channel, kernel_size=1),
            nn.LeakyReLU(),
        )
        self.conv_batch2 = nn.Sequential(
            nn.Conv1d(self.reduced_channel, self.channel_in, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.conv_batch1(x)
        out = self.conv_batch2(out)
        out = layer_norm(out)
        out += residual
        return out


class TCN(nn.Module):

    def __init__(self, config):
        super(TCN, self).__init__()

        self.config = config
        self.cfg = config['classifier']
        self.block_num = self.cfg['block_num']
        self.model_dim = self.cfg['feedforward_dim']
        # self.in_features = config['feature_pyramid']['dim']
        self.in_features = self.cfg['afr_reduced_dim']
        self.out_features = self.cfg['feedforward_dim']

        self.norm = LayerNorm(self.in_features)
        self.querys = clones(nn.Conv1d(self.in_features, self.in_features, kernel_size=7,
                                       stride=1, padding='same'), self.block_num)
        self.tps = clones(TP(self.config), self.block_num)
        self.sublayeroutput = SublayerOutput(self.in_features)
        self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
        self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        self.fc = nn.Linear(self.model_dim, self.cfg['num_classes'])

    def forward(self, x):
        for k in range(self.block_num):
            x = self.querys[k](x)
            ff = self.tps[k]
            x = self.sublayeroutput(x, ff)
        encoded_features = self.norm(x)

        return encoded_features

        # encoded_features = encoded_features.transpose(1, 2)
        # a_states = torch.tanh(self.w_ha(encoded_features))
        # alpha = torch.softmax(self.w_at(a_states), dim=1).view(encoded_features.size(0), 1, encoded_features.size(1))
        # x = torch.bmm(alpha, a_states).view(encoded_features.size(0), -1)
        # out = self.fc(x)

        # return out


class SublayerOutput(nn.Module):
    def __init__(self, size):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, layer):
        return x + layer(self.norm(x))


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return (x - mean) / (std + self.eps)


class TPB(nn.Module):

    def __init__(self, config, dilation):
        super(TPB, self).__init__()

        self.channel_num = config['classifier']['channel_num']
        self.kernel_size = config['classifier']['kernel_size']
        self.strides_num = config['classifier']['strides_num']
        self.in_features = config['classifier']['afr_reduced_dim']
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=self.channel_num,
                      kernel_size=1, stride=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.channel_num, out_channels=self.channel_num,
                      kernel_size=self.kernel_size, stride=self.strides_num, padding='same',
                      dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.channel_num, out_channels=self.in_features,
                      kernel_size=1, stride=1, padding='same')
        )

    def forward(self, x):
        return self.conv(x)


class TP(nn.Module):

    def __init__(self, config):
        super(TP, self).__init__()

        self.config = config
        self.repeat_num = config['classifier']['repeat_num']
        self.tpbs = nn.ModuleList([TPB(self.config, 1) for k in range(self.repeat_num)])

    def forward(self, x):
        for j in range(self.repeat_num):
            x = self.tpbs[j](x)
        return x


# ====================================================================================
##              AttnSleep Feature Extraction
# ====================================================================================
class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class GaborFilterBank(nn.Module):
    """
    5 可微 Gabor 模板，对应 K-复合、纺锤、… 等
    f_targets 为生理上期望的中心频率（Hz）
    """
    def __init__(self,
                 num_filters: int = 5,
                 kernel_size: int = 129,
                 sample_rate: float = 100.0,
                 f_targets: torch.Tensor = None):
        super().__init__()
        self.num = num_filters
        self.ks = kernel_size
        t = torch.linspace(-kernel_size//2, kernel_size//2,
                           steps=kernel_size) / sample_rate
        self.register_buffer('t', t)        # (ks,)
        # 可学习参数
        self.A     = nn.Parameter(torch.ones(self.num))
        self.mu    = nn.Parameter(torch.zeros(self.num))
        self.sigma = nn.Parameter(torch.ones(self.num)*0.1)
        # 初始化 f_k 到生理 target
        if f_targets is None:
            f_targets = torch.linspace(1.0, 15.0, num_filters)
        self.register_buffer('f_target', f_targets)
        self.f     = nn.Parameter(f_targets.clone())
        self.phi   = nn.Parameter(torch.zeros(self.num))

    def forward(self, x):
        # x: (B,1,T) → out: (B,5,T)
        t     = self.t.view(1,1,-1)                # (1,1,ks)
        A     = self.A.view(-1,1,1)                # (5,1,1)
        mu    = self.mu.view(-1,1,1)
        sigma = self.sigma.abs().view(-1,1,1) + 1e-3
        f     = self.f.clamp(0.1, 50.0).view(-1,1,1)
        phi   = self.phi.view(-1,1,1)

        gauss = torch.exp(-((t - mu)**2)/(2*sigma**2))
        sinus = torch.cos(2*torch.pi*f*t + phi)
        kern  = A * gauss * sinus                  # (5,1,ks)
        out   = F.conv1d(x, kern.transpose(1,2))
        return out                                # (B,5,T)

    def gabor_losses(self, λ_freq=1.0, λ_l1=1e-3):
        # 频率对齐损失：(f_k - f_target)^2
        loss_freq = (self.f - self.f_target).pow(2).sum()
        # 稀疏化：振幅与带宽倒数
        loss_l1   = self.A.abs().sum() + (1.0/self.sigma.abs()).sum()
        return λ_freq * loss_freq, λ_l1 * loss_l1


class FourierFilterBank(nn.Module):
    """
    5 可微 Fourier 模板，对应 δ, θ, α, β, γ 频段
    f_targets 为各带中心频率（Hz）
    """
    def __init__(self,
                 num_filters: int = 5,
                 kernel_size: int = 129,
                 sample_rate: float = 100.0,
                 f_targets: torch.Tensor = None):
        super().__init__()
        self.num = num_filters
        self.ks = kernel_size
        t = torch.linspace(-kernel_size//2, kernel_size//2,
                           steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        # 如果用户没传，默认中心频率为 δ,θ,α,β,γ
        if f_targets is None:
            f_targets = torch.tensor([2.0, 6.0, 10.0, 20.0, 35.0])
        self.register_buffer('f_target', f_targets)
        # 学习参数
        self.A   = nn.Parameter(torch.ones(self.num))
        self.f   = nn.Parameter(f_targets.clone())
        self.phi = nn.Parameter(torch.zeros(self.num))

    def forward(self, x):
        # x: (B,1,T) → out: (B,5,T)
        t   = self.t.view(1,1,-1)                   # (1,1,ks)
        A   = self.A.view(-1,1,1)
        f   = self.f.clamp(0.1, 50.0).view(-1,1,1)
        phi = self.phi.view(-1,1,1)

        # 纯正弦
        kern = A * torch.cos(2*torch.pi*f*t + phi)  # (5,1,ks)
        out  = F.conv1d(x, kern.transpose(1,2))
        return out                                 # (B,5,T)

    def fourier_losses(self, λ_freq=1.0, λ_l1=1e-3):
        # 频率对齐
        loss_freq = (self.f - self.f_target).pow(2).sum()
        # 稀疏
        loss_l1   = self.A.abs().sum()
        return λ_freq * loss_freq, λ_l1 * loss_l1


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
x = torch.rand([64,1,30000]).cuda()
out = model(x)
print(out, out.shape)
'''