import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.norm = nn.Layernorm(channels)

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


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1,bias=False,isFTConv=True):
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
                                        stride=stride, padding=padding, dilation=1, groups=groups,bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,bias=False)

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
        if pad_values ==0:
            pad_left = torch.zeros(D_out,D_in,pad_length_left).cuda()
            pad_right = torch.zeros(D_out,D_in,pad_length_right).cuda()
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

        #convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        # self.ffn1act1 = nn.GELU()
        self.ffn1act1 = nn.PReLU()
        self.ffn1norm1 = nn.BatchNorm1d(nvars * dff)
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1norm2 = nn.BatchNorm1d(nvars * dmodel)
        # self.ffn1act2 = nn.GELU()
        self.ffn1act2 = nn.PReLU()
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel
        self.shortcut = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1)

    def forward(self, x):
        print('block shape: ', x.shape)
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M*D, N)

        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B*M, D, N)
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
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
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

        # stem layer & down sampling layers
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

        # cnn backbone
        self.num_stage = 2
        self.stages = nn.ModuleList()
        layer = Stage(4, 1, 51,5, dmodel=64, dw_model=64, nvars=1, small_kernel_merged=False, drop=0.1)
        self.stages.append(layer)
        layer = Stage(4, 1, 31, 5, dmodel=128, dw_model=128, nvars=1, small_kernel_merged=False, drop=0.1)
        self.stages.append(layer)

        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.flatten = nn.Flatten()


    def forward_feature(self, x):
        # x: [B, C, N]
        B, C, N = x.shape

        # 把 C 合并到 D 维度
        x = x.unsqueeze(2)  # [B, C, 1, N]
        x = x.reshape(B, 1, C, N)  # [B, M=1, D=C, N]

        print('motcn in shape: ', x.shape)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            print('down in shape: ', x.shape)
            x = self.downsample_layers[i](x)

            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)

            x = self.stages[i](x)
        return x

    def classification2(self, x, tags=None):
        # lkcnn backbone
        x = self.forward_feature(x).squeeze()
        print("lksleepnet embed shape: ", x.shape)
        x = self.avgpool(x)
        return x

    def forward(self, x, tags=None, pre_stage=2):
        x = self.classification2(x, tags=tags)
        return x



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


model = ModernTCN().cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量 (Total Trainable Params): {total_params} M")

x = torch.rand([8, 64, 30000]).cuda()
start = time.time()
out = model(x)
torch.cuda.synchronize()  # 确保 GPU 完成计算
end = time.time()
print("单次推理耗时: {:.4f} 秒".format(end - start))
print(out.shape)
