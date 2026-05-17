import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import warnings
import argparse

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
# 1. 基础基底生成器 (保持物理先验)
# ====================================================================
class GaborFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A = nn.Parameter(torch.ones(self.num))
        self.mu = nn.Parameter(torch.zeros(self.num))
        self.sigma = nn.Parameter(torch.ones(self.num) * 0.1)
        self.f = nn.Parameter(torch.linspace(1.0, 40.0, num_filters))
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        A = self.A.view(-1, 1, 1)
        mu = self.mu.view(-1, 1, 1)
        sigma = self.sigma.abs().view(-1, 1, 1) + 1e-4
        f = self.f.clamp(0.1, 50.0).view(-1, 1, 1)
        phi = self.phi.view(-1, 1, 1)
        return A * torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * f * t + phi)


class FourierFilterBank(nn.Module):
    def __init__(self, num_filters: int, kernel_size: int, sample_rate: float = 100.0):
        super().__init__()
        self.num, self.ks = num_filters, kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, steps=kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.A = nn.Parameter(torch.ones(self.num))
        self.f = nn.Parameter(torch.linspace(1.0, 40.0, num_filters))
        self.phi = nn.Parameter(torch.zeros(self.num))

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        A = self.A.view(-1, 1, 1)
        f = self.f.clamp(0.1, 50.0).view(-1, 1, 1)
        phi = self.phi.view(-1, 1, 1)
        return A * torch.cos(2 * math.pi * f * t + phi)


# ====================================================================
# 2. LGWDS 提取器 (移除AdaptivePool，自然对齐)
# ====================================================================
class LearnableGaborStem(nn.Module):
    def __init__(self, out_channels=64, kernel_size=63, stride=5):
        super().__init__()
        self.padding = kernel_size // 2
        self.stride = stride
        self.mu_f = nn.Parameter(torch.rand(out_channels) * 30.0 + 0.5)
        self.sigma = nn.Parameter(torch.ones(out_channels) * 10.0)
        t = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) / 100.0
        self.register_buffer('t', t)

    def forward(self, x):
        t = self.t.view(1, 1, -1)
        mu_f, sigma = self.mu_f.view(-1, 1, 1), self.sigma.view(-1, 1, 1)
        env = torch.exp(-0.5 * (t ** 2) / (sigma ** 2))
        w_real = env * torch.cos(2 * math.pi * mu_f * t)
        w_imag = env * torch.sin(2 * math.pi * mu_f * t)

        # [B, 1, 30000] -> [B, 64, 6000]
        out_real = F.conv1d(x, w_real, stride=self.stride, padding=self.padding)
        out_imag = F.conv1d(x, w_imag, stride=self.stride, padding=self.padding)
        mag = torch.sqrt(out_real.pow(2) + out_imag.pow(2) + 1e-8)
        return mag, out_real


class LGWDS_Net(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.stem = LearnableGaborStem(64, kernel_size=63, stride=5)

        # [B, 64, 6000] -> [B, 128, 300] (降采样20倍)
        self.semantic_stream = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=15, stride=4, padding=7), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=7, stride=5, padding=3), nn.BatchNorm1d(128), nn.GELU()
        )
        # 形态流：大核捕获慢波
        self.morph_stream = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=51, stride=4, padding=25), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=21, stride=5, padding=10), nn.BatchNorm1d(128), nn.GELU()
        )
        self.fusion = nn.Conv1d(256, out_dim, kernel_size=1)

    def forward(self, x):
        mag, real = self.stem(x)
        sem = self.semantic_stream(mag)
        mor = self.morph_stream(real)
        return self.fusion(torch.cat([sem, mor], dim=1))


# ====================================================================
# 3. 核心：快速交叉注意力 Prototype 网络
# ====================================================================
class ProtoPNet(nn.Module):
    def __init__(self, config):
        super(ProtoPNet, self).__init__()
        self.cfg = config
        self.c_dim = self.cfg['classifier']['afr_reduced_dim']
        self.proto_num = self.cfg['classifier']['prototype_num']
        self.k_size = self.cfg['classifier']['prototype_shape'][2]
        self.num_classes = self.cfg['classifier']['num_classes']

        self.feature_extractor = LGWDS_Net(out_dim=self.c_dim)
        self.tcn_layer = EnhancedTCN(input_dim=128, num_levels=4)

        # 1. 基础模板库
        self.num_g, self.num_f, self.num_l = 20, 20, 10
        self.total_bases = self.num_g + self.num_f + self.num_l
        self.gabor_bank = GaborFilterBank(self.num_g, self.k_size)
        self.fourier_bank = FourierFilterBank(self.num_f, self.k_size)
        self.learnable_bank = nn.Parameter(torch.randn(self.num_l, 1, self.k_size))

        # 2. 空间-时间 投影权重 (核心创新：解决特征维度物理意义)
        # [原型数, 通道数, 基底数] -> 决定每个通道使用哪些频率波形
        self.mixing_weights = nn.Parameter(torch.randn(self.proto_num, self.c_dim, self.total_bases) * 0.02)

        # 3. 卷积交叉注意力机制 (Q, K 投影)
        self.d_k = 64
        self.W_q = nn.Conv1d(self.c_dim, self.d_k, 1, bias=False)
        self.W_k = nn.Conv1d(self.c_dim, self.d_k, 1, bias=False)

        self.bn = nn.BatchNorm1d(self.proto_num)
        self.fc = nn.Linear(self.proto_num, self.num_classes, bias=False)

        self.max_sim = None
        self.attention_maps = None

    def get_composite_prototypes(self):
        # 提取时域基底 [total_bases, 1, K]
        g_k = self.gabor_bank.get_kernels()
        f_k = self.fourier_bank.get_kernels()
        bases = torch.cat([g_k, f_k, self.learnable_bank], dim=0).squeeze(1)  # [50, K]

        # 生成高维原型 [Proto_num, C, K]
        # 公式: P_{p,c,k} = \sum_{b} W_{p,c,b} * Bases_{b,k}
        prototypes = torch.matmul(self.mixing_weights, bases)
        return prototypes

    def forward(self, x):
        # 1. 提取特征 [B, C, L]
        features = self.feature_extractor(x)
        features = self.tcn_layer(features)

        # 2. 生成原型 [Proto_num, C, K]
        prototypes = self.get_composite_prototypes()

        # 3. 极速交叉注意力 (用卷积实现 Q-K 匹配)
        Q_proto = self.W_q(prototypes)  # [Proto_num, d_k, K]
        K_feat = self.W_k(features)  # [B, d_k, L]

        # 将 Q 翻转作为卷积核 (因为 F.conv1d 内部是互相关计算)
        # S_{b, p, l} = Conv1d(K_feat, Q_proto)
        # 形状: K_feat=[B, d_k, L], Weight=[Proto_num, d_k, K] -> Out=[B, Proto_num, L - K + 1]
        similarity_seq = F.conv1d(K_feat, Q_proto) / math.sqrt(self.d_k)

        # 取最大相似度位置
        self.attention_maps = similarity_seq
        self.max_sim, _ = torch.max(similarity_seq, dim=-1)  # [B, Proto_num]

        # 4. 分类
        logits = self.fc(self.bn(self.max_sim))
        return logits

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
            '1. GaborConv_Layer': self.model.feature_extractor.stem,
            '2. Semantic_Stream': self.model.feature_extractor.semantic_stream,
            '3. Morph_Stream': self.model.feature_extractor.morph_stream,
            '4. Fusion_Pool': self.model.feature_extractor.fusion,
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
                'prototype_shape': [1, 128, 50],
                'prototype_num': 300,
                'num_classes': 5
            }
        }

    config['mode'] = 'normal'

    model = ProtoPNet(config).cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n==============================================")
    print(f"✅ 模型总参数量 (Total Trainable Params): {total_params / 1e6:.4f} M")
    print(f"✅ 目标达成校验: {'通过!' if total_params >= 1.8e6 else '未达标!'} (要求 >= 1.8M)")
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