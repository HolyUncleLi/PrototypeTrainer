import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaborFourierPriorBank(nn.Module):
    def __init__(self, num_filters, kernel_size, sample_rate=100.0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.f_gabor = torch.linspace(0.5, 20.0, num_filters // 2)
        self.f_fourier = torch.linspace(0.5, 40.0, num_filters // 2)

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        f_g = self.f_gabor.view(-1, 1, 1).to(t.device)
        gauss = torch.exp(-((t) ** 2) / (2 * 0.1 ** 2))
        gabor_kernels = gauss * torch.cos(2 * math.pi * f_g * t)

        f_f = self.f_fourier.view(-1, 1, 1).to(t.device)
        fourier_kernels = torch.cos(2 * math.pi * f_f * t)
        return torch.cat([gabor_kernels, fourier_kernels], dim=0)


class RepPhysConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.conv_branch = nn.Conv1d(in_channels, out_channels, kernel_size, stride, self.padding, bias=False)

        if in_channels == 1:
            self.phys_bank = GaborFourierPriorBank(out_channels, kernel_size)
            self.phys_scale = nn.Parameter(torch.ones(out_channels, 1, 1))

        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.conv_branch(x)

        out_conv = self.conv_branch(x)
        if self.in_channels == 1:
            phys_weight = self.phys_bank.get_kernels() * self.phys_scale
            out_phys = F.conv1d(x, phys_weight, stride=self.stride, padding=self.padding)
            return out_conv + out_phys
        return out_conv

    def reparameterize(self):
        if self.deploy: return
        if self.in_channels == 1:
            phys_weight = self.phys_bank.get_kernels() * self.phys_scale
            self.conv_branch.weight.data = self.conv_branch.weight.data + phys_weight.data
        self.deploy = True
        if hasattr(self, 'phys_bank'):
            del self.phys_bank
            del self.phys_scale


import torch
import torch.nn as nn
import torch.nn.functional as F


class STMM_ManyToOne_Accelerator(nn.Module):
    def __init__(self, dim, threshold=0.85):
        super().__init__()
        # 捕捉局部时序上下文
        self.local_tcn = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.threshold = threshold

    def forward(self, x):
        """
        x shape: [B, L, C] -> e.g., [64, 10, 128]
        """
        x_t = x.transpose(1, 2)
        x_t = F.gelu(self.local_tcn(x_t)) + x_t
        x_out = x_t.transpose(1, 2)

        # =======================================================
        # 1. 训练阶段 (Training)：
        # 为保证并行张量对齐与梯度回传，只做数值平滑，不改变物理长度 L
        # =======================================================
        if self.training:
            sim = F.cosine_similarity(x_out[:, :-1, :], x_out[:, 1:, :], dim=-1)
            mask = (sim > self.threshold).float().unsqueeze(-1)
            smoothed_x = x_out.clone()
            # 根据掩码，平稳状态时特征平滑：获取前一时刻冗余特征 (式 5-9)
            smoothed_x[:, 1:, :] = x_out[:, 1:, :] * (1 - mask) + ((x_out[:, :-1, :] + x_out[:, 1:, :]) / 2) * mask
            return smoothed_x, None

        # =======================================================
        # 2. 推理阶段 (Inference/Deploy)：
        # 真正物理删除冗余节点，实现特征序列的动态缩短！
        # =======================================================
        B, L, C = x_out.shape
        merged_batch_list = []
        spans_batch_list = []  # 记录合并后每个节点包含的原始帧数（权重）

        for b in range(B):
            seq_feat = x_out[b]
            sim = F.cosine_similarity(seq_feat[:-1], seq_feat[1:], dim=-1)
            mask = (sim > self.threshold).bool()

            merged_seq = [seq_feat[0]]
            current_span = 1
            spans = []

            for t in range(L - 1):
                if mask[t]:
                    # 状态平稳：两个时间步高度同质化，物理合并为 1 个节点！
                    merged_seq[-1] = (merged_seq[-1] + seq_feat[t + 1]) / 2.0
                    current_span += 1
                else:
                    # 发生瞬态跳变：阻断融合，开辟新节点精准保留瞬变特征！
                    spans.append(current_span)
                    merged_seq.append(seq_feat[t + 1])
                    current_span = 1
            spans.append(current_span)

            # 缩短后的 Tensor，长度变为 L_new (L_new <= 10)
            merged_tensor = torch.stack(merged_seq)

            merged_batch_list.append(merged_tensor)
            spans_batch_list.append(spans)

        return merged_batch_list, spans_batch_list


class RepSleepNet(nn.Module):
    def __init__(self, num_classes=5, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = 128

        # 这里用占位替代你的原代码空间网络
        self.spatial_stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=63, stride=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, self.feature_dim, kernel_size=31, stride=2),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # STMM 时序自适应缩短模块
        self.stome_layer = STMM_ManyToOne_Accelerator(dim=self.feature_dim)

        # 分类器 FC
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.register_buffer('channel_mask', torch.ones(self.feature_dim))

    def forward(self, x):
        B = x.shape[0]
        epoch_len = x.shape[-1] // self.seq_len

        # --- 空间特征提取 ---
        x = x.view(B * self.seq_len, 1, epoch_len)
        feat = self.spatial_stem(x)
        feat = feat.squeeze(-1)
        feat = feat * self.channel_mask.view(1, -1)

        # 恢复时序结构：[B, 10, 128]
        feat_seq = feat.view(B, self.seq_len, self.feature_dim)

        # ========================================================
        # 【训练阶段】：提供 feat_pooled 进行蒸馏，且正常计算 Loss
        # ========================================================
        if self.training:
            # 1. 获取平滑后的时序序列 [B, 10, 128]
            smoothed_seq, _ = self.stome_layer(feat_seq)

            # 2. 计算用于知识蒸馏的聚合特征 feat_pooled [B, 128]
            feat_pooled = smoothed_seq.mean(dim=1)

            # 3. 对序列进行 FC 计算并最终求平均，输出 Many-to-One 预测结果 [B, 5]
            # (基于数学等价性，这与 self.fc(feat_pooled) 结果绝对一致)
            logits_seq = self.fc(smoothed_seq)  # [B, 10, 5]
            final_logits = logits_seq.mean(dim=1)  # [B, 5]

            return final_logits, feat_pooled

        # ========================================================
        # 【推理阶段】：真正发挥 STMM 物理加速作用的地方！
        # ========================================================
        else:
            # 1. 经过 STMM，获取动态缩短后的特征和对应权重
            merged_list, spans_list = self.stome_layer(feat_seq)

            final_logits_list = []
            feat_pooled_list = []

            # 边缘设备通常 B=1，这里用 for 循环模拟真实推理流
            for b in range(B):
                # 假设高度同质化，原本 10 个节点的序列短缩成了 3 个！
                short_feat = merged_list[b]  # 形状: [3, 128]
                spans = spans_list[b]  # 跨度: [3] -> 比如是 [4, 1, 5]

                # ============================================================
                # 全连接层 self.fc 只对缩短后的 3 个节点进行计算！
                # 原本需要做 10 次 128x5 矩阵乘法，现在只需做 3 次
                # ============================================================
                logits_short = self.fc(short_feat)  # 形状: [3, 5]

                # 3. 利用节点原有的 spans 作为权重 将局部决策融合为唯一的输出
                weights = torch.tensor(spans, dtype=logits_short.dtype, device=logits_short.device).unsqueeze(1)

                # print('logit short: ', logits_short.shape, logits_short)
                # print('weight: ',weights.shape,weights)
                # print('span: ', spans)
                final_logit = (logits_short * weights).sum(dim=0, keepdim=True) / sum(spans)

                # [1, 128] 推理时为了统一接口同样返回的 feat_pooled
                feat_pool = (short_feat * weights).sum(dim=0, keepdim=True) / sum(spans)

                final_logits_list.append(final_logit)
                feat_pooled_list.append(feat_pool)

            # 拼接并返回 [B, 5] 和 [B, 128]
            return torch.cat(final_logits_list, dim=0), torch.cat(feat_pooled_list, dim=0)

    def deploy_and_prune(self, prune_ratio=0.2):
        """
        剪枝函数：
        1. 首先执行结构重参数化。
        2. 计算BN层权重的L1范数，利用分位数执行物理通道剪枝。
        """
        # 1. 重参数化
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()

        # 2. 物理剪枝 (基于解释性打分/BN权重相对大小)
        bn_layer = self.spatial_stem[5]
        gamma = bn_layer.weight.data.abs()

        # 计算动态分位数阈值，强行剪掉最不重要的前 prune_ratio (如20%)
        threshold = torch.quantile(gamma, prune_ratio)

        # 寻找存活的通道索引
        alive_indices = torch.nonzero(gamma > threshold).squeeze()
        print(
            f"\n[INFO] 剪枝完成：保留了 {len(alive_indices)}/{self.feature_dim} 个通道！ (剪枝率: {prune_ratio * 100}%)")

        # 将低于阈值的通道特征拦截置零
        self.channel_mask[gamma <= threshold] = 0.0



'''
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

model = RepSleepNet().cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量 (Total Trainable Params): {total_params} M")

x = torch.rand([2, 1, 30000]).cuda()
out = model(x)
print(out, len(out), out[0].shape, out[1].shape)
'''


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
            '6. Entire_ProtoPNet': self.model
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
        print(f" [极限 1ms 版本] 模型各模块耗时分析报告 (GPU Time, 单位: ms)")
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
# 执行入口
# ====================================================================
import math
import warnings
import argparse
import os
import json
import time
if __name__ == '__main__':
    model = RepSleepNet().cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n==============================================")
    print(f"模型总参数量 (Total Trainable Params): {total_params / 1e6:.4f} M")
    print(f"目标达成校验: {'通过!' if total_params >= 1.8e6 else '未达标!'} (要求 >= 1.8M)")
    print(f"==============================================\n")

    profiler = BuiltInProfiler(model)
    model.eval()

    print("[INFO] 正在执行 Warmup 预热...")
    x_warm = torch.rand([8, 1, 30000]).cuda()
    out_warm = model(x_warm)
    out_warm[0].sum().backward()

    profiler.fwd_events.clear()
    profiler.bwd_events.clear()

    print("[INFO] 正在进行真实耗时测试...")
    x = torch.rand([8, 1, 30000]).cuda()
    out = model(x)
    loss = out[0].sum()
    loss.backward()

    profiler.print_report()
    profiler.remove_hooks()

    print("\n[Your Output]:")
    print(out)
    print("Output Shape:", len(out), out[0].shape)