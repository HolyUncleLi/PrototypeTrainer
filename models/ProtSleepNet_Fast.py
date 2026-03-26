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
        # [极限优化]: Stride=10, 序列 30000 -> 3000。耗时直接跌破 1ms
        out_real = F.conv1d(x, w_real, stride=1, padding=self.padding)
        out_imag = F.conv1d(x, w_imag, stride=1, padding=self.padding)
        magnitude = torch.sqrt(out_real.pow(2) + out_imag.pow(2) + 1e-8)
        return magnitude, out_real


class SemanticStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        '''
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=15, stride=3, padding=7, bias=False),
            nn.BatchNorm1d(256), nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=5)  # 序列 1000 -> 200
        )

        self.heavy_block = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(384), nn.GELU(),
            nn.Conv1d(384, 384, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(384), nn.GELU(),
            nn.Conv1d(384, 128, kernel_size=1)
        )
        '''
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
        # 固定尺寸为 194
        self.pool = nn.AdaptiveAvgPool1d(194)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_extractor(x)
        return self.pool(x)  # 输出 [B, 128, 194]


class MorphologicalStream(nn.Module):
    def __init__(self, in_channels=64, out_channels=128):
        super().__init__()

        # ------------------------------------------------------------------
        # [阶段 1]: 超大核步长卷积 (The Wide-Sweep Extractor)
        # 物理意义：kernel=51 相当于 51ms 的物理时间窗，能完整包裹高频尖波/纺锤波的单次震荡。
        # 用 stride=5 替代原来的 AvgPool，让网络自适应学习降采样，防止尖峰特征被“抹平”。
        # 序列变化: 30000 -> 6000
        # 参数量: 64(in) * 96(out) * 51(kernel) = ~313K
        # ------------------------------------------------------------------
        self.stem_large_kernel = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=51, stride=5, padding=25),
            nn.BatchNorm1d(96),
            nn.GELU()
        )

        # ------------------------------------------------------------------
        # [阶段 2]: 峰值保留降采样
        # 物理意义：MaxPool 能够在局部时间窗内保留最显著的脑电形态突变(如波峰/波谷)，
        # 这是典型的 EEG 提纯策略。
        # 序列变化: 6000 -> 1200
        # ------------------------------------------------------------------
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=5)

        # ------------------------------------------------------------------
        # [阶段 3]: 宏观形态匹配滤波器 (The Macro-Wave Matched Filter)
        # 物理意义：[核心亮点] 此时时间序列已经被压缩了 25 倍(5x5)。
        # 这里的 kernel=21 实际上拥有 21 * 25 = 525 个采样点 (0.525秒) 的等效感受野！
        # 这个巨大的感受野刚好完美覆盖了一个 K-复合波或慢波(Delta波)的核心物理跨度。
        # 参数量: 96(in) * 128(out) * 21(kernel) = ~258K
        # ------------------------------------------------------------------
        self.macro_large_kernel = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=21, padding=10),
            nn.BatchNorm1d(128),
            nn.GELU()
        )

        # ------------------------------------------------------------------
        # [阶段 4]: 再次安全压缩并对齐
        # 序列变化: 1200 -> 200
        # ------------------------------------------------------------------
        self.pool2 = nn.MaxPool1d(kernel_size=6, stride=6)

        # ------------------------------------------------------------------
        # [阶段 5]: 最终微调对齐到目标尺寸 (200 -> 194)
        # 由于前方已经将序列安全压缩到了 200，这里的自适应池化只会做极其轻微的插值调整，
        # 彻底避免了原始代码中直接用它压榨长序列带来的灾难性信息丢失。
        # ------------------------------------------------------------------
        self.final_pool = nn.AdaptiveAvgPool1d(194)

    def forward(self, x):
        # 验证输入维度
        x = self.stem_large_kernel(x)  # -> [B, 96, 6000]
        x = self.pool1(x)  # -> [B, 96, 1200]
        x = self.macro_large_kernel(x)  # ->[B, 128, 1200]
        x = self.pool2(x)  # ->[B, 128, 200]
        out = self.final_pool(x)  # ->[B, 128, 194]

        return out


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
# 辅组模块 [保持原型]
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
        '''
        # 传统mrcnn方法
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.GELU()
        )
        self.feature_extractor = EEGNetProto_Slim(
            input_channels=64, afr_reduced_cnn_size=afr_reduced_cnn_size,
            block=ResidualBlock, num_blocks=[2, 2, 2, 2], fixed_output_size=256
        )

        self.tcn_layer = EnhancedTCN(input_dim=afr_reduced_cnn_size, num_levels=4)
        '''


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
        # print('x shape: ', x.shape)
        # x = self.stem(x)
        features = self.feature_extractor(x)
        C = features.shape[1]
        # print('features shape: ', features.shape)
        self.current_gabor_k = self.gabor_basis_bank.get_kernels()
        self.current_fourier_k = self.fourier_basis_bank.get_kernels()
        # print('gabor_k shape: ', self.current_gabor_k.shape)
        # print('fourier_k shape: ', self.current_fourier_k.shape)

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
            '5. Similarity_Einsum': self.model.similarity_calculator,
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


'''
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
'''

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