import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborFilterBank(nn.Module):
    def __init__(self,
                 num_filters: int = 4,
                 kernel_size: int = 129,
                 sample_rate: float = 100.0):
        super().__init__()
        self.num_filters = num_filters
        self.ks = kernel_size
        t = torch.linspace(
            -kernel_size // 2, kernel_size // 2,
            steps=kernel_size
        ) / sample_rate  # 单位秒
        # 初始化可学习参数
        self.A = nn.Parameter(torch.ones(num_filters))  # 振幅
        self.mu = nn.Parameter(torch.zeros(num_filters))  # 中心点（秒）
        self.sigma = nn.Parameter(torch.ones(num_filters) * 0.1)  # 带宽（秒）
        self.f = nn.Parameter(torch.linspace(1, 15, num_filters))  # 频率 Hz
        self.phi = nn.Parameter(torch.zeros(num_filters))  # 相位

        # 保留 t 作为 buffer
        self.register_buffer('t', t)

    def forward(self, x: torch.Tensor):
        """
        x: (B,1,T_in)
        返回: (B, num_filters, T_in)
        """
        # 计算 Gabor 核: (num_filters, kernel_size)
        # p_k(t) = A exp(−(t−μ)^2/(2σ^2)) cos(2π f t + φ)
        t = self.t.unsqueeze(0)  # (1, ks)
        A = self.A.view(-1, 1)  # (num,1)
        mu = self.mu.view(-1, 1)
        sigma = self.sigma.view(-1, 1).abs() + 1e-3
        f = self.f.view(-1, 1).clamp(0.1, 30.0)
        phi = self.phi.view(-1, 1)

        gauss = torch.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
        sinus = torch.cos(2 * torch.pi * f * t + phi)
        kernels = A * gauss * sinus  # (num, ks)
        kernels = kernels.unsqueeze(1)  # (num, 1, ks)

        # 一维卷积
        out = F.conv1d(x, kernels, padding=self.ks // 2)
        return out


class FourierFilterBank(nn.Module):
    def __init__(self,
                 num_filters: int = 4,
                 kernel_size: int = 129,
                 sample_rate: float = 100.0,
                 num_bases: int = 16,
                 f_min: float = 0.5,
                 f_max: float = 30.0):
        super().__init__()
        self.num_filters = num_filters
        self.ks = kernel_size
        self.sample_rate = sample_rate
        self.num_bases = num_bases

        # 预定义频点
        freqs = torch.linspace(f_min, f_max, num_bases)
        t = torch.linspace(-kernel_size // 2, kernel_size // 2,
                           steps=kernel_size) / sample_rate

        # 学习系数 a_{k,n}, b_{k,n}
        self.a = nn.Parameter(torch.randn(num_filters, num_bases) * 0.1)
        self.b = nn.Parameter(torch.randn(num_filters, num_bases) * 0.1)
        # 频带先验权重 w_{k,n}，小值→对应频段鼓励
        self.register_buffer('w', torch.ones(num_filters, num_bases))
        # buffer
        self.register_buffer('t', t)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor):
        """
        x: (B,1,T_in)
        返回: (B, num_filters, T_in)
        """
        # 先构造 cos/sin basis: (num_bases, ks)
        t = self.t.unsqueeze(0)  # (1, ks)
        freqs = self.freqs.view(-1, 1)  # (nb,1)
        cos_basis = torch.cos(2 * torch.pi * freqs * t)  # (nb, ks)
        sin_basis = torch.sin(2 * torch.pi * freqs * t)  # (nb, ks)

        # 对每个滤波器 sum_n a_{k,n} cos + b_{k,n} sin
        # a: (num_filters, nb), cos_basis: (nb, ks)
        kernels = (
                self.a.unsqueeze(2) * cos_basis.unsqueeze(0)
                + self.b.unsqueeze(2) * sin_basis.unsqueeze(0)
        )  # (num_filters, nb, ks)
        kernels = kernels.sum(dim=1)  # (num_filters, ks)
        kernels = kernels.unsqueeze(1)  # (num_filters,1,ks)

        out = F.conv1d(x, kernels, padding=self.ks // 2)
        return out


class ConceptTemplates(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 feature_dim: int = 8):
        super().__init__()
        # 每个阶段对应一个可学习概念向量 p_k ∈ R^feature_dim
        self.p = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.1)

    def forward(self):
        # 返回 (K, feature_dim)
        return self.p


class SleepStageNet(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 sample_rate: float = 100.0):
        super().__init__()
        # 1）可微滤波器组
        self.gabor   = GaborFilterBank( num_filters=4,
                                        kernel_size=129,
                                        sample_rate=sample_rate)
        self.fourier = FourierFilterBank(num_filters=4,
                                         kernel_size=129,
                                         sample_rate=sample_rate,
                                         num_bases=16)
        # 特征维度 C = 4+4 = 8
        C = 8
        # 2）原型模板 p_k ∈ R^C
        self.prototypes = nn.Parameter(torch.randn(num_classes, C)*0.1)

    def forward(self, x):
        # x: (B,1,3000)
        g = self.gabor(x)                 # (B,4,3000)
        f = self.fourier(x)               # (B,4,3000)
        feats = torch.cat([g,f], dim=1)   # (B,8,3000)

        # 全局池化 → 特征向量 c: (B,8)
        c = feats.mean(dim=2)

        # 原型匹配分类：logits_{i,k} = c_i · p_k
        logits = c @ self.prototypes.t()  # (B, K)
        return logits, feats, c


# ----------------------
# 训练示例
# ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SleepStageNet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()

# 超参
λ_align = 1.0
λ_band = 10.0
λ_G_l1 = 1e-3
λ_F_l1 = 1e-3
λ_prior = 1e-2
λ_σ = 1e-2
f_min, f_max = 0.5, 30.0

for epoch in range(100):
    for x, y in dataloader:  # x:(B,1,3000), y:(B,)
        x, y = x.to(device), y.to(device)
        logits, feats, cam = net(x, y)

        # 1) 交叉熵
        loss_ce = ce_loss(logits, y)

        # 2) 概念向量 c^i 提取
        # feats: (B,C,T), cam: (B,1,T)
        w = cam / (cam.sum(dim=2, keepdim=True) + 1e-6)  # (B,1,T)
        c = (feats * w).sum(dim=2)  # (B,C)

        # 3) 模板对齐
        p = net.templates()  # (K, C)
        p_y = p[y]  # (B, C)
        loss_align = F.mse_loss(c, p_y)

        # 4) 频带约束 (对 Gabor f_k)
        f_k = net.gabor.f.clamp(0, 100)
        loss_band = ((F.relu(f_k - f_max)
                      + F.relu(f_min - f_k)) ** 2).sum()

        # 5) Gabor L1 稀疏
        loss_G_l1 = (net.gabor.A.abs()
                     + λ_σ * (1.0 / net.gabor.sigma.abs())).sum()

        # 6) Fourier L1 稀疏
        loss_F_l1 = (net.fourier.a.abs() + net.fourier.b.abs()).sum()

        # 7) Fourier 频带软先验
        w_fb = net.fourier.w
        a, b = net.fourier.a, net.fourier.b
        loss_prior = (w_fb * (a ** 2 + b ** 2)).sum()

        # 汇总
        loss = (
                loss_ce
                + λ_align * loss_align
                + λ_band * loss_band
                + λ_G_l1 * loss_G_l1
                + λ_F_l1 * loss_F_l1
                + λ_prior * loss_prior
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} | Loss {loss.item():.4f}")

