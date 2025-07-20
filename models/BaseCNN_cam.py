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
            -kernel_size//2, kernel_size//2,
            steps=kernel_size
        ) / sample_rate
        self.register_buffer('t', t)           # (ks,)
        # 可学习参数
        self.A     = nn.Parameter(torch.ones(num_filters))
        self.mu    = nn.Parameter(torch.zeros(num_filters))
        self.sigma = nn.Parameter(torch.ones(num_filters)*0.1)
        self.f     = nn.Parameter(torch.linspace(1, 15, num_filters))
        self.phi   = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x: torch.Tensor):
        # x: (B,1,T) → out: (B, num_filters, T)
        t = self.t.view(1, 1, -1)              # (1,1,ks)
        A     = self.A.view(-1, 1, 1)          # (num,1,1)
        mu    = self.mu.view(-1, 1, 1)
        sigma = self.sigma.abs().view(-1, 1, 1) + 1e-3
        f     = self.f.clamp(0.1, 50.0).view(-1, 1, 1)
        phi   = self.phi.view(-1, 1, 1)

        gauss = torch.exp(-((t - mu)**2) / (2 * sigma**2))
        sinus = torch.cos(2*torch.pi*f*t + phi)
        kern = A * gauss * sinus                # (num,1,ks)
        out  = F.conv1d(x, kern, padding=self.ks//2)
        return out                              # (B, num, T)


class FourierFilterBank(nn.Module):
    def __init__(self,
                 num_filters: int = 4,
                 kernel_size: int = 129,
                 sample_rate: float = 100.0,
                 num_bases: int = 16,
                 f_min: float = 0.5,
                 f_max: float = 30.0):
        super().__init__()
        self.ks = kernel_size
        self.register_buffer('t',
            torch.linspace(-kernel_size//2, kernel_size//2,
                           steps=kernel_size) / sample_rate)  # (ks,)
        self.register_buffer('freqs',
            torch.linspace(f_min, f_max, num_bases))        # (nb,)
        # 学习系数 a,b
        self.a = nn.Parameter(torch.randn(num_filters, num_bases)*0.1)
        self.b = nn.Parameter(torch.randn(num_filters, num_bases)*0.1)
        self.register_buffer('w_prior',
            torch.ones(num_filters, num_bases))  # 频带软先验权重

    def forward(self, x: torch.Tensor):
        # x: (B,1,T) → out: (B, num_filters, T)
        t     = self.t.view(1, 1, -1)            # (1,1,ks)
        freqs = self.freqs.view(1, -1, 1)        # (1,nb,1)
        cosb  = torch.cos(2*torch.pi*freqs*t)    # (1,nb,ks)
        sinb  = torch.sin(2*torch.pi*freqs*t)    # (1,nb,ks)

        # 构建核
        # a: (num,nb) → (num,nb,1) * (1,nb,ks) → (num,nb,ks)
        kern = (self.a.unsqueeze(2)*cosb + self.b.unsqueeze(2)*sinb).sum(dim=1)
        kern = kern.unsqueeze(1)                 # (num,1,ks)
        out  = F.conv1d(x, kern, padding=self.ks//2)
        return out                               # (B, num, T)


class Templates(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.p = nn.Parameter(torch.randn(num_classes, feat_dim)*0.1)

    def forward(self):
        return self.p                          # (K, feat_dim)


class BaseCNNNet(nn.Module):
    def __init__(self,
                 num_classes: int = 5,
                 sample_rate: float = 100.0):
        super().__init__()
        # 输入 x: (B,1,3000)
        self.gabor = GaborFilterBank(num_filters=5,
                                    kernel_size=129,
                                    sample_rate=sample_rate)
        self.fourier = FourierFilterBank(num_filters=5,
                                         kernel_size=129,
                                         sample_rate=sample_rate,
                                         num_bases=16)
        # Backbone：输入 8 通道 → 输出 8 通道
        self.backbone = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=51, stride=7, padding=3), nn.ReLU(),
            nn.Conv1d(16,32, kernel_size=4, stride=4), nn.ReLU()
        )
        self.classifier = nn.Linear(32, num_classes)
        self.templates  = Templates(num_classes, feat_dim=10)

    def forward(self, x: torch.Tensor):
        """
        x: (B,1,3000)
        返回:
          logits: (B,5)
          feats:  (B,8,3000)
        """
        # 1) Gabor/ Fourier → (B,4,3000) each
        g = self.gabor(x)
        f = self.fourier(x)
        print(g.shape, f.shape)
        # 2) 拼通道 → (B,8,3000)
        h = torch.cat([g, f], dim=1)
        # 3) Backbone → (B,8,3000)
        feats = self.backbone(h)

        print(feats.shape)
        print(self.templates)

        # 4) 分类
        gap    = feats.mean(dim=-1)             # (B,8)
        logits = self.classifier(gap)           # (B,5)
        return logits, feats


model = BaseCNNNet().cuda()
x = torch.rand([64, 1, 30000]).cuda()
a,b = model(x)
print(a.shape, b.shape)