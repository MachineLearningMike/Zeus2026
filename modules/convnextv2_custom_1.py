import torch
import torch.nn as nn
from timm.layers import DropPath

# -----------------------------
# LayerNorm in NCHW format
# -----------------------------
class LayerNormNCHW(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(1, keepdim=True)
        var = ((x - mean) ** 2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

# -----------------------------
# Global Response Normalization
# -----------------------------
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# -----------------------------
# Cylindric ConvNeXtV2 Block
# -----------------------------
class Block_Cylindric(nn.Module):
    def __init__(self, dim, dropPath=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3,
                                groups=dim, padding_mode='circular')  # fused circular
        self.norm = LayerNormNCHW(dim)
        self.pwconv1 = nn.Conv2d(dim, 4*dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(4*dim)
        self.pwconv2 = nn.Conv2d(4*dim, dim, kernel_size=1)
        self.dropPath = DropPath(dropPath) if dropPath>0 else nn.Identity()

    def forward(self, x):
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.grn(out)
        out = self.pwconv2(out)
        return x + self.dropPath(out)

# -----------------------------
# Full-Resolution ConvNeXtV2
# -----------------------------
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=96, dims=[96]*4, depths=[3,3,9,3], drop_path_rate=0.):
        super().__init__()
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block_Cylindric(dim=dims[i], dropPath=dp_rates[cur+j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final 1x1 conv to match input channels
        self.final_conv = nn.Conv2d(dims[-1], in_chans, kernel_size=1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        return x

# -----------------------------
# Demo / Timing
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    in_chans = 96
    H, W = 144, 288

    x = torch.randn(batch_size, in_chans, H, W).to(device)
    model = ConvNeXtV2(in_chans=in_chans).to(device)
    model.eval()

    import time
    loops = 5
    start = time.time()
    with torch.no_grad():
        for _ in range(loops):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                y = model(x)
    end = time.time()

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)
    print("Average forward time per step: {:.3f} s".format((end-start)/loops))
