import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

# -----------------------------
# LayerNorm for NCHW
# -----------------------------
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# -----------------------------
# Global Response Normalization
# -----------------------------
class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1,1,1,dim))
        self.beta  = nn.Parameter(torch.zeros(1,1,1,dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# -----------------------------
# ConvNeXtV2 Block with cylindrical padding
# -----------------------------
class Block_Cylindric(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.pad = 3
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=0, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4*dim)
        self.act = nn.GELU()
        self.grn = GRN(4*dim)
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()

    def forward(self, x):
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='circular')
        out = self.dwconv(x_padded)
        out = out.permute(0,2,3,1)  # NCHW -> NHWC
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.grn(out)
        out = self.pwconv2(out)
        out = out.permute(0,3,1,2)  # NHWC -> NCHW
        out = x + self.drop_path(out)  # residual
        return out

# -----------------------------
# Cylindrical ConvNeXtV2-Tiny
# -----------------------------
class ConvNeXtV2_Cylindric_Tiny(nn.Module):
    def __init__(self, in_chans=96, drop_path_rate=0.):
        super().__init__()
        dims = [96, 192, 384, 768]
        depths = [3, 3, 9, 3]
        self.depths = depths
        self.downsample_layers = nn.ModuleList()

        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=1, padding=0),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Downsampling layers (stride=1 to preserve size)
        for i in range(3):
            layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=1, padding=0)
            )
            self.downsample_layers.append(layer)

        # Stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block_Cylindric(dim=dims[i], drop_path=dp_rates[cur+j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final 1x1 conv to match input channels
        self.final_conv = nn.Conv2d(dims[-1], in_chans, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.final_conv(x)
        return x

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    in_chans = 96
    H, W = 72, 144

    x = torch.randn(batch_size, in_chans, H, W).to(device)
    model = ConvNeXtV2_Cylindric_Tiny(in_chans=in_chans).to(device)

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
