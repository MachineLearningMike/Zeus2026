import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# LayerNorm for 2D tensors
# ----------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # back to (B, C, H, W)


# ----------------------
# Squeeze-and-Excitation
# ----------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ----------------------
# Horizontal circular 3x3 conv + vertical zero padding
# ----------------------
class FusedHorizontalCircularConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Horizontal circular padding only
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1,
            padding=(0, 1), padding_mode='circular', bias=False
        )

    def forward(self, x):
        # Vertical zero padding
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return self.conv(x)


# ----------------------
# Residual Bottleneck with SE
# ----------------------
class CircularBottleneck(nn.Module):
    def __init__(self, channels, bottlenectRate=0.5):
        super().__init__()
        mid_channels = max(1, int(channels * bottlenectRate))

        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.ln1 = LayerNorm2d(mid_channels)

        self.conv2 = FusedHorizontalCircularConv3x3(mid_channels)
        self.ln2 = LayerNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.ln3 = LayerNorm2d(channels)

        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.act(self.ln1(self.conv1(x)))
        out = self.act(self.ln2(self.conv2(out)))
        out = self.ln3(self.conv3(out))
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out

# ----------------------
# Stack of Bottlenecks preserving shape
# ----------------------
class CircularBottleneckStack(nn.Module):
    def __init__(self, channels, num_blocks=4, bottlenectRate=0.5):
        super().__init__()
        self.blocks = nn.Sequential(*[
            CircularBottleneck(channels, bottlenectRate)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)
    
# ----------------------
# Test speed and shape
# ----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    channels = 24 * 6
    factor10 = 2
    height = 72 * factor10
    width = 144 * factor10
    
    x = torch.randn(1, channels, height, width, device=device)

    # Stack of 5 bottlenecks
    stack = CircularBottleneckStack(channels=channels, num_blocks=20, bottlenectRate=0.3).to(device)

    # Warmup
    for _ in range(5):
        y = stack(x)

    # Timing
    torch.cuda.synchronize()
    import time
    start = time.time()
    loops = 5
    for _ in range(loops):
        y = stack(x)
    torch.cuda.synchronize()
    end = time.time()

    print("Time per run:", (end-start)/loops)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
