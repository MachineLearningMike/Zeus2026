import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


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


class FusedCircularConv3x3(nn.Module):
    """3x3 convolution with fused horizontal circular padding and vertical zero padding."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1,
            padding=(1, 1), padding_mode='circular', bias=False
        )

    def forward(self, x):
        # x: (B, C, H, W)
        # Vertical zero padding is already handled in Conv2d with padding_mode='circular'
        return self.conv(x)


class CircularBottleneck(nn.Module):
    def __init__(self, channels, bottlenectRate=0.5):
        super().__init__()
        mid_channels = max(1, int(channels * bottlenectRate))

        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.ln1 = LayerNorm2d(mid_channels)

        self.conv2 = FusedCircularConv3x3(mid_channels)
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


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 128, device='cuda')
    conv = FusedCircularConv3x3(64).cuda()
    
    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = conv(x)
    torch.cuda.synchronize()
    end = time.time()
    print("Time:", end-start)
    print(x.shape, y.shape)
