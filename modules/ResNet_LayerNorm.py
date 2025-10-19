import torch
import torch.nn as nn
import time

"""
Normalizes across all channels for each spatial location (h,w) individually.
Treats every pixel’s feature vector like a Transformer token embedding.
Better for models where channel relationships matter more than spatial ones (e.g., hybrid CNN-Transformer encoders).
Adds overhead because of tensor permutation (B,C,H,W) → (B,H,W,C) and back.
"""
# ---------- Helper ----------
class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for 2D CNNs (B, C, H, W)
    Normalizes across C only (like GroupNorm with group=1).
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # reshape (B, C, H, W) -> (B, H, W, C) for LayerNorm -> back
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------- Bottleneck Block ----------
class BottleneckSame(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.ln1 = LayerNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                               bias=False, padding_mode='circular')
        self.ln2 = LayerNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.ln3 = LayerNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.ln1(self.conv1(x)))
        out = self.relu(self.ln2(self.conv2(out)))
        out = self.ln3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out


# ---------- FCResNet with head ----------
class FCResNetCylindrical(nn.Module):
    def __init__(self, nVariables=4, nBlocks=18, in_channels=None, mid_channels=32):
        super().__init__()
        if in_channels is None:
            in_channels = 24 * nVariables
        self.in_channels = in_channels

        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                               bias=False, padding_mode='circular')
        self.ln1 = LayerNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Flat residual blocks
        self.blocks = nn.Sequential(
            *[BottleneckSame(in_channels, mid_channels) for _ in range(nBlocks)]
        )

        # self.head = nn.Conv2d(in_channels, nVariables, 1, bias=True)

    def forward(self, x):
        x = self.relu(self.ln1(self.conv1(x)))
        x = self.blocks(x)
        # x = self.head(x)
        return x


# ---------- Demo ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nVariables = 4
    in_channels = 24 * nVariables  # 96

    model = FCResNetCylindrical(nVariables=nVariables, nBlocks=18,
                                mid_channels=32, in_channels=in_channels).to(device)
    model.eval()

    x = torch.randn(2, in_channels, 72, 144, device=device)

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
