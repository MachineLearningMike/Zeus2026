import torch
import torch.nn as nn
import time

"""
Divides the channels into groups (e.g., 32 groups).
Normalizes within each group across spatial positions (H×W).
Good for convolutional spatial invariance and stable training on small batches.
Computation is purely in (B,C,H,W) order → efficient on GPUs.
"""

# ---------- Helper ----------
def get_valid_groups(num_channels, max_groups=32):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1

# ---------- Bottleneck Block ----------
class BottleneckSame(nn.Module):
    def __init__(self, in_channels, mid_channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.gn1 = nn.GroupNorm(get_valid_groups(mid_channels, num_groups), mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                               bias=False, padding_mode='circular')
        self.gn2 = nn.GroupNorm(get_valid_groups(mid_channels, num_groups), mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.gn3 = nn.GroupNorm(get_valid_groups(in_channels, num_groups), in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        out += identity  # in-place residual
        out = self.relu(out)
        return out

# ---------- FCResNet with head ----------
class ResNetCylindrical(nn.Module):
    def __init__(self, nVariables=4, nBlocks=18, in_channels=None, mid_channels=32, num_groups=32):
        super().__init__()
        if in_channels is None:
            in_channels = 24 * nVariables
        self.in_channels = in_channels
        self.nVariables = nVariables

        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                               bias=False, padding_mode='circular')
        self.gn1 = nn.GroupNorm(get_valid_groups(in_channels, num_groups), in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Flat residual blocks
        self.blocks = nn.Sequential(
            *[BottleneckSame(in_channels, mid_channels, num_groups) for _ in range(nBlocks)]
        )

        # Efficient 1x1 head: projects in_channels → nVariables
        # self.head = nn.Conv2d(in_channels, nVariables, 1, bias=True)

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.blocks(x)
        # x = self.head(x)
        return x

# ---------- Demo ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nVariables = 4
    in_channels = 24 * nVariables  # 96

    model = ResNetCylindrical(nVariables=nVariables, nBlocks=36, mid_channels=64,
                                in_channels=in_channels).to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    x = torch.randn(2, in_channels, 72*10, 144*10, device=device)

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
