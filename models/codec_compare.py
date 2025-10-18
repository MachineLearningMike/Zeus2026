import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# -----------------------------
# Original Encoder (looped)
# -----------------------------
class EncoderOriginal(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.cnnPairs = nn.ModuleList([CnnPair(h, C, W) for h in range(H)])
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x):
        slices = [cnnPair(x[:, :, h, :]) for h, cnnPair in enumerate(self.cnnPairs)]
        x = torch.stack(slices, dim=2)
        x = x + self.positional_codes
        return x

# -----------------------------
# Fast Encoder (vectorized)
# -----------------------------
class FastEncoder(nn.Module):
    def __init__(self, C, H, W, max_kernel=65):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.max_kernel = max_kernel

        # Compute alpha for each row (latitude)
        lat_deg = torch.arange(H, dtype=torch.float32) / (W/2) * 180 - 90
        lat_rad = lat_deg * math.pi / 180
        alpha = 1.0 / (torch.cos(lat_rad) + 1e-9)  # avoid division by zero
        alpha = alpha.clamp(1.0, float(max_kernel))

        floor_k = alpha.floor().to(torch.int64)
        ceil_k = alpha.ceil().to(torch.int64)

        self.register_buffer("floor", floor_k)
        self.register_buffer("ceil", ceil_k)

        # interpolation weights
        self.register_buffer("floorW", (ceil_k.float() - alpha))
        self.register_buffer("ceilW", (alpha - floor_k.float()))

        # unique kernel sizes
        unique_sizes = torch.unique(torch.cat([floor_k, ceil_k]))
        self.convs = nn.ModuleDict({
            str(k.item()): nn.Conv1d(C, C, k.item(), padding=0, bias=True) for k in unique_sizes
        })

        # positional encoding
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x):
        B, C, H, W = x.shape
        max_pad = self.max_kernel // 2

        # pad circularly along width (W)
        x_padded = torch.cat([x[:, :, :, -max_pad:], x, x[:, :, :, :max_pad]], dim=3)
        out_slices = []

        for h in range(H):
            f_k, c_k = self.floor[h].item(), self.ceil[h].item()
            pad_f, pad_c = (f_k-1)//2, (c_k-1)//2

            # slice for floor
            x_f = x_padded[:, :, h, max_pad - pad_f : max_pad - pad_f + W + f_k - 1]
            x_f = F.pad(x_f, (pad_f, f_k - 1 - pad_f), mode="circular")
            y_f = self.convs[str(f_k)](x_f)

            if f_k == c_k:
                out_slices.append(y_f)
            else:
                # slice for ceil
                x_c = x_padded[:, :, h, max_pad - pad_c : max_pad - pad_c + W + c_k - 1]
                x_c = F.pad(x_c, (pad_c, c_k - 1 - pad_c), mode="circular")
                y_c = self.convs[str(c_k)](x_c)
                y = self.floorW[h] * y_f + self.ceilW[h] * y_c
                out_slices.append(y)

        x_out = torch.stack(out_slices, dim=2) + self.positional_codes
        return x_out

# -----------------------------
# Simple CnnPair for original encoder
# -----------------------------
class CnnPair(nn.Module):
    def __init__(self, h_index, channels, W):
        super().__init__()
        lat_deg = h_index / (W/2) * 180 - 90
        alpha = 1.0 / (math.cos(math.radians(lat_deg)) + 1e-9)
        alpha = max(1.0, min(alpha, W))
        self.floor = int(math.floor(alpha))
        self.ceil  = int(math.ceil(alpha))
        self.floorW = self.ceil - alpha if self.floor!=self.ceil else 1.0
        self.ceilW = alpha - self.floor if self.floor!=self.ceil else 0.0
        self.floorCnn = nn.Conv1d(channels, channels, self.floor, bias=True)
        self.ceilCnn  = nn.Conv1d(channels, channels, self.ceil,  bias=True) if self.floor!=self.ceil else None

    def forward(self, x):
        pad_f = (self.floor-1)//2
        x_f = F.pad(x, (pad_f, self.floor-1-pad_f), mode="circular")
        y_f = self.floorCnn(x_f)
        if self.ceilCnn is None:
            return y_f
        pad_c = (self.ceil-1)//2
        x_c = F.pad(x, (pad_c, self.ceil-1-pad_c), mode="circular")
        y_c = self.ceilCnn(x_c)
        return self.floorW*y_f + self.ceilW*y_c

# -----------------------------
# Benchmark
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, C, H, W = 2, 4, 64, 128
x = torch.randn(B, C, H, W, device=device)

# Original Encoder
enc_orig = EncoderOriginal(C, H, W).to(device)
torch.cuda.synchronize() if device=="cuda" else None
start = time.time()
y_orig = enc_orig(x)
torch.cuda.synchronize() if device=="cuda" else None
t_orig = time.time() - start
print(f"Original Encoder time: {t_orig:.4f}s")

# Fast Encoder
enc_fast = FastEncoder(C, H, W).to(device)
torch.cuda.synchronize() if device=="cuda" else None
start = time.time()
y_fast = enc_fast(x)
torch.cuda.synchronize() if device=="cuda" else None
t_fast = time.time() - start
print(f"Fast Encoder time: {t_fast:.4f}s")

# Output comparison
diff = (y_orig - y_fast).abs()
print(f"Max absolute difference: {diff.max().item():.6f}")
print(f"Mean absolute difference: {diff.mean().item():.6f}")
print(f"Original mean/std: {y_orig.mean().item():.6f} / {y_orig.std().item():.6f}")
print(f"Fast mean/std:     {y_fast.mean().item():.6f} / {y_fast.std().item():.6f}")
