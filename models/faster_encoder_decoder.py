import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# -----------------------------
# Original Encoder (looped, fixed)
# -----------------------------
class EncoderOriginal(nn.Module):
    def __init__(self, C, H, W, max_kernel=65):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.max_kernel = max_kernel

        self.cnnPairs = nn.ModuleList()
        self.floorWs = []
        self.ceilWs = []

        for h in range(H):
            alpha = math.cos((h / W * 2 * 180 - 90) * math.pi / 180) + 1e-9
            floor = max(1, math.floor(alpha))
            ceil  = max(1, math.ceil(alpha))
            floor = min(floor, max_kernel)
            ceil  = min(ceil, max_kernel)

            self.cnnPairs.append(nn.ModuleDict({
                'floor': nn.Conv1d(C, C, floor, stride=1, padding=0, bias=True),
                'ceil': nn.Conv1d(C, C, ceil, stride=1, padding=0, bias=True)
            }))
            self.floorWs.append(floor - alpha)
            self.ceilWs.append(alpha - floor)

        self.floorWs = torch.tensor(self.floorWs, dtype=torch.float32)
        self.ceilWs = torch.tensor(self.ceilWs, dtype=torch.float32)
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x):
        B, C, H, W = x.shape
        slices = []
        for h in range(H):
            pair = self.cnnPairs[h]
            kf, kc = pair['floor'].kernel_size[0], pair['ceil'].kernel_size[0]
            pf, pc = (kf-1)//2, (kc-1)//2

            xFloor = nn.functional.pad(x[:, :, h, :], (pf, kf-1-pf), mode='circular')
            xFloor = pair['floor'](xFloor)
            xCeil  = nn.functional.pad(x[:, :, h, :], (pc, kc-1-pc), mode='circular')
            xCeil  = pair['ceil'](xCeil)

            x_h = self.floorWs[h]*xFloor + self.ceilWs[h]*xCeil
            slices.append(x_h)
        x_out = torch.stack(slices, dim=2) #+ self.positional_codes
        return x_out


# -----------------------------
# Fast Encoder (vectorized)
# -----------------------------
class EncoderFastExact(nn.Module):
    def __init__(self, C, H, W, max_kernel=65):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.max_kernel = max_kernel

        # compute alpha, floor, ceil, weights exactly like original
        alpha = torch.cos((torch.arange(H, dtype=torch.float32) / W * 2 * 180 - 90) * math.pi / 180) + 1e-9
        floor = torch.clamp(alpha.floor().long(), 1, max_kernel)
        ceil  = torch.clamp(alpha.ceil().long(), 1, max_kernel)
        floorW = (ceil.float() - alpha)
        ceilW  = (alpha - floor.float())

        self.register_buffer("floor", floor)
        self.register_buffer("ceil", ceil)
        self.register_buffer("floorW", floorW)
        self.register_buffer("ceilW", ceilW)

        # build all convs needed (unique sizes)
        unique_sizes = torch.unique(torch.cat([floor, ceil]))
        self.convs = nn.ModuleDict({str(k.item()): nn.Conv1d(C, C, k.item(), stride=1, padding=0, bias=True)
                                    for k in unique_sizes})

        # positional codes
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x):
        B, C, H, W = x.shape
        x_out = []

        # loop only over H (latitude), vectorized over batch and channels
        for h in range(H):
            k_floor = self.floor[h].item()
            k_ceil  = self.ceil[h].item()
            pad_floor = (k_floor - 1) // 2
            pad_ceil  = (k_ceil  - 1) // 2

            # exact circular padding like original
            xFloor = F.pad(x[:, :, h, :], (pad_floor, k_floor - 1 - pad_floor), mode="circular")
            xFloor = self.convs[str(k_floor)](xFloor)

            xCeil = F.pad(x[:, :, h, :], (pad_ceil, k_ceil - 1 - pad_ceil), mode="circular")
            xCeil = self.convs[str(k_ceil)](xCeil)

            x_h = self.floorW[h] * xFloor + self.ceilW[h] * xCeil
            x_out.append(x_h)

        x_out = torch.stack(x_out, dim=2) #+ self.positional_codes
        return x_out

# -----------------------------
# Benchmark
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, C, H, W = 2, 4, 72, 144
x = torch.randn(B, C, H, W, device=device)

# Original Encoder
enc_orig = EncoderOriginal(C, H, W).to(device)
torch.cuda.synchronize() if device=="cuda" else None
start = time.time()
y1 = enc_orig(x)
torch.cuda.synchronize() if device=="cuda" else None
t1 = time.time() - start
print("Original Encoder time: {:.4f}s".format(t1))

# Fast Encoder
enc_fast = EncoderFastExact(C, H, W).to(device)
torch.cuda.synchronize() if device=="cuda" else None
start = time.time()
y2 = enc_fast(x)
torch.cuda.synchronize() if device=="cuda" else None
t2 = time.time() - start
print("Fast Encoder time: {:.4f}s".format(t2))

# Compare outputs
diff = (y1 - y2).abs().max().item()
print("Max absolute difference between outputs:", diff)


# Compute mean and std
mean_orig = y1.mean().item()
std_orig  = y1.std().item()

mean_fast = y2.mean().item()
std_fast  = y2.std().item()

diff_mean = abs(mean_orig - mean_fast)
diff_std  = abs(std_orig - std_fast)

print(f"Original Encoder: mean={mean_orig:.6f}, std={std_orig:.6f}")
print(f"Fast Encoder:     mean={mean_fast:.6f}, std={std_fast:.6f}")
print(f"Difference in mean: {diff_mean:.6f}, difference in std: {diff_std:.6f}")