import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# -----------------------------
# Fast Encoder / Decoder Definition
# -----------------------------
class FastCnnLayer(nn.Module):
    """Latitude-dependent 1D conv, per-latitude floor/ceil convs, truly cylindrical."""
    def __init__(self, C, H, W):
        super().__init__()
        self.C, self.H, self.W = C, H, W

        # Latitude grid
        lat_deg = torch.arange(H, dtype=torch.float32) / (W/2) * 180 - 90
        lat_rad = torch.deg2rad(lat_deg)

        alpha = 1.0 / (torch.cos(lat_rad) + 1e-9)
        alpha = torch.clamp(alpha, 1.0, W)

        floor = torch.floor(alpha).to(torch.int32)
        ceil = torch.ceil(alpha).to(torch.int32)

        floorW = torch.where(floor == ceil, torch.ones_like(alpha), ceil - alpha)
        ceilW = torch.where(floor == ceil, torch.zeros_like(alpha), alpha - floor)

        self.register_buffer("floor", floor)
        self.register_buffer("ceil", ceil)
        self.register_buffer("floorW", floorW)
        self.register_buffer("ceilW", ceilW)

        # Group convs by kernel size
        self.floor_groups = nn.ModuleDict()
        self.ceil_groups = nn.ModuleDict()
        for h in range(H):
            f_k = floor[h].item()
            c_k = ceil[h].item()
            if str(f_k) not in self.floor_groups:
                self.floor_groups[str(f_k)] = nn.Conv1d(C, C, f_k, padding=0, bias=True)
            if f_k != c_k and str(c_k) not in self.ceil_groups:
                self.ceil_groups[str(c_k)] = nn.Conv1d(C, C, c_k, padding=0, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        y_out = torch.empty_like(x)

        for h in range(H):
            f_k = self.floor[h].item()
            c_k = self.ceil[h].item()
            f_w = self.floorW[h].item()
            c_w = self.ceilW[h].item()

            # --- Floor conv with circular padding (true cylinder) ---
            pad_left = (f_k - 1) // 2
            pad_right = f_k - 1 - pad_left
            x_f = F.pad(x[:, :, h, :], (pad_left, pad_right), mode='circular')
            y_f = self.floor_groups[str(f_k)](x_f)
            y_f = y_f[:, :, :W]  # guarantee exact width

            if f_k == c_k:
                y_out[:, :, h, :] = y_f.to(y_out.dtype)
            else:
                pad_left = (c_k - 1) // 2
                pad_right = c_k - 1 - pad_left
                x_c = F.pad(x[:, :, h, :], (pad_left, pad_right), mode='circular')
                y_c = self.ceil_groups[str(c_k)](x_c)
                y_c = y_c[:, :, :W]

                # Interpolation
                y_out[:, :, h, :] = f_w * y_f.to(y_out.dtype) + c_w * y_c.to(y_out.dtype)

        return y_out

class SphericalEncoder(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.C, self.H, self.W = C, H, W
        self.layer = FastCnnLayer(C, H, W)
        self.positional_codes = nn.Parameter(torch.empty(C, H, W).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x):
        x_out = self.layer(x)
        x_out = x_out + self.positional_codes
        return x_out


class SphericalDecoder(SphericalEncoder):
    """Decoder is the same as Encoder, optionally could take different in_C"""
    def __init__(self, in_C, C, H, W):
        assert in_C == C, "in_C must equal C for this simple decoder"
        super().__init__(C, H, W)


# -----------------------------
# Visualization
# -----------------------------
if __name__ == "__main__":
    C, H, W = 8, 72 * 2, 144 * 2
    model = SphericalEncoder(C, H, W)

    # Latitude values
    latitudes = [h / (W/2) * 180 - 90 for h in range(H)]

    # Extract floor, ceil, differences, and interpolation weights
    floor_sizes = model.layer.floor.cpu().numpy()
    ceil_sizes  = model.layer.ceil.cpu().numpy()
    diffs       = ceil_sizes - floor_sizes
    floorW      = model.layer.floorW.cpu().numpy()
    ceilW       = model.layer.ceilW.cpu().numpy()

    # Plot kernel sizes and difference
    # plt.figure(figsize=(10, 4))
    # plt.plot(latitudes, floor_sizes, color="blue", label="floor kernel size")
    # plt.plot(latitudes, diffs, color="red", label="ceil - floor")
    # plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    # plt.xlabel("Latitude (°)")
    # plt.ylabel("Kernel sizes / difference")
    # plt.title("Floor Kernel Size and Ceil-Floor Difference Across Latitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Plot interpolation weights
    # plt.figure(figsize=(10, 4))
    # plt.plot(latitudes, floorW, color="green", label="floorW")
    # plt.plot(latitudes, ceilW, color="orange", label="ceilW")
    # plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    # plt.xlabel("Latitude (°)")
    # plt.ylabel("Interpolation weights")
    # plt.title("Interpolation Weights Across Latitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # -----------------------------
    # Forward pass test
    # -----------------------------
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Random input tensor
    x = torch.randn(2, C, H, W, device=device)  # batch of 2

    # Optional: mixed precision
    import torch.amp as amp
    loops = 10
    start = time.time()
    with torch.no_grad():
        for _ in range(loops):
            with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
                y = model(x)
    end = time.time()

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)
    print("Average forward pass time: {:.3f} s".format((end - start) / loops))