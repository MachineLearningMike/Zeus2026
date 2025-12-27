import torch
import torch.nn as nn

# -------------------------
# Parameters
# -------------------------
B = 1                 # batch size
time_steps = 2
num_vars = 88         # meteorological variables
H, W = 721, 1440      # original spatial grid
conv_kernel = 8
conv_stride = 8
sWin = 18      # in tokens
geo_channels = 16     # example: geo/temporal encoding channels

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Step 0: Dummy input
# -------------------------
x = torch.randn(B, time_steps, num_vars, H, W).to(device)  # (B,2,88,721,1440)
geo_temp = torch.randn(B, geo_channels, H, W).to(device)  # geo/temporal encoding

# -------------------------
# Step 1: Merge time and meteorological channels
# -------------------------
B, T, C, H, W = x.shape
x = x.view(B, T * C, H, W)  # (B, 176, H, W)
print("After concatenating time channels:", x.shape)

# -------------------------
# Step 2: Patch embedding (2D conv downsampling)
# -------------------------
patch_embed = nn.Conv2d(in_channels=T*C,
                        out_channels=T*C,
                        kernel_size=conv_kernel,
                        stride=conv_stride).to(device)

x_patched = patch_embed(x)  # (B, 176, H_patch, W_patch)
B, C, H_patch, W_patch = x_patched.shape
print("After patch embedding:", x_patched.shape)

# -------------------------
# Step 3: Geo/temporal encoding downsampling
# -------------------------
geo_conv = nn.Conv2d(in_channels=geo_channels,
                     out_channels=geo_channels,
                     kernel_size=conv_kernel,
                     stride=conv_stride).to(device)
geo_patched = geo_conv(geo_temp)  # (B, geo_channels, H_patch, W_patch)
print("Geo/temporal embedding:", geo_patched.shape)

# Concatenate along channels
x_cat = torch.cat([x_patched, geo_patched], dim=1)  # (B, C+G, H_patch, W_patch)
print("After concatenating all features:", x_cat.shape)

# -------------------------
# Step 4: Window partition
# -------------------------
def window_partition(x, sWin):
    """
    Args:
        x: (B, C, H, W)
        sWin: int
    Returns:
        windows: (num_windows*B, C, sWin, sWin)
    """
    B, C, H, W = x.shape
    assert H % sWin == 0 and W % sWin == 0, "H and W must be divisible by sWin"
    x = x.view(B, C, H // sWin, sWin, W // sWin, sWin)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, sWin, sWin)
    return windows

windows = window_partition(x_cat, sWin)
num_windows = windows.shape[0]
print(f"Number of windows: {num_windows}, each window shape: {windows.shape[1:]}")
