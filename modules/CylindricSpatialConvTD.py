import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time

# -----------------------------
# CylindricSpatialConvTD
# -----------------------------
class CylindricSpatialConvTD(nn.Module):
    """
    Input:  (B, T, N, D)
    Output: same shape
    Horizontal cylindrical convolution
    Vertical zero padding
    """
    def __init__(self, pm, kernel_size=3):
        pass
    def __init__(self, pm, kernel_size=3):
        super().__init__()
        in_channels_time = pm.T1
        in_channels_dim = pm.patch_dim
        self.H = pm.nPatches_h
        self.W = pm.nPatches_w
        
        # Use circular padding for horizontal axis via PyTorch padding_mode
        self.conv2d_time = nn.Conv2d(
            in_channels_time,
            in_channels_time,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size//2, kernel_size//2),  # (H_pad, W_pad)
            padding_mode='circular',
            bias=True
        )
        
        # Use circular padding for horizontal axis via PyTorch padding_mode
        self.conv2d_dim = nn.Conv2d(
            in_channels_dim,
            in_channels_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size//2, kernel_size//2),  # (H_pad, W_pad)
            padding_mode='circular',
            bias=True
        )

    def forward(self, x):   # (b t n d) -> (b t n d)
        B, T, N, D = x.shape
        H, W = self.H, self.W
        assert H*W == N, "N must equal H*W"

        x = rearrange(x, 'b t (h w) d -> (b d) t h w', h=H, w=W)
        x = self.conv2d_time(x)        
        x = rearrange(x, '(b d) t h w -> (b t) d h w', h=H, w=W, b=B)
        x = self.conv2d_dim(x)
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B, t=T)

        return x


        # x = rearrange(x, 'b t n d -> (b d) t n')
        # x = rearrange(x, 'B t (h w) -> B t h w', h=self.pm.nPatches_h)
        # x = self.convolve(x)    # preserves shape
        # x = rearrange(x, '(b d) t h w -> b t (h w) d', d=self.pm.d_model)

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import sys
    from pathlib import Path
    # Get path to sibling folder
    sibling_path = Path(__file__).resolve().parent.parent / "models"
    sys.path.insert(0, str(sibling_path))
    from config import config
    from params import Params

    config['factor10'] = 10
    config['batch_size'] = 1
    config['T1'] = 24
    
    pm = Params(config)

    B, T, H, W, D = pm.batch_size, pm.T1, pm.nPatches_h, pm.nPatches_w, pm.patch_dim  # example dimensions
    N = H * W
    print("B, T, H, W, D, N", B, T, H, W, D, N)
    x = torch.randn(B, T, N, D, device=device)

    model = CylindricSpatialConvTD(pm, kernel_size=5).to(device)
    model.eval()

    loops = 5
    start = time.time()
    with torch.no_grad():
        for _ in range(loops):
            y = model(x)
    end = time.time()

    print("✅ Input shape:", x.shape)
    print("✅ Output shape:", y.shape)
    print("Average forward time per step: {:.3f} s".format((end-start)/loops))
