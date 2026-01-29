import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.zeus_dataset_Snapshot import Snapshot
from modules.zeus_dateset_geo_tempo import Geo_Tempo
from modules.Flash_Nonlinear_Attention import PhysicalFlashAttention
from modules.RoPE_LatLon import RoPE

# -----------------------
# Zeus model (pre-rotated inputs)
# -----------------------
class Zeus(nn.Module):
    def __init__(self, H, W, C, num_heads=8, num_layers=4, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        
        # Stack multiple PhysicalFlashAttention layers
        self.attn_layers = nn.ModuleList([
            PhysicalFlashAttention(dim=C, H=H, W=W, num_heads=num_heads, dropout=0.0).to(device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

    def forward(self, p, x):
        """
        p, x: (B, H, W, C)
        p: simple physical prediction of y
        """
        B, H, W, C = x.shape
        assert (B, H, W, C) == p.shape
        assert (H, W, C) == (self.H, self.W, self.C)
                                
        # Flatten features
        x_flat = x.reshape(B, N, C)
        p_flat = p.reshape(B, N, C)

        # Apply RoPE
        x_rot = rope.apply_rope(x_flat).reshape(B, H, W, C)
        p_rot = rope.apply_rope(p_flat).reshape(B, H, W, C)
        
        for i, attn in enumerate(self.attn_layers):
            x_rot = attn(p_rot, x_rot)
            p_rot = x_rot
        
        return x_rot + p

# -----------------------
# Test code
# -----------------------
if __name__ == "__main__":
    from modules.zeus_dateset_geo_tempo import Geo_Tempo
    from modules.RoPE_LatLon import RoPE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    factor10 = 2.5
    num_heads = 2
    B = 1

    H, W = int(721*factor10/10), int(1440*factor10/10)
    C = 10 * int(10/factor10 * 10/factor10)
    
    # -----------------------------
    # RoPE
    # -----------------------------
    latlon, _, _ = Geo_Tempo.lat_lon_grid_tensor(H, W)
    latlon = torch.deg2rad(latlon).to(device=device, dtype=dtype)
    lat_grid, lon_grid = latlon[0], latlon[1]

    N = H * W
    lat_flat = lat_grid.reshape(1, N).expand(B, N)
    lon_flat = lon_grid.reshape(1, N).expand(B, N)

    rope = RoPE(lat_flat, lon_flat, dims=C)

    # -----------------------------
    # Dummy inputs
    # -----------------------------
    p = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)

    # -----------------------------
    # Forward through Zeus
    # -----------------------------
    model = Zeus(H, W, C, num_heads=num_heads, num_layers=1, device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )

    print(f"starting...")

    loops = 10
    torch.cuda.synchronize()
    s = time.time()
    
    for loop in range(loops):
        optimizer.zero_grad(set_to_none=True)
        y = model(p, x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        # print(f"loop {loop}")
        
    torch.cuda.synchronize()
    e = time.time()
    print(f"time taken: {(e-s)/loops:.3f}")

    # print("Output shape:", y.shape)
    # print("Output dtype:", y.dtype)
    # print("Output mean:", y.mean().item())

    # print("x.grad:", x.grad is not None)
    # print("p.grad:", p.grad is not None)
    print("✅ Pre-RoPE Zeus test passed successfully.")
