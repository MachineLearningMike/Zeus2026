import torch
import math

class RoPE():
    def __init__(self, lat, lon, dims):
        B, N = lat.shape
        self.dims = dims

        # Split dims for latitude and longitude
        D_lat = dims // 2
        D_lon = dims - D_lat  # ensures D_lat + D_lon = dims

        # -----------------------------
        # Latitude
        # -----------------------------
        # Number of frequencies for latitude
        # We'll duplicate each frequency once, so num_freq = D_lat // 2
        num_freq_lat = D_lat // 2
        if num_freq_lat == 0:
            # Handle very small D_lat
            theta_lat = lat.unsqueeze(-1)  # (B, N, 1)
        else:
            freq_lat = 10000 ** (torch.arange(0, num_freq_lat, device=lat.device, dtype=lat.dtype) * 2 / D_lat)
            theta_lat = lat[..., None] / freq_lat  # (B, N, num_freq_lat)
            # Duplicate each frequency once
            theta_lat = torch.cat([theta_lat, theta_lat], dim=-1)  # (B, N, 2*num_freq_lat)

        # If duplication < D_lat, pad with zeros
        if theta_lat.shape[-1] < D_lat:
            pad = D_lat - theta_lat.shape[-1]
            theta_lat = torch.cat([theta_lat, torch.zeros(B, N, pad, device=lat.device, dtype=lat.dtype)], dim=-1)
        # Truncate if overshoot
        theta_lat = theta_lat[..., :D_lat]

        # -----------------------------
        # Longitude
        # -----------------------------
        K = D_lon // 2
        if K == 0:
            theta_lon = lon.unsqueeze(-1)
        else:
            k_harmonics = torch.arange(1, K+1, device=lat.device, dtype=lat.dtype)
            theta_lon = lon[..., None] * k_harmonics  # (B, N, K)
            theta_lon = torch.cat([theta_lon, theta_lon], dim=-1)  # duplicate
        # Pad / truncate
        if theta_lon.shape[-1] < D_lon:
            pad = D_lon - theta_lon.shape[-1]
            theta_lon = torch.cat([theta_lon, torch.zeros(B, N, pad, device=lat.device, dtype=lat.dtype)], dim=-1)
        theta_lon = theta_lon[..., :D_lon]

        # -----------------------------
        # Final concatenation
        # -----------------------------
        self.theta = torch.cat([theta_lat, theta_lon], dim=-1)  # (B, N, dims)
        assert self.theta.shape[-1] == dims, f"theta last dim {self.theta.shape[-1]} != dims {dims}"

    def _rotate_90(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x, theta):
        return x * torch.cos(theta) + self._rotate_90(x) * torch.sin(theta)

    def apply_rope(self, x):
        return self._apply_rope(x, self.theta)

# -----------------------------
# Test code
# -----------------------------
if __name__ == "__main__":
    
    from Flash_Nonlinear_Attention import PhysicalFlashAttention
    from zeus_dateset_geo_tempo import Geo_Tempo
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Grid / batch dimensions
    B, H, W, C = 2, 180, 360, 10
    num_heads = 5

    #--------------------------------------------------- lat_flat, lon_flat : (B, N)
    # Latitude [-pi/2, pi/2], Longitude [0, 2pi)
    latlon, _, _ = Geo_Tempo.lat_lon_grid_tensor(H, W)
    # Degrees → radians
    latlon = torch.deg2rad(latlon).to(device=device, dtype=dtype)

    # Build full spatial grids
    lat_grid = latlon[0]
    lon_grid = latlon[1]

    # Flatten
    N = H * W
    lat_flat = lat_grid.reshape(1, N).repeat(B, 1)  # (B, N)
    lon_flat = lon_grid.reshape(1, N).repeat(B, 1)  # (B, N)
    
    rope = RoPE(lat_flat, lon_flat, dims=C)

    #--------------------------------------------------- x_flat, p_flat: (B, N, C)
    # Dummy inputs
    x = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)
    p = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)

    # Flatten spatial dims
    N = H * W
    x_flat = x.reshape(B, N, C)
    p_flat = p.reshape(B, N, C)

    #--------------------------------------------------- x_rot, p_rot: (B, N, C)    
    # --- Apply RoPE externally ---
    x_rot = rope.apply_rope(x_flat)
    p_rot = rope.apply_rope(p_flat)

    #--------------------------------------------------- x_rot, p_rot: (B, H, W, C)
    # Reshape back to (B, H, W, C)
    x_rot = x_rot.reshape(B, H, W, C)
    p_rot = p_rot.reshape(B, H, W, C)

    # --- Call attention module ---
    attn = PhysicalFlashAttention(dim=C, H=H, W=W, num_heads=num_heads, dropout=0.0).to(device=device, dtype=dtype)
    y = attn(x_rot, p_rot)

    print("Output shape:", y.shape)
    print("Output mean:", y.mean().item())
