import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ---------------------------
# MLP
# ---------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ---------------------------
# Window partition / reverse (no padding)
# ---------------------------
def window_partition_3d(x, window_size):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    win_D, win_H, win_W = window_size
    assert D % win_D == 0 and H % win_H == 0 and W % win_W == 0, \
        f"Input dims {(D,H,W)} must be multiples of window_size {window_size}"

    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W

    x = x.view(B, C, nWd, win_D, nWh, win_H, nWw, win_W)
    # (B, nWd, nWh, nWw, C, win_D, win_H, win_w) -> (-1, C, win_D, win_H, win_W)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, C, win_D, win_H, win_W)
    return windows, nWd, nWh, nWw

def window_reverse_3d(windows, window_size, D, H, W):
    win_D, win_H, win_W = window_size
    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W
    B = int(windows.shape[0] // (nWd * nWh * nWw))
    C = windows.shape[1]

    x = windows.view(B, nWd, nWh, nWw, C, win_D, win_H, win_W)
    # (B, C, nWd, win_D, nWh, win_H, nWw, win_W) -> (B, C, D, H, W)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, C, D, H, W)
    return x

# ---------------------------
# Position bias learner
# ---------------------------
class PositionBiasLearner3D(nn.Module):
    def __init__(self, input_resolution, window_size, num_heads, hidden_dim=6, shift_size=(0,0,0)):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.window_volume = window_size[0]*window_size[1]*window_size[2]
        self.num_heads = num_heads
        self.shift_size = shift_size  # shift applied when creating flat_pair

        D, H, W = input_resolution
        lat_step = 180.0 / H
        lon_step = 360.0 / W
        depth_step = 1.0

        lat = torch.linspace(-90 + lat_step/2, 90 - lat_step/2, H)
        lon = torch.linspace(-180 + lon_step/2, 180 - lon_step/2, W)
        depth = torch.arange(D, dtype=torch.float32) * depth_step

        self.lat = lat[None,:,None].expand(D,H,W)
        self.lon = lon[None,None,:].expand(D,H,W)
        self.depth = depth[:,None,None].expand(D,H,W)

        # Precompute flat_pair with the given shift
        shifted_flat_pair_pos_code_on_windows, nWd, nWh, nWw = self.pair_position_partition(
            self.lat, self.lon, self.depth, shift_size=self.shift_size
        )        
        self.register_buffer("shifted_flat_pair_pos_code_on_windows", shifted_flat_pair_pos_code_on_windows)
        
        self.nWindows = nWd*nWh*nWw

        # Learnable layers
        self.fc1 = nn.Linear(4, hidden_dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_heads)
        self.scale = nn.Parameter(torch.ones(1)*1.5)

    def pair_position_partition(self, lat, lon, depth, shift_size):
        shift_D, shift_H, shift_W = shift_size

        shifted_depth_lat_lon = torch.stack([depth, lat, lon], dim=-1).permute(3,0,1,2).contiguous().unsqueeze(0)   # (B, C, D, H, W)

        if any(s != 0 for s in self.shift_size):
            shifted_depth_lat_lon = torch.roll(shifted_depth_lat_lon, shifts=self.shift_size, dims=(2,3,4))     # x: (B, C, D, H, W)

        windows, nWd, nWh, nWw = window_partition_3d(
            shifted_depth_lat_lon,        # (B, C, D, H, W)
            self.window_size
        )

        N = windows.shape[0]
        P = self.window_volume
        flat = windows.view(N,3,P).permute(0,2,1).contiguous()

        A = flat.unsqueeze(2)
        B = flat.unsqueeze(1)
        depth1, lat1, lon1 = A[...,0], A[...,1], A[...,2]
        depth2, lat2, lon2 = B[...,0], B[...,1], B[...,2]

        delta_lon = ((lon1 - lon2 + 180) % 360) - 180
        dlat = torch.deg2rad(lat1 - lat2)
        dlon = torch.deg2rad(delta_lon)
        cos_lat1 = torch.cos(torch.deg2rad(lat1)).expand(-1,-1,P)
        cos_lat2 = torch.cos(torch.deg2rad(lat2)).expand(-1,P,-1)
        a = torch.sin(dlat/2)**2 + cos_lat1*cos_lat2*torch.sin(dlon/2)**2
        great_circle = 2*torch.asin(torch.sqrt(a))
        delta_depth = depth1 - depth2

        # pair_pos_code_on_windows = torch.stack([cos_lat1, cos_lat2, great_circle], dim=-1)
        pair_pos_code_on_windows = torch.stack([cos_lat1, cos_lat2, great_circle, delta_depth], dim=-1)

        return pair_pos_code_on_windows.view(N*P*P, pair_pos_code_on_windows.shape[-1]), nWd, nWh, nWw

    def forward(self):
        flat_pair = self.shifted_flat_pair_pos_code_on_windows
        h = self.act(self.fc1(flat_pair))
        h = self.norm(h)
        h = self.fc2(h)
        return h.view(self.nWindows, self.window_volume, self.window_volume, self.num_heads) * self.scale

# ---------------------------
# WindowAttention3D
# ---------------------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.nWindows = None

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, position_bias, mask=None):
        B_, N, C = x.shape  # B * nWindows, win_volume, C

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2,-1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1./0.01).to(self.logit_scale.device))).exp()
        attn = attn * logit_scale

        position_bias = position_bias.permute(0,3,1,2).contiguous() # (nWindows, num_heads, win_volume, win_volume)
        if self.nWindows is None:
            self.nWindows = position_bias.shape[0]
        B = B_ // self.nWindows
        attn = attn.view(B, self.nWindows, self.num_heads, N, N)    # (B, nWindows, num_heads, win_volume, win_volume)
        position_bias = 16 * torch.sigmoid(position_bias)
        attn = attn + position_bias         # propagate to B
        attn = attn.view(B_, self.num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ---------------------------
# Swin Block 3D
# ---------------------------
class Zeus_SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(2,4,4), shift_size=(0,0,0),
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        assert any(s==0 for s in shift_size) or any(s!=0 for s in shift_size)
        assert dim % num_heads == 0, f"dim {dim} is not divisible with num_heads {num_heads}"

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_volume = window_size[0] * window_size[1] * window_size[2]
        self.shift_size = shift_size

        for i in range(3):
            if self.input_resolution[i] <= self.window_size[i]:
                self.shift_size = tuple([0,0,0])
                self.window_size = tuple(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, input_resolution, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.positionBiasLearner = PositionBiasLearner3D(input_resolution, window_size, num_heads, shift_size=shift_size, hidden_dim=6)

        if any(s != 0 for s in self.shift_size):
            # calculate attention mask for SW-MSA
            D, H, W = self.input_resolution
            img_mask = torch.zeros((1, 1, D, H, W))  # (batch, channel, D, H, W)
            d_slices = (slice(0, -self.window_size[0]),
                        slice(-self.window_size[0], -self.shift_size[0]),
                        slice(-self.shift_size[0], None))
            h_slices = (slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None))
            w_slices = (slice(0, -self.window_size[2]),
                        slice(-self.window_size[2], -self.shift_size[2]),
                        slice(-self.shift_size[2], None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, :, d, h, w] = cnt
                        cnt += 1

            mask_windows, _, _, _ = window_partition_3d(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_volume)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D*H*W, f"Input feature has wrong size {L}, expected {D*H*W}"

        shortcut = x
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        # cyclic shift
        if any(s != 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=self.shift_size, dims=(2,3,4))     # x: (B, C, D, H, W)
        else:
            shifted_x = x

        # partition windows
        x_windows, nWd, nWh, nWw = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, C, self.window_volume).permute(0,2,1).contiguous()

        # posision bias
        position_bias = self.positionBiasLearner()      # (nWindows, win_volume, win_volume, num_heads)

        # attention
        attn_windows = self.attn(x_windows, position_bias, mask=self.attn_mask)

        # reverse windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.window_size)
        x = window_reverse_3d(attn_windows, self.window_size, D, H, W)

        # reverse cyclic shift
        if any(s != 0 for s in self.shift_size):
            x = torch.roll(x, shifts=[-s for s in self.shift_size], dims=(2,3,4))

        # flatten back
        x = x.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()

        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def show_on_map(data):  # np.array

    # # Example: your data tensor (replace with your real tensor)
    # data = np.random.rand(720, 1440)  # shape (lat, lon)
    mean = data.mean()
    # Latitude and longitude vectors at cell centers
    H, W = data.shape
    
    lat_step = 180.0 / H
    lon_step = 360.0 / W
    lat = torch.linspace(-90 + lat_step/2, 90 - lat_step/2, H)
    lon = torch.linspace(-180 + lon_step/2, 180 - lon_step/2, W)
    lat = lat.numpy()
    lon = lon.numpy()

    # Create a meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Plot
    fig = plt.figure(figsize=(12,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    # Plot the data as a colormap
    c = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add colorbar
    cb = plt.colorbar(c, ax=ax, orientation='vertical', label='Value')

    # Add country boundaries
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Add gridlines for lat/lon
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title(f"Geographical Distribution. mean = {mean}")
    plt.show()

import time

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    factor10 = 2  # 2, 3
    H, W, D = int(90/10*factor10), int(180/10*factor10), 14
    print(H, W, D)
    C, num_heads = 16, 4
    window_size = (2, int(10/10*factor10), int(10/10*factor10))  # (win_D, win_H, win_W)

    input_resolution=(D, H, W)

    # Initialize 3D Swin Transformer Blocks
    blocks3D = [
        Zeus_SwinTransformerBlock3D(
            dim=C,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0,0,0) if idx % 2 == 0 else (window_size[0]//2, window_size[1]//2, window_size[2]//2),
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0
        ).to(device)
        for idx in range(30)  # reduced number for testing
    ]

    # Random input: (B, C, D, H, W)
    x_in = torch.randn(1, C, D, H, W).to(device).to(device)

    # Flatten input for block input
    B, C, D, H, W = x_in.shape
    x_flat = x_in.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()

    # Forward pass
    s = time.time()
    for b in blocks3D:
        x_flat = b(x_flat)
    e = time.time()

    # Reshape back to (B, C, D, H, W)
    x_out = x_flat.view(B, D, H, W, C).permute(0,4,1,2,3).contiguous()
    print(f"Output shape: {x_out.shape}, time: {(e-s):.3f} sec")
    assert x_out.shape == x_in.shape


    PBLearner = PositionBiasLearner3D(input_resolution=(D, H, W), window_size=window_size, num_heads=num_heads)
    pos_code = PBLearner.shifted_flat_pair_pos_code_on_windows # (nWindows * window_volume * window_volume, 4)
    
    pos_code = pos_code.view(PBLearner.nWindows, PBLearner.window_volume, PBLearner.window_volume, 4) # (nWindows, indow_volume, window_volume, 4)
    pos_code = pos_code[:, 0, :, :]
    pos_code = pos_code.view(D//window_size[0], H//window_size[1], W//window_size[2], window_size[0], window_size[1], window_size[2],  4)
    pos_code = pos_code.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    pos_code = pos_code.view(D, H, W, 4)[0] # (H, W, 4).
    data = pos_code[:, :, 2].numpy()    # cos_lat1, cos_lat2, great_circle, delta_depth

    show_on_map(data)