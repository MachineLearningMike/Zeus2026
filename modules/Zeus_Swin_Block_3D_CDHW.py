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

import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.config_Mike1 import config
from modules.zeus_dateset_geo_tempo import Geo_Tempo

class PositionBiasLearner3D(nn.Module):
    pair_bias_shared, shifted_pair_bias_shared = None, None
    
    def __init__(self, input_resolution, window_size, num_heads, hidden_dim, shift_size, pair_bias_dim):
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

        depth = torch.arange(D, dtype=torch.float32) * depth_step
        
        # Do we need half-step? -----------------------------------------
    
        # lat = torch.linspace(-90 + lat_step/2, 90 - lat_step/2, H + 1)
        # lon = torch.linspace(-180 + lon_step/2, 180 - lon_step/2, W + 1)
        
        lat = torch.linspace(-90, 90, H + 1)
        lon = torch.linspace(-180, 180, W + 1)

        lat = lat[1:] if Geo_Tempo.respect_north else lat[:-1]
        lon = lon[1:] if Geo_Tempo.respect_east else lon[:-1]

        self.lat = lat[None,:,None].expand(D,H,W)
        self.lon = lon[None,None,:].expand(D,H,W)
        self.depth = depth[:,None,None].expand(D,H,W)

        # Precompute flat_pair with the given shift
        pair_bias, nWd, nWh, nWw = self.pair_position_partition(
            self.lat, self.lon, self.depth, shift_size=self.shift_size, pair_bias_dim=pair_bias_dim
        )
        self.nWindows = nWd*nWh*nWw
        
        #---------------------------------------------------------------------------------------------------------------------
        # pair_bias: (nWindows * window_volume * window_volume, c)
        # nWindows = D / window_size[0] * H / window_size[1] * W / window_size[2]
        # pair.pair_bias.shape[0] = D / window_size[0] * H / window_size[1] * W / window_size[2] * window_volume * window_volume
        # = D * H * W * window_volume
        # = total_swin_volume * window_volume   ------------- window_volume decides the size of pair_bias
        #
        # nWindows = total_swin_volume / wondow_volume
        #---------------------------------------------------------------------------------------------------------------------
        
        self.shifted = any(s != 0 for s in shift_size)
        
        if not self.shifted:
            if PositionBiasLearner3D.pair_bias_shared is None:
                PositionBiasLearner3D.pair_bias_shared = pair_bias

            self.register_buffer(
                "pair_bias", 
                PositionBiasLearner3D.pair_bias_shared,
                persistent=False
            )
        else:
            if PositionBiasLearner3D.shifted_pair_bias_shared is None:
                PositionBiasLearner3D.shifted_pair_bias_shared = pair_bias

            self.register_buffer(
                "pair_bias",
                PositionBiasLearner3D.shifted_pair_bias_shared,
                persistent=False
            )
        
        # Learnable layers
        self.fc1 = nn.Linear(pair_bias_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, num_heads)
        
        self.scale = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)

    def pair_position_partition(self, lat, lon, depth, shift_size, pair_bias_dim):

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

        dlat = lat1 - lat2
        dlon = ((lon1 - lon2 + 180) % 360) - 180

        dlat = torch.deg2rad(dlat)
        dlon = torch.deg2rad(dlon)

        cos_lat1 = torch.cos(torch.deg2rad(lat1)).expand(-1,-1,P)
        cos_lat2 = torch.cos(torch.deg2rad(lat2)).expand(-1,P,-1)
        a = torch.sin(dlat/2)**2 + cos_lat1*cos_lat2*torch.sin(dlon/2)**2
        great_circle = 2*torch.asin(torch.sqrt(a))
        
        # It's -1, 0 or 1 in a window, and is not a constant.
        # Issue: -1 and +1, directionally different, yet imply similar physical meaning.
        # So, ... 
        # Do we need depth besides great_circle?
        delta_depth = depth1 - depth2

        # Why cos**2 ? Because it represents the volume of a grid cell linearly.
        # pair_bias = torch.stack([cos_lat1 ** 2, cos_lat2 ** 2, great_circle], dim=-1)
        pair_bias = torch.stack([cos_lat1 ** 2, cos_lat2 ** 2, great_circle, delta_depth, torch.abs(delta_depth)], dim=-1)
        # pair_bias = torch.stack([cos_lat1 ** 2, cos_lat2 ** 2, great_circle, torch.abs(delta_depth)], dim=-1)
        assert pair_bias.shape[-1] == pair_bias_dim

        return pair_bias.view(N*P*P, pair_bias.shape[-1]), nWd, nWh, nWw

    def forward(self):
        flat_pair = self.pair_bias
        h = self.act1(self.fc1(flat_pair))
        h = self.act2(self.fc2(h))
        h = self.fc3(h)
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
            self.q_bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
            self.v_bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    #------------------------------------------------ One-shot attention 
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

        if position_bias is not None:
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

    # #------------------------------------------------ Multi-shot attention
    # def forward(self, x, position_bias, mask=None):
    #     B_, N, C = x.shape  # B * nWindows, win_volume, C
        
    #     qkv_bias = None
    #     if self.q_bias is not None:
    #         qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
    #     qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
    #     qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
    #     q, k, v = qkv[0], qkv[1], qkv[2]
        
    #     # Normalize queries and keys
    #     q = F.normalize(q, dim=-1)
    #     k = F.normalize(k, dim=-1)
        
    #     logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1./0.01).to(self.logit_scale.device))).exp()
        
    #     # Prepare position bias
    #     position_bias = position_bias.permute(0, 3, 1, 2).contiguous()  # (nWindows, num_heads, win_volume, win_volume)
    #     position_bias = 16 * torch.sigmoid(position_bias)
        
    #     if self.nWindows is None:
    #         self.nWindows = position_bias.shape[0]
    #     B = B_ // self.nWindows
        
    #     # Initialize output
    #     output = torch.zeros(B_, self.num_heads, N, C // self.num_heads, dtype=x.dtype, device=x.device)
        
    #     # Process in chunks to save memory
    #     Times = 4  # You can adjust this value
    #     chunk_size = N // Times
    #     if chunk_size == 0:
    #         chunk_size = 1
    #         Times = N
        
    #     for i in range(Times):
    #         start_idx = i * chunk_size
    #         end_idx = min((i + 1) * chunk_size, N)
    #         chunk_length = end_idx - start_idx
            
    #         # Calculate attention for current chunk
    #         attn_chunk = torch.matmul(q[:, :, start_idx:end_idx], k.transpose(-2, -1))
    #         attn_chunk = attn_chunk * logit_scale
            
    #         # Reshape attention for position bias addition
    #         # attn_chunk shape: (B_, num_heads, chunk_length, N)
    #         attn_chunk = attn_chunk.view(B, self.nWindows, self.num_heads, chunk_length, N)
            
    #         # Add position bias
    #         # position_bias shape: (nWindows, num_heads, N, N)
    #         # We need to extract the right slice for the chunk
    #         position_bias_chunk = position_bias[:, :, start_idx:end_idx, :]  # (nWindows, num_heads, chunk_length, N)
    #         attn_chunk = attn_chunk + position_bias_chunk.unsqueeze(0)
            
    #         # Reshape back
    #         attn_chunk = attn_chunk.view(B_, self.num_heads, chunk_length, N)
            
    #         # Apply mask if provided
    #         if mask is not None:
    #             nW = mask.shape[0]
    #             # Reshape for mask addition
    #             attn_chunk_view = attn_chunk.view(B_ // nW, nW, self.num_heads, chunk_length, N)
                
    #             # Get the appropriate mask slice
    #             mask_chunk = mask[:, start_idx:end_idx, :]  # (nW, chunk_length, N)
                
    #             attn_chunk = attn_chunk_view + mask_chunk.unsqueeze(1).unsqueeze(0)
    #             attn_chunk = attn_chunk.view(-1, self.num_heads, chunk_length, N)
            
    #         # Apply softmax and attention dropout
    #         attn_chunk = self.softmax(attn_chunk)
    #         attn_chunk = self.attn_drop(attn_chunk)
            
    #         # Apply attention to values
    #         output_chunk = torch.matmul(attn_chunk, v)  # (B_, num_heads, chunk_length, head_dim)
    #         output[:, :, start_idx:end_idx] = output_chunk
        
    #     # Transpose and reshape output
    #     x = output.transpose(1, 2).reshape(B_, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
        
    #     return x

class AttentionMask3D(nn.Module):     # original, mine
    #------------------------------------------------------------------------------------------------------------------------
    # consistent attn_mask: ([roll negative, attend, roll positive], assert all(s >= 0 for s in shift_size), AttentionMask3D)
    #------------------------------------------------------------------------------------------------------------------------
    attn_mask_shared = {}  # key: shift_size tuple, value: mask tensor

    def __init__(self, input_resolution, window_size, shift_size):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        assert all(s >= 0 for s in shift_size)
        self.shift_size = shift_size  # tuple (shift_D, shift_H, shift_W)

        # Convert to hashable key for sharing
        key = tuple(shift_size)
        if key not in AttentionMask3D.attn_mask_shared:
            # Compute mask once
            attn_mask = self.compute_mask(input_resolution, window_size, shift_size)
            AttentionMask3D.attn_mask_shared[key] = attn_mask

        # Register buffer for this instance
        self.register_buffer("attn_mask", AttentionMask3D.attn_mask_shared[key], persistent=False)

    @staticmethod
    def compute_mask(input_resolution, window_size, shift_size):
        D, H, W = input_resolution
        if all(s == 0 for s in shift_size):
            return None  # no shift, no mask

        # Create mask
        img_mask = torch.zeros((1, 1, D, H, W))
        d_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        h_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        
        w_slices = (slice(0, -window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None))

        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    # img_mask[:, :, d, h, w] = cnt
                    img_mask[:, :, d, h] = cnt          #----------------------- This should give cylindric effects on W dim.
                    cnt += 1

        # Partition windows
        windows, nWd, nWh, nWw = window_partition_3d(img_mask, window_size)
        mask_windows = windows.view(-1, window_size[0]*window_size[1]*window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    

# ---------------------------
# Swin Block 3D
# ---------------------------
class Zeus_SwinTransformerBlock3D(nn.Module):
    #------------------------------------------------------------------------------------------------------------------------
    # consistent attn_mask: ([roll negative, attend, roll positive], assert all(s >= 0 for s in shift_size), AttentionMask3D)
    #------------------------------------------------------------------------------------------------------------------------
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size, pair_bias_dim,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_pos_bias=True):
        super().__init__()
        
        assert any(s==0 for s in shift_size) or any(s!=0 for s in shift_size)
        assert dim % num_heads == 0, f"dim {dim} is not divisible with num_heads {num_heads}"

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_volume = window_size[0] * window_size[1] * window_size[2]
        assert all(s >= 0 for s in shift_size)
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
        
        if use_pos_bias:
            self.positionBiasLearner = PositionBiasLearner3D(input_resolution, window_size, num_heads, shift_size=shift_size, pair_bias_dim=pair_bias_dim, hidden_dim=pair_bias_dim * 4)
        else:
            self.positionBiasLearner = None

        self.attn_mask_learner = AttentionMask3D(input_resolution, window_size, shift_size)

    #---------------------------------------- One-shot FFN
    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D*H*W, f"Input feature has wrong size {L}, expected {D*H*W}"

        shortcut = x
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        # cyclic shift
        if any(s != 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=[ - s for s in self.shift_size], dims=(2,3,4))     # x: (B, C, D, H, W)
        else:
            shifted_x = x

        # partition windows
        x_windows, nWd, nWh, nWw = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, C, self.window_volume).permute(0,2,1).contiguous()

        # posision bias
        if self.positionBiasLearner is not None:
            position_bias = self.positionBiasLearner()      # (nWindows, win_volume, win_volume, num_heads)
        else:
            position_bias = None

        # attention
        attn_windows = self.attn(x_windows, position_bias, mask=self.attn_mask_learner.attn_mask)

        # reverse windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.window_size)
        x = window_reverse_3d(attn_windows, self.window_size, D, H, W)

        # reverse cyclic shift
        if any(s != 0 for s in self.shift_size):
            x = torch.roll(x, shifts=[s for s in self.shift_size], dims=(2,3,4))

        # flatten back
        x = x.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()

        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x
    
    # #--------------------------------------- Multi-shot FFN
    # def forward(self, x):
    #     """Modified forward with chunked MLP - minimal changes"""
    #     D, H, W = self.input_resolution
    #     B, L, C = x.shape
    #     assert L == D*H*W, f"Input feature has wrong size {L}, expected {D*H*W}"

    #     # --- ATTENTION PART (unchanged) ---
    #     shortcut = x
    #     x = x.transpose(1, 2).reshape(B, C, D, H, W)

    #     if any(s != 0 for s in self.shift_size):
    #         shifted_x = torch.roll(x, shifts=[ - s for s in self.shift_size], dims=(2,3,4))
    #     else:
    #         shifted_x = x

    #     x_windows, nWd, nWh, nWw = window_partition_3d(shifted_x, self.window_size)
    #     x_windows = x_windows.view(-1, C, self.window_volume).permute(0,2,1).contiguous()

    #     position_bias = self.positionBiasLearner()
    #     attn_windows = self.attn(x_windows, position_bias, mask=self.attn_mask_learner.attn_mask)

    #     attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.window_size)
    #     x = window_reverse_3d(attn_windows, self.window_size, D, H, W)

    #     if any(s != 0 for s in self.shift_size):
    #         x = torch.roll(x, shifts=[s for s in self.shift_size], dims=(2,3,4))

    #     x = x.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()

    #     # --- OPTIMIZED FFN PART ---
    #     # First residual
    #     x = shortcut + self.drop_path(self.norm1(x))
        
    #     # Chunked MLP to reduce peak memory
    #     B, L, C = x.shape
        
    #     # Auto-tune chunk size based on sequence length
    #     # Aim for ~4MB per chunk (adjust based on your GPU)
    #     target_chunk_tokens = 64  # Start with this, tune based on your needs
    #     chunk_size = min(L, target_chunk_tokens)
        
    #     # Process MLP in chunks
    #     mlp_output = torch.empty_like(x)
    #     for i in range(0, L, chunk_size):
    #         end_idx = min(i + chunk_size, L)
    #         chunk = x[:, i:end_idx, :]
    #         # norm2 + MLP
    #         processed_chunk = self.norm2(chunk)
    #         processed_chunk = self.mlp(processed_chunk)
    #         mlp_output[:, i:end_idx, :] = processed_chunk
        
    #     # Apply drop_path and final residual
    #     x = x + self.drop_path(mlp_output)
    #     return x


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


def show_global_attn_mask(attn_mask, window_size, input_resolution, depth_idx=0):
    if attn_mask is None:
        print("No attention mask (no shift).")
        return

    D, H, W = input_resolution
    win_D, win_H, win_W = window_size
    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W
    window_volume = win_D * win_H * win_W

    # Initialize global mask
    global_mask = torch.zeros((D, H, W), device=attn_mask.device)

    idx = 0
    for d in range(nWd):
        for h in range(nWh):
            for w in range(nWw):
                mask_win = attn_mask[idx]  # (window_volume, window_volume)
                mask_win_6d = mask_win.view(win_D, win_H, win_W, win_D, win_H, win_W)
                # Take attention of queries at depth_idx slice
                mask_slice = mask_win_6d[depth_idx, :, :, depth_idx, :, :].sum(dim=(2,3))
                # Place it in the global mask
                global_mask[d*win_D + depth_idx, h*win_H:(h+1)*win_H, w*win_W:(w+1)*win_W] = mask_slice
                idx += 1

    mask_2d = global_mask[depth_idx].cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(mask_2d, cmap='viridis', origin='lower')
    plt.colorbar(label='Attention Mask')
    plt.title(f"Aggregated Attention Mask (depth={depth_idx})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


import cartopy.crs as ccrs
import cartopy.feature as cfeature

def show_global_attn_mask_map(attn_mask, window_size, input_resolution, depth_idx=0):
    if attn_mask is None:
        print("No attention mask (no shift).")
        return

    D, H, W = input_resolution
    win_D, win_H, win_W = window_size
    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W
    window_volume = win_D * win_H * win_W

    global_mask = torch.zeros((D, H, W), device=attn_mask.device)

    idx = 0
    for d in range(nWd):
        for h in range(nWh):
            for w in range(nWw):
                mask_win = attn_mask[idx]  # (window_volume, window_volume)
                mask_win_6d = mask_win.view(win_D, win_H, win_W, win_D, win_H, win_W)
                mask_slice = mask_win_6d[depth_idx, :, :, depth_idx, :, :].sum(dim=(2,3))
                global_mask[d*win_D + depth_idx, h*win_H:(h+1)*win_H, w*win_W:(w+1)*win_W] = mask_slice
                idx += 1

    mask_2d = global_mask[depth_idx].cpu().numpy()

    lat_step = 180.0 / H
    lon_step = 360.0 / W
    lat = np.linspace(-90 + lat_step/2, 90 - lat_step/2, H)
    lon = np.linspace(-180 + lon_step/2, 180 - lon_step/2, W)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    c = ax.pcolormesh(lon_grid, lat_grid, mask_2d, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    plt.colorbar(c, ax=ax, orientation='vertical', label='Attention Mask')
    plt.title(f"Global Attention Mask at depth={depth_idx}")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch

def show_global_position_bias(PBLearner, pair_dim_idx=2, depth_idx=0):
    """
    Visualize the aggregated position bias of a given pair dimension
    (e.g., great-circle distance) at a specified depth slice.
    """
    device = next(PBLearner.parameters()).device
    D, H, W = PBLearner.input_resolution
    win_D, win_H, win_W = PBLearner.window_size
    nWd, nWh, nWw = D // win_D, H // win_H, W // win_W
    window_volume = win_D * win_H * win_W
    pair_bias_dim = PBLearner.fc1.in_features  # original input dim

    # Select the correct shared tensor
    pair_bias_ref = PBLearner.pair_bias
    # Reshape flat pair_bias to (nWindows, win_D, win_H, win_W, win_D, win_H, win_W, pair_bias_dim)
    pair_bias_grid = pair_bias_ref.reshape(
        nWd * nWh * nWw,
        win_D, win_H, win_W,
        win_D, win_H, win_W,
        pair_bias_dim
    )

    # Initialize global bias
    global_bias = torch.zeros((D, H, W), device=device)

    # Map each window back to its global position
    idx = 0
    for d in range(nWd):
        for h in range(nWh):
            for w in range(nWw):
                window_bias = pair_bias_grid[idx]  # (win_D, win_H, win_W, win_D, win_H, win_W, pair_bias_dim)
                # Take bias for queries at depth_idx, aggregate keys along D/H/W
                slice_bias = window_bias[depth_idx, :, :, depth_idx, :, :, pair_dim_idx]  # (win_H, win_W, win_H, win_W)
                slice_bias_2d = slice_bias.sum(dim=(2,3))  # (win_H, win_W)
                # Assign to global bias
                global_bias[d*win_D + depth_idx, h*win_H:(h+1)*win_H, w*win_W:(w+1)*win_W] = slice_bias_2d
                idx += 1

    # Move to CPU and convert to numpy
    bias_2d = global_bias[depth_idx].cpu().numpy()
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.imshow(bias_2d, cmap='viridis', origin='lower')
    plt.colorbar(label=f'Position Bias dim={pair_dim_idx}')
    plt.title(f"Aggregated Position Bias (depth={depth_idx})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()




#-------------------------------------------------------------------------------------------------------------------------
import time

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    factor10 = 2  # 2, 3
    H, W, D = int(90/10*factor10), int(180/10*factor10), 14
    print(H, W, D)
    C, num_heads = 16, 4
    window_size = (2, int(10/10*factor10), int(10/10*factor10))  # (win_D, win_H, win_W)
    pair_bias_dim = 5

    input_resolution=(D, H, W)

    # Initialize 3D Swin Transformer Blocks
    blocks3D = [
        Zeus_SwinTransformerBlock3D(
            dim=C,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=(0,0,0) if idx % 2 == 0 else (window_size[0]//2, window_size[1]//2, window_size[2]//2),
            pair_bias_dim = pair_bias_dim,
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

    shift_size = (0,0,0)
    PBLearner = PositionBiasLearner3D(input_resolution=(D, H, W), window_size=window_size,shift_size=shift_size, num_heads=num_heads, pair_bias_dim=pair_bias_dim, hidden_dim=pair_bias_dim*2)
    pair_bias = PBLearner.pair_bias # (nWindows * window_volume * window_volume, 4)
    
    pair_bias = pair_bias.view(PBLearner.nWindows, PBLearner.window_volume, PBLearner.window_volume, pair_bias_dim) # (nWindows, indow_volume, window_volume, 4)
    pair_bias = pair_bias[:, 0, :, :]
    pair_bias = pair_bias.view(D//window_size[0], H//window_size[1], W//window_size[2], window_size[0], window_size[1], window_size[2],  pair_bias_dim)
    pair_bias = pair_bias.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    pair_bias = pair_bias.view(D, H, W, pair_bias_dim)[0] # (H, W, 4).
    data = pair_bias[:, :, 2].numpy()    # cos_lat1, cos_lat2, great_circle, delta_depth
    
    # show_global_position_bias(PBLearner, pair_dim_idx=4, depth_idx=0)
    # pair_bias = torch.stack([cos_lat1 ** 2, cos_lat2 ** 2, great_circle, delta_depth, torch.abs(delta_depth)], dim=-1)

    # show_on_map(data)    
    
    attn_mask = blocks3D[1].attn_mask_learner.attn_mask
    # show_global_attn_mask(attn_mask, window_size=window_size, input_resolution=input_resolution, depth_idx=0)
    show_global_attn_mask_map(attn_mask, window_size=window_size, input_resolution=input_resolution, depth_idx=0)