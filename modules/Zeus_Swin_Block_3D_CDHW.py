import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import copy

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
def window_partition_3d(x, sWin):
    # x: (B, C, D, H, W)
    B, C, D, H, W = x.shape
    win_D, win_H, win_W = sWin
    assert D % win_D == 0 and H % win_H == 0 and W % win_W == 0, \
        f"Input dims {(D,H,W)} must be multiples of sWin {sWin}"

    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W

    x = x.view(B, C, nWd, win_D, nWh, win_H, nWw, win_W)
    # (B, nWd, nWh, nWw, C, win_D, win_H, win_w) -> (-1, C, win_D, win_H, win_W)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, C, win_D, win_H, win_W)
    return windows, nWd, nWh, nWw

def window_reverse_3d(windows, sWin, D, H, W):
    win_D, win_H, win_W = sWin
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

class PositionBiasLearner3D_Shpere(nn.Module):
    pair_bias_shared, shifted_pair_bias_shared = None, None
    
    def __init__(self, i_reso, sWin, num_heads, hidden_dim, shift_size, pair_bias_dim):
        super().__init__()
        self.i_reso = i_reso
        self.sWin = sWin
        self.vWin = sWin[0]*sWin[1]*sWin[2]
        self.num_heads = num_heads
        self.shift_size = shift_size  # shift applied when creating flat_pair

        D, H, W = i_reso
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
        self.nWin = nWd*nWh*nWw
        
        #---------------------------------------------------------------------------------------------------------------------
        # THIS structure avoids a pair_bias matrix shared between windows, and allows each window has its own pair_bias matrix.
        #
        # pair_bias: (nWin * vWin * vWin, pair_bias_dim)
        # nWin = D / sWin[0] * H / sWin[1] * W / sWin[2]
        # pair.pair_bias.shape[0] = D / sWin[0] * H / sWin[1] * W / sWin[2] * vWin * vWin
        # = D * H * W * vWin
        # = total_swin_volume * vWin   ------------- vWin decides the size of pair_bias
        #
        # nWin = total_swin_volume / wondow_volume
        #---------------------------------------------------------------------------------------------------------------------
        
        self.shifted = any(s != 0 for s in shift_size)
        
        if not self.shifted:
            if PositionBiasLearner3D_Shpere.pair_bias_shared is None:
                PositionBiasLearner3D_Shpere.pair_bias_shared = pair_bias

            self.register_buffer(
                "pair_bias", 
                PositionBiasLearner3D_Shpere.pair_bias_shared,
                persistent=False
            )
        else:
            if PositionBiasLearner3D_Shpere.shifted_pair_bias_shared is None:
                PositionBiasLearner3D_Shpere.shifted_pair_bias_shared = pair_bias

            self.register_buffer(
                "pair_bias",
                PositionBiasLearner3D_Shpere.shifted_pair_bias_shared,
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
            self.sWin
        )
        
        N = windows.shape[0]
        P = self.vWin
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
        return h.view(self.nWin, self.vWin, self.vWin, self.num_heads) * self.scale

# ---------------------------
# WindowAttention3D
# ---------------------------
class WindowAttention3D(nn.Module):
    def __init__(self, dim, i_reso, sWin, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.i_reso = i_reso
        self.sWin = sWin
        self.nWin = None

        # I gave up this technique. Read forward()
        # self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.qkv = nn.Linear(dim, dim*3, bias=False)   # qkv_bais==True leads to 'gradient not received'
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
        B_, N, C = x.shape  # B * nWin, win_volume, C

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        head_dim = q.shape[-1]
        attn = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # cosine attention ----------- This block is given up: attn = attn * logit_scale causes OOM on 80BG GPUs.
        # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        # device = self.logit_scale.device
        # logit_scale = torch.clamp(self.logit_scale, min=torch.log(torch.tensor(1e-6, device=device)), max=torch.log(torch.tensor(100., device=device))).exp()
        # attn = attn * logit_scale #--------------- This line causes OOM on 80GB GPUs, like A100 and H100.
        # We have to content ourselves with the flexibility of embedded position_bias and softmax.

        if position_bias is not None:   # (nWin, win_volume, win_volume, num_heads)
            position_bias = position_bias.permute(0,3,1,2).contiguous() # (nWin, num_heads, win_volume, win_volume)
            if self.nWin is None:
                self.nWin = position_bias.shape[0]
            B = B_ // self.nWin
            attn = attn.view(B, self.nWin, self.num_heads, N, N)    # (B, nWin, num_heads, win_volume, win_volume)
            
            # position_bias = 16 * torch.sigmoid(position_bias)  #--------------- This line causes OOM on 80GB GPUs, like A100 and H100.
            # position_bias is an embedded tensor, like lonBand, it can freely scale withouth this technique.
            
            attn = attn + position_bias         # propagate to B
            attn = attn.view(B_, self.num_heads, N, N)  # (B * nWin_ num_heads, win_vol, win_vol)

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
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

    # #------------------------------------------------ Multi-shot attention
    # def forward(self, x, position_bias, mask=None):
    #     B_, N, C = x.shape  # B * nWin, win_volume, C
        
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
    #     position_bias = position_bias.permute(0, 3, 1, 2).contiguous()  # (nWin, num_heads, win_volume, win_volume)
    #     position_bias = 16 * torch.sigmoid(position_bias)
        
    #     if self.nWin is None:
    #         self.nWin = position_bias.shape[0]
    #     B = B_ // self.nWin
        
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
    #         attn_chunk = attn_chunk.view(B, self.nWin, self.num_heads, chunk_length, N)
            
    #         # Add position bias
    #         # position_bias shape: (nWin, num_heads, N, N)
    #         # We need to extract the right slice for the chunk
    #         position_bias_chunk = position_bias[:, :, start_idx:end_idx, :]  # (nWin, num_heads, chunk_length, N)
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

    def __init__(self, i_reso, sWin, shift_size, padding=None):
        super().__init__()
        self.i_reso = i_reso
        self.sWin = sWin
        assert all(s >= 0 for s in shift_size)
        self.shift_size = shift_size  # tuple (shift_D, shift_H, shift_W)

        # Convert to hashable key for sharing
        padding_key = () if padding is None else padding
        key = (*i_reso, *sWin, *shift_size, *padding_key)
        if key not in AttentionMask3D.attn_mask_shared:
            # Compute mask once
            attn_mask = self.compute_mask(i_reso, sWin, shift_size, padding)
            AttentionMask3D.attn_mask_shared[key] = attn_mask

        # Register buffer for this instance
        self.register_buffer("attn_mask", AttentionMask3D.attn_mask_shared[key], persistent=False)

    @staticmethod
    def compute_mask(i_reso, sWin, shift_size, padding):
        D, H, W = i_reso
        if all(s == 0 for s in shift_size) and padding is None:
            return None

        #------------------------- torch.roll ------------------------------------------------------------------------------------
        #   x = torch.tensor([0, 1, 2, 3, 4, 5])
        #   shifted_x = torch.roll(x, shifts=-1, dims=0)    # tensor([1, 2, 3, 4, 5, 0])
        #
        #   When this mask is created, org_x = [0, 1, 2, 3, 4, 5] is already shifted negatively forming shifted_z = [1, 2, 3, 4, 5, 0],
        #   assuming sWin == 2 and shift_size == 1.
        #   So, the window-wide attention mask should be (0,0; 0,0), (0,0; 0,0), (0, -inf; -inf,0) for windows, respectively.
        #
        #   NOTE: A negative shift or positive shift, there is no imparity like which side of indices are respected more.
        #-------------------------------------------------------------------------------------------------------------------------
        def get_slices(s_size, w_size, pad):
            # ASSUMPTION: pad is attached leftward, or to lower indices.
            
            if s_size != 0 and pad == 0:  # shift only, it's a standard Swin mask.
                slices = (slice(0, -w_size),            # L = total - w_size        # (0, 1, 2, 3)              sWin[0] == 2
                            slice(-w_size, -s_size),    # L = w_size - s_size       # (4)                       shift_size[0] == 1
                            slice(-s_size, None))       # L = s_size                # (5)
                                                                                    # (t, t, t, t, t1, t2)            after filling in with type_id
                                                                                    # (t, t), (t, t), (t1, t2)        after window_partition_3d
                                                                                    # (0,0; 0,0), (0,0; 0,0), (0,-100; -100,0)    after masked_fill
                                                                                    # This means mask is palced on the higher-index side.
                                                                                    # This is consistent with negative shift, where the higher-index side has imigrants, like 0.
            elif s_size == 0 and pad != 0:              # pad only
                slices = (slice(0, pad),                # pad.    L = pad                   # Left pad columns are no-values.
                            slice(pad, None))           # main.   L = total - pad

            elif s_size != 0 and pad != 0:
                if pad <= s_size:                               # The whole pad is rolled to the right.
                    slices = (slice(0, -w_size),                # main.         L = total - w_size
                                slice(-w_size, -s_size),        # originals.    L = w_size - s_size
                                slice(-s_size, -pad),           # immigrants.   L = s_size - pad
                                slice(-pad, None))              # pad.          L = pad
                else:
                    slices = (slice(0, pad - s_size),           # pad2.         L = pad - s_size
                                slice(pad - s_size, -w_size),   # main.         L = total - pad + s_size - w_size
                                slice(-w_size, -s_size),        # originals.    L = w_size - s_size
                                slice(-s_size, None))           # pad1.         L = s_size
            elif s_size == 0 and pad == 0:              # s_size == 0 includes total <= w_size
                slices = (slice(None),)                      # total L = total
            
            return slices      
      
        #------------------------------------------ D ------------------------------------------------------------  
        idx = 0
        pad = padding[idx] if padding is not None else 0
        w_size = sWin[idx]
        s_size = shift_size[idx]
        assert pad < w_size
        
        d_slices = get_slices(s_size, w_size, pad)

        #------------------------------------------ H ------------------------------------------------------------  
        idx = 1
        pad = padding[idx] if padding is not None else 0
        w_size = sWin[idx]
        s_size = shift_size[idx]
        assert pad < w_size
        
        h_slices = get_slices(s_size, w_size, pad)
        
        #------------------------------------------ W ------------------------------------------------------------  
        idx = 2
        pad = padding[idx] if padding is not None else 0
        w_size = sWin[idx]
        s_size = shift_size[idx]
        assert pad < w_size

        # w_slices = (slice(0, -w_size),
        #             slice(-w_size, -shift_size),
        #             slice(-s_size, None))

        # This line replaces the above block and creates an attention mask that's cylindric on W dim.
        
        if s_size != 0 and pad == 0:                        # shift only, in a cylindric mode.  Attend free.
            w_slices = (slice(None),)                       # total L = total
            
        elif s_size == 0 and pad != 0:                      # The whole pad is on the left.
            w_slices = (slice(0, pad),                      # pad.   L = pad
                        slice(pad, None))                   # main.  L = total - pad   ----------- total L = total
            
        elif s_size != 0 and pad != 0:
            if pad <= s_size:                               # The whole pad is rolled to the right.
                w_slices = (slice(0, -pad),                 # main.  L = total - pad
                            slice(-pad, None))              # pad.   L = pad   --------- total L = total
            else:
                w_slices = (slice(0, pad - s_size),         # pad2.  L = pad - s_size
                            slice(pad - s_size, -s_size),   # main.  L = total - pad + s_size - s_size = total - pad
                            slice(-s_size, None))           # pad1.  L = s_size    ------- total L = total
        else:
            w_slices = (slice(None),)                       # total L = total

        #-------------------------------------- Build --------------------------------------
        # Create mask
        img_mask = torch.zeros((1, 1, D, H, W))
        
        region_id = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, :, d, h, w] = region_id
                    region_id += 1

        # Partition windows
        windows, nWd, nWh, nWw = window_partition_3d(img_mask, sWin)
        mask_windows = windows.view(-1, sWin[0]*sWin[1]*sWin[2])
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
    def __init__(self, dim, i_reso, num_heads, sWin, shift_size, pair_bias_dim,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, type_pos_bias='lonBand', padding=None, inherited_padding=None,
                 is_padding_block=False, in_encoder=True):
        
        # is_padding_block == True : this block is either the entry of an encoding layer or the exit of a decoding layer.   Responsisble for padding / un-padding
        # padding is shared between all blocks in the layer.
        
        super().__init__()
        
        # assert all(s==0 for s in shift_size) or all(s!=0 for s in shift_size)
        assert dim % num_heads == 0, f"dim {dim} is not divisible with num_heads {num_heads}"

        self.dim = dim
        self.i_reso = i_reso
        self.input_vol = i_reso[0] * i_reso[1] * i_reso[2]
        self.num_heads = num_heads
        self.sWin = sWin
        self.vWin = sWin[0] * sWin[1] * sWin[2]
        self.nWin = (i_reso[0] // sWin[0]) * (i_reso[1] // sWin[2]) * (i_reso[2] // sWin[2])
        assert all(s >= 0 for s in shift_size)      # We accept non-nagative shift only. The sign is managed by this module.
        self.shift_size = shift_size

        # !!!!!  This block is generated by GPT, leading to a crazy behavior when resolution <= sWin
        # for i in range(3):
        #     if self.i_reso[i] <= self.sWin[i]:
        #         self.shift_size = tuple([0,0,0])
        #         self.sWin = tuple(self.i_reso)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, i_reso, sWin, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.type_pos_bias = type_pos_bias
        
        if type_pos_bias == 'shpere':
            self.positionBiasLearner = PositionBiasLearner3D_Shpere(i_reso, sWin, num_heads, shift_size=shift_size, pair_bias_dim=pair_bias_dim, hidden_dim=pair_bias_dim * 4)
        elif type_pos_bias == 'lonBand':
            D, H, W = i_reso
            wd, wh, ww = sWin
            # (n_windows on a lon band of windows, win_vol, win_vol, num_heads)
            self.learnable_pair_types = nn.Parameter(torch.empty(int(D/wd) * int(H/wh), wd*wd, wh*wh, (2*ww-1), num_heads), requires_grad=True)
            nn.init.trunc_normal_(self.learnable_pair_types, std=0.02)
                
        mask_padding = [p + ip for (p, ip) in zip(padding if padding is not None else (0,0,0), inherited_padding if inherited_padding is not None else (0,0,0))]
        mask_padding = None if sum(mask_padding) == 0 else mask_padding       
        self.attn_mask = AttentionMask3D(i_reso, sWin, shift_size, padding=mask_padding)     # All blocks in the layer have the same final mask
            
        self.unpadded_resolution = i_reso    
        if padding is not None:
            self.unpadded_resolution = [i_reso[dim] - padding[dim] for dim in range(3)]  # resolutions for D, H and W each
        self.unpadded_vol = self.unpadded_resolution[0] * self.unpadded_resolution[1] * self.unpadded_resolution[2]
                
        # (dim_left, dim_right, w_left, w_right, h_left, h_right, d_left, d_right, B_left, B_right)
        self.padding_config = (0,0, padding[2],0, padding[1],0, padding[0],0, 0,0) if padding is not None else None 
        self.pad_sum = 0 if padding is None else sum(self.padding_config)
            
        self.is_padding_block = is_padding_block
        self.in_encoder = in_encoder
        
        # nChunks = 20
        # nChunks = self.find_min_divisor_above(self.nWin, nChunks)
        # self.nWin_chunks = self.create_gpu_chunks(self.nWin, nChunks, device='cuda')

    def build_pair_type_matrix_sphere(self):
        """
        Build pair_type_matrix covering full sphere by repeating along longitude.
        Returns: pair_type_matrix of shape (total_windows, N, N, num_heads)
        """
        wd, wh, ww = self.sWin
        D, H, W = self.i_reso
        device = self.learnable_pair_types.device
        G, _, _, _, num_heads = self.learnable_pair_types.shape
        N = wd * wh * ww

        # --- Token coordinates inside a window
        d = torch.arange(wd, device=device)
        h = torch.arange(wh, device=device)
        w = torch.arange(ww, device=device)
        coords = torch.stack(torch.meshgrid(d, h, w, indexing="ij")).view(3, -1)  # (3, N)
        d1, d2 = coords[0][:, None], coords[0][None, :]
        h1, h2 = coords[1][:, None], coords[1][None, :]
        w1, w2 = coords[2][:, None], coords[2][None, :]

        idx_d = d1 * wd + d2
        idx_h = h1 * wh + h2
        idx_w = (w1 - w2 + (ww - 1)) % (2*ww-1)  # cylindric along W

        # Expand along longitude
        G_idx = torch.arange(G, device=device)[:, None, None, None]
        W_idx = torch.arange(W // ww, device=device)[None, :, None, None]  # number of windows along longitude

        D_idx = idx_d[None, None, :, :]
        H_idx = idx_h[None, None, :, :]
        W_idx_tokens = idx_w[None, None, :, :]

        # repeat learnable_pair_types along longitude
        pair_type_matrix = self.learnable_pair_types[:, None, :, :, :, :].repeat(1, W // ww, 1, 1, 1, 1)

        # advanced indexing to select correct pair types
        pair_type_matrix = pair_type_matrix[
            G_idx, W_idx, D_idx, H_idx, W_idx_tokens
        ]  # (G, num_lon_windows, N, N, num_heads)

        # merge G and longitude dims
        pair_type_matrix = pair_type_matrix.view(G * (W // ww), N, N, num_heads)
        return pair_type_matrix

    def find_min_divisor_above(self, n, nChunks):
        if n <= 0 or nChunks <= 0:
            return None
        
        # Check if nChunks itself divides n
        if n % nChunks == 0:
            return nChunks
        
        # Search upwards from nChunks
        for m in range(nChunks, n + 1):
            if n % m == 0:
                return m
        
        return None  # No divisor found (shouldn't happen since n divides n)

    def create_gpu_chunks(self, total_size, n_chunks, device='cuda'):
        # Create indices on GPU
        indices = torch.arange(total_size, device=device)
        
        # Calculate chunk boundaries
        chunk_size = total_size // n_chunks
        remainder = total_size % n_chunks
        
        chunks = []
        start = 0
        
        for i in range(n_chunks):
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            
            # Get slice of indices tensor (creates view, no copy)
            chunk_indices = indices[start:end]
            
            chunks.append((start, end, chunk_indices))
            start = end
        
        return chunks

    #---------------------------------------- One-shot FFN
    def forward(self, x):
        
        # Wrap the standard Swin Transformer Block forward with padding and de-padding
        
        if self.in_encoder and self.is_padding_block and self.pad_sum > 0:      # Pad as an entrry of a encoding BasicBlock
            [D, H, W] = self.unpadded_resolution
            
            B, L, dim = x.shape
            x = x.view(B, D, H, W, dim)
            x = F.pad(x, self.padding_config, mode='constant')  # padding with zeros
            x = x.view(B, -1, dim)
        
        D, H, W = self.i_reso
        B, L, C = x.shape
        assert L == D*H*W, f"Input feature has wrong size {L}, expected {D*H*W}"

        shortcut = x
        x = x.transpose(1, 2).reshape(B, C, D, H, W)

        #------------------------------------------------------------------------------------------------------------------------
        # consistent attn_mask: ([roll negative, attend, roll positive], assert all(s >= 0 for s in shift_size), AttentionMask3D)
        #------------------------------------------------------------------------------------------------------------------------

        #------------------------- torch.roll ----------------------------------------
        #   x = torch.tensor([1, 2, 3])
        #   shifted_x = torch.roll(x, shifts=-1, dims=0)    # tensor([2, 3, 1])
        #-----------------------------------------------------------------------------

        # cyclic shift
        if any(s != 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=[ - s for s in self.shift_size], dims=(2,3,4))     # x: (B, C, D, H, W)
        else:
            shifted_x = x

        # partition windows
        x_windows, nWd, nWh, nWw = window_partition_3d(shifted_x, self.sWin)
        x_windows = x_windows.view(-1, C, self.vWin).permute(0,2,1).contiguous()

        # posision bias
        
        if self.type_pos_bias == 'sphere':
            position_bias = self.positionBiasLearner()      # (nWin, win_volume, win_volume, num_heads)
        elif self.type_pos_bias == 'lonBand':
            position_bias = self.build_pair_type_matrix_sphere()

        # attn_windows: (nWin, vWin, dim), x_windows: (nWin, vWin, dim), posision_bias: (nWin, vWin, vWin, nHeads), mask: None or (nWin, vWin, vWin)
        attn_windows = self.attn(x_windows, position_bias, mask=self.attn_mask.attn_mask)
        
        # results = []
        # for start, end, chunk_indices in self.nWin_chunks:
        #     x_windows_chunk, position_bias_chunk, mask_chunk = x_windows[chunk_indices], position_bias[chunk_indices], None if self.attn_mask.attn_mask is None else self.attn_mask.attn_mask[chunk_indices]
        #     attn_windows_chunk = self.attn(x_windows_chunk, position_bias_chunk, mask_chunk)
        #     results.append(attn_windows_chunk)
        # attn_windows = torch.cat(results, dim=0)        

        # reverse windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.sWin)
        x = window_reverse_3d(attn_windows, self.sWin, D, H, W)

        #------------------------- torch.roll ----------------------------------------
        #   x = torch.tensor([2, 3, 1])
        #   shifted_x = torch.roll(x, shifts=+1, dims=0)    # tensor([1, 2, 3])
        #-----------------------------------------------------------------------------

        # reverse cyclic shift
        if any(s != 0 for s in self.shift_size):
            x = torch.roll(x, shifts=[s for s in self.shift_size], dims=(2,3,4))

        # flatten back
        x = x.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()

        # FFN
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        
        if not self.in_encoder and self.is_padding_block and self.pad_sum > 0:  # Un-Pad as a exit of a decoding BasicBlock
            # This is the exit block of a decoding BasicLayer. It's time to un-pad what the corresponding entry block of encoding BasicLayer paded. 
            [D, H, W] = self.i_reso
            B, L, dim = x.shape
            x = x.view(B, D, H, W, dim)
            padding_config = [-c for c in self.padding_config]  # de-padding
            x = F.pad(x, tuple(padding_config), mode='constant')  # zeros
            x = x.view(B, -1, dim)
        
        return x
    
    def flops(self):
        flops = 0
        D, H, W = self.i_reso
        # norm1
        flops += self.dim * D * H * W
        # W-MSA/SW-MSA
        nW = D * H * W / self.vWin
        flops += nW * self.attn.flops(self.vWin)
        # mlp
        flops += 2 * D * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * D * H * W
        return flops
    
    # #--------------------------------------- Multi-shot FFN
    # def forward(self, x):
    #     """Modified forward with chunked MLP - minimal changes"""
    #     D, H, W = self.i_reso
    #     B, L, C = x.shape
    #     assert L == D*H*W, f"Input feature has wrong size {L}, expected {D*H*W}"

    #     # --- ATTENTION PART (unchanged) ---
    #     shortcut = x
    #     x = x.transpose(1, 2).reshape(B, C, D, H, W)

    #     if any(s != 0 for s in self.shift_size):
    #         shifted_x = torch.roll(x, shifts=[ - s for s in self.shift_size], dims=(2,3,4))
    #     else:
    #         shifted_x = x

    #     x_windows, nWd, nWh, nWw = window_partition_3d(shifted_x, self.sWin)
    #     x_windows = x_windows.view(-1, C, self.vWin).permute(0,2,1).contiguous()

    #     position_bias = self.positionBiasLearner()
    #     attn_windows = self.attn(x_windows, position_bias, mask=self.attn_mask.attn_mask)

    #     attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.sWin)
    #     x = window_reverse_3d(attn_windows, self.sWin, D, H, W)

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


class PatchMerging3D(nn.Module):
    r""" 3D Patch Merging Layer.

    Args:
        i_reso (tuple[int]): (D, H, W)
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer.
    """

    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim

        # 8C -> 2C reducing capability, not -> 8**0.5 * C, because I don't wantt to preserve physics.
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: (B, D*H*W, C)
        """
        D, H, W = self.i_reso
        B, L, C = x.shape

        assert L == D * H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, \
            f"x size ({D}*{H}*{W}) must be even"

        x = x.view(B, D, H, W, C)

        x000 = x[:, 0::2, 0::2, 0::2, :]
        x100 = x[:, 1::2, 0::2, 0::2, :]
        x010 = x[:, 0::2, 1::2, 0::2, :]
        x110 = x[:, 1::2, 1::2, 0::2, :]
        x001 = x[:, 0::2, 0::2, 1::2, :]
        x101 = x[:, 1::2, 0::2, 1::2, :]
        x011 = x[:, 0::2, 1::2, 1::2, :]
        x111 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat(
            [x000, x100, x010, x110, x001, x101, x011, x111],
            dim=-1
        )  # (B, D/2, H/2, W/2, 8C)

        x = x.view(B, -1, 8 * C)
        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"i_reso={self.i_reso}, dim={self.dim}"

    def flops(self):
        D, H, W = self.i_reso
        flops = (D // 2) * (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += D * H * W * self.dim // 2
        return flops


class PatchUnmerging3D(nn.Module):
    r"""3D Patch Unmerging Layer (dual of PatchMerging3D)

    Args:
        i_reso (tuple[int]): (D, H, W)
        dim (int): Input channel dimension C
        norm_layer (nn.Module): Normalization layer
    """

    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        assert dim % 2 == 0, "dim must be even since C_out = C//2"

        self.out_dim = dim // 2

        # C -> 8*(C//2) = 4C
        self.expansion = nn.Linear(dim, 8 * self.out_dim, bias=False)
        self.norm = norm_layer(8 * self.out_dim)

    def forward(self, x):
        """
        x: (B, D*H*W, C)
        """
        D, H, W = self.i_reso
        B, L, C = x.shape

        assert L == D * H * W, "input feature has wrong size"
        assert C == self.dim

        # expand channels
        x = self.expansion(x)          # (B, D*H*W, 4C)
        x = self.norm(x)

        # reshape to grid
        x = x.view(B, D, H, W, 8 * self.out_dim)

        C_out = self.out_dim

        # split channels into 8 groups
        x000 = x[..., 0*C_out:1*C_out]
        x100 = x[..., 1*C_out:2*C_out]
        x010 = x[..., 2*C_out:3*C_out]
        x110 = x[..., 3*C_out:4*C_out]
        x001 = x[..., 4*C_out:5*C_out]
        x101 = x[..., 5*C_out:6*C_out]
        x011 = x[..., 6*C_out:7*C_out]
        x111 = x[..., 7*C_out:8*C_out]

        # scatter to higher resolution
        out = torch.zeros(
            B, 2*D, 2*H, 2*W, C_out,
            device=x.device,
            dtype=x.dtype
        )

        out[:, 0::2, 0::2, 0::2, :] = x000
        out[:, 1::2, 0::2, 0::2, :] = x100
        out[:, 0::2, 1::2, 0::2, :] = x010
        out[:, 1::2, 1::2, 0::2, :] = x110
        out[:, 0::2, 0::2, 1::2, :] = x001
        out[:, 1::2, 0::2, 1::2, :] = x101
        out[:, 0::2, 1::2, 1::2, :] = x011
        out[:, 1::2, 1::2, 1::2, :] = x111

        # flatten back
        out = out.view(B, (2*D)*(2*H)*(2*W), C_out)

        return out

    def extra_repr(self) -> str:
        return (
            f"i_reso={self.i_reso}, "
            f"dim={self.dim}, out_dim={self.out_dim}"
        )



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        i_reso (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        sWin (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        resampler (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_sWin (int): Local window size in pre-training.
    """

    def __init__(self, dim, i_reso, depth, num_heads, sWin, pair_bias_dim,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, resampler=PatchMerging3D, use_checkpoint=False,  #------- use_checkpoint = True for enhanced BasicLayer
                 pretrained_sWin=0, type_pos_bias='lonBand', padding=None, inherited_padding=None, in_encoder=True):

        super().__init__()
        self.dim = dim
        self.i_reso = i_reso
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.padding = padding
        self.in_encoder = in_encoder
        
        self.o_reso = i_reso
        if resampler is not None:
            if in_encoder:
                self.o_reso = (r//2 for r in i_reso)
            else:
                self.o_reso = (r*2 for r in i_reso)

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # This shift_size doesn't mask attention if total_size <= sWin. It doesn't divide them into attention regions.
            shift_size=(0,0,0) if (i % 2 == 0) else \
            (
                sWin[0] // 2 if i_reso[0] > sWin[0] else 0,
                sWin[1] // 2 if i_reso[1] > sWin[1] else 0, 
                sWin[2] // 2 if i_reso[2] > sWin[2] else 0,
            )
        
            block = Zeus_SwinTransformerBlock3D(dim=dim, i_reso=i_reso,
                                num_heads=num_heads, sWin=sWin,
                                # odd windows_sizes are allowed here.
                                shift_size=shift_size,
                                pair_bias_dim = pair_bias_dim,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,
                                #  pretrained_sWin=pretrained_sWin,
                                type_pos_bias = type_pos_bias, padding=padding, inherited_padding=inherited_padding,
                                is_padding_block=(i==0),
                                in_encoder = in_encoder,
                                )
            self.blocks.append(block)

        # patch merging layer
        if resampler is not None:
            incoming_resolution = i_reso if in_encoder else self.blocks[0].unpadded_resolution
            
            self.resampler = resampler(incoming_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.resampler = None

    def forward(self, x, ):
        layer_output = []
        blocks = self.blocks if self.in_encoder else reversed(self.blocks)
        for blk in blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        layer_output.append(x)
        if self.resampler is not None:
            x = self.resampler(x)
        layer_output.append(x)
        return layer_output
    
    def enhance_layer_irreversible(self, layer, nEnhancementPairs=1):
        """
        Device-safe enhancement.
        """
        if hasattr(layer, 'gates'):
            return layer
        
        # Get current device
        try:
            device = next(layer.parameters()).device
        except StopIteration:
            # Layer has no parameters? Use CPU
            device = torch.device('cpu')
        
        print(f"  Enhancing on: {device}")
        
        # #--------------------- Define the parallels ---------------------- 
        # # This was not a success. The gates, zero initially, grew VERY slowly blocking the flow of gradient.

        # # 1. Gates on correct device        
        # layer.gates = nn.ParameterList([
        #     nn.Parameter(torch.normal(0.0, 0.01, size=(1,), device=device, requires_grad=True)) 
        #     for _ in layer.blocks
        # ])
        
        # for i, gate in enumerate(layer.gates):
        #     layer.register_parameter(f'gate_{i}', gate)
        
        # # 2. Parallel blocks
        # layer.parallel_blocks = nn.ModuleList()
        
        # for i, orig_block in enumerate(layer.blocks):
        #     # Deep copy
        #     parallel_block = copy.deepcopy(orig_block)
        #     parallel_block = parallel_block.to(device)
            
        #     # Re-initialize with DIFFERENT initialization than original
        #     for name, param in parallel_block.named_parameters():
        #         if param.dim() >= 2:
        #             # Use orthogonal initialization for better gradient flow
        #             if 'weight' in name:
        #                 nn.init.orthogonal_(param)
        #             else:
        #                 nn.init.normal_(param, mean=0.0, std=0.02)
        #         elif param.dim() == 1:
        #             if 'bias' in name:
        #                 nn.init.constant_(param, 0.1)  # Small positive bias
        #             else:
        #                 nn.init.zeros_(param)
            
        #     layer.parallel_blocks.append(parallel_block)
        
        #------------ A 2nd attempt -----------------------------
        #----- insert 1 or 2 pairs of blocks in the sequence of 8 blocks.
        
        assert len(self.blocks) >= 6, f" len(self.blocks) == { len(self.blocks)}"
        
        def copy_and_zero(block):
            new_block = copy.deepcopy(block)
            
            # FIXED: Modify new_block, not the original block!
            with torch.no_grad():
                nn.init.normal_(new_block.attn.proj.weight, mean=0.0, std=0.001)
                nn.init.zeros_(new_block.attn.proj.bias)          #  new_block, not block
                nn.init.normal_(new_block.mlp.fc2.weight, mean=0.0, std=0.001)  #  new_block
                nn.init.zeros_(new_block.mlp.fc2.bias)            #  new_block
            
            return new_block  #  Return the modified block

        # Clone blocks 2 and 3
        enhancement_pairs = []
        for p in range(nEnhancementPairs):
            even_block = copy_and_zero(self.blocks[2])  # W-MSA block
            odd_block = copy_and_zero(self.blocks[3])   # SW-MSA block
            enhancement_pairs += [even_block, odd_block]

        # CORRECTED: Insert after block 3 (between pair 1 and pair 2)
        # Original: [0, 1, 2, 3, 4, 5, 6, 7]
        # Insert after position 3 (after block 3)
        new_blocks_list = list(self.blocks)  # Convert ModuleList to list
        new_blocks_list = new_blocks_list[:4] + enhancement_pairs + new_blocks_list[4:]
        self.blocks = nn.ModuleList(new_blocks_list)  # Convert back to ModuleList
 
        #-----------------------------------------------------------------       
                
        # 4. Forward
        original_forward = layer.forward
        
        def diverse_enhanced_forward(self, x):
            if not hasattr(layer, 'gates'):
                return original_forward(x)
            
            layer_output = []
            
            sources = list(zip(
                layer.blocks,
                layer.parallel_blocks,
                layer.gates
            ))
            
            # Process with diverse parallel blocks
            for orig_block, parallel_block, gate in sources if self.in_encoder else reversed(sources):
                orig_out = orig_block(x)  # Proven wisdom                
                parallel_out = parallel_block(x)  # Random explorer
                x = orig_out + gate * parallel_out    

            layer_output.append(x)      # x before, possible, resampling

            if self.resampler is not None:
                x = self.resampler(x)
            layer_output.append(x)
                
            return layer_output
        
        
        import types
        layer.forward = types.MethodType(diverse_enhanced_forward, layer)
        
        return layer

    def extra_repr(self) -> str:
        return f"dim={self.dim}, i_reso={self.i_reso}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.resampler is not None:
            flops += self.resampler.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class Encoder(nn.Module):
    def __init__(self, i_reso = (8, 180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(2, 6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,  
                # absolute position eocoding is required in geo-shpere, because geo-cat and geo-dense is not enough.
                use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=5, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.skippable = skippable

        num_patches = i_reso[0] * i_reso[1] * i_reso[2]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        prev_resolution = (i_reso[0] * 2, i_reso[1] * 2, i_reso[2] * 2)
        inherited_padding = [0,0,0]
        
        for i_layer in range(self.num_layers):

            #--------------------- Define i_reso -----------------
            # 1) There is no general mothod to find compact input resolution that is both a multiple of window size and an even number.
            # A even number resolution is required if the BasicLayer has PatchMerghing3D
            # 2) If i_reso[dim] % (2 ** i_layer) > 0, x.shape and i_reso mismatches in forward passes.

            for dim in range(3):
                assert prev_resolution[dim] % 2 == 0, f"Input resolution will mismatch with incoming x.shape in BasicLayer {i_layer}"

            # This will be the output resolution from the previous BasicLayer, if it has downsampler as we ensure here.
            resolution=[prev_resolution[0] // 2,
                        prev_resolution[1] // 2,
                        prev_resolution[2] // 2,
            ]

            padding = []
            for i in range(3):
                # Ensure resolution is a multiple of window size
                pad = 0 if resolution[i] % sWin[i] == 0 else sWin[i] - resolution[i] % sWin[i]
                if i_layer != self.num_layers - 1: assert pad % 2 == 0     # for patch-merging  # Keep it even in decoder
                if pad != 0:
                    print(f"!!!!!!!!!! Encoder layer {i_layer}: input resolution[{i}] {resolution[i]} is increased by {pad} to be a multiple of window size {sWin[i]}")
                resolution[i] += pad
                padding.append(pad)
            
            donwsample = PatchMerging3D if (i_layer < self.num_layers - 1) else None
            if donwsample is not None:
                for i in range(3):
                    assert resolution[i] % 2 == 0, f"Encoder: BasicLayer {i_layer} has an odd input resolution {resolution} while having PatchMerging."
            
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               i_reso=resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               sWin=sWin,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               resampler=donwsample,
                               use_checkpoint=use_checkpoint,
                               pretrained_sWin=pretrained_sWins[i_layer],
                               pair_bias_dim = pair_bias_dim, padding=padding, inherited_padding=inherited_padding,
                               in_encoder=True)
            inherited_padding = [p//2 for p in padding]
            self.layers.append(layer)            
            prev_resolution = resolution

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, x):
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        encoder_output = []

        for idx, layer in enumerate(self.layers):
            layer_output = layer(x)
            x = layer_output[1]

            if self.skippable or (not self.skippable and idx == len(self.layers)-1):
                encoder_output.append(layer_output)

        # for idx, (enc_layer, (before_ds, after_ds)) in enumerate(zip(self.layers, encoder_output)):
        #     print(f"{idx}, enc_layer: {enc_layer.dim}, {enc_layer.i_reso}, enc_before_ds: {before_ds.shape}, enc_after_ds: {after_ds.shape}")

        return encoder_output


class Decoder(nn.Module):
    def __init__(self, i_reso = (8, 180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(2, 6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,  
                # absolute position eocoding is required in geo-shpere, because geo-cat and geo-dense is not enough.
                use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=5, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        skip_learnable = True   # no time to put it in the config

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.skip_learnable = skip_learnable
        self.skippable = skippable

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        if skip_learnable:
            self.skip_learners = nn.ModuleList()
        prev_resolution = (i_reso[0] * 2, i_reso[1] * 2, i_reso[2] * 2)
        inherited_padding = [0,0,0]
        
        for i_layer in range(self.num_layers):

            #--------------------- Define i_reso -----------------
            # 1) There is no general mothod to find compact input resolution that is both a multiple of window size and an even number.
            # A even number resolution is required if the BasicLayer has PatchMerghing3D
            # 2) If i_reso[dim] % (2 ** i_layer) > 0, x.shape and i_reso mismatches in forward passes.

            for dim in range(3):
                assert prev_resolution[dim] % 2 == 0, f"Input resolution will mismatch with incoming x.shape in BasicLayer {i_layer}"

            # This will be the output resolution from the previous BasicLayer, if it has downsampler as we ensure here.
            resolution=[prev_resolution[0] // 2,
                        prev_resolution[1] // 2,
                        prev_resolution[2] // 2,
            ]
            dim = int(embed_dim * 2 ** i_layer)

            padding = []
            for i in range(3):
                # Ensure resolution is a multiple of window size
                pad = 0 if resolution[i] % sWin[i] == 0 else sWin[i] - resolution[i] % sWin[i]
                if i_layer != self.num_layers - 1: assert pad % 2 == 0     # for patch-merging  # Keep it even in decoder
                if pad != 0:
                    print(f"!!!!!!!!!! Decoder layer {i_layer}: input resolution[{i}] {resolution[i]} is increased by {pad} to be a multiple of window size {sWin[i]}")
                resolution[i] += pad
                padding.append(pad)
            
            donwsample = PatchMerging3D if (i_layer < self.num_layers - 1) else None
            if donwsample is not None:
                for i in range(3):
                    assert resolution[i] % 2 == 0, f"Decoder: BasicLayer {i_layer} has an odd input resolution {resolution} while having PatchMerging."
            
            layer = BasicLayer(dim=dim,
                               i_reso=resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               sWin=sWin,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               resampler=PatchUnmerging3D if i_layer > 0 else None, # (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_sWin=pretrained_sWins[i_layer],
                               pair_bias_dim = pair_bias_dim, padding=padding, inherited_padding=inherited_padding,
                               in_encoder=False)
            inherited_padding = [p//2 for p in padding]
            self.layers.append(layer)
            prev_resolution = resolution

            if self.skippable and skip_learnable:
                skip_learner = nn.Linear(dim, dim, bias=True)
                torch.nn.init.eye_(skip_learner.weight)
                torch.nn.init.zeros_(skip_learner.bias)
                self.skip_learners.append(skip_learner)            

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward(self, enc_x_by_layer):        

        if self.skippable:
            dec_x_after_us = torch.zeros_like(enc_x_by_layer[-1][0])
            
            if self.skip_learnable:            
                for idx, (dec_layer, (enc_x_before_ds, _), skip_learner) in enumerate(zip(reversed(self.layers), reversed(enc_x_by_layer), reversed(self.skip_learners))):
                    # print(f"{idx}, dec_layer: {dec_layer.dim}, {dec_layer.i_reso}, enc_before_ds: {enc_x_before_ds.shape}")
                    input = dec_x_after_us + skip_learner(enc_x_before_ds) 
                    (_, dec_x_after_us) = dec_layer(input)
            else:
                for idx, (dec_layer, (enc_x_before_ds, _)) in enumerate(zip(reversed(self.layers), reversed(enc_x_by_layer))):
                    # print(f"{idx}, dec_layer: {dec_layer.dim}, {dec_layer.i_reso}, enc_before_ds: {enc_x_before_ds.shape}")
                    input = dec_x_after_us + enc_x_before_ds 
                    (_, dec_x_after_us) = dec_layer(input)
    
        else:
            dec_x_after_us = enc_x_by_layer[-1][0]
            for layer in reversed(self.layers):
                (_, dec_x_after_us) = layer(dec_x_after_us)
            
        return dec_x_after_us

class Encoder_Decoder(nn.Module):
    def __init__(self, i_reso = (8, 180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(2, 6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,  
                # absolute position eocoding is required in geo-shpere, because geo-cat and geo-dense is not enough.
                use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=5, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        self.encoder = Encoder(i_reso = i_reso, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                sWin=sWin, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=nn.LayerNorm, ape=True,  
                # absolute position eocoding is required in geo-shpere, because geo-cat and geo-dense is not enough.
                use_checkpoint=use_checkpoint, pretrained_sWins=pretrained_sWins, pair_bias_dim=pair_bias_dim, 
                skippable=skippable, skip_learnable=skip_learnable, **kwargs)
        
        self.decoder = Decoder(i_reso = i_reso, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                sWin=sWin, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=nn.LayerNorm, ape=True,  
                # absolute position eocoding is required in geo-shpere, because geo-cat and geo-dense is not enough.
                use_checkpoint=use_checkpoint, pretrained_sWins=pretrained_sWins, pair_bias_dim=pair_bias_dim, 
                skippable=skippable, skip_learnable=skip_learnable, **kwargs)
        
        if skippable and skip_learnable:
            self.skip_learner = nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.eye_(self.skip_learner.weight)
            torch.nn.init.zeros_(self.skip_learner.bias)
        self.skippable = skippable
        self.skip_learnable = skip_learnable
        
        
    def forward(self, x_flat):
        if self.skippable:
            x_skip = x_flat     # .clone() ------ no clone() because self.encoder doesn't modify its input arguments.
        x_by_layer = self.encoder(x_flat)
        x = self.decoder(x_by_layer)
        if self.skippable:
            if self.skip_learnable:
                x = x + self.skip_learner(x_skip)
            else:
                x = x + x_skip
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


def show_global_attn_mask(attn_mask, sWin, i_reso, depth_idx=0):
    if attn_mask is None:
        print("No attention mask (no shift).")
        return

    D, H, W = i_reso
    win_D, win_H, win_W = sWin
    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W
    vWin = win_D * win_H * win_W

    # Initialize global mask
    global_mask = torch.zeros((D, H, W), device=attn_mask.device)

    idx = 0
    for d in range(nWd):
        for h in range(nWh):
            for w in range(nWw):
                mask_win = attn_mask[idx]  # (vWin, vWin)
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

def show_global_attn_mask_map(attn_mask, sWin, i_reso, depth_idx=0, title="", projection='HW'):
    if attn_mask is None:
        print("No attention mask (no shift).")
        return

    D, H, W = i_reso
    win_D, win_H, win_W = sWin
    nWd = D // win_D
    nWh = H // win_H
    nWw = W // win_W
    vWin = win_D * win_H * win_W

    global_mask = torch.zeros((D, H, W), device=attn_mask.device)

    idx = 0
    for d in range(nWd):
        for h in range(nWh):
            for w in range(nWw):
                mask_win = attn_mask[idx]  # (vWin, vWin)
                mask_win_6d = mask_win.view(win_D, win_H, win_W, win_D, win_H, win_W)
                
                if projection == 'HW':
                    mask_slice = mask_win_6d[depth_idx, :, :, depth_idx, :, :].sum(dim=(2,3))
                    global_mask[d*win_D + depth_idx, h*win_H:(h+1)*win_H, w*win_W:(w+1)*win_W] = mask_slice
                elif projection == 'DH':
                    mask_slice = mask_win_6d[:, :, depth_idx, :, :, depth_idx].sum(dim=(2,3))                
                    global_mask[d*win_D:(d+1)*win_D, h*win_H:(h+1)*win_H, w*win_W + depth_idx] = mask_slice
                
                idx += 1

    if projection == 'HW':
        mask_2d = global_mask[depth_idx].cpu().numpy()
    elif projection == 'DH':
        mask_2d = global_mask[:, :, depth_idx].cpu().numpy()

    if projection == 'HW':
        lat_step = 180.0 / H
        lon_step = 360.0 / W
        lat = np.linspace(-90 + lat_step/2, 90 - lat_step/2, H)
        lon = np.linspace(-180 + lon_step/2, 180 - lon_step/2, W)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
    elif projection == 'DH':
        lat_step = 8.0 / mask_2d.shape[0]
        lon_step = 180.0 / mask_2d.shape[1]
        lat = np.linspace(-4 + lat_step/2, 4 - lat_step/2, H)
        lon = np.linspace(-90 + lon_step/2, 90 - lon_step/2, W)
        lon_grid, lat_grid = np.meshgrid(lon, lat)


    fig = plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    c = ax.pcolormesh(lon_grid, lat_grid, mask_2d, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    plt.colorbar(c, ax=ax, orientation='vertical', label='Attention Mask')
    plt.title(f"Maks: depth {depth_idx}, R {i_reso}, WS {sWin}, " + title, fontsize=9)
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
    D, H, W = PBLearner.i_reso
    win_D, win_H, win_W = PBLearner.sWin
    nWd, nWh, nWw = D // win_D, H // win_H, W // win_W
    vWin = win_D * win_H * win_W
    pair_bias_dim = PBLearner.fc1.in_features  # original input dim

    # Select the correct shared tensor
    pair_bias_ref = PBLearner.pair_bias
    # Reshape flat pair_bias to (nWin, win_D, win_H, win_W, win_D, win_H, win_W, pair_bias_dim)
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

    # factor10 = 2  # 2, 3
    # H, W, D = int(180/10*factor10), int(360/10*factor10), 8
    # print("H, W, D", H, W, D)
    # C, num_heads = 16, 4
    # sWin = (2, int(10/10*factor10), int(10/10*factor10))  # (win_D, win_H, win_W)
    # pair_bias_dim = 5

    # i_reso=(D, H, W)
    
    # #---------------- block --------------------

    # # Initialize 3D Swin Transformer Blocks
    # blocks3D = [
    #     Zeus_SwinTransformerBlock3D(
    #         dim=C,
    #         i_reso=i_reso,
    #         num_heads=num_heads,
    #         sWin=sWin,
    #         shift_size=(0,0,0) if idx % 2 == 0 else (sWin[0]//2, sWin[1]//2, sWin[2]//2),
    #         pair_bias_dim = pair_bias_dim,
    #         mlp_ratio=4.,
    #         qkv_bias=True,
    #         drop=0.0,
    #         attn_drop=0.0,
    #         drop_path=0.0
    #     ).to(device)
    #     for idx in range(30)  # reduced number for testing
    # ]

    # # Random input: (B, C, D, H, W)
    # x_in = torch.randn(1, C, D, H, W).to(device).to(device)

    # # Flatten input for block input
    # B, C, D, H, W = x_in.shape
    # x_flat = x_in.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()
    
    # print(f"x_flat: ({x_flat.shape})")

    # #------------------- blocks -------------------------

    # x_flat_clone = x_flat.clone()
    # # Forward pass
    # s = time.time()
    # for b in blocks3D:
    #     x_flat_clone = b(x_flat_clone)
    # e = time.time()
    
    # print(f"x_flat after blocks: ({x_flat.shape}), time after 30 blocks: {e-s}")

    # #------------------------------------------------------------------------------------------------------------------
    # #------------------------ Basic layer -----------------------
    
    # factor10 = 2  # 2, 3
    # H, W, D = int(180/10*factor10), int(360/10*factor10), 8
    # print("H, W, D", H, W, D)
    # C, num_heads = 16, 4
    # sWin = (2, int(10/10*factor10), int(10/10*factor10))  # (win_D, win_H, win_W)
    # pair_bias_dim = 5

    # i_reso=(D, H, W)
    
    # # Random input: (B, C, D, H, W)
    # x_in = torch.randn(1, C, D, H, W).to(device).to(device)

    # # Flatten input for block input
    # B, C, D, H, W = x_in.shape
    # x_flat = x_in.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()
    
    
    # depth = 6
    # merging = False          
    # bLayer = BasicLayer(dim=C, i_reso=i_reso, depth=depth, num_heads=num_heads, sWin=sWin, pair_bias_dim=pair_bias_dim,
    #              mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
    #              drop_path=0., norm_layer=nn.LayerNorm, resampler=PatchMerging3D if merging else PatchMerging3D, use_checkpoint=False,
    #              pretrained_sWin=0, type_pos_bias='lonBand').to(device)
    
    # # Forward pass
    # s = time.time()
    # layer_output = bLayer(x_flat)
    # e = time.time()

    # print(f"x_flat after basicLayer of depth 6 with last patch merging, befor_ds: ({layer_output[0].shape}), after_ds: ({layer_output[1].shape}), time: {e-s}")
    
    print("====================================== Big Test ==============================================")
    
    factor10, sWin = 2, (2, 4, 4)
    # factor10, sWin = 10, (2, 6, 6) # (2, 6, 12), (2, 4, 8)
    
    depths, nHeads = (4, 8, 4), (4, 8, 16)
    patch_size = 4
    H, W, D = int(int(720/patch_size)/10*factor10), int(int(1440/patch_size)/10*factor10), 8
    C, num_heads = 8, 4
    pair_bias_dim = 5
    i_reso=(D, H, W)

    skippable = True
    skip_learnable = True

    # Random input: (B, C, D, H, W)
    x_in = torch.randn(1, C, D, H, W).to(device).to(device)
    # Flatten input for block input
    B, C, D, H, W = x_in.shape
    x_flat = x_in.permute(0,2,3,4,1).contiguous().reshape(B, D*H*W, C).contiguous()
    
    print(f"f10 {factor10}, i_reso {i_reso}, win_size {sWin}, depths {depths}, nHeads {nHeads}, skippable {skippable}, x_flat {x_flat.shape}")
      
    print("\n-------------- Encoder ------------")

    # # mask = AttentionMask3D(i_reso=(4, 9, 18), sWin=(2, 3, 3), shift_size=(1, 1, 1))
    # # print("i_reso=(4, 9, 18), sWin=(2, 3, 3), shift_size=(1, 1, 1)", mask.attn_mask.shape)
    # mask = AttentionMask3D(i_reso=(2, 9, 18), sWin=(2, 3, 3), shift_size=(0, 1, 1))    # shift == 0 if D == sWin[0]
    # print("i_reso=(2, 9, 18), sWin=(2, 3, 3), shift_size=(1, 1, 1)", mask.attn_mask.shape)

    # encoder = Encoder(i_reso = (D, H, W), embed_dim=192, depths=[4, 8, 4], num_heads=[4, 8, 16],
    encoder = Encoder(i_reso = (D, H, W), embed_dim=C, depths=depths, num_heads=nHeads,    #--------- num_heads to revise for deployment.
                sWin=sWin, mlp_ratio=4., qkv_bias=True,     #---------- sWin=(2, 6, 6) for deployment
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True,
                use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], skippable=skippable, skip_learnable=skip_learnable).to(device)

    # s = time.time()
    # encoder_output = encoder(x_flat)
    # e = time.time()    

    # print(f"-------------- output after encoder. time: {e-s}")
    # if skippable:
    #     for idx, (layer, out) in enumerate(zip(encoder.layers, encoder_output)):
    #         i_res = list(layer.i_reso)
    #         assert out[0].shape[1] == i_res[0] * i_res[1] * i_res[2]
    #         o_res = list(layer.o_reso)
    #         assert out[1].shape[1] == o_res[0] * o_res[1] * o_res[2]
    #         print(f"skippable, basicLayer: {idx}, before ds: {out[0].shape} / {i_res}; after_ds: {out[1].shape} / {o_res}")
    # else:
    #     out = encoder_output[-1]
    #     layer = encoder.layers[-1]
    #     i_res = list(layer.i_reso)
    #     assert out[0].shape[1] == i_res[0] * i_res[1] * i_res[2]
    #     o_res = list(layer.o_reso)
    #     assert out[1].shape[1] == o_res[0] * o_res[1] * o_res[2]
    #     print(f"un-skippable, basicLayer: before ds: {out[0].shape} / {i_res}; after_ds: {out[1].shape} / {o_res}")
    

    print("\n-------------- Decoder --------------")
    
    decoder = Decoder(i_reso = (D, H, W), embed_dim=C, depths=depths, num_heads=nHeads,    #--------- num_heads to revise for deployment.
            sWin=sWin, mlp_ratio=4., qkv_bias=True,     #---------- sWin=(2, 5, 5) for deployment
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
            norm_layer=nn.LayerNorm, ape=True,
            use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], skippable=skippable, skip_learnable=skip_learnable).to(device)
        
    # s = time.time()
    # x = decoder(encoder_output)
    # e = time.time()

    # print(f"-------------- output after decoder. time: {e-s}")
    
    # o_res = decoder.layers[0].o_reso
    # assert x.shape[1] ==  o_res[0] * o_res[1] * o_res[2]
    # print(f"output {x.shape}, <{o_res}")
    
    # for idx, layer in enumerate(decoder.layers):
    #     i_res = list(layer.i_reso)
    #     o_res = list(layer.o_reso)
    #     print(f"input_resulition {i_res}; o_reso {o_res}")

    print("\n-------------- Encoder-Decoder --------------")
    
    codec = Encoder_Decoder(i_reso = (D, H, W), embed_dim=C, depths=depths, num_heads=nHeads,    #--------- num_heads to revise for deployment.
            sWin=sWin, mlp_ratio=4., qkv_bias=True,     #---------- sWin=(2, 5, 5) for deployment
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
            norm_layer=nn.LayerNorm, ape=True,
            use_checkpoint=False, pretrained_sWins=[0, 0, 0, 0], skippable=skippable, skip_learnable=skip_learnable).to(device)

    # print(f"Encoder_Decoder. x_flat: {x_flat.shape}")

    # s = time.time()
    # x = codec(x_flat)
    # e = time.time()
    
    # print(f"-------------- output after encoder_decoder. x: {x.shape} time: {e-s}")
        
    #-----------------------------------------------------------------

    # # Reshape back to (B, C, D, H, W)
    # x_out = x_flat.view(B, D, H, W, C).permute(0,4,1,2,3).contiguous()
    # print(f"Output shape: {x_out.shape}, time: {(e-s):.3f} sec")
    # assert x_out.shape == x_in.shape

    # shift_size = (0,0,0)
    # PBLearner = PositionBiasLearner3D_Shpere(i_reso=(D, H, W), sWin=sWin,shift_size=shift_size, num_heads=num_heads, pair_bias_dim=pair_bias_dim, hidden_dim=pair_bias_dim*2)
    # pair_bias = PBLearner.pair_bias # (nWin * vWin * vWin, 4)
    
    # pair_bias = pair_bias.view(PBLearner.nWin, PBLearner.vWin, PBLearner.vWin, pair_bias_dim) # (nWin, indow_volume, vWin, 4)
    # pair_bias = pair_bias[:, 0, :, :]
    # pair_bias = pair_bias.view(D//sWin[0], H//sWin[1], W//sWin[2], sWin[0], sWin[1], sWin[2],  pair_bias_dim)
    # pair_bias = pair_bias.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
    # pair_bias = pair_bias.view(D, H, W, pair_bias_dim)[0] # (H, W, 4).
    # data = pair_bias[:, :, 2].numpy()    # cos_lat1, cos_lat2, great_circle, delta_depth
    
    # show_global_position_bias(PBLearner, pair_dim_idx=4, depth_idx=0)
    # pair_bias = torch.stack([cos_lat1 ** 2, cos_lat2 ** 2, great_circle, delta_depth, torch.abs(delta_depth)], dim=-1)

    # show_on_map(data)    
    
    print("----------------------------------- printing ... ----------------------------------")
    
    encoder = codec.encoder
    projection = 'HW'
    
    nLayers = len(encoder.layers)
    for layer_idx in range(nLayers):
        nBlocks = len(encoder.layers[layer_idx].blocks)
        for block_idx in range(nBlocks):
            block = encoder.layers[layer_idx].blocks[block_idx]
            attn_mask = block.attn_mask.attn_mask
            u_res = block.unpadded_resolution
            title = f"L {layer_idx}/{nLayers}, B {block_idx}/{nBlocks}, UnpaddedR {u_res}"
            show_global_attn_mask_map(attn_mask, block.sWin, block.i_reso, depth_idx=0, title=title, projection=projection)