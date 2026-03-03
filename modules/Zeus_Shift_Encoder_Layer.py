import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.util_snapshots_auto import save, ctx

from modules.Zeus_Flash_Encoder_Layer import FlashEncoderLayer

# SwiGLU (unchanged)
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)
        return swiglu

# ---------------------------
# MLP (unchanged)
# ---------------------------
class MLP(nn.Module):
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

class StableLayerNorm(nn.Module):
    """LayerNorm with numerical stability improvements for mixed precision"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # Use double precision for variance calculation
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x_float - mean) / torch.sqrt(var + self.eps)
        x_normalized = x_normalized.to(x.dtype)
        return x_normalized * self.weight + self.bias

class PositionwiseFeedForward(nn.Module):
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
        return x


# ---------------------------
# Window partition / reverse (2D version)
# ---------------------------
def window_partition_2d(x, sWin):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    win_H, win_W = sWin
    assert H % win_H == 0 and W % win_W == 0, f"Input dims {(H,W)} must be divisible with sWin {sWin}"

    nWh = H // win_H
    nWw = W // win_W

    x = x.view(B, C, nWh, win_H, nWw, win_W)
    # (B, nWh, nWw, C, win_H, win_W) -> (-1, C, win_H, win_W)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, win_H, win_W)    # (total_nWin, C, win_H, win_W)
    return windows, nWh, nWw

def window_reverse_2d(windows, sWin, H, W):
    # windows: (total_nWin, C, win_H, win_W)
    win_H, win_W = sWin
    nWh = H // win_H
    nWw = W // win_W
    B = int(windows.shape[0] // (nWh * nWw))
    C = windows.shape[1]

    x = windows.view(B, nWh, nWw, C, win_H, win_W)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)   # (B, C, H, W)
    return x


class AttentionMask2D(nn.Module):
    attn_mask_shared = {}

    def __init__(self, i_reso, sWin, shift_size, padding=None):
        super().__init__()
        self.i_reso = i_reso
        self.sWin = sWin
        assert all(s >= 0 for s in shift_size)
        self.shift_size = shift_size

        padding_key = () if padding is None else (0, 0)  # FIXED: Use tuple instead of None
        key = (*i_reso, *sWin, *shift_size, *padding_key)
        if key not in AttentionMask2D.attn_mask_shared:
            attn_mask = self.compute_mask(i_reso, sWin, shift_size, padding)
            AttentionMask2D.attn_mask_shared[key] = attn_mask

        self.register_buffer("attn_mask", AttentionMask2D.attn_mask_shared[key], persistent=False)

    @staticmethod
    def compute_mask(i_reso, sWin, shift_size, padding):
        H, W = i_reso
        if all(s == 0 for s in shift_size) and padding is None:
            return None

        def get_slices(s_size, w_size, pad):
            if s_size != 0 and pad == 0:
                slices = (slice(0, -w_size),
                          slice(-w_size, -s_size),
                          slice(-s_size, None))
            elif s_size == 0 and pad != 0:
                slices = (slice(0, pad),
                          slice(pad, None))
            elif s_size != 0 and pad != 0:
                if pad <= s_size:
                    slices = (slice(0, -w_size),
                              slice(-w_size, -s_size),
                              slice(-s_size, -pad),
                              slice(-pad, None))
                else:
                    slices = (slice(0, pad - s_size),
                              slice(pad - s_size, -w_size),
                              slice(-w_size, -s_size),
                              slice(-s_size, None))
            elif s_size == 0 and pad == 0:
                slices = (slice(None),)
            return slices
      
        # H dimension
        idx = 0
        pad = padding[idx] if padding is not None else 0
        w_size = sWin[idx]
        s_size = shift_size[idx]
        assert pad < w_size
        h_slices = get_slices(s_size, w_size, pad)

        # W dimension (with cylindrical handling)
        idx = 1
        pad = padding[idx] if padding is not None else 0
        w_size = sWin[idx]
        s_size = shift_size[idx]
        assert pad < w_size

        if s_size != 0 and pad == 0:
            w_slices = (slice(None),)
        elif s_size == 0 and pad != 0:
            w_slices = (slice(0, pad),
                        slice(pad, None))
        elif s_size != 0 and pad != 0:
            if pad <= s_size:
                w_slices = (slice(0, -pad),
                            slice(-pad, None))
            else:
                w_slices = (slice(0, pad - s_size),
                            slice(pad - s_size, -s_size),
                            slice(-s_size, None))
        else:
            w_slices = (slice(None),)        
        
        # idx = 1
        # pad = padding[idx] if padding is not None else 0
        # w_size = sWin[idx]
        # s_size = shift_size[idx]
        # assert pad < w_size
        # w_slices = get_slices(s_size, w_size, pad)

        # Build mask
        img_mask = torch.zeros((1, 1, H, W))  # torch.float32
        
        region_id = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = region_id
                region_id += 1

        # Partition windows
        # img_mask: (1, 1, H, W)
        windows, nWh, nWw = window_partition_2d(img_mask, sWin) # (total_nWin, 1, sWin_h, sWin_w)
        mask_windows = windows.view(-1, sWin[0]*sWin[1])    # (total_nWin, vWin)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # (total_nWin, 1, vWin) - (total_nWin, vWin, 1) = (total_nWin, vWin, vWin)
        unique = region_id  # unique must meet abs(unique) > 4
        # Standards attention kills attentoins for zero-masked pairs.
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 1).masked_fill(attn_mask != 0, 0)
        return attn_mask[:, None, :, :]    # (total_nWin, propagation to num_heads, vWin, vWin) with values 0 or 1.

# ---------------------------
# Swin Block 2D WITHOUT autocast inside
# ---------------------------
class ShiftEncoderLayer(nn.Module):
    def __init__(self, H, W, d_model, num_heads, nWin, shift=False, act_layer=nn.Sigmoid, skip_learn=False):
        
        super().__init__()
        
        self.H, self.W = H, W

        assert d_model % num_heads == 0, f"d_model {d_model} is not divisible with num_heads {num_heads}"

        # We have techniques to asure this, eliminating padding.
        assert H % nWin[0] == 0 and W % nWin[1] == 0, f"Total size ({H}, {W}) is not divisible with number of windows {nWin}"
        assert H // nWin[0] % 2 == 0 and W // nWin[1] % 2 == 0, f"Window size ({H // nWin[0]}, {W // nWin[1]}) are not even numbers."

        self.d_model = d_model
        self.num_heads = num_heads
        
        self.sWin = (H // nWin[0], W // nWin[1])
        
        self.input_vol = H * W
        self.vWin = self.sWin[0] * self.sWin[1]
        self.total_nWin = nWin[0] * nWin[1]

        # self.shift = shift
        self.sShift = (self.sWin[0]//2, self.sWin[1]//2) if shift else (0, 0)

        self.enc_layer = FlashEncoderLayer(d_model, num_heads, dropout=0., use_flash=True, act_layer=act_layer, eps=1e-9, skip_learn=skip_learn)
        
        # Remove position bias related code
        self.type_pos_bias = None
        self.positionBiasLearner = None
        self.learnable_pair_types = None
        
        # mask_padding = [p + ip for (p, ip) in zip(padding if padding is not None else (0,0), inherited_padding if inherited_padding is not None else (0,0))]
        # mask_padding = None if sum(mask_padding) == 0 else mask_padding     
        self.attn_mask = AttentionMask2D((H, W), self.sWin, self.sShift, padding=None)
            
    def forward(self, x, pos_bias_er=None, ss=None):        
        B, H, W, C = x.shape
        assert H == self.H and W == self.W, f"Input feature has wrong size ({H}, {W}), expected ({self.H}, {self.W})"

        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        if any(s != 0 for s in self.sShift):
            x = torch.roll(x, shifts=[ - s for s in self.sShift], dims=(2,3))

        # partition windows
        x_windows, nWh, nWw = window_partition_2d(x, self.sWin)     # (total_nWin, C, win_H, win_W)
        x_windows = x_windows.view(-1, C, self.vWin).permute(0,2,1).contiguous()    # (total_nWin, self.vWin, C) = (batch, seq_len, dim)

        mask=self.attn_mask.attn_mask if hasattr(self.attn_mask, 'attn_mask') else None

        attn_windows = self.enc_layer(x_windows, mask=mask, pos_bias_er=pos_bias_er, ss=ss)    # (total_nWin, self.vWin, C) = (batch, seq_len, dim)

        # reverse windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.sWin) # (total_nWin, C, win_H, win_W)
        x = window_reverse_2d(attn_windows, self.sWin, H, W)    # (B, C, H, W)

        # reverse cyclic shift
        if any(s != 0 for s in self.sShift):
            x = torch.roll(x, shifts=[s for s in self.sShift], dims=(2,3))

        x = x.permute(0,2,3,1).contiguous()     # (B, H, W, C)
                
        return x
