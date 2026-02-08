import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import copy

# Try to import FlashAttention-2
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("FlashAttention-2 not installed. Using standard attention.")

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.DeltaOperator import DeltaOperatorMultiChannel

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
# Window partition / reverse (2D version)
# ---------------------------
def window_partition_2d(x, sWin):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    win_H, win_W = sWin
    assert H % win_H == 0 and W % win_W == 0, \
        f"Input dims {(H,W)} must be multiples of sWin {sWin}"

    nWh = H // win_H
    nWw = W // win_W

    x = x.view(B, C, nWh, win_H, nWw, win_W)
    # (B, nWh, nWw, C, win_H, win_W) -> (-1, C, win_H, win_W)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, win_H, win_W)
    return windows, nWh, nWw

def window_reverse_2d(windows, sWin, H, W):
    win_H, win_W = sWin
    nWh = H // win_H
    nWw = W // win_W
    B = int(windows.shape[0] // (nWh * nWw))
    C = windows.shape[1]

    x = windows.view(B, nWh, nWw, C, win_H, win_W)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x

# ---------------------------
# FlashAttention-based WindowAttention2D WITHOUT autocast inside
# ---------------------------
class WindowAttention2D(nn.Module):
    def __init__(self, dim, i_reso, sWin, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.i_reso = i_reso
        self.sWin = sWin
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Use FlashAttention-2 if available
        self.use_flash_attention = HAS_FLASH_ATTN
        if self.use_flash_attention:
            pass
            # print("Using FlashAttention-2 for WindowAttention2D")
        else:
            # print("FlashAttention-2 not available, using standard attention")
            pass
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, position_bias=None, mask=None):
        B_, N, C = x.shape  # B * nWin, win_volume, C
        
        # Prepare qkv with bias if needed
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        
        if self.use_flash_attention:
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            
            dropout_p = self.attn_drop if self.training else 0.0
            
            if mask is None:
                attn_output = flash_attn_func(q, k, v, dropout_p, softmax_scale=self.scale, causal=False)
            else:
                nW = mask.shape[0]
                batch_per_window = B_ // nW
                
                # Initialize output
                attn_output = torch.zeros_like(q)
                
                # Process each window separately
                for w in range(nW):
                    start_idx = w * batch_per_window
                    end_idx = (w + 1) * batch_per_window
                    
                    q_window = q[start_idx:end_idx]  # (batch_per_window, N, heads, dim)
                    k_window = k[start_idx:end_idx]
                    v_window = v[start_idx:end_idx]
                    
                    # Get mask for this window
                    window_mask = mask[w]  # (N, N) with -100/0
                    
                    # Find connected components in this window's mask
                    # Tokens i and j are connected if window_mask[i,j] == 0 (can attend)
                    adjacency = (window_mask == 0).float().cpu().numpy()
                    
                    # Find connected components using graph algorithm
                    import numpy as np
                    from scipy import sparse
                    from scipy.sparse.csgraph import connected_components
                    
                    # Convert to sparse matrix for efficiency
                    sparse_adj = sparse.csr_matrix(adjacency)
                    
                    # Find connected components
                    n_components, labels = connected_components(
                        csgraph=sparse_adj, 
                        directed=False, 
                        return_labels=True
                    )
                    
                    # Process each region in this window
                    for region_id in range(n_components):
                        # Get indices of tokens in this region
                        region_indices = np.where(labels == region_id)[0]
                        
                        if len(region_indices) == 0:
                            continue
                        
                        region_indices_tensor = torch.tensor(region_indices, device=q.device)
                        
                        # Extract region tokens
                        q_region = q_window[:, region_indices_tensor, :, :]
                        k_region = k_window[:, region_indices_tensor, :, :]
                        v_region = v_window[:, region_indices_tensor, :, :]
                        
                        # Compute attention for this region
                        region_output = flash_attn_func(
                            q_region, k_region, v_region,
                            dropout_p=dropout_p,
                            softmax_scale=self.scale,
                            causal=False
                        )
                        
                        # Place back in output
                        attn_output[start_idx:end_idx, region_indices_tensor, :, :] = region_output
            
            attn_output = attn_output.reshape(B_, N, C)
            
        else:
            # Standard attention fallback
            qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # Apply attention mask if provided
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            
            attn = self.softmax(attn)
            attn = F.dropout(attn, p=self.attn_drop) if self.training else attn
            attn_output = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Projection
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x

    
    def flash_attn_func_equivalent(self, q, k, v, dropout_p=0.0, causal=False):
        # q, k, v shapes: (batch, seq, heads, dim)
        
        # Transpose to (batch, heads, seq, dim) for attention computation
        q_t = q.transpose(1, 2)  # (batch, heads, seq, dim)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if causal:
            seq_len = q_t.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q_t.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v_t)
        
        # Transpose back to (batch, seq, heads, dim)
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output
        
    def flash_attn_func_equivalent_with_mask(self, q, k, v, dropout_p=0.0, mask=None):
        # q, k, v shapes: (B_, N, num_heads, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
                
        # Apply mask BEFORE softmax (this is the correct way!)
        if mask is not None:
            # mask has values: -100 (block) and 0 (allow)
            # Add mask to attn_scores (mask is already -100 for blocked)
            
            nW = mask.shape[0]
            B_, num_heads, N, _ = attn_scores.shape
            
            # Add mask WITHOUT creating new tensor
            # Reshape attn_scores and add mask in-place
            attn_scores = attn_scores.view(B_ // nW, nW, num_heads, N, N)
            
            # Expand mask to the right shape
            mask_expanded = mask.unsqueeze(0).unsqueeze(2)  # (1, nW, 1, N, N)
            
            # Use add_ for in-place addition (no new tensor)
            attn_scores.add_(mask_expanded)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
    def flash_attn_func_equivalent_with_mask_trick(self, q, k, v, dropout_p=0.0, mask=None):
        # q, k, v shapes: (B_, N, num_heads, head_dim)
        
        # Always transpose to (B_, num_heads, N, head_dim) for matmul
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
        
        # Softmax first (no mask yet)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply Swin mask AFTER softmax
        if mask is not None:
            # mask has values: -100 (no attention) and 0 (allow attention)
            # After softmax, we need to zero out the -100 positions
            
            nW = mask.shape[0]
            B_, num_heads, N, _ = attn_weights.shape
            
            # Reshape attn_weights
            attn_weights = attn_weights.view(B_ // nW, nW, num_heads, N, N)
            
            # Create a boolean mask from the -100 values
            # mask == 0 means "allow attention", mask == -100 means "block attention"
            attention_mask = (mask != -100).float()  # 1.0 for allow, 0.0 for block
            
            # Apply the mask by multiplying (zero out blocked attention)
            attn_weights = attn_weights * attention_mask.unsqueeze(0).unsqueeze(2)
            
            # Renormalize (optional but good for stability)
            # attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            attn_weights = attn_weights.view(-1, num_heads, N, N)
        
        # Apply dropout
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v_t)
        
        # Transpose back
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output
        
    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops

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

        # Build mask
        img_mask = torch.zeros((1, 1, H, W))
        
        region_id = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = region_id
                region_id += 1

        # Partition windows
        windows, nWh, nWw = window_partition_2d(img_mask, sWin)
        mask_windows = windows.view(-1, sWin[0]*sWin[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

# ---------------------------
# Swin Block 2D WITHOUT autocast inside
# ---------------------------
class Zeus_SwinTransformerBlock2D(nn.Module):
    def __init__(self, dim, i_reso, num_heads, sWin, shift_size, pair_bias_dim=None,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, type_pos_bias=None, padding=None, inherited_padding=None,
                 is_padding_block=False, in_encoder=True):
        
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} is not divisible with num_heads {num_heads}"

        self.dim = dim
        self.i_reso = i_reso
        self.input_vol = i_reso[0] * i_reso[1]
        self.num_heads = num_heads
        self.sWin = sWin
        self.vWin = sWin[0] * sWin[1]
        self.nWin = (i_reso[0] // sWin[0]) * (i_reso[1] // sWin[1])
        assert all(s >= 0 for s in shift_size)
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        # Remove position_bias from WindowAttention2D initialization
        self.attn = WindowAttention2D(dim, i_reso, sWin, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Remove position bias related code
        self.type_pos_bias = None
        self.positionBiasLearner = None
        self.learnable_pair_types = None
        
        mask_padding = [p + ip for (p, ip) in zip(padding if padding is not None else (0,0), inherited_padding if inherited_padding is not None else (0,0))]
        mask_padding = None if sum(mask_padding) == 0 else mask_padding       
        self.attn_mask = AttentionMask2D(i_reso, sWin, shift_size, padding=mask_padding)
            
        self.unpadded_resolution = i_reso    
        if padding is not None:
            self.unpadded_resolution = [i_reso[dim] - padding[dim] for dim in range(2)]
        self.unpadded_vol = self.unpadded_resolution[0] * self.unpadded_resolution[1]
                
        self.padding_config = (0,0, padding[1],0, padding[0],0, 0,0) if padding is not None else None 
        self.pad_sum = 0 if padding is None else sum(self.padding_config) if self.padding_config is not None else 0
            
        self.is_padding_block = is_padding_block
        self.in_encoder = in_encoder

    def forward(self, x):
        if self.in_encoder and self.is_padding_block and self.pad_sum > 0:
            [H, W] = self.unpadded_resolution
            B, L, dim = x.shape
            x = x.view(B, H, W, dim)
            x = F.pad(x, self.padding_config, mode='constant')
            x = x.view(B, -1, dim)
        
        H, W = self.i_reso
        B, L, C = x.shape
        assert L == H*W, f"Input feature has wrong size {L}, expected {H*W}"

        shortcut = x
        
        # LayerNorm will be handled by autocast - no manual conversion needed
        x_norm = self.norm1(x)
        
        x = x_norm.transpose(1, 2).reshape(B, C, H, W)

        if any(s != 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=[ - s for s in self.shift_size], dims=(2,3))
        else:
            shifted_x = x

        # partition windows
        x_windows, nWh, nWw = window_partition_2d(shifted_x, self.sWin)
        x_windows = x_windows.view(-1, C, self.vWin).permute(0,2,1).contiguous()

        # No position bias computation - FlashAttention doesn't support it
        attn_windows = self.attn(x_windows, position_bias=None, mask=self.attn_mask.attn_mask if hasattr(self.attn_mask, 'attn_mask') else None)

        # reverse windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(-1, C, *self.sWin)
        x = window_reverse_2d(attn_windows, self.sWin, H, W)

        # reverse cyclic shift
        if any(s != 0 for s in self.shift_size):
            x = torch.roll(x, shifts=[s for s in self.shift_size], dims=(2,3))

        # flatten back
        x = x.permute(0,2,3,1).contiguous().reshape(B, H*W, C).contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x_norm2 = self.norm2(x)
        x = x + self.drop_path(self.mlp(x_norm2))
        
        if not self.in_encoder and self.is_padding_block and self.pad_sum > 0:
            [H, W] = self.i_reso
            B, L, dim = x.shape
            x = x.view(B, H, W, dim)
            padding_config = [-c for c in self.padding_config]
            x = F.pad(x, tuple(padding_config), mode='constant')
            x = x.view(B, -1, dim)
        
        return x
    
    def flops(self):
        flops = 0
        H, W = self.i_reso
        flops += self.dim * H * W
        nW = H * W / self.vWin
        flops += nW * self.attn.flops(self.vWin)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

# ---------------------------
# PatchMerging2D (unchanged)
# ---------------------------
class PatchMerging2D(nn.Module):
    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.i_reso
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) must be even"

        x = x.view(B, H, W, C)

        x00 = x[:, 0::2, 0::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x11 = x[:, 1::2, 1::2, :]

        x = torch.cat([x00, x10, x01, x11], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x_norm = self.norm(x)  # No manual conversion needed
        return x_norm

    def flops(self):
        H, W = self.i_reso
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops

# ---------------------------
# PatchUnmerging2D (updated for LayerNorm)
# ---------------------------
class PatchUnmerging2D(nn.Module):
    def __init__(self, i_reso, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        assert dim % 2 == 0, "dim must be even since C_out = C//2"

        self.out_dim = dim // 2
        self.expansion = nn.Linear(dim, 4 * self.out_dim, bias=False)
        self.norm = norm_layer(4 * self.out_dim)

    def forward(self, x):
        H, W = self.i_reso
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        assert C == self.dim

        x = self.expansion(x)
        x_norm = self.norm(x)  # No manual conversion needed
        x = x_norm.view(B, H, W, 4 * self.out_dim)

        C_out = self.out_dim
        x00 = x[..., 0*C_out:1*C_out]
        x10 = x[..., 1*C_out:2*C_out]
        x01 = x[..., 2*C_out:3*C_out]
        x11 = x[..., 3*C_out:4*C_out]

        out = torch.zeros(
            B, 2*H, 2*W, C_out,
            device=x.device,
            dtype=x.dtype
        )

        out[:, 0::2, 0::2, :] = x00
        out[:, 1::2, 0::2, :] = x10
        out[:, 0::2, 1::2, :] = x01
        out[:, 1::2, 1::2, :] = x11

        out = out.view(B, (2*H)*(2*W), C_out)
        return out

# ---------------------------
# BasicLayer (2D version) - Updated to remove position_bias_dim
# ---------------------------
class BasicLayer2D(nn.Module):
    def __init__(self, dim, i_reso, depth, num_heads, sWin, pair_bias_dim=None,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, resampler=PatchMerging2D, use_checkpoint=False,
                 pretrained_sWin=0, type_pos_bias=None, padding=None, inherited_padding=None, in_encoder=True):

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
                self.o_reso = (i_reso[0]//2, i_reso[1]//2)
            else:
                self.o_reso = (i_reso[0]*2, i_reso[1]*2)

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size=(0,0) if (i % 2 == 0) else \
            (
                sWin[0] // 2 if i_reso[0] > sWin[0] else 0,
                sWin[1] // 2 if i_reso[1] > sWin[1] else 0,
            )
        
            block = Zeus_SwinTransformerBlock2D(dim=dim, i_reso=i_reso,
                                num_heads=num_heads, sWin=sWin,
                                shift_size=shift_size,
                                pair_bias_dim=pair_bias_dim,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,
                                type_pos_bias=type_pos_bias, padding=padding, inherited_padding=inherited_padding,
                                is_padding_block=(i==0),
                                in_encoder=in_encoder,
                                )
            self.blocks.append(block)

        # patch merging/unmerging layer
        if resampler is not None:
            incoming_resolution = i_reso if in_encoder else self.blocks[0].unpadded_resolution
            self.resampler = resampler(incoming_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.resampler = None

    def forward(self, x):
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

# ---------------------------
# Encoder (2D version) - Updated to remove pair_bias_dim
# ---------------------------
class Encoder2D(nn.Module):
    def __init__(self, i_reso = (180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True, use_checkpoint=False, 
                pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=None, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.mlp_ratio = mlp_ratio
        self.skippable = skippable

        num_patches = i_reso[0] * i_reso[1]

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        prev_resolution = (i_reso[0] * 2, i_reso[1] * 2)
        inherited_padding = [0,0]
        
        for i_layer in range(self.num_layers):
            for dim in range(2):
                assert prev_resolution[dim] % 2 == 0, f"Input resolution will mismatch"

            resolution=[prev_resolution[0] // 2,
                        prev_resolution[1] // 2]

            padding = []
            for i in range(2):
                pad = 0 if resolution[i] % sWin[i] == 0 else sWin[i] - resolution[i] % sWin[i]
                if i_layer != self.num_layers - 1: 
                    assert pad % 2 == 0
                if pad != 0:
                    print(f"!!!!!!!!!! Encoder layer {i_layer}: input resolution[{i}] {resolution[i]} is increased by {pad} to be a multiple of window size {sWin[i]}")
                resolution[i] += pad
                padding.append(pad)
            
            donwsample = PatchMerging2D if (i_layer < self.num_layers - 1) else None
            if donwsample is not None:
                for i in range(2):
                    assert resolution[i] % 2 == 0, f"Encoder: BasicLayer {i_layer} has an odd input resolution {resolution} while having PatchMerging."
            
            layer = BasicLayer2D(dim=int(embed_dim * 2 ** i_layer),
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
                               pair_bias_dim=pair_bias_dim, padding=padding, inherited_padding=inherited_padding,
                               in_encoder=True)
            inherited_padding = [p//2 for p in padding]
            self.layers.append(layer)            
            prev_resolution = resolution

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

        return encoder_output

# ---------------------------
# Decoder (2D version) - Updated to remove pair_bias_dim
# ---------------------------
class Decoder2D(nn.Module):
    def __init__(self, i_reso = (180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True, use_checkpoint=False,
                pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=None, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.skip_learnable = skip_learnable
        self.skippable = skippable

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        if skip_learnable:
            self.skip_learners = nn.ModuleList()
        prev_resolution = (i_reso[0] * 2, i_reso[1] * 2)
        inherited_padding = [0,0]
        
        for i_layer in range(self.num_layers):
            for dim in range(2):
                assert prev_resolution[dim] % 2 == 0, f"Input resolution will mismatch"

            resolution=[prev_resolution[0] // 2,
                        prev_resolution[1] // 2]
            dim = int(embed_dim * 2 ** i_layer)

            padding = []
            for i in range(2):
                pad = 0 if resolution[i] % sWin[i] == 0 else sWin[i] - resolution[i] % sWin[i]
                if i_layer != self.num_layers - 1: 
                    assert pad % 2 == 0
                if pad != 0:
                    print(f"!!!!!!!!!! Decoder layer {i_layer}: input resolution[{i}] {resolution[i]} is increased by {pad} to be a multiple of window size {sWin[i]}")
                resolution[i] += pad
                padding.append(pad)
            
            layer = BasicLayer2D(dim=dim,
                               i_reso=resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               sWin=sWin,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               resampler=PatchUnmerging2D if i_layer > 0 else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_sWin=pretrained_sWins[i_layer],
                               pair_bias_dim=pair_bias_dim, padding=padding, inherited_padding=inherited_padding,
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, enc_x_by_layer):        
        if self.skippable:
            dec_x_after_us = torch.zeros_like(enc_x_by_layer[-1][0])
            
            if self.skip_learnable:            
                for idx, (dec_layer, (enc_x_before_ds, _), skip_learner) in enumerate(zip(reversed(self.layers), reversed(enc_x_by_layer), reversed(self.skip_learners))):
                    input = dec_x_after_us + skip_learner(enc_x_before_ds) 
                    (_, dec_x_after_us) = dec_layer(input)
            else:
                for idx, (dec_layer, (enc_x_before_ds, _)) in enumerate(zip(reversed(self.layers), reversed(enc_x_by_layer))):
                    input = dec_x_after_us + enc_x_before_ds 
                    (_, dec_x_after_us) = dec_layer(input)
        else:
            dec_x_after_us = enc_x_by_layer[-1][0]
            for layer in reversed(self.layers):
                (_, dec_x_after_us) = layer(dec_x_after_us)
            
        return dec_x_after_us

# ---------------------------
# Encoder-Decoder (2D version) - Updated to remove pair_bias_dim
# ---------------------------
class Encoder_Decoder2D(nn.Module):
    def __init__(self, i_reso = (180, 360), embed_dim=192, depths=[4, 10, 2], num_heads=[4, 8, 16],
                sWin=(6, 12), mlp_ratio=4., qkv_bias=True,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                norm_layer=nn.LayerNorm, ape=True, use_checkpoint=False,
                pretrained_sWins=[0, 0, 0, 0], pair_bias_dim=None, skippable=False, skip_learnable=False, **kwargs):
        super().__init__()

        self.encoder = Encoder2D(i_reso = i_reso, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                sWin=sWin, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, ape=True,
                use_checkpoint=use_checkpoint, pretrained_sWins=pretrained_sWins, pair_bias_dim=pair_bias_dim, 
                skippable=skippable, skip_learnable=skip_learnable, **kwargs)
        
        self.decoder = Decoder2D(i_reso = i_reso, embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                sWin=sWin, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=nn.LayerNorm, ape=True,
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
            x_skip = x_flat
        x_by_layer = self.encoder(x_flat)
        x = self.decoder(x_by_layer)
        if self.skippable:
            if self.skip_learnable:
                x = x + self.skip_learner(x_skip)
            else:
                x = x + x_skip
        return x

# ---------------------------
# Visualization functions (2D version)
# ---------------------------
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def show_global_attn_mask_map(attn_mask, sWin, i_reso, title=""):
    if attn_mask is None:
        print("No attention mask (no shift).")
        return

    H, W = i_reso
    win_H, win_W = sWin
    nWh = H // win_H
    nWw = W // win_W
    vWin = win_H * win_W

    global_mask = torch.zeros((H, W), device=attn_mask.device)

    idx = 0
    for h in range(nWh):
        for w in range(nWw):
            mask_win = attn_mask[idx]
            mask_win_4d = mask_win.view(win_H, win_W, win_H, win_W)
            mask_slice = mask_win_4d.sum(dim=(2,3))
            global_mask[h*win_H:(h+1)*win_H, w*win_W:(w+1)*win_W] = mask_slice
            idx += 1

    mask_2d = global_mask.cpu().numpy()
    
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
    plt.title(f"Mask: R {i_reso}, WS {sWin}, " + title, fontsize=9)
    plt.show()

# ---------------------------
# Main test (2D version) with autocast support
# ---------------------------
import time

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test configuration for 2D weather data
    factor10, sWin = 2.5, (60, 60)  # Window size for 2D       # f=2.5 : (180, 360)
    depths, nHeads = (4,), (4,) # (4, 8, 4), (4, 8, 16)
    patch_size = 1
    H, W = int(int(720/patch_size)/10*factor10), int(int(1440/patch_size)/10*factor10)
    C = 12
    i_reso = (H, W)

    skippable = True
    skip_learnable = True

    print(f"2D Weather Model Test with FlashAttention-2")
    print(f"FlashAttention-2 available: {HAS_FLASH_ATTN}")
    print(f"Resolution: {i_reso}, Window size: {sWin}")
    print(f"Depths: {depths}, Heads: {nHeads}")
    
    # Create 2D encoder-decoder
    codec = Encoder_Decoder2D(
        i_reso=i_reso,
        embed_dim=C,
        depths=depths,
        num_heads=nHeads,
        sWin=sWin,
        mlp_ratio=4.,
        qkv_bias=True,
        skippable=skippable,
        skip_learnable=skip_learnable
    ).to(device)
    
    # Test 1: BF16 with autocast
    print("\n-------------- Testing with BF16 input (autocast) --------------")
    x_in_bf16 = torch.randn(1, C, H, W).to(device).to(torch.bfloat16)
    x_flat_bf16 = x_in_bf16.permute(0,2,3,1).contiguous().reshape(1, H*W, C).contiguous()
    
    print(f"Input shape: {x_flat_bf16.shape}")
    print(f"Input dtype: {x_flat_bf16.dtype}")
    
    # Get device type from tensor
    device_type = x_flat_bf16.device.type
    
    # Warmup
    print("Warming up (3 iterations)...")
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for _ in range(3):
            _ = codec(x_flat_bf16)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    loops = 10
    print(f"Benchmarking {loops} iterations...")
    
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(loops):
                output_bf16 = codec(x_flat_bf16)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
    else:
        # CPU fallback
        import time
        start_time = time.perf_counter()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            for _ in range(loops):
                output_bf16 = codec(x_flat_bf16)
        elapsed_time = time.perf_counter() - start_time
    
    avg_time = elapsed_time / loops
    print(f"BF16 forward pass:")
    print(f"  Total time for {loops} iterations: {elapsed_time:.3f} seconds")
    print(f"  Average time per iteration: {avg_time:.3f} seconds")
    print(f"  Throughput: {loops/elapsed_time:.2f} iterations/second")
    
    print(f"Output shape: {output_bf16.shape}")
    print(f"Output dtype: {output_bf16.dtype}")
    
    # Test 2: FP32 with autocast
    print("\n-------------- Testing with FP32 input (autocast) --------------")
    x_in_fp32 = torch.randn(1, C, H, W).to(device).to(torch.float32)
    x_flat_fp32 = x_in_fp32.permute(0,2,3,1).contiguous().reshape(1, H*W, C).contiguous()
    
    print(f"Input shape: {x_flat_fp32.shape}")
    print(f"Input dtype: {x_flat_fp32.dtype}")
    
    # Get device type from tensor
    device_type = x_flat_fp32.device.type
    
    # Warmup
    print("Warming up (3 iterations)...")
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        for _ in range(3):
            _ = codec(x_flat_fp32)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    loops = 10
    print(f"Benchmarking {loops} iterations...")
    
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(loops):
                output_fp32 = codec(x_flat_fp32)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
    else:
        # CPU fallback
        import time
        start_time = time.perf_counter()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            for _ in range(loops):
                output_fp32 = codec(x_flat_fp32)
        elapsed_time = time.perf_counter() - start_time
    
    avg_time = elapsed_time / loops
    print(f"FP32 forward pass (autocast BF16):")
    print(f"  Total time for {loops} iterations: {elapsed_time:.3f} seconds")
    print(f"  Average time per iteration: {avg_time:.3f} seconds")
    print(f"  Throughput: {loops/elapsed_time:.2f} iterations/second")
    
    print(f"Output shape: {output_fp32.shape}")
    print(f"Output dtype: {output_fp32.dtype}")
    
    # Test 3: Mixed precision with autocast (FP16)
    print("\n-------------- Testing with mixed precision (autocast FP16) --------------")
    x_in_mixed = torch.randn(1, C, H, W).to(device).to(torch.float32)
    x_flat_mixed = x_in_mixed.permute(0,2,3,1).contiguous().reshape(1, H*W, C).contiguous()
    
    print(f"Input shape: {x_flat_mixed.shape}")
    print(f"Input dtype: {x_flat_mixed.dtype}")
    
    # Get device type from tensor
    device_type = x_flat_mixed.device.type
    
    # Warmup
    print("Warming up (3 iterations)...")
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        for _ in range(3):
            _ = codec(x_flat_mixed)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    loops = 10
    print(f"Benchmarking {loops} iterations...")
    
    if device_type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for _ in range(loops):
                output_mixed = codec(x_flat_mixed)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
    else:
        # CPU fallback
        import time
        start_time = time.perf_counter()
        with torch.autocast(device_type='cpu', dtype=torch.float16):
            for _ in range(loops):
                output_mixed = codec(x_flat_mixed)
        elapsed_time = time.perf_counter() - start_time
    
    avg_time = elapsed_time / loops
    print(f"Mixed precision forward pass (autocast FP16):")
    print(f"  Total time for {loops} iterations: {elapsed_time:.3f} seconds")
    print(f"  Average time per iteration: {avg_time:.3f} seconds")
    print(f"  Throughput: {loops/elapsed_time:.2f} iterations/second")
    
    print(f"Output shape: {output_mixed.shape}")
    print(f"Output dtype: {output_mixed.dtype}")
    
    # Test that output matches input dimensions
    assert output_bf16.shape == x_flat_bf16.shape, f"BF16 Output shape {output_bf16.shape} doesn't match input shape {x_flat_bf16.shape}"
    assert output_fp32.shape == x_flat_fp32.shape, f"FP32 Output shape {output_fp32.shape} doesn't match input shape {x_flat_fp32.shape}"
    assert output_mixed.shape == x_flat_mixed.shape, f"Mixed Output shape {output_mixed.shape} doesn't match input shape {x_flat_mixed.shape}"
    print("\n✅ All tests passed!")
    
    # ------------------ Visualization test ------------------
    print("\n----------------------------------- printing attention masks ... ----------------------------------")
    
    # encoder = codec.encoder
    
    # nLayers = len(encoder.layers)
    # for layer_idx in range(nLayers):
    #     nBlocks = len(encoder.layers[layer_idx].blocks)
    #     for block_idx in range(nBlocks):
    #         block = encoder.layers[layer_idx].blocks[block_idx]
    #         attn_mask = block.attn_mask.attn_mask
    #         u_res = block.unpadded_resolution
    #         title = f"L {layer_idx}/{nLayers}, B {block_idx}/{nBlocks}, UnpaddedR {u_res}"
    #         show_global_attn_mask_map(attn_mask, block.sWin, block.i_reso, title=title)