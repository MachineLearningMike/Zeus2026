import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------
# Utilities
# -------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def make_stochastic_depth_probs(depth, drop_rate=0.):
    return [x.item() for x in torch.linspace(0, drop_rate, depth)]

# -------------------------
# Patch embedding
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, D
        x = self.norm(x)
        return x, (H, W)

# -------------------------
# Window utils
# -------------------------
def window_partition(x, sWin: int):
    B, H, W, C = x.shape
    x = x.view(B, H // sWin, sWin, W // sWin, sWin, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, sWin * sWin, C)
    return windows

def window_reverse(windows, sWin: int, H: int, W: int):
    B = int(windows.shape[0] / (H // sWin * W // sWin))
    x = windows.view(B, H // sWin, W // sWin, sWin, sWin, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

# -------------------------
# Window Attention
# -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, sWin, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.sWin = sWin if isinstance(sWin, tuple) else (sWin, sWin)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias
        Wh, Ww = self.sWin
        num_relative_positions = (2 * Wh - 1) * (2 * Ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, num_heads))

        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w), dim=0)
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# -------------------------
# MoE
# -------------------------
class MoE(nn.Module):
    def __init__(self, dim, hidden_dim, n_experts=4, capacity_factor=1.25, drop=0.):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(dim, n_experts)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        logits = self.gate(x)
        top1 = logits.argmax(dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (top1 == e)
            if mask.any():
                out[mask] = self.experts[e](x[mask])
        return self.drop(out)

# -------------------------
# FeedForward / MLP
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0., use_moe=False, moe_params=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)
        self.use_moe = use_moe
        if use_moe:
            moe_params = moe_params or {}
            hidden_dim_moe = moe_params.pop('hidden_dim', hidden_dim)
            self.moe = MoE(dim, hidden_dim_moe, **moe_params)
        else:
            self.fc1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_dim, dim)
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_moe:
            B, N, C = x.shape
            y = self.moe(x.reshape(-1, C))
            return y.view(B, N, C)
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

# -------------------------
# XiFuBlock
# -------------------------
class XiFuBlock(nn.Module):
    def __init__(self, dim, i_reso: Tuple[int,int], num_heads, sWin=7,
                 shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 use_moe=False, moe_params=None, layer_scale_init_value=1e-5):
        super().__init__()
        self.dim = dim
        self.i_reso = i_reso
        self.num_heads = num_heads
        self.sWin = sWin
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, sWin=sWin, num_heads=num_heads,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim*mlp_ratio), drop=drop, use_moe=use_moe, moe_params=moe_params)

        if layer_scale_init_value > 0:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)))
        else:
            self.gamma1 = self.gamma2 = None

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        H, W = self.i_reso
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.sWin)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, self.sWin, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
        else:
            x = shifted_x

        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(self.gamma1*x if self.gamma1 is not None else x)

        shortcut2 = x
        x = self.norm2(x)
        y = self.mlp(x)
        x = shortcut2 + self.drop_path(self.gamma2*y if self.gamma2 is not None else y)
        return x

# -------------------------
# Patch Merging
# -------------------------
class PatchMerging(nn.Module):
    def __init__(self, i_reso: Tuple[int,int], dim):
        super().__init__()
        self.i_reso = i_reso
        self.dim = dim
        self.norm = nn.LayerNorm(4*dim)
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        H, W = self.i_reso
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0,x1,x2,x3], dim=-1)
        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# -------------------------
# XiFu2 backbone
# -------------------------
class XiFu2(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=4, in_chans=3,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 sWin=7, mlp_ratio=4., drop_rate=0., attn_drop=0.,
                 drop_path_rate=0.1, use_moe_stages=(), moe_params=None):
        super().__init__()
        if isinstance(img_size,int):
            img_size=(img_size,img_size)
        self.img_size = img_size

        self.patch_embed = PatchEmbed(in_ch=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        H, W = img_size[0]//patch_size, img_size[1]//patch_size
        self.pos_drop = nn.Dropout(drop_rate)

        self.stages = nn.ModuleList()
        total_blocks = sum(depths)
        dpr = make_stochastic_depth_probs(total_blocks, drop_path_rate)
        block_idx = 0
        dims = [embed_dim*(2**i) for i in range(len(depths))]
        cur_resolution = (H, W)
        cur_dim = embed_dim

        for stage_idx, (depth, num_head, dim) in enumerate(zip(depths, num_heads, dims)):
            blocks = []
            for i in range(depth):
                shift_size = 0 if (i%2==0) else sWin//2
                use_moe = stage_idx in use_moe_stages
                blocks.append(XiFuBlock(dim=cur_dim,
                                        i_reso=cur_resolution,
                                        num_heads=num_head,
                                        sWin=sWin,
                                        shift_size=shift_size,
                                        mlp_ratio=mlp_ratio,
                                        drop=drop_rate,
                                        attn_drop=attn_drop,
                                        drop_path=dpr[block_idx],
                                        use_moe=use_moe,
                                        moe_params=moe_params))
                block_idx+=1
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx < len(depths)-1:
                down = PatchMerging(cur_resolution, cur_dim)
                self.stages.append(down)
                cur_resolution = (cur_resolution[0]//2, cur_resolution[1]//2)
                cur_dim = cur_dim*2

        self.norm = nn.LayerNorm(cur_dim)
        self.head = nn.Linear(cur_dim,1000)

    def forward(self, x):
        B = x.shape[0]
        x, (H, W) = self.patch_embed(x)
        x = self.pos_drop(x)
        cur_H, cur_W = H, W

        for module in self.stages:
            if isinstance(module, PatchMerging):
                x = module(x)
                cur_H, cur_W = cur_H//2, cur_W//2
            else:
                for blk in module:
                    if blk.shift_size>0:
                        Hp = int(math.ceil(cur_H/blk.sWin)*blk.sWin)
                        Wp = int(math.ceil(cur_W/blk.sWin)*blk.sWin)
                        img_mask = torch.zeros((1,Hp,Wp), device=x.device, dtype=torch.int32)
                        h_slices = (slice(0,-blk.sWin),
                                    slice(-blk.sWin,-blk.shift_size),
                                    slice(-blk.shift_size,None))
                        w_slices = (slice(0,-blk.sWin),
                                    slice(-blk.sWin,-blk.shift_size),
                                    slice(-blk.shift_size,None))
                        cnt = 0
                        for h in h_slices:
                            for w in w_slices:
                                img_mask[:,h,w]=cnt
                                cnt+=1
                        mask_windows = window_partition(img_mask.unsqueeze(-1), blk.sWin)
                        mask_windows = mask_windows.view(-1, blk.sWin*blk.sWin)
                        attn_mask = mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)
                        attn_mask = attn_mask.float() * -100.0
                    else:
                        attn_mask=None
                    x = blk(x, attn_mask=attn_mask)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x



# -------------------------
# Quick test / usage
# -------------------------
def _test():
    model = XiFu2(img_size=224,
                  patch_size=4,
                  embed_dim=64,
                  depths=(2, 2, 6, 2),
                  num_heads=(2, 4, 8, 16),
                  sWin=7,
                  use_moe_stages=(2,),  # enable MoE in stage 2
                  moe_params=dict(n_experts=4, hidden_dim=256))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    _test()
