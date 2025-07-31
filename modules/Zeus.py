import numpy as np
import torch
from torch import nn
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from timm.models.layers import to_2tuple

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):   # shape -> shape
        return self.net(x)
     
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):                   # (b, n, dim)  -> (b, n, dim)
        b, n, dim, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)                                   # (b, n, dim_head * heads * 3)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # (b, heads, n, dim_head *3)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale         # (b, heads, n, n)

        attn = dots.softmax(dim=-1)                                             # (b, heads, n, n)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)                    # (b, heads, n, dim_head * 3)
        out = rearrange(out, 'b h n d -> b n (h d)')                            # (b, n, dim_head * heads * 3)
        out = self.to_out(out)                                                  # (b, n, dim)
        return out
    
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')    # (1, n_channels, dim)

class GatedAttention(nn.Module):      # (b, n, dim) -> (b, n, dim)
    def __init__(self, dim, layers, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))

        self.apply(self._init_weights)  # recursive to the whole tree of children.

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)    # defaults: mean=0, std=0, a=-2, b=2
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)
    

class ZeusLayer(nn.Module):
    def __init__(self, dim, attLayers, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(ZeusLayer, self).__init__()

        self.ts_temporal_transformer = GatedAttention(dim, attLayers, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedAttention(dim, attLayers, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_space_transformer = GatedAttention(dim, attLayers, heads, dim_head, 
                                                mlp_dim, dropout, attn_dropout, drop_path)
        self.st_temporal_transformer = GatedAttention(dim, attLayers, heads, dim_head, 
                                                   mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):   # (B, T, N, dim) -> (B, T, N, dim)
        b, t, n, dim = x.shape
        x_ts, x_ori = x, x      # (B, T, N, dim)
        
        # temporal attention
        x_ts = rearrange(x_ts, 'b t n d -> (b n) t d')  # (B*N, T, dim)
        x_ts = self.ts_temporal_transformer(x_ts)       # (B*N, T, dim)
        
        # spatial attention
        x_ts = rearrange(x_ts, '(b n) t d -> (b t) n d', b=b)       # (B*T, N, dim)
        x_ts = self.ts_space_transformer(x_ts)                      # (B*T, N, dim)

        # spatial attention
        x_st = self.st_space_transformer(x_ts)                      # (B*T, N, dim)
        
        # tenporal attention
        x_st = rearrange(x_st, '(b t) n d -> (b n) t d', b=b)       # (B*N, T, dim)
        x_st = self.st_temporal_transformer(x_st)                   # (B*N, T, dim)

        # output
        x_st = rearrange(x_st, '(b n) t d -> b t n d', b=b)         # (B, T, N, dim)
        
        return x_st
    

class Zeus(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_height = config['height']                        # 720
        self.image_width = config['width']                          # 1,440
        self.nFramesIn = config['T1'] * config['FG']                # 2 * 24 = 48           # temporal seq_len
        self.nFramesOut = 1 * config['FG']                          # 24

        self.patch_size = config['patch_size']                      # 16
        self.nPatches_h = self.image_height // self.patch_size      # 720 / 16 = 45
        self.nPatches_w = self.image_width // self.patch_size       # 1,440 / 16 = 90
        self.nPatches = self.nPatches_h * self.nPatches_w           # 45 * 90 = 4,050       # spatial seq_len
        self.nSequences = self.nFramesIn * self.nPatches            # 48 * 4,050 = 194,400

        self.nChannels = config['channels']                         # 5
        self.patch_dim = self.nChannels * self.patch_size ** 2      # 16 * 16 * 5 = 1,280
        self.dim = self.patch_dim
        
        self.nClasses = self.nChannels
        self.nHeads = config['heads']
        self.dim_head = config['dim_head']
        self.dropout = config['dropout']
        self.attn_dropout = config['attn_dropout']
        self.drop_path = config['drop_path']
        self.scale_dim = config['scale_dim']
        
        self.attLayers = config['attLayers']
        self.zeusLayers = config['zeusLayers']
        
        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
            )
        
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.nSequences, self.dim),
                                               requires_grad=False).view(1, self.nFramesIn, self.nPatches, self.dim)

        self.blocks = nn.ModuleList([
            ZeusLayer(self.dim, self.attLayers, self.nHeads, self.dim_head, self.dim * self.scale_dim, self.dropout, self.attn_dropout, self.drop_path)
            for i in range(self.zeusLayers)
        ])

        self.reduce_frames = nn.Linear(self.nFramesIn, self.nFramesOut)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.nChannels * self.patch_size ** 2)  # just restore.
        )


    def forward(self, x):
        B, T1FG, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)  # (B, T1FG, H/p * W/p, self.patch_dim = p * p * C) -> (B, T1FG, H/p * W/p, dim)
        
        # Posion Embedding
        x += self.pos_embedding.to(x.device)    # pos_embedding: (1, T1FG * Hp * W/p, dim) -> (1, T1FG, H/p * W/p, dim) 
        
        # Add a mask?

        # PredFormer Encoder
        for blk in self.blocks:
            x = blk(x)          # (B, T1FG, H/p * W/p, dim)

        # Reduce frames
        x = rearrange(x, 'b t p d -> b p d t')              # (B, H/p * W/p, dim, T1FG)
        x = self.reduce_frames(x)                           # (B, H/p * W/p, dim, FG)
        x = rearrange(x, 'b p d t -> b t p d')              # (B, H/p * W/p, dim, FG)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T1FG, self.nPatches_h, self.nPatches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T1FG, C, H, W)

        return x

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    config = {
        # image h w c
        'height': 72,
        'width': 144,
        'channels': 2,
        # video length in and out
        'T1': 1,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 8,
        # transformer
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
    }

    model = Zeus(config)

    x = torch.rand(1, 24, 2, 72, 144)
    output = model(x)
    print(output.shape)  # [B, T, C, H, W]
    # # Calculate FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f'Number of flops: {flop_count_table(flops)}')

