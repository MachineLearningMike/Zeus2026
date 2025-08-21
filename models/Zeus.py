import numpy as np
import torch
from torch import nn
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from einops import rearrange

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

from timm.models.layers import to_2tuple

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.params import Params


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
     
class Attention(nn.Module):     # (b, n, dim)  -> (b, n, dim)
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
        
    def efficient_attention(self, q, k, scale, query_chunk_size=128):
        """
        q: (b, h, i, d)
        k: (b, h, j, d)
        scale: scalar to scale attention scores
        query_chunk_size: chunk size along query dimension
        """
        b, h, i, d = q.shape
        _, _, j, _ = k.shape

        dots_chunks = []

        for start in range(0, i, query_chunk_size):
            end = min(start + query_chunk_size, i)
            q_chunk = q[:, :, start:end, :]  # (b, h, chunk_size, d)

            # Compute attention scores head by head for this chunk
            dot_chunk = []
            for head in range(h):
                # einsum: (b, chunk_size, d) x (b, j, d) -> (b, chunk_size, j)
                dot_head = torch.einsum('bid,bjd->bij', q_chunk[:, head], k[:, head])
                dot_chunk.append(dot_head.unsqueeze(1))  # restore head dim

            # concat heads for this chunk: (b, h, chunk_size, j)
            dot_chunk = torch.cat(dot_chunk, dim=1)
            dots_chunks.append(dot_chunk)

        # concat all chunks along query dimension: (b, h, i, j)
        dots = torch.cat(dots_chunks, dim=2)

        # apply scaling
        dots = dots * scale
        return dots

    def forward(self, x):                   # (b, n, dim)  -> (b, n, dim)
        b, n, dim, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)                                   # (b, n, dim_head * heads) x 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # (b, heads, n, dim_head) x 3
        
        # This is the notorious attention memory bottleneck. Replace it!
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale         # (b, heads, n, n)
        # dots = self.efficient_attention(q, k, self.scale)     # !!! This doesn't mitigate memory peak !!!

        attn = dots.softmax(dim=-1)                                             # (b, heads, n, n)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)                    # (b, heads, n, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')                            # (b, n, dim_head * heads)
        out = self.to_out(out)                                                  # (b, n, dim)
        return out                                                              # (b, n, dim)
    
def sinusoidal_embedding_1d(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe # rearrange(pe, '... -> 1 ...')    # (1, n_channels, dim)

def sinusoidal_embedding_3d(sPos, dim):    # shape of positions, dim. -> (1,) + sPos + (dim,)
    assert len(sPos) == 3
    assert dim % 3 == 0
    pe0 = sinusoidal_embedding_1d(sPos[0], dim//3)[:, None, None, :].expand(-1, sPos[1], sPos[2], -1)
    pe1 = sinusoidal_embedding_1d(sPos[1], dim//3)[None, :, None, :].expand(sPos[0], -1, sPos[2], -1)
    pe2 = sinusoidal_embedding_1d(sPos[2], dim//3)[None, None, :, :].expand(sPos[0], sPos[1], -1, -1)
    pe = torch.cat((pe0, pe1, pe2), dim=-1)  # sPos + (dim,)
    return pe # (s0, s1, s2, d)

class GatedTemporalAttention(nn.Module):      # ((b n), t, dim) -> ((b n), t, dim) 
    def __init__(self, model_dict, dim, layers, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.model_dict = model_dict
        self.convolve = nn.ModuleList([
            Rearrange('(b n) t d -> (b d) t n', n=self.model_dict['nPatches_h']*self.model_dict['nPatches_w']),
            Rearrange('B t (h w) -> B t h w', h=self.nPatches_h),
            nn.Conv2d(model_dict['nFramesIn'], model_dict['nFramesIn'], 5, 1, 2),  # preserve shape
            Rearrange('(b d) t h w -> (b w h) t d', d=self.model_dict['dim'])
        ])
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
            
    def forward(self, x):   # x: ((b n), t, d)
        x = self.convolve(x)    # ((b n), t, d) -> ((b n), t, d)

        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class GatedAttention(nn.Module):      # (b, n, dim) -> (b, n, dim)
    def __init__(self, pm):
        super().__init__()

        dim = pm.dim
        layers = pm.attLayers
        nHeads = pm.nHeads
        dim_head = pm.dim_head
        mlp_dim = pm.mlp_dim
        dropout = pm.dropout
        attn_dropout = pm.attn_dropout
        drop_path = pm.drop_path

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])          

        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=nHeads, dim_head=dim_head, dropout=attn_dropout)),
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
    
class SpatialConvolution(nn.Module):
    def __init__(self, pm):
        self.pm = pm
        super(SpatialConvolution, self).__init__()
        self.convolve = nn.Conv2d(pm.nFramesIn, pm.nFramesIn, 3, 1, 1)  # preserve shape

    def forward(self, x):   # (B, T1FG, H/p * W/p, dim) -> preserve shape
        x = rearrange(x, 'b t n d -> (b d) t n')
        x = rearrange(x, 'B t (h w) -> B t h w', h=self.pm.nPatches_h)
        x = self.convolve(x)    # preserves shape
        x = rearrange(x, '(b d) t h w -> b t (h w) d', d=self.pm.dim)

        return x

class ZeusLayer(nn.Module):
    def __init__(self, pm):
        super(ZeusLayer, self).__init__()
        self.pm = pm

        self.spatial_colvolution_1 = SpatialConvolution(pm)
        self.temporal_transformer_1 = GatedAttention(pm)

        self.space_transformer_1 = GatedAttention(pm)
        self.space_transformer_2 = GatedAttention(pm)
        
        self.spacial_colvolution_2 = SpatialConvolution(pm)
        self.temporal_transformer_2 = GatedAttention(pm)

    def forward(self, x):   # (B, T, N, dim) -> (B, T, N, dim)      # (B, T1FG, H/p * W/p, dim)
        b, t, n, dim = x.shape                  # (B, T, N, dim), or (B, T1FG, H/p * W/p, dim)
        
        x = self.spatial_colvolution_1(x)   # (B, T, N, dim) -> (B, T, N, dim)
        x = rearrange(x, 'b t n d -> (b n) t d')  # (B*N, T, dim)
        x = self.temporal_transformer_1(x)       # (B*N, T, dim)
        
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b)       # (B*T, N, dim)
        x = self.space_transformer_1(x)                      # (B*T, N, dim)
        x = self.space_transformer_2(x)                      # (B*T, N, dim)
        
        x = rearrange(x, '(b t) n d -> b t n d', b=b)       # (B, T, N, dim)
        x = self.spacial_colvolution_2(x)                   # (B, T, N, dim)
        x = rearrange(x, 'b t n d -> (b n) t d')        # (B*N, T, dim)
        x = self.temporal_transformer_2(x)                   # (B*N, T, dim)

        # output
        x = rearrange(x, '(b n) t d -> b t n d', b=b)         # (B, T, N, dim)
        
        return x

class Zeus(nn.Module):
    def __init__(self, config, nDevices):
        super().__init__()
        self.pm = Params(config)
        self.nDevices = nDevices
        self.deployed_to_gpus = False

        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.dim),
            nn.Tanh(),
            nn.Linear(2 * self.pm.dim, self.pm.dim),
            )
        if self.nDevices > 1: self.to_patch_embedding = self.to_patch_embedding.to('cuda:0')
                    
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFramesId, self.pm.nPatches_h, self.pm.nPatches_w, self.pm.dim)
        pe = sinusoidal_embedding_3d((self.pm.nFramesIn, self.pm.nPatches_h, self.pm.nPatches_w), self.pm.dim)
        self.pos_embedding = nn.Parameter(rearrange(pe, 't h w dim -> 1 t (h w) dim'))
        if self.nDevices > 1: self.pos_embedding = nn.Parameter(self.pos_embedding.to('cuda:0'))

        #-------------- zeus layers -------------------------
        if self.nDevices > 1: 
            assert self.nDevices == self.pm.zeusLayers, "The numbers of devices and zeud layers must be the same if nDevices > 1"
        self.zeusLayers = nn.ModuleList([       # (B, T, N, dim) -> (B, T, N, dim)
            ZeusLayer(self.pm)
            for i in range(self.pm.zeusLayers)
        ])
        if self.nDevices > 1:
            for idx, layer in enumerate(self.zeusLayers):
                layer = layer.to('cuda:' + str(idx))

        # self.reduce_frames = nn.Sequential(
        #     Rearrange('b t p d -> b p d t'),
        #     nn.Linear(self.pm.nFramesIn, self.pm.nFramesOut),       # (B, H/p * W/p, dim, FG)
        #     Rearrange('b p d t -> b t p d')
        # )
        
        #--------------- reduce -----------------------------
        if self.pm.nFramesIn != self.pm.nFramesOut:
            self.reduce_frames = nn.Sequential(
                Rearrange('b t (h w) d -> (b d) t h w', h=self.pm.nPatches_h),
                nn.Conv2d(self.pm.nFramesIn, self.pm.nFramesOut, 3, 1, 1),       # (B, H/p * W/p, dim, 1FG)
                Rearrange('(b d) t h w -> b t (h w) d', d=self.pm.dim)
            )
            if self.nDevices > 1: self.reduce_frames = self.reduce_frames.to('cuda:' + str(self.nDevices-1))

        #--------------- ffn --------------------------------
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pm.dim),
            nn.Linear(self.pm.dim, self.pm.nChannels * self.pm.patch_size ** 2)  # just restore.
        )
        if self.nDevices > 1: self.mlp_head = self.mlp_head.to('cuda:' + str(self.nDevices-1))

        #---------------- flag it ---------------------------
        if self.nDevices > 1: self.deployed_to_gpus = True


    def forward(self, x):
        B, T1FG, C, H, W = x.shape
        
        # Patch Embedding
        x = self.to_patch_embedding(x)  # (B, T1FG, H/p * W/p, self.patch_dim = p * p * C) -> (B, T1FG, H/p * W/p, dim)   
        
        # Posion Embedding
        x += self.pos_embedding.to(x.device)    # pos_embedding: (1, T1FG * Hp * W/p, dim) -> (1, T1FG, H/p * W/p, dim) 
        
        # Add a mask?

        # Encoders
        for idx, layer in enumerate(self.zeusLayers):
            # !!! This line fails to mitigate memory peak !!!
            # x = checkpoint(layer.forward, x, use_reentrant=False)          # (B, T1FG, H/p * W/p, dim)
            # x = layer(x)          # (B, T1FG, H/p * W/p, dim)
            if idx > 0:
                x = x.to('cuda:' + str(min(idx, self.nDevices-1)))
                x = layer(x)
 
        # Reduce frames
        if self.pm.nFramesIn != self.pm.nFramesOut:
            x = self.reduce_frames(x)   # (B, FG, H/p * W/p, dim)
        
        # MLP head        
        x = self.mlp_head(x.reshape(-1, self.pm.dim))
        x = x.view(B, self.pm.nFramesOut, self.pm.nPatches_h, self.pm.nPatches_w, C, self.pm.patch_size, self.pm.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.pm.nFramesOut, C, H, W)

        return x    # (B,FG,C,H,W)

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    from models.config import config

    model = Zeus(config)

    x = torch.rand(1, config['T1'] * config['FG'], len(config['variables']), config['height'], config['width'])
    print("input", x.shape)  # [B, T, C, H, W]
    output = model(x)
    print("output", output.shape)  # [B, T, C, H, W]
    # # Calculate FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f'Number of flops: {flop_count_table(flops)}')

