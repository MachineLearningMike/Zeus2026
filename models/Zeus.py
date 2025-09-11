import numpy as np
import torch
from torch import nn
import gc
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
    
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe    # (n_channels, dim)
    
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
    
class PatchEmbedWithPos(nn.Module):
    def __init__(self, to_patch_embedding, pos_embedding):
        super().__init__()
        # Assign with the name you want in state_dict
        self.to_patch_embedding = to_patch_embedding  # nn.Linear
        self.pos_embedding = nn.Parameter(pos_embedding)    # trainable

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding
        return x
    
class PatchReconstruct(nn.Module):
    # x = x.view(B, self.pm.nFramesOut, self.pm.nPatches_h, self.pm.nPatches_w, C, self.pm.patch_size, self.pm.patch_size)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.pm.nFramesOut, C, H, W)
    def __init__(self, nFramesOut, nPatches_h, nPatches_w, patch_size, C, H, W):
        super().__init__()
        self.nFramesOut = nFramesOut
        self.nPatches_h = nPatches_h
        self.nPatches_w = nPatches_w
        self.patch_size = patch_size
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        B = x.size(0)
        # Step 1: view into patch blocks
        x = x.view(B, self.nFramesOut, self.nPatches_h, self.nPatches_w, self.C, self.patch_size, self.patch_size)
        # Step 2: permute and merge patches back into H x W
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.nFramesOut, self.C, self.H, self.W)
        return x

class Zeus(nn.Module):
    def __init__(self, config, nDevices, device_type='cuda', base_device=0, layers_home='device', compute_place='data_find_layer'):
    # layers_home: gpu, cpu
    # layers on gpu
    # layers on cpu
        super().__init__()
        self.pm = Params(config)
        self.nDevices = nDevices
        self.device_type = device_type
        self.base_device = base_device
        self.layers_home = layers_home
        self.compute_place = compute_place

        def dstring(device_type, id):
            return device_type + ':' + str(id)

        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.dim),
            nn.Tanh(),
            nn.Linear(2 * self.pm.dim, self.pm.dim),
            )
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFramesId, self.pm.nPatches_h, self.pm.nPatches_w, self.pm.dim)
        pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
        
        self.patch_module = PatchEmbedWithPos(to_patch_embedding, pos_embedding)

        if self.layers_home == 'device':
            self.patch_module = self.patch_module.to(dstring(self.device_type, self.base_device))  # from the defalut devoce to the 1st device.
        elif self.layers_home == 'cpu': pass

        #-------------- zeus layers -------------------------
        if self.nDevices > 1: 
            assert self.nDevices >= self.pm.zeusLayers, "The numbers of devices and zeud layers must be the same if nDevices > 1"
        self.zeusLayers = nn.ModuleList([       # (B, T, N, dim) -> (B, T, N, dim)
            ZeusLayer(self.pm)
            for i in range(self.pm.zeusLayers)
        ])
        if self.layers_home == 'device':
            for idx, layer in enumerate(self.zeusLayers):
                layer = layer.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, idx)))  # to the i-th device
        elif self.layers_home == 'cpu': pass
            
        # self.reduce_frames = nn.Sequential(
        #     Rearrange('b t p d -> b p d t'),
        #     nn.Linear(self.pm.nFramesIn, self.pm.nFramesOut),       # (B, H/p * W/p, dim, FG)
        #     Rearrange('b p d t -> b t p d')
        # )
        
        assert self.pm.nFramesIn == self.pm.nFramesOut
        # #--------------- reduce -----------------------------
        # if self.pm.nFramesIn != self.pm.nFramesOut:
        #     self.reduce_frames = nn.Sequential(
        #         Rearrange('b t (h w) d -> (b d) t h w', h=self.pm.nPatches_h),
        #         nn.Conv2d(self.pm.nFramesIn, self.pm.nFramesOut, 3, 1, 1),       # (B, H/p * W/p, dim, 1FG)
        #         Rearrange('(b d) t h w -> b t (h w) d', d=self.pm.dim)
        #     )
        #     if self.layers_home == 'device':
        #         self.reduce_frames = self.reduce_frames.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, self.pm.zeusLayers-1))) # to the last device

        #--------------- ffn --------------------------------
        reconstruct = PatchReconstruct(
            nFramesOut=self.pm.nFramesOut, nPatches_h=self.pm.nPatches_h, nPatches_w=self.pm.nPatches_w, patch_size=self.pm.patch_size,
            C=self.pm.nChannels, H=self.pm.image_height, W=self.pm.image_width
        )       
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.pm.dim),
            nn.Linear(self.pm.dim, self.pm.nChannels * self.pm.patch_size ** 2),
            reconstruct
        )
        if self.layers_home == 'device':
            self.mlp_head = self.mlp_head.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, self.pm.zeusLayers-1)))    # to the last device
        elif self.layers_home == 'cpu': pass
        
    def get_modules(self):
        return [self.patch_module] + list(self.zeusLayers) + [self.mlp_head]

    def forward(self, x):
        # compute_place: data_find_layer, all_find_device_0
        # data_find_layer: move data to layers' device
        # all_find_device_0: move layers and data to cuda:0
        B, T1FG, C, H, W = x.shape
        
        if self.compute_place == 'all_find_device_0':
            x = x.to(self.device_type + ':0')
            
        if self.compute_place == 'data_find_layer':
            device = next(self.patch_module.parameters()).device
            x = x.to(device, non_blocking=True)
            x = self.patch_module(x)  # (B, T1FG, H/p * W/p, self.patch_dim = p * p * C) -> (B, T1FG, H/p * W/p, dim)
            torch.cuda.empty_cache()    # free the space where x was.
        elif self.compute_place == 'all_find_device_0':
            layer = self.patch_module.to(x.device)  # layer is not a copy.
            x = layer(x)
            layer.to('cpu:0')
            torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs
        else: pass

        for idx, layer in enumerate(self.zeusLayers):
            # !!! This line fails to mitigate memory peak !!!
            # x = checkpoint(layer.forward, x, use_reentrant=False)          # (B, T1FG, H/p * W/p, dim)
            # x = layer(x)          # (B, T1FG, H/p * W/p, dim)
            if self.compute_place == 'data_find_layer':
                device = next(layer.parameters()).device
                x = x.to(device, non_blocking=True)
                x = layer(x)
                torch.cuda.empty_cache()    # free the space where x was.
            elif self.compute_place == 'all_find_device_0':
                layer.to(x.device)
                x = layer(x)
                layer.to('cpu:0')
                torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs
            else: pass
 
        # # Reduce frames
        # if self.pm.nFramesIn != self.pm.nFramesOut:
        #     if self.compute_place == 'data_find_layer':
        #         device = next(self.reduce_frames.parameters()).device
        #         x = x.to(device)
        #         x = self.reduce_frames(x)   # (B, FG, H/p * W/p, dim)
        #         torch.cuda.empty_cache()    # free the space where x was.
        #     elif self.compute_place == 'all_find_device_0':
        #         layer = self.reduce_frames.to(inference_device)
        #         x = layer(x)
        #         layer.to('cpu:0')
        #         torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs
                
        # MLP head
        if self.compute_place == 'data_find_layer':
            device = next(self.mlp_head.parameters()).device
            x = x.to(device, non_blocking=True)
            x = self.mlp_head(x.reshape(-1, self.pm.dim))
            torch.cuda.empty_cache()    # free the space where x was.
        elif self.compute_place == 'all_find_device_0':
            layer = self.mlp_head.to(x.device)
            x = x.reshape(-1, self.pm.dim)
            x = layer(x)
            layer.to('cpu:0')
            torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs
        else: pass

        return x    # (B,FG,C,H,W)
    
class DoubleZeus(nn.Module):
    def __init__(self, config, multi_gpu, nDevices, device_type='cuda', trainable=True, freeze_1st_half=True):
        super().__init__()
        self.pm = Params(config)
        self.multi_gpu = multi_gpu
        self.nDevices = nDevices
        self.device_type = device_type
        self.trainable = trainable
        self.freeze_1st_half = freeze_1st_half
        
        if self.multi_gpu:
            if self.device_type == 'cuda' and self.nDevices >= 2:
                # goal: deployed on [gpus + gpus if trainable else cpu]
                self.zeus1 = Zeus(config, nDevices=nDevices//2, device_type=device_type, base_device=0, layers_home='device', compute_place='data_find_layer')
                self.zeus2 = Zeus(config, nDevices=nDevices//2, device_type=device_type, base_device=nDevices//2, layers_home='device', compute_place='data_find_layer')
            else:
                raise Exception("DoubleZeus cannot be created in multi_gpu mode with less than 2 GPUs")
        else:   #------------------------------------------------------- works on 1-GPU Windows and 1-GPU VM ---------------------
            if self.device_type == 'cuda' and self.nDevices == 1:
                # goal: deployed on [cpu], for sinlge-gpu offload parallel inference and train
                self.zeus1 = Zeus(config, nDevices=1, device_type=device_type, base_device=0, layers_home='cpu', compute_place='all_find_device_0')
                self.zeus2 = Zeus(config, nDevices=1, device_type=device_type, base_device=0, layers_home='cpu', compute_place='all_find_device_0')      
            else:
                raise Exception("DoubleZeus cannot be created withoug GPU")    
        
        self.modules = self.get_modules()
        
        if freeze_1st_half:
            self.zeus1.eval()
            for param in self.zeus1.parameters():
                param.requires_grad = False
                
    def get_modules(self):  # on either (distributed) GPU(s) on CPU(s)
        return self.zeus1.get_modules() + self.zeus2.get_modules()
    
    def forward(self, x):
        x = self.zeus1(x)
        x = self.zeus2(x)
        return x
    
    def single_gpu_inference(self, x):        
        x = x.to("cuda:0")  # input on GPU

        for i, module in enumerate(self.modules):
            module.to("cuda:0")
            with torch.set_grad_enabled(False):
                x = module(x)
                module.to("cpu:0")       # move module back to CPU
                torch.cuda.empty_cache()
        return x.to("cpu:0")
    
    def single_gpu_forth_and_back(self, data, target, loss_fn, denormal_fn, batch_diff, era5_data_vars):
        nFirstFrozenModules = len(self.modules) // 2 if self.pm.freeze_1st_half else 0
        
        # data, target = data.to("cpu"), target.to("cpu")
        acts_cpu = []
        x = data.to("cuda:0")  # input on GPU
        
        # ---------------- Forward Pass ----------------
        for i, module in enumerate(self.modules):
            module.to("cuda:0")

            if i < len(self.modules) - 1:
                with torch.no_grad():
                    x = module(x)       # intermediate module, no grad
                x = x.detach()          # safe, disconnect from graph
                acts_cpu.append(x.cpu())  # save for recomputation
                module.to("cpu")
                torch.cuda.empty_cache()
            else:
                # Last module
                final_input = x
                final_module = module
                final_input.requires_grad_(True)   # leaf tensor
                final_out = final_module(final_input)    # compute output with grad

        # ---------------- Backward Pass ----------------
        # 1) Last module and loss
        target = target.to('cuda:0')
        final_out = denormal_fn(final_out)
        target = denormal_fn(target)
        lossAndMetrics = loss_fn(final_out, target, batch_diff, era5_data_vars)     
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        loss.backward()                         # grads for last module & final_input
        final_module.to("cpu")                  # Safe place to move back.
        grad_next = final_input.grad.detach().cpu()

        # 2) Recompute previous modules in reverse
        for i in reversed(range(len(self.modules)-1)):
            if i < nFirstFrozenModules: continue

            module = self.modules[i]
            module.to("cuda:0")

            prev_act = acts_cpu[i-1].to("cuda:0") if i > 0 else data.to("cuda:0")
            prev_act.requires_grad_(True)

            with torch.set_grad_enabled(True):
                out = module(prev_act)
                if grad_next is None:
                    grad_next = torch.ones_like(out)
                out.backward(grad_next.to(out.device))

            grad_next = prev_act.grad.detach().cpu()
            del out, prev_act
            torch.cuda.empty_cache()
            module.to("cpu")
            
        return lossAndMetrics

    
    def load_state_dict(self, state_dict, strict=True):
        # Detect if state_dict belongs to a single Zeus model
        sample_keys = list(state_dict.keys())
        if sample_keys and all(not k.startswith("zeus2") for k in sample_keys):
            # assume it's a single Zeus, load into zeus1
            return self.zeus1.load_state_dict(state_dict, strict=strict)
        else:
            # default behavior: load as usual
            return super().load_state_dict(state_dict, strict=strict)
        

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


