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
            skip=False
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
        if skip:
            nn.init.constant_(self.fc2.weight, 0.0)
            nn.init.constant_(self.fc2.bias, 0.0)    
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., skip=False):
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
        
        if skip:
            if project_out:
                nn.init.constant_(self.to_out[0].weight, 0.0)
                nn.init.constant_(self.to_out[0].bias, 0.0)
            else:
                nn.init.constant_(self.to_out.weight, 0.0)
                nn.init.constant_(self.to_out.bias, 0.0)

    def forward(self, x):                   # (b, n, dim)  -> (b, n, dim)
        # return zeros initially
        b, n, dim, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)                                   # (b, n, dim_head * heads) x 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) # (b, heads, n, dim_head) x 3
        
        # This is the notorious attention memory bottleneck. Replace it!
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale         # (b, heads, n, n)

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
    def __init__(self, pm, skip):
        super().__init__()
        self.skip = skip

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
                PreNorm(dim, Attention(dim, heads=nHeads, dim_head=dim_head, dropout=attn_dropout, skip=self.skip)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout, skip=self.skip)),
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
        # return intact x initially
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))     # x + zeros (initially)
            x = x + drop_path2(ff(x))       # x + zeors (initially)
        return self.norm(x)
    
class SpatialConvolution(nn.Module):
    def __init__(self, pm, skip=False):
        self.pm = pm
        super(SpatialConvolution, self).__init__()
        self.convolve = nn.Conv2d(pm.nFramesIn, pm.nFramesIn, 3, 1, 1)  # preserve shape
        if skip:
            nn.init.constant_(self.convolve.weight, 0.0)
            nn.init.constant_(self.convolve.bias, 0.0)

    def forward(self, x):   # (B, T1FG, H/p * W/p, dim) -> preserve shape
        # return zero initially
        x = rearrange(x, 'b t n d -> (b d) t n')
        x = rearrange(x, 'B t (h w) -> B t h w', h=self.pm.nPatches_h)
        x = self.convolve(x)    # preserves shape
        x = rearrange(x, '(b d) t h w -> b t (h w) d', d=self.pm.dim)

        return x

class ZeusLayer(nn.Module):
    def __init__(self, pm, skip=False):
        super(ZeusLayer, self).__init__()
        self.pm = pm
        self.skip = skip
        
        self.only_transformer = GatedAttention(pm, skip=self.skip)

        # # self.spatial_colvolution_1 = SpatialConvolution(pm, skip=self.skip)  # zero initially
        # self.temporal_transformer_1 = GatedAttention(pm, skip=self.skip)     # identity initially

        # self.space_transformer_1 = GatedAttention(pm, skip=self.skip)         # identity initially
        # self.space_transformer_2 = GatedAttention(pm, skip=self.skip)         # identity initially
        
        # # self.spacial_colvolution_2 = SpatialConvolution(pm, skip=self.skip)  # zero initially
        # self.temporal_transformer_2 = GatedAttention(pm, skip=self.skip)      # identity initially

    def forward(self, x):   # (B, T, N, dim) -> (B, T, N, dim)      # (B, T1FG, H/p * W/p, dim)
        # returns x initially
        if self.skip: x_skip = x
        
        t = x.shape[1]
        x = rearrange(x, 'b t n d -> b (t n) d')
        x = self.only_transformer(x)
        x = rearrange(x, 'b (t n) d -> b t n d', t=t)
        
        # # x = self.spatial_colvolution_1(x)   # (B, T, N, dim) -> (B, T, N, dim)
        # x = rearrange(x, 'b t n d -> (b n) t d')  # (B*N, T, dim)
        # x = self.temporal_transformer_1(x)       # (B*N, T, dim)
        
        # x = rearrange(x, '(b n) t d -> (b t) n d', b=b)       # (B*T, N, dim)
        # x = self.space_transformer_1(x)                      # (B*T, N, dim)
        # x = self.space_transformer_2(x)                      # (B*T, N, dim)
        
        # x = rearrange(x, '(b t) n d -> b t n d', b=b)       # (B, T, N, dim)
        # # x = self.spacial_colvolution_2(x)                   # (B, T, N, dim)
        # x = rearrange(x, 'b t n d -> (b n) t d')        # (B*N, T, dim)
        # x = self.temporal_transformer_2(x)                   # (B*N, T, dim)

        # # output
        # x = rearrange(x, '(b n) t d -> b t n d', b=b)         # (B, T, N, dim)
        
        if self.skip: x = x + x_skip
        
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

class ReshapeForMLP(nn.Module):
    # x = x.view(B, self.pm.nFramesOut, self.pm.nPatches_h, self.pm.nPatches_w, C, self.pm.patch_size, self.pm.patch_size)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.pm.nFramesOut, C, H, W)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # b t (h w) (p1 p2 c)
        x = x.reshape(-1, self.dim)     # ((b t h w) (p1 p2 c))
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
        B = x.numel() // (self.nFramesOut * self.nPatches_h * self.nPatches_w * self.C * self.patch_size * self.patch_size)
        # Step 1: view into patch blocks
        # ((b t h w) (p1 p2 c))
        x = x.view(B, self.nFramesOut, self.nPatches_h, self.nPatches_w, self.C, self.patch_size, self.patch_size)
        # Step 2: permute and merge patches back into H x W
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.nFramesOut, self.C, self.H, self.W)
        return x

class Zeus(nn.Module):
    def __init__(self, config, nDevices, device_type='cuda', base_device=0, abundant_GPUs=True):
        super().__init__()
        self.pm = Params(config)
        self.nDevices = nDevices
        self.device_type = device_type
        self.base_device = base_device
        self.abundant_GPUs = abundant_GPUs

        def dstring(device_type, id):
            return device_type + ':' + str(id)

        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        to_patch_embedding = nn.Sequential(             # Mike!!! (p1, p2, c) is wrong. should be (c, p1, p2). No touch, as we have learned parameters.
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.dim),
            nn.Tanh(),
            nn.Linear(2 * self.pm.dim, self.pm.dim),
            )
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFramesId * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
        
        self.patch_module = PatchEmbedWithPos(to_patch_embedding, pos_embedding)

        if self.abundant_GPUs:
            self.patch_module = self.patch_module.to(dstring(self.device_type, self.base_device))  # from the defalut devoce to the 1st device.
        elif not self.abundant_GPUs: self.patch_module

        #-------------- zeus layers -------------------------
        if self.nDevices > 1: 
            assert self.nDevices >= self.pm.zeusLayers, "The numbers of devices and zeud layers must be the same if nDevices > 1"
        self.zeusLayers = nn.ModuleList([       # (B, T, N, dim) -> (B, T, N, dim)
            ZeusLayer(self.pm)
            for i in range(self.pm.zeusLayers)
        ])
        if self.abundant_GPUs:
            for idx, layer in enumerate(self.zeusLayers):
                layer = layer.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, idx)))  # to the i-th device
        elif not self.abundant_GPUs: pass

        #--------------- ffn --------------------------------
        reshapeForMlp = ReshapeForMLP(self.pm.dim)
        reconstruct = PatchReconstruct(
            nFramesOut=self.pm.nFramesOut, nPatches_h=self.pm.nPatches_h, nPatches_w=self.pm.nPatches_w, patch_size=self.pm.patch_size,
            C=self.pm.nChannels, H=self.pm.image_height, W=self.pm.image_width
        )       
        
        self.mlp_head = nn.Sequential(
            reshapeForMlp,
            nn.LayerNorm(self.pm.dim),
            nn.Linear(self.pm.dim, self.pm.nChannels * self.pm.patch_size ** 2),
            reconstruct
        )
        if self.abundant_GPUs:
            self.mlp_head = self.mlp_head.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, self.pm.zeusLayers-1)))    # to the last device
        elif not self.abundant_GPUs: pass
        
        self.modules = self.get_modules()
        
    def get_modules(self):
        return [self.patch_module] + list(self.zeusLayers) + [self.mlp_head]

    def forward(self, x):
        # compute_place: data_find_layer, all_find_device_0
        # data_find_layer: move data to layers' device
        # all_find_device_0: move layers and data to cuda:0
        
        if not self.abundant_GPUs:
            device = self.device_type + ':0'
            x = x.to(device)
        
        for module in self.modules:                 #========================= both training and inference
            if self.abundant_GPUs:
                device = next(module.parameters()).device
                x = x.to(device, non_blocking=True)
                x = module(x)  # (B, T1FG, H/p * W/p, self.patch_dim = p * p * C) -> (B, T1FG, H/p * W/p, dim)
                # backwarding is possible, because module is on the device it was forwarded on.
                torch.cuda.empty_cache()    # free the space where x was.
            elif not self.abundant_GPUs:       #========================== inference only
                module.to(x.device)  # layer is not a copy.
                x = module(x)
                module.to('cpu:0')
                # backwarding is NOT possible, because module NOT is on the device it was forwarded on.
                torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs 

        return x    # (B,FG,C,H,W)
    
    def sigle_GPU_forward_for_gradient(self, data, target, loss_fn, denormal_fn, batch_diff, era5_data_vars):
        acts_on_cpu = False
                
        if acts_on_cpu:
            acts = [data.cpu()]     # module[-1].act, or act[-1]
        else:
            acts = [data.to('cuda:0')]
            
        x = data.to("cuda:0")
        
        self.modules = self.get_modules()
                
        # ---------------- Forward Pass ----------------
        for i, module in enumerate(self.modules):
            module.to("cuda:0")

            if i < len(self.modules) - 1:      # 0, 1, 2, ..., n-2: all except the last module n-1
                with torch.no_grad():
                    x = module(x)       # 0, 1, ..., n-2
                if i < len(self.modules) - 2:      # 0, 1, ..., n-3
                    if acts_on_cpu:
                        acts.append(x.cpu())  # 0, 1, ..., n-3
                    else:
                        acts.append(x)  # 0, 1, ..., n-3
                module.to("cpu")
                # torch.cuda.empty_cache()
            else:                               # 3 = n-1
                act_n_2 = x                  # act[n-2]. no save as it's used now
                module_n_1 = module
                act_n_2.requires_grad_(True)   # act[n-2] as a leaf tensor
                act_n_2.retain_grad()
                act_n_1 = module_n_1(act_n_2)    # act[n-1], no save as it's used now

        # ---------------- Backward Pass ----------------
        # 1) Last module and loss
        # target = target.to('cuda:0')
        target = denormal_fn(target)
        pred = denormal_fn(act_n_1)
        lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars)     
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        loss.backward()             # grads for last_input (i.e. act[n-2]) -> last_module -> act[n-1] -> loss_fn -> loss.

        # metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else None
        # if metrics is not None: metrics = metrics.detach()
        # loss = loss.detach()
        
        grad_next = act_n_2.grad    # No need to move to cpu, if memory is enough.
        module_n_1.to("cpu")        # After backward(), safe place to move back. 
        
        del act_n_2, act_n_1, target, pred
        torch.cuda.empty_cache()

        # 2) Recompute previous modules in reverse. n-1 is already done.
        for module in reversed(self.modules[0 : -1]):     # n-2, ...,  2, 1, 0
            module.to("cuda:0")
            prev_act = acts.pop()  # acts: n-3, ...  1, 0, -1
            if acts_on_cpu:
                prev_act = prev_act.to("cuda:0")  # acts: n-3, ...  1, 0, -1
            prev_act.requires_grad_(True)
            prev_act.retain_grad()
            out = module(prev_act)

            out.backward(grad_next)     # grad_next is on cuda.
            grad_next = prev_act.grad   # 
                                   
            module.to("cpu")
            del out, prev_act
            torch.cuda.empty_cache()
            
        return lossAndMetrics

class PatchEmbedWithPos_Skip(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.pm = pm
        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.dim),
            nn.Tanh(),
            nn.Linear(2 * self.pm.dim, self.pm.dim),
            )
        nn.init.constant_(to_patch_embedding[3].weight, 0.0)
        nn.init.constant_(to_patch_embedding[3].bias, 0.0)
        
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFrames * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pe = torch.zeros(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        # pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
                
        # Assign with the name you want in state_dict
        self.to_patch_embedding = to_patch_embedding  # nn.Linear
        self.pos_embedding = nn.Parameter(pos_embedding)    # trainable

    def forward(self, x):
        # Assuming self.pm.patch_dim == self.pm.dim
        # returns x initially
        # (b t c H W)
        x_skip = rearrange(x, 'b t c (h p1) (w p2) -> b t (h w) (c p1 p2)', p1=self.pm.patch_size, p2=self.pm.patch_size)
        x = self.to_patch_embedding(x)  # zero
        x = x + self.pos_embedding      # zero
        x = x + x_skip
        return x
    
class MLP_Skip(nn.Module):
    def __init__(self, pm):
        super().__init__()
        self.pm = pm
    
        self.reshapeForMlp = ReshapeForMLP(self.pm.dim)
        self.reconstruct = PatchReconstruct(
            nFramesOut=self.pm.nFramesOut, nPatches_h=self.pm.nPatches_h, nPatches_w=self.pm.nPatches_w, patch_size=self.pm.patch_size,
            C=self.pm.nChannels, H=self.pm.image_height, W=self.pm.image_width
        )
        self.mlp_head = nn.Sequential(
            self.reshapeForMlp,
            nn.LayerNorm(self.pm.dim),
            nn.Linear(self.pm.dim, self.pm.nChannels * self.pm.patch_size ** 2),
            self.reconstruct
        )
        nn.init.constant_(self.mlp_head[2].weight, 0.0)
        nn.init.constant_(self.mlp_head[2].bias, 0.0)
        
    def forward(self, x):
        # Assuing self.pm.dim == self.pm.nChannels * self.pm.patch_size ** 2
        x_skip = self.reconstruct(self.reshapeForMlp(x))
        x = self.mlp_head(x)
        x = x + x_skip
        return x
        
class Zeus_Skip(nn.Module):
    def __init__(self, config, nDevices, device_type='cuda', base_device=0, abundant_GPUs=True):
        super().__init__()
        self.pm = Params(config)
        self.nDevices = nDevices
        self.device_type = device_type
        self.base_device = base_device
        self.abundant_GPUs = abundant_GPUs

        def dstring(device_type, id):
            return device_type + ':' + str(id)

        #--------------- patch embedding with position ---------------        
        self.patch_module = PatchEmbedWithPos_Skip(self.pm)

        if self.abundant_GPUs:
            self.patch_module = self.patch_module.to(dstring(self.device_type, self.base_device))  # from the defalut devoce to the 1st device.
        elif not self.abundant_GPUs: pass

        #-------------- zeus layers -------------------------
        if self.nDevices > 1: 
            assert self.nDevices >= self.pm.zeusLayers, "The numbers of devices and zeud layers must be the same if nDevices > 1"
        self.zeusLayers = nn.ModuleList([       # (B, T, N, dim) -> (B, T, N, dim)
            ZeusLayer(self.pm, skip=True)
            for i in range(self.pm.zeusLayers)
        ])
        if self.abundant_GPUs:
            for idx, layer in enumerate(self.zeusLayers):
                layer = layer.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, idx)))  # to the i-th device
        elif not self.abundant_GPUs: pass

        #--------------- ffn --------------------------------
        self.mlp_head = MLP_Skip(self.pm)
        
        if self.abundant_GPUs:
            self.mlp_head = self.mlp_head.to(dstring(self.device_type, self.base_device + min(self.nDevices - 1, self.pm.zeusLayers-1)))    # to the last device
        elif not self.abundant_GPUs: pass
        
        self.modules = self.get_modules()
        
    def get_modules(self):
        return [self.patch_module] + list(self.zeusLayers) + [self.mlp_head]

    def forward(self, x):
        # compute_place: data_find_layer, all_find_device_0
        # data_find_layer: move data to layers' device
        # all_find_device_0: move layers and data to cuda:0
        
        if not self.abundant_GPUs:
            device = self.device_type + ':0'
            x = x.to(device)
        
        for module in self.modules:                 #========================= both training and inference
            if self.abundant_GPUs:
                device = next(module.parameters()).device
                x = x.to(device, non_blocking=True)
                x = module(x)  # (B, T1FG, H/p * W/p, self.patch_dim = p * p * C) -> (B, T1FG, H/p * W/p, dim)
                # backwarding is possible, because module is on the device it was forwarded on.
                torch.cuda.empty_cache()    # free the space where x was.
            elif not self.abundant_GPUs:       #========================== inference only
                module.to(x.device)  # layer is not a copy.
                x = module(x)
                module.to('cpu:0')
                # backwarding is NOT possible, because module NOT is on the device it was forwarded on.
                torch.cuda.empty_cache()    # free the space where layer was if there are multiple GPUs 

        return x    # (B,FG,C,H,W)
    
    def sigle_GPU_forward_for_gradient(self, data, target, loss_fn, denormal_fn, batch_diff, era5_data_vars):        
        # data, target = data.to("cpu"), target.to("cpu")
        acts_cpu = []
        x = data.to("cuda:0")  # input on GPU
        
        # ---------------- Forward Pass ----------------
        for i, module in enumerate(self.get_modules()):
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
                last_input = x
                last_module = module
                last_input.requires_grad_(True)   # leaf tensor
                last_out = last_module(last_input)    # compute output with grad

        # ---------------- Backward Pass ----------------
        # 1) Last module and loss
        target = target.to('cuda:0')
        last_out = denormal_fn(last_out)
        target = denormal_fn(target)
        lossAndMetrics = loss_fn(last_out, target, batch_diff, era5_data_vars)     
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        loss.backward()                         # grads for last module & last_input
        last_module.to("cpu")                  # Safe place to move back.
        grad_next = last_input.grad.detach().cpu()

        # 2) Recompute previous modules in reverse
        for i in reversed(range(len(self.modules)-1)):

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
    
class DoubleZeus(nn.Module):
    def __init__(self, config, nDevices, device_type='cuda', abundant_GPUs=True, freeze_zeus1=True):
        super().__init__()
        self.pm = Params(config)
        self.nDevices = nDevices
        self.device_type = device_type
        self.abundant_GPUs = abundant_GPUs
        self.freeze_zeus1 = freeze_zeus1
        
        if self.abundant_GPUs:
            if self.device_type == 'cuda' and self.nDevices >= 2:
                # goal: deployed on [gpus + gpus if trainable else cpu]
                self.zeus1 = Zeus(config, nDevices=nDevices//2, device_type=device_type, base_device=0, abundant_GPUs=self.abundant_GPUs)
                self.zeus2 = Zeus(config, nDevices=nDevices//2, device_type=device_type, base_device=nDevices//2, abundant_GPUs=self.abundant_GPUs)
            else:
                raise Exception("DoubleZeus cannot be created in abundance_GPU mode with less than 2 GPUs")
        else:   #------------------------------------------------------- works on 1-GPU Windows and 1-GPU VM ---------------------
            if self.device_type == 'cuda' and self.nDevices == 1:
                # goal: deployed on [cpu], for sinlge-gpu offload parallel inference and train
                self.zeus1 = Zeus(config, nDevices=1, device_type=device_type, base_device=0, abundant_GPUs=self.abundant_GPUs)
                if not self.pm.skip_net_for_zeus2:
                    self.zeus2 = Zeus(config, nDevices=1, device_type=device_type, base_device=0, abundant_GPUs=self.abundant_GPUs)
                else:
                    self.zeus2 = Zeus_Skip(config, nDevices=1, device_type=device_type, base_device=0, abundant_GPUs=self.abundant_GPUs)      
            else:
                raise Exception("DoubleZeus cannot be created without GPU")    
        
        self.modules = self.get_modules()
        self.modules1 = self.zeus1.get_modules()
        self.modules2 = self.zeus2.get_modules()
        
        if freeze_zeus1:
            self.zeus1.eval()
            for param in self.zeus1.parameters():
                param.requires_grad = False
                
    def get_modules(self):  # on either (distributed) GPU(s) on CPU(s)
        return self.zeus1.get_modules() + self.zeus2.get_modules()
    
    def forward(self, x):
        x = self.zeus1(x)
        # if not self.pm.freeze_zeus1:
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
    
    
    def sigle_GPU_forward_for_gradient(self, data, target, loss_fn, denormal_fn, batch_diff, era5_data_vars):
        acts_on_cpu = False
        
        if self.pm.freeze_zeus1:
            data, target = data.to("cpu"), target.to("cpu")
            data = self.zeus1(data.to("cuda:0"))
            self.modules = self.modules2
        else:
            self.modules = self.modules

        if acts_on_cpu:
            acts = [data.cpu()]     # module[-1].act, or act[-1]
        else:
            acts = [data.to('cuda:0')]
            
        x = data.to("cuda:0")
                
        # ---------------- Forward Pass ----------------
        for i, module in enumerate(self.modules):
            module.to("cuda:0")

            if i < len(self.modules) - 1:      # 0, 1, 2, ..., n-2: all except the last module n-1
                with torch.no_grad():
                    x = module(x)       # 0, 1, ..., n-2
                if i < len(self.modules) - 2:      # 0, 1, ..., n-3
                    if acts_on_cpu:
                        acts.append(x.cpu())  # 0, 1, ..., n-3
                    else:
                        acts.append(x)  # 0, 1, ..., n-3
                module.to("cpu")
                # torch.cuda.empty_cache()
            else:                               # 3 = n-1
                act_n_2 = x                  # act[n-2]. no save as it's used now
                module_n_1 = module
                act_n_2.requires_grad_(True)   # act[n-2] as a leaf tensor
                act_n_2.retain_grad()
                act_n_1 = module_n_1(act_n_2)    # act[n-1], no save as it's used now

        # ---------------- Backward Pass ----------------
        # 1) Last module and loss
        # target = target.to('cuda:0')
        target = denormal_fn(target)
        pred = denormal_fn(act_n_1)
        lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars)     
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        loss.backward()             # grads for last_input (i.e. act[n-2]) -> last_module -> act[n-1] -> loss_fn -> loss.

        # metrics = lossAndMetrics[1] if isinstance(lossAndMetrics, (tuple, list)) else None
        # if metrics is not None: metrics = metrics.detach()
        # loss = loss.detach()
        
        grad_next = act_n_2.grad    # No need to move to cpu, if memory is enough.
        module_n_1.to("cpu")        # After backward(), safe place to move back. 
        
        del act_n_2, act_n_1, target, pred
        torch.cuda.empty_cache()

        # 2) Recompute previous modules in reverse. n-1 is already done.
        for module in reversed(self.modules[0 : -1]):     # n-2, ...,  2, 1, 0
            module.to("cuda:0")
            prev_act = acts.pop()  # acts: n-3, ...  1, 0, -1
            if acts_on_cpu:
                prev_act = prev_act.to("cuda:0")  # acts: n-3, ...  1, 0, -1
            prev_act.requires_grad_(True)
            prev_act.retain_grad()
            out = module(prev_act)

            out.backward(grad_next)     # grad_next is on cuda.
            grad_next = prev_act.grad   # 
                                   
            module.to("cpu")
            del out, prev_act
            torch.cuda.empty_cache()
            
        return lossAndMetrics


    def sigle_GPU_forward_for_gradient_old(self, data, target, loss_fn, denormal_fn, batch_diff, era5_data_vars):
        n_modules = len(self.modules2)
        acts_on_cpu = False

        # Forward through zeus1
        x = self.zeus1(data.to("cuda:0"))
        acts = [x.cpu()] if acts_on_cpu else [x]

        # ---------------- Forward Pass ----------------
        for i, module in enumerate(self.modules2):
            module.to("cuda:0")

            if i < n_modules - 1:  # all except last module
                with torch.no_grad():
                    x = module(x)
                if i < n_modules - 2:  # store intermediate activations
                    acts.append(x.cpu() if acts_on_cpu else x)
                module.to("cpu")
                torch.cuda.empty_cache()
            else:  # last module
                act_prev = x
                act_prev.requires_grad_(True)
                act_prev.retain_grad()  # important for non-leaf tensors
                x = checkpoint(lambda inp: module(inp), act_prev, use_reentrant=False)
                module_last = module

        # ---------------- Loss & Backward ----------------
        target = denormal_fn(target.to(x.device))
        pred = denormal_fn(x)
        loss_and_metrics = loss_fn(target, pred, batch_diff, era5_data_vars)
        loss = loss_and_metrics[0] if isinstance(loss_and_metrics, (tuple, list)) else loss_and_metrics
        loss.backward()  # grads for last module & last input
        grad_next = act_prev.grad
        module_last.to("cpu")

        # ---------------- Backward through previous modules ----------------
        for module in reversed(self.modules2[:-1]):
            module.to("cuda:0")

            prev_act = acts.pop()
            if acts_on_cpu:
                prev_act = prev_act.to("cuda:0")
            prev_act.requires_grad_(True)
            prev_act.retain_grad()

            out = checkpoint(lambda inp: module(inp), prev_act, use_reentrant=False)

            if grad_next is None:
                grad_next = torch.ones_like(out)
            out.backward(grad_next)
            grad_next = prev_act.grad

            module.to("cpu")
            del out, prev_act
            torch.cuda.empty_cache()

        return loss_and_metrics
    
    def load_state_dict(self, state_dict, strict=True):
        # Detect if state_dict belongs to a single Zeus model
        # sample_keys = list(state_dict.keys())
        # if sample_keys and all(not k.startswith("zeus2") for k in sample_keys):
        #     # assume it's a single Zeus, load into zeus1
        #     return self.zeus1.load_state_dict(state_dict, strict=strict)
        # else:
        #     # default behavior: load as usual
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


