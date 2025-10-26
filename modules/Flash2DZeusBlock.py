import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
from einops.layers.torch import Rearrange
from einops import rearrange

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Flash2DTransformer import Flash2DTransformer
from modules.CylindricSpatialConvTD import CylindricSpatialConvTD
from modules.utils import check_gradient

class SpatialConvolution(nn.Module):
    def __init__(self, pm):
        self.pm = pm
        super(SpatialConvolution, self).__init__()
        self.convolve = nn.Conv2d(pm.nFramesIn, pm.nFramesIn, 3, 1, 1)  # preserve shape

    def forward(self, x):   # (B, T1FG, H/p * W/p, dim) -> preserve shape
        # return zero initially
        x = rearrange(x, 'b t n d -> (b d) t n')
        x = rearrange(x, 'B t (h w) -> B t h w', h=self.pm.nPatches_h)
        x = self.convolve(x)    # preserves shape
        x = rearrange(x, '(b d) t h w -> b t (h w) d', d=self.pm.d_model)

        return x


class Flash2DZeusBlock(nn.Module):
    def __init__(
        self, pm, device=None):  
        print("Flash2DZeusBlock: layers {}, dim {}, nHeads {}, heads_per_chunk {}, tokens_per_chunk {}, dim_fwd {}, layerFFN {}"
              .format(pm.nLayers, pm.d_model, pm.nHeads, pm.heads_per_chunk, pm.tokens_per_chunk, pm.dim_layerFFN, pm.layerFFN))
        super().__init__()
        
        self.add_spatial_conv = pm.add_spatial_conv
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.add_spatial_conv:
            self.spaial_conv_1 = CylindricSpatialConvTD(pm, kernel_size=5)
            self.spaial_conv_2 = CylindricSpatialConvTD(pm, kernel_size=5)
               
        self.temporal_transformer_1 = Flash2DTransformer(
            nLayers = pm.nLayers,
            d_model = pm.d_model,
            nHeads = pm.nHeads, 
            heads_per_chunk = pm.heads_per_chunk,
            tokens_per_chunk = pm.tokens_per_chunk,
            dim_layerFFN = pm.dim_layerFFN,
            kv_full_sequence = pm.kv_full_sequence,
            dropout=pm.dropout, 
            layerFFN = pm.layerFFN, 
            device=device
        )
        self.spatial_transformer_1 = Flash2DTransformer(
            nLayers = pm.nLayers,
            d_model = pm.d_model,
            nHeads = pm.nHeads, 
            heads_per_chunk = pm.heads_per_chunk,
            tokens_per_chunk = pm.tokens_per_chunk,
            dim_layerFFN = pm.dim_layerFFN,
            kv_full_sequence = pm.kv_full_sequence,
            dropout=pm.dropout, 
            layerFFN = pm.layerFFN, 
            device=device
        )
        self.spatial_transformer_2 = Flash2DTransformer(
            nLayers = pm.nLayers,
            d_model = pm.d_model,
            nHeads = pm.nHeads, 
            heads_per_chunk = pm.heads_per_chunk,
            tokens_per_chunk = pm.tokens_per_chunk,
            dim_layerFFN = pm.dim_layerFFN,
            kv_full_sequence = pm.kv_full_sequence,
            dropout=pm.dropout, 
            layerFFN = pm.layerFFN, 
            device=device
        )
        self.temporal_transformer_2 = Flash2DTransformer(
            nLayers = pm.nLayers,
            d_model = pm.d_model,
            nHeads = pm.nHeads, 
            heads_per_chunk = pm.heads_per_chunk,
            tokens_per_chunk = pm.tokens_per_chunk,
            dim_layerFFN = pm.dim_layerFFN,
            kv_full_sequence = pm.kv_full_sequence,
            dropout=pm.dropout, 
            layerFFN = pm.layerFFN, 
            device=device
        )
        
    def save_cpu_activation(self, tensor, name):
        buf = getattr(self, name, None)
        if buf is None or buf.shape != tensor.shape:
            buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
            setattr(self, name, buf)
        buf.copy_(tensor, non_blocking=True)
        
    def forward(self, x, train=False, modules_to_cpu=True): # (B, T, N, dim) -> (B, T, N, dim)      # (B, T1FG, H/p * W/p, dim)
        b, t, n, dim = x.shape                  # (B, T, N, dim), or (B, T1FG, H/p * W/p, dim)
        
        x = x.to(self.device)                                       # assume it's contiguous
        
        # if train: 
        #     # self.in_temporal1_saved = x.to("cpu")                 # (B, T, N, dim)
        #     self.save_cpu_activation(x, "in_temporal1_saved")
        if self.add_spatial_conv:
            self.spaial_conv_1.to(self.device)
            x = self.spaial_conv_1(x, train=train, modules_to_cpu=modules_to_cpu)                               # (B, T, N, dim)
            if modules_to_cpu: self.spaial_conv_1.to("cpu")
        x = rearrange(x, 'b t n d -> (b n) t d').contiguous()       # (B*N, T, dim)
        self.temporal_transformer_1.to(self.device)
        x = self.temporal_transformer_1(x, train=train, modules_to_cpu=modules_to_cpu)                          # (B*N, T, dim)
        if modules_to_cpu: self.temporal_transformer_1.to("cpu")

        # if train: 
        #     # self.in_spatial1_saved = x.to("cpu")                  # (B*N, T, dim)
        #     self.save_cpu_activation(x, "in_spatial1_saved")
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b).contiguous()# (B*T, N, dim)
        x = self.spatial_transformer_1(x, train=train, modules_to_cpu=modules_to_cpu)                           # (B*T, N, dim)
        if modules_to_cpu: self.spatial_transformer_1.to("cpu")
        
        # if train: 
        #     # self.in_spatial2_saved = x.to("cpu")                  # (B*T, N, dim)
        #     self.save_cpu_activation(x, "in_spatial2_saved")
        x = self.spatial_transformer_2(x, train=train, modules_to_cpu=modules_to_cpu)                           # (B*T, N, dim)
        if modules_to_cpu: self.spatial_transformer_2.to("cpu")

        # if train: 
        #     # self.in_temporal2_saved = x.to("cpu")                 # (B*T, N, dim)
        #     self.save_cpu_activation(x, "in_temporal2_saved")
        if self.add_spatial_conv:
            self.spaial_conv_2.to(self.device)
            x = rearrange(x, '(b t) n d -> b t n d', t=t)           # contiguous
            x = self.spaial_conv_2(x, train=train, modules_to_cpu=modules_to_cpu)                               # (B, T, N, dim)
            x = rearrange(x, 'b t n d -> (b t) n d').contiguous()   # contiguous
            if modules_to_cpu: self.spaial_conv_2.to("cpu")
        x = rearrange(x, '(b t) n d -> (b n) t d', t=t, n=n).contiguous()
        # x = rearrange(x, '(b t) n d -> b t n d', b=b).contiguous()  # (B, T, N, dim)  # contiguous
        # x = rearrange(x, 'b t n d -> (b n) t d').contiguous()       # (B*N, T, dim)    # contiguous
        x = self.temporal_transformer_2(x, train=train, modules_to_cpu=modules_to_cpu)                          # (B*N, T, dim)        
        x = rearrange(x, '(b n) t d -> b t n d', b=b).contiguous()  # (B, T, N, dim)   # contiguous
        if modules_to_cpu: self.temporal_transformer_2.to("cpu")
              
        return x                                                    # (B, T, N, dim)

    def modules_to_cpu(self):
        self.temporal_transformer_1.modules_to_cpu()
        self.spatial_transformer_1.modules_to_cpu()
        self.spatial_transformer_2.modules_to_cpu()
        self.temporal_transformer_2.modules_to_cpu()

    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.to(self.device)                          # redundant
        b, t, n, dim = gradient.shape      
        
        # # detached leaf on correct device
        # src = self.in_temporal2_saved.to(self.device).detach().requires_grad_(True)
        # with torch.enable_grad():   # redundant for readability
        #     if self.add_spatial_conv:
        #         # self.spaial_conv_2.to(self.device)
        #         x = rearrange(src, '(b t) n d -> b t n d', t=t)           # contiguous
        #         x = self.spaial_conv_2(x, train=False, modules_to_cpu=False)                               # (B, T, N, dim)
        #         x = rearrange(x, 'b t n d -> (b t) n d')   # contiguous
        #     else:
        #         x = rearrange(src, 'b t n d -> (b t) n d')   # contiguous

        #     x = rearrange(x, '(b t) n d -> b t n d', b=b).contiguous()  # (B, T, N, dim)  # contiguous
        #     x = rearrange(x, 'b t n d -> (b n) t d').contiguous()       # (B*N, T, dim)    # contiguous
        #     # self.temporal_transformer_2.to(self.device)
        #     x = self.temporal_transformer_2(x, train=False, modules_to_cpu=False)                          # (B*N, T, dim)        
        #     x = rearrange(x, '(b n) t d -> b t n d', b=b).contiguous()  # (B, T, N, dim)   # contiguous

        # x.backward(gradient, retain_graph=False)
        # gradient = src.grad                                         # (B*T, N, dim)
        # if self.add_spatial_conv:
        #     self.spaial_conv_2.to("cpu")
        # self.temporal_transformer_2.to("cpu")
        
        gradient = rearrange(gradient, 'b t n d -> (b n) t d').contiguous()
        gradient = self.temporal_transformer_2.backward(gradient)
        gradient = rearrange(gradient, '(b n) t d -> (b t) n d', n=n, t=t).contiguous()
        # gradient = rearrange(gradient, '(b n) t d -> b t n d', n=n).contiguous()
        # gradient = rearrange(gradient, 'b t n d -> (b t) n d')
        if self.add_spatial_conv:
            gradient = rearrange(gradient, '(b t) n d -> b t n d', t=t)
            gradient = self.spaial_conv_2.backward(gradient)
            gradient = rearrange(gradient, 'b t n d -> (b t) n d')
        else:
            pass
        
        # src = self.in_spatial2_saved.to(self.device).detach().requires_grad_(True)
        # self.spatial_transformer_2.to(self.device)
        # with torch.enable_grad():   # redundant for readability
        #     x = self.spatial_transformer_2(src, train=False, modules_to_cpu=False)                 # (B*T, N, dim)
        # x.backward(gradient, retain_graph=False)
        # gradient = src.grad                                         # (B*T, N, dim)
        # self.spatial_transformer_2.to("cpu")
        
        gradient = self.spatial_transformer_2.backward(gradient)

        # src = self.in_spatial1_saved.to(self.device).detach().requires_grad_(True)
        # self.spatial_transformer_1.to(self.device)
        # with torch.enable_grad():   # redundant for readability
        #     x = rearrange(src, '(b n) t d -> (b t) n d', b=b).contiguous()     # (B*T, N, dim)
        #     x = self.spatial_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*T, N, dim) 
        # x.backward(gradient, retain_graph=False)
        # gradient = src.grad                                         # (B*N, T, dim)
        # self.spatial_transformer_1.to("cpu")
        
        gradient = self.spatial_transformer_1.backward(gradient)
        gradient = rearrange(gradient, '(b t) n d -> (b n) t d', t=t)
        
        
        # src = self.in_temporal1_saved.to(self.device).detach().requires_grad_(True)
        # src.requires_grad_(True)
        # src.retain_grad()
        # self.temporal_transformer_1.to(self.device)
        # with torch.enable_grad():   # redundant for readability
        #     if self.add_spatial_conv:
        #         self.spaial_conv_1.to(self.device)
        #         x = self.spaial_conv_1(src)                         # (B, T, N, dim)
        #         x = rearrange(x, 'b t n d -> (b n) t d').contiguous()
        #     else:
        #         x = rearrange(src, 'b t n d -> (b n) t d')            # (B*N, T, dim)
        #     x = self.temporal_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*N, T, dim) 
        # x.backward(gradient, retain_graph=False)
        # gradient = src.grad                                         # (B, N, T, dim)
        # if self.add_spatial_conv:
        #     self.spaial_conv_1.to("cpu")
        # self.temporal_transformer_1.to("cpu")
        
        gradient = self.temporal_transformer_1.backward(gradient)
        if self.add_spatial_conv:
            gradient = rearrange(gradient, '(b n) t d -> b t n d', n=n).contiguous()
            gradient = self.spaial_conv_1.backward(gradient)
        else:
            gradient = rearrange(gradient, '(b n) t d -> b t n d', n=n)
                
        return gradient

    def backward_old(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.to(self.device)                          # redundant
        b, t, n, dim = gradient.shape      
        
        # detached leaf on correct device
        src = self.in_temporal2_saved.to(self.device).detach().requires_grad_(True)
        with torch.enable_grad():   # redundant for readability
            if self.add_spatial_conv:
                # self.spaial_conv_2.to(self.device)
                x = rearrange(src, '(b t) n d -> b t n d', t=t)           # contiguous
                x = self.spaial_conv_2(x, train=False, modules_to_cpu=False)                               # (B, T, N, dim)
                x = rearrange(x, 'b t n d -> (b t) n d')   # contiguous
            else:
                x = rearrange(src, 'b t n d -> (b t) n d')   # contiguous
            x = rearrange(x, '(b t) n d -> b t n d', b=b).contiguous()  # (B, T, N, dim)  # contiguous
            x = rearrange(x, 'b t n d -> (b n) t d').contiguous()       # (B*N, T, dim)    # contiguous
            # self.temporal_transformer_2.to(self.device)
            x = self.temporal_transformer_2(x, train=False, modules_to_cpu=False)                          # (B*N, T, dim)        
            x = rearrange(x, '(b n) t d -> b t n d', b=b).contiguous()  # (B, T, N, dim)   # contiguous

        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                         # (B*T, N, dim)
        if self.add_spatial_conv:
            self.spaial_conv_2.to("cpu")
        self.temporal_transformer_2.to("cpu")
        
        
        # gradient = rearrange(gradient, 'b t n d -> (b n) t d').contiguous()
        # gradient = self.temporal_transformer_2.backward(gradient)
        # gradient = rearrange(gradient, '(b n) t d -> b t n d')
        # if self.add_spatial_conv:
        #     gradient = self.spaial_conv_2.backward(gradient)
        # gradient = rearrange(gradient, 'b t n d -> (b t) n d')

        
        src = self.in_spatial2_saved.to(self.device).detach().requires_grad_(True)
        self.spatial_transformer_2.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = self.spatial_transformer_2(src, train=False, modules_to_cpu=False)                 # (B*T, N, dim)
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                         # (B*T, N, dim)
        self.spatial_transformer_2.to("cpu")

        src = self.in_spatial1_saved.to(self.device).detach().requires_grad_(True)
        self.spatial_transformer_1.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = rearrange(src, '(b n) t d -> (b t) n d', b=b).contiguous()     # (B*T, N, dim)
            x = self.spatial_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*T, N, dim) 
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                         # (B*N, T, dim)
        self.spatial_transformer_1.to("cpu")
        
        # in_temporal1_saved.grad
        src = self.in_temporal1_saved.to(self.device).detach().requires_grad_(True)
        src.requires_grad_(True)
        src.retain_grad()
        self.temporal_transformer_1.to(self.device)
        with torch.enable_grad():   # redundant for readability
            if self.add_spatial_conv:
                self.spaial_conv_1.to(self.device)
                x = self.spaial_conv_1(src)                         # (B, T, N, dim)
                x = rearrange(x, 'b t n d -> (b n) t d').contiguous()
            else:
                x = rearrange(src, 'b t n d -> (b n) t d')            # (B*N, T, dim)
            x = self.temporal_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*N, T, dim) 
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                         # (B, N, T, dim)
        if self.add_spatial_conv:
            self.spaial_conv_1.to("cpu")
        self.temporal_transformer_1.to("cpu")
        
        return gradient

# ---- Run Demo ----
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"   
    
    import time
    import sys
    from pathlib import Path
    # Get path to sibling folder
    sibling_path = Path(__file__).resolve().parent.parent / "models"
    sys.path.insert(0, str(sibling_path))
    from config import config
    from params import Params

    # config['factor10'] = 2
    # config['batch_size'] = 1
    # config['patch_size'] = 12
    # config['T1'] = 24
    # config['nLayers'] = 5
    # config['nHeads'] = 72
    # config['tokens_per_chunk'] = None
    # config['heads_per_chunk'] = None
    # config['kv_full_sequence'] = True
    # config['self.add_spatial_conv'] = False
    # config['layerFFN'] = "GLU"    
    
    config['factor10'] = 2
    config['nLayers'] = 1
    config['add_spatial_conv'] = True
    
    pm = Params(config)
    batch_size, T1, N, d_model = pm.batch_size, pm.T1, pm.nPatches, pm.d_model

    lr = 1e-3               # -4
    steps = 100
       
    model = Flash2DZeusBlock(pm, device = device)

    x = torch.randn(batch_size, T1, N, d_model)

    # with autocast(device_type="cuda", dtype=torch.bfloat16):
    #     with torch.set_grad_enabled(False):
    #         out = model(x, train=True, modules_to_cpu=True)
    # print("Output:", out.shape, out.device)
    # gradient = torch.ones_like(x)    
    # with autocast(device_type="cuda", dtype=torch.bfloat16):
    #     gradient = model.backward(gradient)
    # print("Gradient:", gradient.shape, gradient.device)
    
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, T1, N, d_model)
    y_train = torch.randn(batch_size, T1, N, d_model)

    chunkBackward = True

    start = time.time()
    print("start: ", start)
    for step in range(steps):
        optimizer.zero_grad()

        # model.to(device)
        # x_train.requires_grad_(True)   # it's False by default.

        # forward pass (offloads chunks to GPU internally)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.set_grad_enabled(not chunkBackward):
                out = model(x_train, train=chunkBackward, modules_to_cpu=chunkBackward)

        if chunkBackward:
            out.requires_grad_(chunkBackward)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.enable_grad():   # redundant for readability
                loss = criterion(out, y_train.to(out.device))
            loss.backward(retain_graph=False)

        if chunkBackward:
            gradient = out.grad
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                gradient = model.backward(gradient)
                
        check_gradient(model)
        
        optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")