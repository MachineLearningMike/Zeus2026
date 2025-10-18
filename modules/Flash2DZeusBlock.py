import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from einops import rearrange

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Flash2DTransformer import Flash2DTransformer

class Flash2DZeusBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        causal: bool = False,
        heads_per_chunk: int = None,
        tokens_per_chunk: int = None,
        kv_full_sequence: bool = True,        
        ffn: Optional[str] = "FFN",  # "FFN", "GLU"
        eps: float = 1e-5,
        device: Optional[str] = None,
        norm_layer: Optional[nn.Module] = None,
    ):  
        print("Flash2DZeusBlock: layers {}, dim {}, heads {}, heads_per_chunk {}, tokens_per_chunk {}, dim_fwd {}, ffn {}".format(num_layers, d_model, num_heads, heads_per_chunk, tokens_per_chunk, dim_feedforward, ffn))

        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
               
        self.temporal_transformer_1 = Flash2DTransformer(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads, 
            heads_per_chunk = heads_per_chunk,
            tokens_per_chunk = tokens_per_chunk,
            dim_feedforward = 4*d_model,
            kv_full_sequence = kv_full_sequence,
            dropout=0.0, 
            ffn = ffn, 
            device=device
        )
        self.spatial_transformer_1 = Flash2DTransformer(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads, 
            heads_per_chunk = heads_per_chunk,
            tokens_per_chunk = tokens_per_chunk,
            dim_feedforward = 4*d_model,
            kv_full_sequence = kv_full_sequence,
            dropout=0.0, 
            ffn = ffn, 
            device=device
        )
        self.spatial_transformer_2 = Flash2DTransformer(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads, 
            heads_per_chunk = heads_per_chunk,
            tokens_per_chunk = tokens_per_chunk,
            dim_feedforward = 4*d_model,
            kv_full_sequence = kv_full_sequence,
            dropout=0.0, 
            ffn = ffn, 
            device=device
        )
        self.temporal_transformer_2 = Flash2DTransformer(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads, 
            heads_per_chunk = heads_per_chunk,
            tokens_per_chunk = tokens_per_chunk,
            dim_feedforward = 4*d_model,
            kv_full_sequence = kv_full_sequence,
            dropout=0.0, 
            ffn = ffn, 
            device=device
        )
        
    def forward(self, x, train=False, modules_to_cpu=True): # (B, T, N, dim) -> (B, T, N, dim)      # (B, T1FG, H/p * W/p, dim)
        b, t, n, dim = x.shape                  # (B, T, N, dim), or (B, T1FG, H/p * W/p, dim)
        
        if train: self.in_temporal1_saved = x.to("cpu")     # (B, T, N, dim)
        x = rearrange(x, 'b t n d -> (b n) t d')            # (B*N, T, dim)
        self.temporal_transformer_1.to(self.device)
        x = self.temporal_transformer_1(x)                  # (B*N, T, dim)
        if modules_to_cpu: self.temporal_transformer_1.to("cpu")

        if train: self.in_spatial1_saved = x.to("cpu")      # (B*N, T, dim)
        x = rearrange(x, '(b n) t d -> (b t) n d', b=b)     # (B*T, N, dim)
        x = self.spatial_transformer_1(x)                   # (B*T, N, dim)
        if modules_to_cpu: self.spatial_transformer_1.to("cpu")
        
        if train: self.in_spatial2_saved = x.to("cpu")      # (B*T, N, dim)
        x = self.spatial_transformer_2(x)                   # (B*T, N, dim)
        if modules_to_cpu: self.spatial_transformer_2.to("cpu")

        if train: self.in_temporal2_saved = x.to("cpu")     # (B*T, N, dim)
        x = rearrange(x, '(b t) n d -> b t n d', b=b)       # (B, T, N, dim)
        x = rearrange(x, 'b t n d -> (b n) t d')            # (B*N, T, dim)
        x = self.temporal_transformer_2(x)                  # (B*N, T, dim)        
        x = rearrange(x, '(b n) t d -> b t n d', b=b)       # (B, T, N, dim)
        if modules_to_cpu: self.temporal_transformer_2.to("cpu")
               
        return x                                            # (B, T, N, dim)

    def modules_to_cpu(self):
        self.temporal_transformer_1.modules_to_cpu()
        self.spatial_transformer_1.modules_to_cpu()
        self.spatial_transformer_2.modules_to_cpu()
        self.temporal_transformer_2.modules_to_cpu()

    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.to(self.device)  # redundant
        b, t, n, dim = gradient.shape
        
        # in_temporal2_saved.grad
        src = self.in_temporal2_saved.to(self.device)             # (B*T, N, dim)
        src.requires_grad_(True)
        src.retain_grad()
        self.temporal_transformer_2.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = rearrange(src, '(b t) n d -> b t n d', b=b)       # (B, T, N, dim)
            x = rearrange(x, 'b t n d -> (b n) t d')            # (B*N, T, dim)
            x = self.temporal_transformer_2(x, train=False, modules_to_cpu=False)                  # (B*N, T, dim) 
            x = rearrange(x, '(b n) t d -> b t n d', b=b)     # (B, T, N, dim)
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                       # (B*T, N, dim)
        self.temporal_transformer_2.to("cpu")

        # in_spatial2_saved.grad
        src = self.in_spatial2_saved.to(self.device)              # (B*T, N, dim)
        src.requires_grad_(True)
        src.retain_grad()
        self.spatial_transformer_2.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = self.spatial_transformer_2(src, train=False, modules_to_cpu=False)                 # (B*T, N, dim)
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                       # (B*T, N, dim)
        self.spatial_transformer_2.to("cpu")

        # in_spatial1_saved.grad
        src = self.in_spatial1_saved.to(self.device)              # (B*N, T, dim)
        src.requires_grad_(True)
        src.retain_grad()
        self.spatial_transformer_1.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = rearrange(src, '(b n) t d -> (b t) n d', b=b)     # (B*T, N, dim)
            x = self.spatial_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*T, N, dim) 
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                       # (B*N, T, dim)
        self.spatial_transformer_1.to("cpu")
        
        # in_temporal1_saved.grad
        src = self.in_temporal1_saved.to(self.device)              # (B, N, T, dim)
        src.requires_grad_(True)
        src.retain_grad()
        self.temporal_transformer_1.to(self.device)
        with torch.enable_grad():   # redundant for readability
            x = rearrange(src, 'b t n d -> (b n) t d')            # (B*N, T, dim)
            x = self.temporal_transformer_1(x, train=False, modules_to_cpu=False)                                  # (B*N, T, dim) 
        x.backward(gradient, retain_graph=False)
        gradient = src.grad                                       # (B, N, T, dim)
        self.temporal_transformer_1.to("cpu")
        
        return gradient

# ---- Run Demo ----
if __name__ == "__main__":
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    factor10 = 2
    batch_size = 1         # it's also the data simplicity in this test.
    T1 = 24
    N = round(72 * factor10 * 144 * factor10 /16/16)
    seq_len = round(T1 * 72 * factor10 * 144 * factor10 /16/16)    # 97_200 for factor10 == 10
    num_layers = 5
    d_model = 1024
    num_heads = 128
    tokens_per_chunk = seq_len // 10
    heads_per_chunk = 32
    kv_full_sequence = True
    lr = 1e-4
    steps = 100
    
    ffn = "GLU"  # "FFN", "GLU"
    
    model = Flash2DZeusBlock(
        num_layers = num_layers,
        d_model = d_model,
        num_heads = num_heads,
        dim_feedforward = d_model * 4,
        dropout = 0.0,
        causal = False,
        heads_per_chunk = heads_per_chunk,
        tokens_per_chunk = tokens_per_chunk,
        kv_full_sequence = True,
        ffn = ffn,
        device = device,
    )

    x = torch.randn(batch_size, T1, N, d_model)
    with torch.no_grad():
        out = model(x, train=True, modules_to_cpu=True)
    print("Output:", out.shape, out.device)
    gradient = torch.ones_like(x)
    gradient = model.backward(gradient)
    model.modules_to_cpu()
    print("Gradient:", gradient.shape, gradient.device)
    
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, T1, N, d_model)
    y_train = torch.randn(batch_size, T1, N, d_model)

    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()

        # forward pass (offloads chunks to GPU internally)
        with torch.no_grad():
            out = model(x_train, train=True, modules_to_cpu=True)
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with torch.enable_grad():   # redundant for readability
            loss = criterion(out, y_train.to(out.device))
            loss.backward()
        gradient = out.grad
        gradient = model.backward(gradient)
        optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")