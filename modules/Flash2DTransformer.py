import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import torch.optim as optim

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Flash2DEncLayer import Flash2DEncLayer
# from modules.Flash2DEncLayer_MSave import Flash2DEncLayer     # Weird, its slower.

class Flash2DTransformer(nn.Module):
    def __init__(
        self,
        nLayers: int,
        d_model: int,
        nHeads: int,
        dim_layerFFN: int = 2048,
        dropout: float = 0.0,
        causal: bool = False,
        heads_per_chunk: int = None,
        tokens_per_chunk: int = None,
        kv_full_sequence: bool = True,        
        layerFFN: Optional[str] = "FFN",  # "FFN", "GLU"
        eps: float = 1e-5,
        device: Optional[str] = None,
        norm_layer: Optional[nn.Module] = None,
    ):  
        print("transformer Chunk2D: layers {}, dim {}, nHeads {}, heads_per_chunk {}, tokens_per_chunk {}, dim_fwd {}, layerFFN {}".format(nLayers, d_model, nHeads, heads_per_chunk, tokens_per_chunk, dim_layerFFN, layerFFN))
        
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        layers = []
        for _ in range(nLayers):
            layers.append(
                Flash2DEncLayer(
                    d_model=d_model,
                    nHeads=nHeads,
                    dim_layerFFN=dim_layerFFN,
                    dropout=dropout,
                    causal=causal,
                    tokens_per_chunk=tokens_per_chunk,
                    heads_per_chunk=heads_per_chunk,
                    kv_full_sequence=kv_full_sequence,
                    layerFFN = layerFFN,
                    device=device,
                    eps=eps,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer(d_model) if norm_layer is not None else None
        
        self.src_saved = None
        
    def save_cpu_activation(self, tensor, name):
        buf = getattr(self, name, None)
        if buf is None or buf.shape != tensor.shape:
            buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
            setattr(self, name, buf)
        buf.copy_(tensor, non_blocking=True)

    def forward(
        self,
        src: torch.Tensor,                       # (B, S, E) typically on CPU
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        train: bool = False,
        modules_to_cpu: bool = True
    ) -> torch.Tensor:
        layered = src.to(self.device)
        for layer in self.layers:
            layered = layer(layered, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, train=train, modules_to_cpu=modules_to_cpu)        

        if train and self.norm is not None:
            self.save_cpu_activation(layered, "layered_saved")

        output = layered
        if self.norm is not None:
            self.norm.to(self.device)
            output = self.norm(layered)
            if modules_to_cpu:
                self.norm.to("cpu")
        # output is on GPU (self.device)

        # del layered
        # gc.collect()
        # torch.cuda.empty_cache()
        
        return output
    
    def modules_to_cpu(self):
        for layer in self.layers:
            layer.modules_to_cpu()
        if self.norm is not None:
            self.norm.to("cpu")
        return
        
    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        # gradient = gradient.to(self.device)  # redundant
        
        if self.norm is not None:
            layered_saved = self.layered_saved.to(self.device, non_blocking=True)
            layered_saved.requires_grad_(True)
            layered_saved.retain_grad()
            self.norm.to(self.device)
            with torch.enable_grad():   # redundant for readability
                out = self.norm(layered_saved)
            out.backward(gradient, retain_graph=False)
            gradient = layered_saved.grad
            self.norm.to("cpu")
            del layered_saved, out
            gc.collect()
            torch.cuda.empty_cache()
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
if __name__ == "__main__":
    """
    1. For large seq_len, the more layers the more loss.
    """
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 1         # it's also the data simplicity in this test.
    T1 = 24
    N = round(72 * factor10 * 144 * factor10 /16/16)
    seq_len = round(T1 * 72 * factor10 * 144 * factor10 /16/16)    # 97_200 for factor10 == 10
    nLayers = 1
    d_model = 1024
    nHeads = 128
    tokens_per_chunk = None
    heads_per_chunk = None
    kv_full_sequence = True
    lr = 1e-3
    steps = 100
    
    layerFFN = "GLU"  # "FFN", "GLU"

    # nLayers == 1 is faster than a Flash2DEncLayer.
    
    model = Flash2DTransformer(
        nLayers = nLayers,
        d_model = d_model,
        nHeads = nHeads, 
        heads_per_chunk = heads_per_chunk,
        tokens_per_chunk = tokens_per_chunk,
        dim_layerFFN = 4*d_model,
        kv_full_sequence = kv_full_sequence,
        dropout=0.0, 
        layerFFN = layerFFN, 
        device=device
    )

    x = torch.randn(batch_size, seq_len, d_model)  # (B, S, E)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            out = model(x, train=True, modules_to_cpu=False)
    print("Output:", out.shape, out.device)
    gradient = torch.ones_like(x)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        gradient = model.backward(gradient)
    print("Gradient:", gradient.shape, gradient.device)
    
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, d_model)
    y_train = torch.randn(batch_size, seq_len, d_model)

    model.layers[0].src_saved = None    # test in test

    chunkBackward = False

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
        
        optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")