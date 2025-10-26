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

from modules.Flash2DMHA import Flash2DMHA
from modules.ChunkFFN import FFN
from modules.ChunkGLU import GLU
from modules.utils import check_gradient


class Flash2DEncLayer(nn.Module):
    """
    implements:   
      x = x + dropout(MHA(norm1(x)))        # pre_LN to MHA
      x = norm2( x + dropout( ff(x) ) )     # standard post_LN to ff

    WHY?
    ff = "GLU" shows no significant advantage over "FFN" yet is slower, 
        although GLU is millions of times more efficient when used in isolation.
        
    Single encoder layer using ChunkMHA, which stands for offloaded chunked MHA.
    Pre-LN style (LayerNorm before attention and before FFN).
    All weights start on CPU; moved temporarily to GPU during computation.
    """
    def __init__(
        self,
        d_model: int,
        nHeads: int,
        dim_layerFFN: int = 2048,
        dropout: float = 0.0,
        causal: bool = False,
        tokens_per_chunk: int = None,
        heads_per_chunk: int = None,
        kv_full_sequence: bool = True,        
        layerFFN: Optional[str] = "FFN",  # "FFN", "GLU"
        eps: float = 1e-5,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)
        
        self.norm1 = nn.LayerNorm(d_model, eps=eps)   
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)      # shared between mha and layerFFN
        
        self.self_attn = Flash2DMHA(
            d_model=d_model,
            nHeads=nHeads,
            dropout=dropout,
            causal=causal,
            tokens_per_chunk=tokens_per_chunk,
            heads_per_chunk=heads_per_chunk,
            kv_full_sequence=kv_full_sequence,
            device=device,
        )

        if layerFFN == "GLU":
            self.layerFFN = GLU(in_features=d_model, hidden_features=dim_layerFFN, out_features=d_model, act_layer=nn.SiLU, device=device)
        elif layerFFN == "FFN":
            self.layerFFN = FFN(d_model=d_model, dim_layerFFN=dim_layerFFN, activation=F.relu, device=device)
        else: assert False

        self.src_saved, self.src_mask_saved, self.src_key_padding_mask_saved, self.attn_out_saved = None, None, None, None

    def save_cpu_activation(self, tensor, name):
        buf = getattr(self, name, None)
        if buf is None or buf.shape != tensor.shape:
            buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
            setattr(self, name, buf)
        buf.copy_(tensor, non_blocking=True)

    #---------------------------- pre_mha --------------------------
    def apply_norm_to_src_before_mha(self, src, modules_to_cpu=True):
        # --- Pre-LN ---           
        self.norm1.to(self.gpu_device)
        src_ln = self.norm1(src.to(self.gpu_device))

        if modules_to_cpu:
            # move back to CPU
            self.norm1.to("cpu")
                
        return src_ln
    
    def merge_mask(self, src_mask, src_key_padding_mask):

        # --- Merge masks for attention ---
        mask = None
        if src_mask is not None:
            mask = src_mask
        if src_key_padding_mask is not None:
            kpm = src_key_padding_mask
            if mask is None:
                mask = kpm
            elif mask.dtype == torch.bool:
                mask = mask & kpm
                
        return mask
    
    def modules_to_cpu_pre_mha(self):
        self.norm1.to("cpu")
        return

    #---------------------------- post_mha --------------------------
    def apply_remaining_after_mha(self, src, attn_out):
        # return res on self.gpu_device
        assert attn_out.device == self.gpu_device
        
        attn_out = self.dropout(attn_out)
        # Residual add: move src to GPU
        res = src.to(self.gpu_device) + attn_out
        return res
   
    def apply_post_LN_to_ff(self, src, ff, modules_to_cpu=True):
        # return src_out on self.gpu_device
        ff = self.dropout(ff)
        # --- Post-LN ---
        self.norm2.to(self.gpu_device)
        src_out = self.norm2(src + ff)
        if modules_to_cpu:
            self.norm2.to("cpu")
        return src_out
    
    def modules_to_cpu_post_mha(self):
        self.layerFFN.modules_to_cpu()
        self.norm2.to("cpu")
        return

    def modules_to_cpu(self):
        self.modules_to_cpu_pre_mha()
        self.self_attn.modules_to_cpu()
        self.modules_to_cpu_post_mha()
        
    #-----------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,                       # [B, S, D] or [B, T, N, D] depending on your layout
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        train: bool = False,
        modules_to_cpu: bool = True
    ) -> torch.Tensor:
        """
        x: input tensor on CPU (will be moved to GPU internally)
        src_mask: optional attention mask
        src_key_padding_mask: optional padding mask
        train: if True, save activations for backward
        modules_to_cpu: if True, move submodules to CPU after use
        """
        
        if train:
            self.save_cpu_activation(src, "src_saved")
            if src_mask is not None:
                self.save_cpu_activation(src_mask, "src_mask_saved")
            if src_key_padding_mask is not None:
                self.save_cpu_activation(src_key_padding_mask, "src_key_padding_mask_saved")

        x = src.to(self.gpu_device)
        self.to(self.gpu_device)

        # --- Self-Attention block ---  
        # x = x + dropout(attn(norm1(x), attn_mask=mask(src_mask,src_key_padding_mask)))
        
        mask = None
        if src_mask is not None:
            mask = src_mask
        if src_key_padding_mask is not None:
            kpm = src_key_padding_mask
            if mask is None:
                mask = kpm
            elif mask.dtype == torch.bool:
                mask = mask & kpm
        
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            x_norm,
            attn_mask=mask,
            train=False,        # train=False disables future custom backward.
            modules_to_cpu=modules_to_cpu
        )
        attn_out = self.dropout(attn_out)
        x = residual + attn_out  # residual connection

        if modules_to_cpu:
            self.self_attn.to("cpu")
            self.norm1.to("cpu")

        # --- Feedforward block ---
        # x = x + layerFFN(norm2(x))
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.layerFFN(x_norm, train=False, modules_to_cpu=modules_to_cpu) # # train=False disables future custom backward.
        x = residual + ffn_out

        if modules_to_cpu:
            self.layerFFN.to("cpu")
            self.norm2.to("cpu")

        return x

    
    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        initial: all submodules; norm1, norm2, linear1, and linear2 ; are on cpu
        return: gradient from self.src_saved through to gradient, on cpu
        state_change: gradinets are stored to parameters, for optmizer.step
        final: all submodules are back on cpu
        """        
        assert self.src_saved is not None
        assert self.src_saved.shape == gradient.shape 

        src = self.src_saved.to(self.gpu_device, non_blocking=True)
        src_mask = self.src_mask_saved.to(self.gpu_device) if self.src_mask_saved is not None else None
        src_key_padding_mask = self.src_key_padding_mask.to(self.gpu_device) if self.src_key_padding_mask_saved is not None else None
        
        self.to(self.gpu_device)
        gradient = gradient.to(self.gpu_device)
        
        src.requires_grad_(True)
        out = self(src, src_mask, src_key_padding_mask, train=False, modules_to_cpu=False)
        out.backward(gradient)
        gradient = src.grad
        self.to("cpu")
        
        return gradient
    
if __name__ == "__main__":
    
    import time
    import sys
    from pathlib import Path
    # Get path to sibling folder
    sibling_path = Path(__file__).resolve().parent.parent / "models"
    sys.path.insert(0, str(sibling_path))
    from config import config
    from params import Params
    from models.AdaptiveLRLearner import AdaptiveLRLearner
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # !!! This code can handle factor10=10 and batch_size=1 at 75-second interval.

    factor10 = 2
    batch_size = 1         # it's also the data simplicity in this test.
    T1 = 24
    N = round(72 * factor10 * 144 * factor10 /16/16)
    seq_len = round(T1 * 72 * factor10 * 144 * factor10 /16/16)    # 97_200 for factor10 == 10
    nLayers = 5
    d_model = 1024
    nHeads = 128
    tokens_per_chunk = None
    heads_per_chunk = None
    kv_full_sequence = True
    lr = 1e-3 # 1e-3
    steps = 100

    layerFFN = "GLU"  # "GLU", "FFN"
    model = Flash2DEncLayer(d_model=d_model, nHeads=nHeads, heads_per_chunk=heads_per_chunk, dim_layerFFN=4*d_model, dropout=0.0, layerFFN=layerFFN, device=device)

    x = torch.randn(batch_size, seq_len, d_model)  # (B, S, E)

    # with autocast(device_type="cuda", dtype=torch.bfloat16):
    #     with torch.no_grad():
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
    x_train = torch.randn(batch_size, seq_len, d_model)
    y_train = torch.randn(batch_size, seq_len, d_model)

    learnLR = True

    if learnLR:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_controller = AdaptiveLRLearner(optimizer, accum_steps=5)


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
        
        # check_gradient(model)
        
        if learnLR:
            performed_step, lr, grad_norm = lr_controller.stepOptimizer(loss.item())
            # print(f"performed_step: {performed_step}, new_lr: {lr:.7f}")
        else:
            optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")


