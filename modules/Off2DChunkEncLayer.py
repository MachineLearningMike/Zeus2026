import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Off2DChunkMHA import Off2DChunkMHA
from modules.OffChunkFFN import FFN
from modules.OffChunkGLU import GLU


class Off2DChunkEncLayer(nn.Module):
    """
    implements:   
      x = x + dropout(MHA(norm1(x)))        # pre_LN to MHA
      x = norm2( x + dropout( ff(x) ) )     # standard post_LN to ff

    WHY?
    ff = "GLU" shows no significant advantage over "FFN" yet is slower, 
        although GLU is millions of times more efficient when used in isolation.
        
    Single encoder layer using OffChunkMHA, which stands for offloaded chunked MHA.
    Pre-LN style (LayerNorm before attention and before FFN).
    All weights start on CPU; moved temporarily to GPU during computation.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        causal: bool = False,
        tokens_per_chunk: int = None,
        heads_per_chunk: int = None,
        kv_full_sequence: bool = True,        
        ffn: Optional[str] = "FFN",  # "FFN", "GLU"
        eps: float = 1e-5,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)
        
        # FFN and LayerNorm: start on CPU
        self.norm1 = nn.LayerNorm(d_model, eps=eps)   

        # Offloaded attention (weights on CPU, chunks on GPU)
        self.self_attn = Off2DChunkMHA(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
            tokens_per_chunk=tokens_per_chunk,
            heads_per_chunk=heads_per_chunk,
            kv_full_sequence=kv_full_sequence,
            device=device,
        )

        if ffn == "GLU":
            self.ffn = GLU(in_features=d_model, hidden_features=dim_feedforward, out_features=d_model, act_layer=nn.SiLU, device=device)
        elif ffn == "FFN":
            self.ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward, activation=F.relu, device=device)
        else: assert False
        
        self.dropout = nn.Dropout(dropout)      # shared between mha and ffn
        self.norm2 = nn.LayerNorm(d_model, eps=eps)

        self.src_saved, self.mask_saved, self.attn_out_saved \
        = None, None, None

    #---------------------------- pre_mha --------------------------
    def apply_norm_to_src_before_mha(self, src, modules_to_cpu=True):
        # --- Pre-LN ---           
        norm1_gpu = self.norm1.to(self.gpu_device)
        src_ln = norm1_gpu(src.to(self.gpu_device))

        if modules_to_cpu:
            # move back to CPU
            norm1_gpu.to("cpu")
                
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
        norm2_gpu = self.norm2.to(self.gpu_device)
        src_out = norm2_gpu(src + ff)
        if modules_to_cpu:
            norm2_gpu.to("cpu")
        return src_out
    
    def modules_to_cpu_post_mha(self):
        self.ffn.modules_to_cpu()
        return

    def modules_to_cpu(self):
        self.modules_to_cpu_pre_mha()
        self.self_attn.modules_to_cpu()
        self.modules_to_cpu_post_mha()
        
    #-----------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,                       # (B, S, E)
        src_mask: Optional[torch.Tensor] = None, # (B, S) or (B, S, S)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
        train: bool = False,
        modules_to_cpu: bool = True
    ) -> torch.Tensor:
        
        # grad unit ------------------------------------------ apply_norm_to_src_before_mha ------------------------
        # input: src, output: src, src_ln
        # backward: grad_in: src_ln.grad, grad_out: src.grad + src_ln.grad

        # move norm1 weights temporarily to GPU
        if train:
            self.src_saved = src.to("cpu")
        src_ln = self.apply_norm_to_src_before_mha(src, modules_to_cpu=modules_to_cpu)
        mask = self.merge_mask(src_mask, src_key_padding_mask)

        # grad unit -------------------------------------------- MHA_attention -----------------------
        # input: src_ln, output: attn_out
        # backward: grad_in: attn_out.grad, grad_out: src_ln.grad

        if train:
            pass
            # !!! saving and reusing src_ln and mask are internal to self_attn.
        # --- Attention ---
        attn_out = self.self_attn.forward(src_ln, attn_mask=mask, train=train, modules_to_cpu=modules_to_cpu)  # output on GPU
        # Note: We have self.self.attn.backward defined, which leaves gradient on cpu.        
        del src_ln
        # gc.collect()
        # torch.cuda.empty_cache()
                
        # grad unit ------------------------ self.linear1, self.linear2, self.norm2 ------------------
        # input: src, attn_out, output: src_out
        # backward: grad_in: from outside, grad_out: src.grad (= res.grad), attn_out.grad (= res.grad)
        
        if train:
            self.attn_out_saved = attn_out.to("cpu") 
        res = self.apply_remaining_after_mha(src, attn_out)
        del attn_out
        gc.collect()
        torch.cuda.empty_cache()
        # --- FFN ---
        ff = self.ffn.forward(res, modules_to_cpu=modules_to_cpu)   #
        src_out = self.apply_post_LN_to_ff(res, ff, modules_to_cpu=modules_to_cpu)
        # No need to delete res and ff - operation is almost over.
        # Result stays on GPU
        return src_out
    
    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        initial: all submodules; norm1, norm2, linear1, and linear2 ; are on cpu
        return: gradient from self.src_saved through to gradient, on cpu
        state_change: gradinets are stored to parameters, for optmizer.step
        final: all submodules are back on cpu
        """
        assert self.src_saved is not None
        assert self.src_saved.shape == gradient.shape 

        # backward -------------------- post_mha --------------
        src_saved = self.src_saved.to(self.gpu_device)
        attn_out_saved = self.attn_out_saved.to(self.gpu_device)
        src_saved.requires_grad_(True)
        src_saved.retain_grad()
        attn_out_saved.requires_grad_(True)
        attn_out_saved.retain_grad()
        gradient = gradient.to(self.gpu_device)
        with torch.enable_grad():   # redundant for readability
            res = self.apply_remaining_after_mha(src_saved, attn_out_saved)
            ff = self.ffn(res, modules_to_cpu=False)
            src_out = self.apply_post_LN_to_ff(res, ff, modules_to_cpu=False)
            src_out.backward(gradient, retain_graph=False)  # gradients are saved to out_proj_gpu
        src_gradient = src_saved.grad
        attn_out_gradient = attn_out_saved.grad
        self.modules_to_cpu_post_mha()
        del res, ff, src_saved, attn_out_saved, src_out
        gc.collect()
        torch.cuda.empty_cache()

        # backward -------------------- mha --------------
        src_ln_gradient = self.self_attn.backward(attn_out_gradient)    # mask is interally used.
        
        # backward -------------------- pre_mha --------------
        src_saved = self.src_saved.to(self.gpu_device)
        attn_out_saved = self.attn_out_saved.to(self.gpu_device)
        src_saved.requires_grad_(True)
        src_saved.retain_grad()
        with torch.enable_grad():   # redundant for readability
            src_ln = self.apply_norm_to_src_before_mha(src_saved, modules_to_cpu=False)
            # No need: mask = self.merge_mask(self.src_mask_saved, self.src_key_padding_mask_saved)
            src_ln_gradient = src_ln_gradient.to(self.gpu_device)
            src_ln.backward(src_ln_gradient)
        src_gradient1 = src_saved.grad
        self.modules_to_cpu_pre_mha()
        del src_saved, src_ln
        gc.collect()
        torch.cuda.empty_cache()
        
        src_gradient = src_gradient + src_gradient1
        
        return src_gradient
    
if __name__ == "__main__":
    
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 1         # it's also the data simplicity in this test.
    T1 = 1
    seq_len = round(T1 * 72 * factor10 * 144 * factor10 /16/16)    # 97_200 for factor10 == 10
    num_layers = 5
    d_model = 1024
    num_heads = 128
    tokens_per_chunk = seq_len // 10
    heads_per_chunk = 32
    kv_full_sequence = True
    lr = 1e-4
    steps = 100

    ffn = "GLU"  # "GLU", "FFN"
    model = Off2DChunkEncLayer(d_model=d_model, num_heads=num_heads, heads_per_chunk=heads_per_chunk, dim_feedforward=4*d_model, dropout=0.0, ffn=ffn, device=device)

    x = torch.randn(batch_size, seq_len, d_model)  # (B, S, E)
    with torch.no_grad():
        out = model.forward(x, train=True, modules_to_cpu=True)
    print("Output:", out.shape, out.device)
    gradient = torch.ones_like(x)
    gradient = model.backward(gradient)
    print("Gradient:", gradient.shape, gradient.device)
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, d_model)
    y_train = torch.randn(batch_size, seq_len, d_model)

    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()

        # forward pass (offloads chunks to GPU internally)
        with torch.no_grad():
            out = model.forward(x_train, train=True, modules_to_cpu=True)
            
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