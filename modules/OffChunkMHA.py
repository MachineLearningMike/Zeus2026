
import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.optim as optim
import time

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

class OffChunkMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        heads_per_chunk: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_head_chunks = num_heads // heads_per_chunk
        self.dropout = dropout

        assert heads_per_chunk > 0
        rem = num_heads % self.num_head_chunks
        self.head_chunks = [heads_per_chunk + (1 if i < rem else 0) for i in range(self.num_head_chunks)]
        
        print(self.head_chunks)

        # qkv per-chunk linears on CPU
        self.qkv_linears = nn.ModuleList()
        for heads in self.head_chunks:
            d_local = heads * self.head_dim
            lin = nn.Linear(embed_dim, 3 * d_local, bias=bias)
            self.qkv_linears.append(lin.cpu())

        # out proj kept on CPU until used
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias).cpu()
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.input_saved = None
        self.mask_saved = None
        self.concat_saved = None


    def foreward_attention(self, src, mask, idx, heads, modules_to_cpu=False):
        assert src.device == self.gpu_device
        
        d_local = heads * self.head_dim
        B, S, E = src.shape

        # Bring chunk qkv to GPU
        lin = self.qkv_linears[idx].to(self.gpu_device)     # The content tensors of self.qkv_linears[idx] are moved to gpu, down-referencing tensors on cpu.
        qkv = lin(src)  # (B, S, 3 * d_local)
        del src                   # src is now down-referenced on gpu.
        
        # Move qkv linear back to CPU (weights) to free GPU memory        
        if modules_to_cpu:
            lin.to("cpu")   # The content tensors of lin, so, of self.qkv_linears[idx], are moved to cpu, down-referencing the tensors on gpu.

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        del qkv
        q = q.view(B, S, heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.view(B, S, heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, heads, self.head_dim).transpose(1, 2)

        # Release the gpu memory used by lin tensors and qkv to the cuda driver, as they are now zero-referenced.
        gc.collect()                # de-allocate zero-referenced tensors. The memory used by the de-allocated cpu / gpu tensor is returned to the system / torch
        torch.cuda.empty_cache()    # Torch returns the memory used by the de-allocated gpu tensor to the cuda driver.

        # !!! THIS is the place where the memory footprint reaches maximum.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, S, S)
        del q, k    # delelet as much and early as possible.

        gc.collect()                # de-allocate zero-referenced tensors. The memory used by the de-allocated cpu / gpu tensor is returned to the system / torch
        torch.cuda.empty_cache()    # Torch returns the memory used by the de-allocated gpu tensor to the cuda driver.

        if mask is not None:
            mask_dev = mask.to(self.gpu_device)
            if mask_dev.dtype == torch.bool:
                # assume mask shape (B, S) -> keys valid; or (B, S, S)
                if mask_dev.ndim == 2:
                    attn_logits = attn_logits.masked_fill(~mask_dev.unsqueeze(1).unsqueeze(2), float("-inf"))
                else:
                    attn_logits = attn_logits.masked_fill(~mask_dev.unsqueeze(1), float("-inf"))
            else:
                # additive mask (float) broadcastable
                attn_logits = attn_logits + mask_dev.unsqueeze(1)

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, S, head_dim)

        # No mode need to clean memory, because the remaining operation has no more memory footprint.
        # del v, attn_logits, attn
        # gc.collect()
        # torch.cuda.empty_cache()
        
        out = out.transpose(1, 2).contiguous().view(B, S, d_local)  # (B, S, d_local)
    
        return out
    
    def modules_to_cpu_foreward_attention(self, idx):
        self.qkv_linears[idx].to("cpu")
        return
    
    def modules_to_cpu(self):
        self.out_proj.to("cpu")
        for idx, _ in enumerate(self.head_chunks):
            self.modules_to_cpu_foreward_attention(idx)
        return

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, train: bool = False, modules_to_cpu=True) -> torch.Tensor:
        """
        x: (B, S, E) - input may be on CPU or GPU; chunk processing moves parts to GPU.
        mask: optional mask broadcastable to (B, 1, S, S) or (S, S) or boolean key mask shape (B, S)
        returns: tensor on gpu_device with shape (B, S, E)
        """
        #----------------------------------------------- from input
        if train:
            self.input_saved = x.to("cpu")
            if mask is not None:
                self.mask_saved = mask.to("cpu")
        
        B, S, E = x.shape
        assert E == self.embed_dim

        per_chunk_outputs = []

        # Keep original input on GPU for chunk computation to avoid repeated input transfers:
        x_dev = x.to(self.gpu_device)

        for idx, heads in enumerate(self.head_chunks):
            # This line requires maximum memory footprint and leaves some tensors allocated.
            out = self.foreward_attention(x_dev, mask, idx, heads, modules_to_cpu=modules_to_cpu)
            gc.collect()                # Return memory used by zero-referenced cpu / gpu 
            torch.cuda.empty_cache()
            per_chunk_outputs.append(out)

        # concat on cuda
        concat = torch.cat(per_chunk_outputs, dim=-1)  # (B, S, E) on CPU
        del per_chunk_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        #------------------------------------------------ from concat
        if train:
            self.concat_saved = concat.to("cpu")

        # final projection: move out_proj to GPU, apply, then keep result on GPU
        out_proj_gpu = self.out_proj.to(self.gpu_device)
        out = out_proj_gpu(concat)
        # move out_proj weights back to CPU to keep GPU footprint small
        if modules_to_cpu:
            self.out_proj.to("cpu")
        #------------------------------------------------- out

        return out  # on gpu_device
    
    def backward(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        initial: all submodules, self.out_proj and self.qkv_linears, are on cpu
        return: gradient from self.input_saved through to gradient, on cpu
        state_change: gradinets are stored to parameters, for optmizer.step
        final: all submodules are back on cpu
        """
        assert self.input_saved is not None
        assert self.input_saved.shape == gradient.shape        

        #-------------------------------------------- back to concat

        # Commenting out this block let error diverges from 1 to 40 for 100 loops, 
        # instead of converging from 1 to 0.294.
        concat = self.concat_saved.to(self.gpu_device)
        concat.requires_grad_(True)
        # concat.retain_grad()  # No need, it's a leaf tensor.
        out_proj_gpu = self.out_proj.to(self.gpu_device)
        gradient = gradient.to(self.gpu_device)
        # with torch.enable_grad():   # redundant for readability
        out = out_proj_gpu(concat)
        out.backward(gradient, retain_graph=False)  # gradients are saved to out_proj_gpu
        gradient = concat.grad
        out_proj_gpu.to("cpu")  # This line can be commented out. why?
        del concat, out
        gc.collect()
        torch.cuda.empty_cache()
        
        #----------------------------------------------- back to input
        
        gradient = gradient.to(self.gpu_device)     # redundant, deliberately, for readability
        per_chunk_gradient = []
        gradient_chunks = torch.chunk(gradient, self.num_head_chunks, dim=-1)
        
        # Keep original input on GPU for chunk computation to avoid repeated input transfers:
        input = self.input_saved.to(self.gpu_device)
        input.requires_grad_(True)
        # input.retain_grad()   # No need, it's a leaf tensor.
        mask = self.mask_saved.to(self.gpu_device) if self.mask_saved is not None else None

        # Commenting out this block reduces loss from 1 to 0.967, not to 0.294, for 100 loops.
        for idx, heads in enumerate(self.head_chunks):
            # with torch.enable_grad():   # redundant for readability
            out = self.foreward_attention(input, mask, idx, heads, modules_to_cpu=False)
            out.backward(gradient_chunks[idx], retain_graph=False)
            grad = input.grad
            per_chunk_gradient.append(grad.to("cpu"))
            self.modules_to_cpu_foreward_attention(idx)
            del out, grad
            gc.collect()
            torch.cuda.empty_cache()
        gradient = torch.stack(per_chunk_gradient, dim=0).sum(dim=0)    # Whatever head chunking should give the same sum. right?
        
        return gradient


if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 1         # it's also the data simplicity in this test.
    seq_len = round(1 * 72 * factor10 * 144 * factor10 /16/16)    # 97_200 for factor10 == 10
    embed_dim = 1024
    num_heads = 128
    seq_per_chunk = seq_len // 10
    heads_per_chunk = 32
    kv_full_sequence = True
    lr = 1e-3
    steps = 100
    
    model = OffChunkMHA(embed_dim=embed_dim, num_heads=num_heads, heads_per_chunk=heads_per_chunk, device=device)

    #------------------- simple test -------------------
    x = torch.randn(batch_size, seq_len, embed_dim)  # (B, S, E)
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
    x_train = torch.randn(batch_size, seq_len, embed_dim)
    y_train = torch.randn(batch_size, seq_len, embed_dim)

    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()

        # with autocast(device_type='cuda', dtype=torch.bfloat16):
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

        gradient = model.backward(gradient) # modules are on cpu.
        optimizer.step()
        
        print(f"Step {step+1}/{steps} - Loss: {loss.item():.10f}")
        
    end = time.time()
    duration = end - start
    print(f"Training done. time taken: {duration: .3f} seconds")
    