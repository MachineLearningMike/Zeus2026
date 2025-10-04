
import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OffChunkMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_chunks: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_chunks = num_chunks
        self.dropout = dropout

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)

        base = num_heads // num_chunks
        assert base > 0
        rem = num_heads % num_chunks
        self.heads_per_chunk = [base + (1 if i < rem else 0) for i in range(num_chunks)]

        # qkv per-chunk linears on CPU
        self.qkv_linears = nn.ModuleList()
        for heads in self.heads_per_chunk:
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


    def foreward_attention(self, x_dev, mask, idx, heads, modules_to_cpu=False):
        assert x_dev.device == self.gpu_device
        
        d_local = heads * self.head_dim
        B, S, E = x_dev.shape

        # Bring chunk qkv to GPU
        lin = self.qkv_linears[idx].to(self.gpu_device)
        qkv = lin(x_dev)  # (B, S, 3 * d_local)
        del x_dev
        torch.cuda.empty_cache()
        
        # Move qkv linear back to CPU (weights) to free GPU memory        
        if modules_to_cpu:
            lin.to("cpu")
            torch.cuda.empty_cache()

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        del qkv
        torch.cuda.empty_cache()
        
        q = q.view(B, S, heads, self.head_dim).transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.view(B, S, heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, heads, self.head_dim).transpose(1, 2)

        # This is the place where the memory footprint reaches maximum.
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, S, S)
        del q, k    # delelet as much and early as possible.
        torch.cuda.empty_cache()

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
        # del v, attn
        # torch.cuda.empty_cache()
        
        out = out.transpose(1, 2).contiguous().view(B, S, d_local)  # (B, S, d_local)
    
        return out
    
    def modules_to_cpu_foreward_attention(self, idx):
        self.qkv_linears[idx].to("cpu")
        return

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, train: bool = False) -> torch.Tensor:
        """
        x: (B, S, E) - input may be on CPU or GPU; chunk processing moves parts to GPU.
        mask: optional mask broadcastable to (B, 1, S, S) or (S, S) or boolean key mask shape (B, S)
        returns: tensor on gpu_device with shape (B, S, E)
        """
        if train:
            self.input_saved = x.to("cpu")
            if mask is not None:
                self.mask_saved = mask.to("cpu")
        
        B, S, E = x.shape
        assert E == self.embed_dim

        per_chunk_outputs = []

        # Keep original input on GPU for chunk computation to avoid repeated input transfers:
        x_dev = x.to(self.gpu_device)

        for idx, heads in enumerate(self.heads_per_chunk):
            
            out = self.foreward_attention(x_dev, mask, idx, heads, modules_to_cpu=True)
            torch.cuda.empty_cache()
            per_chunk_outputs.append(out.to("cpu"))

        # concat on CPU
        concat = torch.cat(per_chunk_outputs, dim=-1)  # (B, S, E) on CPU
        
        if train:
            self.concat_saved = concat.to("cpu")

        # final projection: move out_proj to GPU, apply, then keep result on GPU
        out_proj_gpu = self.out_proj.to(self.gpu_device)
        out = out_proj_gpu(concat.to(self.gpu_device))
        # move out_proj weights back to CPU to keep GPU footprint small
        out_proj_gpu.to("cpu")
        del concat
        torch.cuda.empty_cache()

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

        B, S, E = gradient.shape

        # Commenting out this block let error diverges from 1 to 40 for 100 loops, 
        # instead of converging from 1 to 0.294.
        x_dev = self.concat_saved.to(self.gpu_device)
        x_dev.requires_grad_(True)
        x_dev.retain_grad()
        out_proj_gpu = self.out_proj.to(self.gpu_device)
        with torch.enable_grad():   # redundant for readability
            out = out_proj_gpu(x_dev)
            gradient = gradient.to(self.gpu_device)
            out.backward(gradient, retain_graph=False)  # gradients are saved to out_proj_gpu
        gradient = x_dev.grad
        out_proj_gpu.to("cpu")  # This line can be commented out. why?
        del x_dev, out
        torch.cuda.empty_cache()
        
        gradient = gradient.to(self.gpu_device)     # redundant, deliberately.
        per_chunk_gradient = []
        gradient_chunks = torch.chunk(gradient, self.num_chunks, dim=-1)
        
        # Keep original input on GPU for chunk computation to avoid repeated input transfers:
        x_dev = self.input_saved.to(self.gpu_device)
        x_dev.requires_grad_(True)
        x_dev.retain_grad()

        # Commenting out this block reduces loss from 1 to 0.967, not to 0.294, for 100 loops.
        for idx, heads in enumerate(self.heads_per_chunk):
            with torch.enable_grad():   # redundant for readability
                out = self.foreward_attention(x_dev, self.mask_saved, idx, heads, modules_to_cpu=False)
                torch.cuda.empty_cache()
                out.backward(gradient_chunks[idx], retain_graph=False)
            self.modules_to_cpu_foreward_attention(idx)
            grad = x_dev.grad
            per_chunk_gradient.append(grad.to("cpu"))
            del out, grad
            torch.cuda.empty_cache()
        gradient = torch.stack(per_chunk_gradient, dim=0).sum(dim=0)
        
        return gradient


if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 24         # it's also the data simplicity in this test.
    seq_len = round(1 * 72 * factor10 * 144 * factor10 /16/16)
    embed_dim = 1024
    num_heads = 8
    num_chunks = 4
    lr = 1e-3
    steps = 500

    model = OffChunkMHA(embed_dim=embed_dim, num_heads=num_heads, num_chunks=num_chunks, device=device)

    #------------------- simple test -------------------
    x = torch.randn(batch_size, seq_len, embed_dim)  # (B, S, E)
    out = model(x, train=True)
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

    for step in range(steps):
        optimizer.zero_grad()

        # forward pass (offloads chunks to GPU internally)
        with torch.no_grad():
            out = model.forward(x_train, train=True)
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with torch.enable_grad():   # redundant for readability
            loss = criterion(out, y_train.to(out.device))
            loss.backward()
        gradient = out.grad

        gradient = model.backward(gradient)

        optimizer.step()

        print(f"Step {step+1}/{steps} - Loss: {loss.item():.6f}")

    print("Training done.")
    