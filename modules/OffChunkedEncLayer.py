import math
from typing import Optional, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from OffChunkedMHA import OffChunkedMHA


class OffChunkedEncLayer(nn.Module):
    """
    Single encoder layer using OffloadedChunkedMHA.
    Pre-LN style (LayerNorm before attention and before FFN).
    All weights start on CPU; moved temporarily to GPU during computation.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_chunks: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable = F.relu,
        device: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)
        
        # FFN and LayerNorm: start on CPU
        self.norm1 = nn.LayerNorm(embed_dim, eps=eps)   

        # Offloaded attention (weights on CPU, chunks on GPU)
        self.self_attn = OffChunkedMHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_chunks=num_chunks,
            dropout=dropout,
            device=device,
        )

        # FFN and LayerNorm: start on CPU
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)   
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.src_saved, self.src_ln_saved, self.mask_saved, self.attn_out_saved, self.src_mask_saved, self.src_key_padding_mask_saved \
        = None, None, None, None, None, None

    #---------------------------- pre_mha --------------------------
    def norm1_and_merge_mask(self, src, src_mask, src_key_padding_mask, modules_to_cpu=True):
        # --- Pre-LN ---           
        norm1_gpu = self.norm1.to(self.gpu_device)
        src_ln = norm1_gpu(src.to(self.gpu_device))

        if modules_to_cpu:
            # move back to CPU
            norm1_gpu.to("cpu")

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
                
        return src_ln, mask
    
    def modules_to_cpu_pre_mha(self):
        self.norm1.to("cpu")
        return

    #---------------------------- post_mha --------------------------
    def attn_dropout_residual(self, src, attn_out):
        # return res on self.gpu_device
        assert attn_out.device == self.gpu_device
        
        attn_out = self.dropout(attn_out)
        # Residual add: move src to GPU
        res = src.to(self.gpu_device) + attn_out
        return res

    def ffn(self, res, modules_to_cpu=True):
        # return ff on self.gpu_device        
        # Move FFN weights to GPU temporarily
        linear1_gpu = self.linear1.to(self.gpu_device)
        linear2_gpu = self.linear2.to(self.gpu_device)
        ff = linear2_gpu(self.activation(linear1_gpu(res)))
        if modules_to_cpu:
            linear1_gpu.to("cpu")
            linear2_gpu.to("cpu")        
        return ff
    
    def ff_dropout_norm2_residual(self, res, ff, modules_to_cpu=True):
        # return src_out on self.gpu_device
        ff = self.dropout(ff)
        # --- Post-LN ---
        norm2_gpu = self.norm2.to(self.gpu_device)
        src_out = norm2_gpu(res + ff)
        if modules_to_cpu:
            norm2_gpu.to("cpu")
        return src_out
    
    def modules_to_cpu_post_mha(self):
        self.linear1.to("cpu")
        self.linear2.to("cpu")
        self.norm2.to("cpu")
        return

    #-----------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,                       # (B, S, E)
        src_mask: Optional[torch.Tensor] = None, # (B, S) or (B, S, S)
        src_key_padding_mask: Optional[torch.Tensor] = None,  # (B, S)
        train: bool = False,
        modules_to_cpu: bool = True
    ) -> torch.Tensor:
        
        # grad unit ------------------------------------------ norm1_and_merge_mask ------------------------
        # input: src, output: src, src_ln
        # backward: grad_in: src_ln.grad, grad_out: src.grad + src_ln.grad

        # move norm1 weights temporarily to GPU
        if train:
            self.src_saved = src.to("cpu")
            if src_mask is not None:
                self.src_mask_saved = src_mask.to("cpu")
            if src_key_padding_mask is not None:
                self.src_key_padding_mask_saved = src_key_padding_mask.to("cpu")

        src_ln, mask = self.norm1_and_merge_mask(src, src_mask, src_key_padding_mask, modules_to_cpu=modules_to_cpu)

        # grad unit -------------------------------------------- MHA_attention -----------------------
        # input: src_ln, output: attn_out
        # backward: grad_in: attn_out.grad, grad_out: src_ln.grad

        if train:
            self.src_ln_saved = src_ln.to("cpu")
            # saving and reusing mask is internal to self_attn.
            
        # --- Attention ---
        attn_out = self.self_attn.forward(src_ln, mask=mask, train=train)  # output on GPU
        # Note: We have self.self.attn.backward defined, which leaves gradient on cpu.
        
        del src_ln
        torch.cuda.empty_cache()
                
        # grad unit ------------------------ self.linear1, self.linear2, self.norm2 ------------------
        # input: src, attn_out, output: src_out
        # backward: grad_in: from outside, grad_out: src.grad (= res.grad), attn_out.grad (= res.grad)
        
        if train:
            self.attn_out_saved = attn_out.to("cpu")
 
        # --- attn_dropout_residual ---
        res = self.attn_dropout_residual(src, attn_out)

        del attn_out
        torch.cuda.empty_cache()


        # --- FFN ---
        ff = self.ffn(res, modules_to_cpu=modules_to_cpu)
        # keep res

        src_out = self.ff_dropout_norm2_residual(res, ff, modules_to_cpu=modules_to_cpu)
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
        with torch.enable_grad():   # redundant for readability
            # --- attn_dropout_residual ---
            res = self.attn_dropout_residual(src_saved, attn_out_saved)
            # --- FFN ---
            ff = self.ffn(res, modules_to_cpu=False)
            # keep res
            src_out = self.ff_dropout_norm2_residual(res, ff, modules_to_cpu=False)
            del res, ff
            torch.cuda.empty_cache()
            gradient = gradient.to(self.gpu_device)
            src_out.backward(gradient, retain_graph=False)  # gradients are saved to out_proj_gpu
        src_gradient = src_saved.grad
        attn_out_gradient = attn_out_saved.grad
        del src_saved, attn_out_saved, src_out
        torch.cuda.empty_cache()
        self.modules_to_cpu_post_mha()

        # backward -------------------- mha --------------
        src_ln_gradient = self.self_attn.backward(attn_out_gradient)    # mask is interally used.
        
        # backward -------------------- pre_mha --------------
        src_saved = self.src_saved.to(self.gpu_device)
        attn_out_saved = self.attn_out_saved.to(self.gpu_device)
        src_saved.requires_grad_(True)
        src_saved.retain_grad()
        with torch.enable_grad():   # redundant for readability
            src_ln, _ = self.norm1_and_merge_mask(src_saved, self.src_mask_saved, self.src_key_padding_mask_saved, modules_to_cpu=False)
            src_ln_gradient = src_ln_gradient.to(self.gpu_device)
            src_ln.backward(src_ln_gradient)
        src_gradient1 = src_saved.grad
        del src_saved, src_ln
        torch.cuda.empty_cache()
        self.modules_to_cpu_pre_mha()
        
        src_gradient = src_gradient + src_gradient1
        
        return src_gradient
    
if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 24
    seq_len = round(72 * factor10 * 144 * factor10 /16/16)
    embed_dim = 1024

    model = OffChunkedEncLayer(embed_dim=embed_dim, num_heads=8, num_chunks=4, dim_feedforward=4*embed_dim, dropout=0.0, activation=F.relu, device=device)

    x = torch.randn(batch_size, seq_len, embed_dim)  # (B, S, E)
    out = model(x)
    print("Output:", out.shape, out.device)

    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, embed_dim)
    y_train = torch.randn(batch_size, seq_len, embed_dim)

    steps = 100
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