import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
from torch.amp import autocast, GradScaler
import torch.optim as optim
import time

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import check_dir, check_gradient

"""
Standard MHA does:
scores = Q @ K^T / sqrt(d_k)          # (B, H, Lq, Lk)
attn = softmax(scores)                # apply mask, dropout
out = attn @ V

This requires storing scores, which is size (B, H, Lq, Lk) — huge for long sequences.
E.g., B=1, H=16, L=8192, d_k=64 → scores = 8 GB in float32!

FlashAttention solves this by:
Computing softmax in a streaming fashion
Only a small chunk of QK^T is in memory at any time.
Fusing operations
QK^T → masked softmax → dropout → multiply with V in one kernel.
Supporting fp16/bf16
Reduces memory footprint without significant precision loss.
Optional causal masking
Handles autoregressive models efficiently.

Important constraints:
Inputs must be fp16 or bf16 (cannot use float32).
Must run on CUDA (GPU).
Supports optional causal masking for autoregressive models

NOTE: FlashAttention shines with long sequences!!!!!!!!!!

"""

class Flash2DMHA(nn.Module):
    """
    FlashAttention-2 (Windows) MHA with:
      - head chunking (nHeads split evenly)
      - sequence chunking (local attention per sequence chunk by default)
      - optional kv_full_sequence (if True, q-chunk attends to full k/v)
      - autocast for mixed precision
    Requirements:
      - nHeads % heads_per_chunk == 0 (if heads_per_chunk provided)
      - flash_attn_func is callable (Windows wheel)
    """
    def __init__(self, d_model, nHeads, dropout=0.0, causal=False,
                 tokens_per_chunk=None, heads_per_chunk=None, kv_full_sequence=True, device="cuda:0"):
        super().__init__()
        assert d_model % nHeads == 0, "d_model must be divisible by nHeads"
        if heads_per_chunk is not None:
            assert nHeads % heads_per_chunk == 0, "heads_per_chunk must divide nHeads"

        self.d_model = d_model
        self.nHeads = nHeads 
        self.head_dim = d_model // nHeads
        self.dropout = dropout
        self.causal = causal
        self.device = device

        # chunking params
        self.tokens_per_chunk = tokens_per_chunk
        self.heads_per_chunk = heads_per_chunk or nHeads  # default: no head chunking
        self.kv_full_sequence = kv_full_sequence

        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward_attention(self, q_h, k_h, v_h, s_start, s_end, attn_mask):
                
        # slice query in sequence
        q_seq = q_h[:, :, s_start:s_end, :]  # [B, heads_per_chunk, Lc, head_dim]

        # choose k_seq/v_seq: either full sequence (kv_full_sequence) or the same window
        if self.kv_full_sequence:
            k_seq = k_h  # [B, heads_per_chunk, L, head_dim]
            v_seq = v_h
            # If attn_mask is provided and is full LxL, slice columns accordingly
            mask_chunk = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:  # [B, L] padding
                    # keys dimension across full sequence -> use as [B,1,1,L]
                    mask_chunk = attn_mask.unsqueeze(1).unsqueeze(1)
                elif attn_mask.dim() == 3:  # [B, L, L]
                    mask_chunk = attn_mask[:, s_start:s_end, :].unsqueeze(1)  # [B,1,Lc,L]
                elif attn_mask.dim() == 4:  # [B,1,L,L]
                    mask_chunk = attn_mask[:, :, s_start:s_end, :]
        else:
            # local attention: keys/values are same slice as queries
            k_seq = k_h[:, :, s_start:s_end, :]
            v_seq = v_h[:, :, s_start:s_end, :]
            # mask slice for local window
            mask_chunk = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:  # [B, L] padding
                    mask_chunk = attn_mask[:, s_start:s_end].unsqueeze(1).unsqueeze(1)
                elif attn_mask.dim() == 3:  # [B, L, L] -> take submatrix
                    mask_chunk = attn_mask[:, s_start:s_end, s_start:s_end].unsqueeze(1)
                elif attn_mask.dim() == 4:  # [B, 1, L, L]
                    mask_chunk = attn_mask[:, :, s_start:s_end, s_start:s_end]

        # call flash_attn_func (it expects [B, L_q, H, head_dim] or [B, H, L, head_dim]?
        # Our wheel expects [B, L, nheads, head_dim], so transpose to [B, Lc, heads_per_chunk, head_dim]
        # Current q_seq shape: [B, nHeads, Lc, head_dim] -> permute to [B, Lc, nHeads, head_dim]
        q_seq_for = q_seq.permute(0, 2, 1, 3).contiguous()
        k_seq_for = k_seq.permute(0, 2, 1, 3).contiguous()
        v_seq_for = v_seq.permute(0, 2, 1, 3).contiguous()

        # This is required to spare memory space before calling attn_func, which takes huge memory.        
        del q_seq, k_seq, v_seq
        if attn_mask is not None: del attn_mask

        # call FlashAttention (pass mask_chunk if available)
        # signature differs across builds; this variant uses keyword args for clarity
        if mask_chunk is not None:
            out_seq = flash_attn_func(q_seq_for, k_seq_for, v_seq_for,
                                        attn_mask=mask_chunk,
                                        dropout_p=self.dropout,
                                        softmax_scale=None,
                                        causal=self.causal)
        else:
            out_seq = flash_attn_func(q_seq_for, k_seq_for, v_seq_for,
                                        dropout_p=self.dropout,
                                        softmax_scale=None,
                                        causal=self.causal)
            
        return out_seq
        
    def forward(self, input, attn_mask=None):
        # return input    # NoMHA
        
        """
        x: [B, L, D]
        attn_mask: optional boolean mask. Supports shapes:
                   - [B, L] (padding mask)
                   - [B, L, L]
                   - [B, 1, L, L]
        Returns: [B, L, D]
        """
        # No reason not to load the whole module, because
        # its effect on memory footprint is very small compared to the effect ot the sequence length of data.
        self.to(self.device)
        
        B, L, D = input.shape
        assert D == self.d_model
        #---------------------------------------------------------------------------------------    from input
        
        # Project QKV under autocast so weights/inputs are cast consistently
        qkv = self.W_qkv(input)  # [B, L, 3*D], weights/autocast -> fp16 for kernel
        qkv = qkv.view(B, L, self.nHeads, 3 * self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        # split into q,k,v: each [B, H, L, head_dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Prepare head splitting (split along H dimension). Each split has identical head count.
        heads_per_chunk = self.heads_per_chunk
        # torch.split returns equal-sized chunks because we asserted divisibility
        q_head_splits = list(q.split(heads_per_chunk, dim=1))
        k_head_splits = list(k.split(heads_per_chunk, dim=1))
        v_head_splits = list(v.split(heads_per_chunk, dim=1))

        # Sequence chunking iterator
        seq_ranges = [(0, L)] if self.tokens_per_chunk is None else [(i, min(i + self.tokens_per_chunk, L)) for i in range(0, L, self.tokens_per_chunk)]

        out_head_chunks = []  # will collect per-head-chunk outputs (for all sequence positions)
        # For each head chunk, compute attention over the whole sequence by concatenating sequence-chunk outputs
        for h_idx, (q_h, k_h, v_h) in enumerate(zip(q_head_splits, k_head_splits, v_head_splits)):
            # q_h/k_h/v_h shapes: [B, heads_per_chunk, L, head_dim]
            seq_out_parts = []
            for s_idx, (s_start, s_end) in enumerate(seq_ranges):
                                
                out_seq = self.forward_attention(q_h, k_h, v_h, s_start, s_end, attn_mask)

                # out_seq: [B, tokens_per_chunk, heads_per_chunk, head_dim] -> append
                seq_out_parts.append(out_seq)
        
            # concat sequence chunks back along sequence dim -> [B, L, heads_per_chunk, head_dim]
            out_h = torch.cat(seq_out_parts, dim=1)     # tracking: (B, s_idx * tokens_per_chunk : (s_idx+1) * tokens_per_chunk, heads_per_chunk, head_dim )
            out_head_chunks.append(out_h)

        # concat head chunks along head dim -> [B, L, H, head_dim]
        out_all = torch.cat(out_head_chunks, dim=2)
        # tracking: (B, s_idx * tokens_per_chunk : (s_idx+1) * tokens_per_chunk, h_idx * heads_per_chunk : (h_idx+1) * heads_per_chunk, head_dim)

        # permute and merge nHeads -> [B, L, D]
        out_final = out_all.contiguous().view(B, L, D)
        # tracking: G(s_idx, h_idx) = out_final[B, s_idx * tokens_per_chunk : (s_idx+1) * tokens_per_chunk, h_idx * heads_per_chunk * head_dim : (h_idx+1) * heads_per_chunk * head_dim]
        
        # SO, each of (q, k, v) is mapped to G(s_idx, h_idx).     q.Grad = sum_{s_idx, h_idx}{backward(G(s_idx, h_idx).Grad)}
        # = sum_{s_idx, h_idx}{out_final.grad[B, s_idx * tokens_per_chunk : (s_idx+1) * tokens_per_chunk, h_idx * heads_per_chunk * head_dim : (h_idx+1) * heads_per_chunk * head_dim]}
        
        #-------------------------------------------------------------------------     from out_final
        # final output projection (you can wrap in autocast as needed)
        out_proj = self.W_o(out_final)

        return out_proj

if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # B, L, D = 100_000, 72, 768
    # x = torch.randn(B, L, D, dtype=torch.float32, device=device, requires_grad=True)

    # model = Flash2DMHA(d_model=1024, nHeads=8, dropout=0.0, causal=False,
    #                     tokens_per_chunk=10000, heads_per_chunk=4).cuda()
    # out = model(x)
    # loss = out.sum()
    # loss.backward()
    # grad = x.grad
    # print(grad.shape, grad.dtype)


    #------------------ training test --------------------
    
    batch_size = 1000         # it's also the data simplicity in this test.
    seq_len = 72
    d_model = 768
    nHeads = 16
    tokens_per_chunk = None
    heads_per_chunk = None
    kv_full_sequence = True
    lr = 1e-3
    steps = 100

    model = Flash2DMHA(d_model=d_model, nHeads=nHeads, dropout=0.0, causal=False,
                        tokens_per_chunk=tokens_per_chunk, heads_per_chunk=heads_per_chunk, kv_full_sequence=kv_full_sequence).cuda()
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, d_model).to(device)
    y_train = torch.randn(batch_size, seq_len, d_model).to(device)

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
                out = model(x_train)

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

        optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")