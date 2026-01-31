import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.DeltaOperator import DeltaOperatorMultiChannel


# SwiGLU (unchanged)
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)
        return swiglu

# ---------------------------
# MLP (unchanged)
# ---------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LearnableEyeLayer(nn.Module):
    """
    Learnable identity initialization.
    Starts as identity, can learn small deviations.
    """
    def __init__(self, dim, init_scale=0.01):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.init_scale = init_scale
        
        # Add small noise to make it learnable
        with torch.no_grad():
            self.weight.data += init_scale * torch.randn_like(self.weight)
    
    def forward(self, x):
        # x: [..., dim]
        return x @ self.weight.T + self.bias

class PhysicalFlashAttention(nn.Module):
    def __init__(self, dim, H, W, nHeads=8, dropout=0.0, eps=1e-5):
        super().__init__()
        assert dim % nHeads == 0

        self.dim = dim
        self.H = H
        self.W = W
        self.nHeads = nHeads
        self.head_dim = dim // nHeads
        self.dropout = dropout

        # Norms
        self.norm_x = nn.LayerNorm(dim, eps=eps)
        self.norm_p = nn.LayerNorm(dim, eps=eps)

        # QK projections from model state
        self.q_proj = SwiGLU(dim) # nn.Linear(dim, dim)   # SwiGLU(dim)
        self.k_proj = SwiGLU(dim) # nn.Linear(dim, dim)   # SwiGLU(dim)

        # Output projection
        self.out_proj = LearnableEyeLayer(dim, init_scale=1.) # SwiGLU(dim) # Mlp(dim, dim*2, act_layer=torch.sigmoid, drop=0), nn.Linear(dim, dim), LearnableEyeLayer(dim, init_scale=1)
                            
        # Physics error corrector
        rate = 50   # time @ rate: 50.790 @ 10, 1.139 @ 50, 1.075 @ 100, 1.029 @ 200, 
        self.delta_operator = DeltaOperatorMultiChannel(
            H=H, W=W, C=dim, latent_rank=(H*W*dim)//rate, gate_init_scale=0.01
        )

    def _reshape_heads(self, x, B, N):
        return x.view(B, N, self.nHeads, self.head_dim)

    def forward(self, x, p):
        """
        x: (B, H, W, dim)   – model state
        p: (B, H, W, dim)   – physical prediction (x_phys)
        """
        B, H, W, _ = x.shape
        N = H * W

        # Flatten
        x = x.view(B, N, self.dim)
        p = p.view(B, N, self.dim)

        # Norm
        x = self.norm_x(x)
        p = self.norm_p(p)

        # Q, K
        q = self.q_proj(x) * (self.head_dim ** -0.5)
        k = self.k_proj(x)

        q = self._reshape_heads(q, B, N)
        k = self._reshape_heads(k, B, N)

        # ---- Physics correction ----
        p_corr = p + self.delta_operator(p.view(B, H, W, self.dim)).view(B, N, self.dim)

        # ---- V: use corrected physical directly ----
        v = p_corr.view(B, N, self.nHeads, self.head_dim)  # split dims for multi-head

        # Ensure all tensors have the same dtype for FlashAttention
        if q.dtype != v.dtype:
            # Cast v to match q's dtype (usually happens when norms are in float32)
            v = v.to(q.dtype)

        # FlashAttention
        att = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout,
            causal=False
        )

        # Merge heads
        att = att.reshape(B, N, self.dim)

        # Back to physical space
        out = self.out_proj(att)

        return out.view(B, H, W, self.dim)


def test_physical_flash_attention():
    assert torch.cuda.is_available(), "CUDA required for FlashAttention"

    device = torch.device("cuda")
    dtype = torch.bfloat16  # FlashAttention requirement

    # -------------------------
    # Dimensions
    # -------------------------
    B = 1
    H, W = 90, 180  # 120, 240
    dim = 4       # model state and physical channels
    nHeads = 2

    # -------------------------
    # Inputs
    # -------------------------
    x = torch.randn(
        B, H, W, dim,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    p = torch.randn(
        B, H, W, dim,
        device=device,
        dtype=dtype,
        requires_grad=True
    )

    # -------------------------
    # Module
    # -------------------------
    attn = PhysicalFlashAttention(
        dim=dim,
        H=H,
        W=W,
        nHeads=nHeads,
        dropout=0.0
    ).to(device=device)

    # Cast entire model to bfloat16
    attn = attn.to(dtype=dtype)
    
    # Simple optimizer for testing
    optimizer = torch.optim.AdamW(attn.parameters(), lr=1e-3)
    
    # Create CUDA events for synchronized timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up runs with autocast
    autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for _ in range(3):
        with torch.amp.autocast(autocast_device, dtype=dtype):
            y = attn(x, p)
            loss = y.mean()
        
        # Backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Clear input gradients for next iteration
        x.grad = None
        p.grad = None
    
    # Synchronize before starting timing
    torch.cuda.synchronize()
    
    # Start timing
    start_event.record()
    
    loops = 5
    for _ in range(loops):
        with torch.amp.autocast(autocast_device, dtype=dtype):
            # Forward
            y = attn(x, p)

            # Check shape
            assert y.shape == (B, H, W, dim), f"Output shape mismatch: {y.shape}"

            # Compute loss
            loss = y.mean()
        
        # Backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Clear input gradients for next iteration
        x.grad = None
        p.grad = None

    # End timing and synchronize
    end_event.record()
    torch.cuda.synchronize()
    
    # Calculate elapsed time in milliseconds
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / loops

    # Run one more forward pass for final checks
    with torch.amp.autocast(autocast_device, dtype=dtype):
        y = attn(x, p)
    
    # Do a final backward pass to check gradients
    loss = y.mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "x.grad is None"
    assert p.grad is not None, "p.grad is None"

    # DeltaOperator gradients
    delta_params = list(attn.delta_operator.parameters())
    assert any(p.grad is not None for p in delta_params), "Δ(x) has no gradients"
    
    # Check optimizer state
    print(f"Optimizer has {len(optimizer.param_groups)} parameter groups")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr={group['lr']}, params={len(group['params'])}")

    print("✅ PhysicalFlashAttention test passed")
    print(f"Using dtype: {dtype}")
    print(f"Output mean: {y.mean().item():.4f}")
    print(f"Output dtype: {y.dtype}")
    print(f"x.grad norm: {x.grad.norm().item():.4f}")
    print(f"p.grad norm: {p.grad.norm().item():.4f}")
    print(f"Total time: {elapsed_time_ms:.2f}ms")
    print(f"Time per forward+backward+step loop: {avg_time_ms:.2f}ms")
    
    # Print model parameters count
    total_params = sum(p.numel() for p in attn.parameters())
    trainable_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    test_physical_flash_attention()