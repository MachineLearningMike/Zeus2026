import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn import flash_attn_func

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.DeltaOperator import DeltaOperatorMultiChannel

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
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Physics error corrector
        self.delta_operator = DeltaOperatorMultiChannel(
            H=H, W=W, C=dim, latent_rank=16, gate_init_scale=0.01
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
    H = 120
    W = 240
    dim = 6       # model state and physical channels
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
    ).to(device=device, dtype=dtype)

    import time
    s = time.time()
    loops = 5
    for _ in range(loops):
        # Forward
        y = attn(x, p)

        # Check shape
        assert y.shape == (B, H, W, dim), f"Output shape mismatch: {y.shape}"

        # Backward
        loss = y.mean()
        loss.backward()

        assert x.grad is not None, "x.grad is None"
        assert p.grad is not None, "p.grad is None"

    e = time.time()

    # DeltaOperator gradients
    delta_params = list(attn.delta_operator.parameters())
    assert any(p.grad is not None for p in delta_params), "Δ(x) has no gradients"

    print("✅ PhysicalFlashAttention test passed")
    print(f"Output mean: {y.mean().item():.4f}")
    print(f"x.grad norm: {x.grad.norm().item():.4f}")
    print(f"p.grad norm: {p.grad.norm().item():.4f}")
    print(f"Time per forward+backward loop: {(e-s)/loops:.3f}s")

if __name__ == "__main__":
    test_physical_flash_attention()