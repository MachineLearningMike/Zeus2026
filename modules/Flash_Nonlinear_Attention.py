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
    def __init__(self, dim, phys_channels, H, W, nHeads=8, dropout=0.0, eps=1e-5):
        super().__init__()
        assert dim % nHeads == 0

        self.dim = dim
        self.C = phys_channels
        self.H = H
        self.W = W
        self.nHeads = nHeads
        self.head_dim = dim // nHeads
        self.dropout = dropout

        # Norms
        self.norm_x = nn.LayerNorm(dim, eps=eps)
        self.norm_p = nn.LayerNorm(self.C, eps=eps)

        # QK from model state
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

        # --- PHYSICS-AWARE V PROJECTION (PER-HEAD, LOW-RANK) ---
        # (nHeads, C, head_dim)
        self.v_head_proj = nn.Parameter(
            torch.randn(nHeads, self.C, self.head_dim) * 0.02
        )

        # Output: bring attention result back to physical space
        self.out_proj = nn.Linear(dim, self.C)

        # Physics error corrector
        self.delta_operator = DeltaOperatorMultiChannel(
            H=H, W=W, C=self.C, latent_rank=16, gate_init_scale=0.01
        )

    def _reshape_heads(self, x, B, N):
        return x.view(B, N, self.nHeads, self.head_dim)

    def forward(self, x, p):
        """
        x: (B, H, W, dim)   – model state
        p: (B, H, W, C)     – physical prediction (x_phys)
        """
        B, H, W, _ = x.shape
        N = H * W

        # Flatten
        x = x.view(B, N, self.dim)
        p = p.view(B, N, self.C)

        # Norm
        x = self.norm_x(x)
        p = self.norm_p(p)

        # Q, K
        q = self.q_proj(x) * (self.head_dim ** -0.5)
        k = self.k_proj(x)

        q = self._reshape_heads(q, B, N)
        k = self._reshape_heads(k, B, N)

        # ---- Physics correction ----
        p_corr = p + self.delta_operator(p.view(B, H, W, self.C)).view(B, N, self.C)

        # ---- V: project physics directly into head space ----
        # v: (B, N, nHeads, head_dim)
        v = torch.einsum(
            "bnc,hcd->bnhd",
            p_corr,
            self.v_head_proj
        )

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

        return out.view(B, H, W, self.C)


if __name__ == "__main__":

    def test_physical_flash_attention():
        import torch
        from datetime import datetime

        assert torch.cuda.is_available(), "CUDA required for FlashAttention"

        device = torch.device("cuda")
        dtype = torch.bfloat16  # FlashAttention requirement

        # -------------------------
        # Dimensions
        # -------------------------
        B = 1
        H = 180
        W = 360
        dim = 10          # model state channels
        C = 1            # physical channels (e.g. u100, v100)
        nHeads = 5

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
            B, H, W, C,
            device=device,
            dtype=dtype,
            requires_grad=True
        )

        # -------------------------
        # Module
        # -------------------------
        attn = PhysicalFlashAttention(
            dim=dim,
            phys_channels=C,
            H=H,
            W=W,
            nHeads=nHeads,
            dropout=0.0
        ).to(device=device, dtype=dtype)

        import time
        s = time.time()
        loops = 5
        for _ in range(loops):
            # -------------------------
            # Forward
            # -------------------------
            y = attn(x, p)

            # -------------------------
            # Checks
            # -------------------------
            assert y.shape == (B, H, W, C), f"Output shape mismatch: {y.shape}"

            loss = y.mean()
            loss.backward()

            assert x.grad is not None, "x.grad is None"
            assert p.grad is not None, "p.grad is None"
        e = time.time()
        
        print(f"tike taken: {(e-s)/loops:.3f}")

        # DeltaOperator gradients
        delta_params = list(attn.delta_operator.parameters())
        assert any(p.grad is not None for p in delta_params), "Δ(x) has no gradients"

        print("✅ PhysicalFlashAttention test passed")
        print(f"Output mean: {y.mean().item():.4f}")
        print(f"x.grad norm: {x.grad.norm().item():.4f}")
        print(f"p.grad norm: {p.grad.norm().item():.4f}")

    test_physical_flash_attention()