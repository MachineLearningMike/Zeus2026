import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from flash_attn import flash_attn_func

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.DeltaOperator import DeltaOperator


class SwiGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        return x1 * F.silu(x2) * math.sqrt(2)

class TanGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        return x1 * F.tanh(x2) * math.sqrt(2)

class PhysicalFlashAttention(nn.Module):
    """
    Multi-head Physical Attention:
      q = Q(p) * Em
      k = K(x)
      v = V(x)   (nonlinear)
      out = O(att)
    """

    def __init__(self, dim, H, W, nHeads=8, dropout=0.0, eps=1e-5):
        super().__init__()
        assert dim % nHeads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = nHeads
        self.head_dim = dim // nHeads
        self.dropout = dropout

        # LayerNorm for input features
        self.norm_x = nn.LayerNorm(dim, eps=eps)
        self.norm_p = nn.LayerNorm(dim, eps=eps)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = TanGLU(dim, dim)          # Aggressive all-directional non-linearity
        self.out_proj = TanGLU(dim, dim)        # Aggressive all-directional non-linearity
        
        self.delta_operator = DeltaOperator(input_dim=H*W, latent_rank=16, device='cuda')

        # self.register_parameter("Em", nn.Parameter(torch.ones(1, H, W, 1)))     #============== Determine the strategy.
        
        self.apply(self.init_linear)
        
    def init_linear(self, m):
        if isinstance(m, nn.Linear):
            if m is self.out_proj.proj:  # SwiGLU inner Linear
                nn.init.normal_(m.weight, std=1)    # std = 1, not 0.02, for aggressive non-linearity
            else:
                nn.init.normal_(m.weight, std=1)    # std = 1, not 0.02, for aggressive non-linearity
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _reshape_heads(self, x, B, N):
        """
        (B, N, C) -> (B, N, H, D)
        """
        return x.reshape(B, N, self.num_heads, self.head_dim)

    def forward(self, x, p):
        """
        x, p: (B, H, W, C)
        """
        B, H, W, C = x.shape
        N = H * W

        # Flatten spatial dims
        x = x.reshape(B, N, C)
        p = p.reshape(B, N, C)

        # Apply LayerNorm
        x = self.norm_x(x)
        p = self.norm_p(p)

        print(f"x: {x}")
        print(f"p: {p}")

        # Projections
        q = self.q_proj(x)
        q = q * (self.head_dim ** -0.5)     # This is non-negotiable with FlashAttention.
        k = self.k_proj(x)
        
        err = self.delta_operator(p)
        v = self.v_proj(p + err) * 1.5

        # Multi-head reshape
        q = self._reshape_heads(q, B, N)
        k = self._reshape_heads(k, B, N)
        v = self._reshape_heads(v, B, N)

        # FlashAttention-2
        att = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout,
            causal=False
        )

        # Merge heads safely
        att = att.reshape(B, N, C)

        out = self.out_proj(att)
        return out.reshape(B, H, W, C)


if __name__ == "__main__":
    def test_physical_flash_attention():
        import torch

        assert torch.cuda.is_available(), "CUDA is required for FlashAttention-2"

        device = torch.device("cuda")
        dtype = torch.bfloat16  # fp16 or bf16 only

        B, H, W, C = 1, 180, 360, 10
        nHeads = 2

        # Inputs
        x = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)
        p = torch.randn(B, H, W, C, device=device, dtype=dtype, requires_grad=True)

        # Module
        attn = PhysicalFlashAttention(dim=C, H=H, W=W, nHeads=nHeads, dropout=0.0).to(device=device, dtype=dtype)

        # Forward + backward
        y = attn(x, p)
        loss = y.mean()
        loss.backward()

        print("Output shape:", y.shape)
        print("Output mean:", y.mean().item())
        print("x.grad:", x.grad is not None)
        print("p.grad:", p.grad is not None)
        print("✅ PhysicalFlashAttention with LayerNorm test passed")

    test_physical_flash_attention()
