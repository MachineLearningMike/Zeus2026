import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.DeltaOperator import DeltaOperatorMultiChannel

# -------------------- Utility Modules --------------------
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        return swish * self.linear_2(x)

class TwiGLU(nn.Module):
    def __init__(self, dimension, initial_eye=False, initial_scale=1.0):
        super().__init__()
        self.dimension = dimension
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)
        
        # Initialize the linear layers
        self._init_weights(initial_eye, initial_scale)
    
    def _init_weights(self, initial_eye, initial_scale):
        """Initialize weights based on the given parameters."""
        if initial_eye:
            # Initialize as scaled identity matrix: eye * initial_scale
            scaled_eye = torch.eye(self.dimension) * initial_scale
            with torch.no_grad():
                self.linear_1.weight.data.copy_(scaled_eye)
                self.linear_2.weight.data.copy_(scaled_eye)
                
                # Initialize biases to zero for scaled identity transformation
                if self.linear_1.bias is not None:
                    self.linear_1.bias.data.zero_()
                if self.linear_2.bias is not None:
                    self.linear_2.bias.data.zero_()
        else:
            # Initialize with normal distribution with specified std (initial_scale)
            for linear in [self.linear_1, self.linear_2]:
                nn.init.normal_(linear.weight, mean=0.0, std=initial_scale)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
    
    def forward(self, x):
        output = self.linear_1(x)
        twish = output * torch.tanh(output)  # Swish-like activation
        return twish * self.linear_2(x)

class LearnableEyeLayer(nn.Module):
    def __init__(self, dim, init_scale=0.01):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.weight.data += init_scale * torch.randn_like(self.weight)

    def forward(self, x):
        return x @ self.weight.T + self.bias

# -------------------- Axial FlashAttention --------------------
class AxialFlashAttention(nn.Module):
    def __init__(self, dim, H, W, n_heads=8, eps=1e-5):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.H, self.W = H, W

        self.norm_x = nn.LayerNorm(dim, eps=eps)
        self.norm_p = nn.LayerNorm(dim, eps=eps)

        self.q_proj = TwiGLU(dim, initial_eye=True, initial_scale=1.)
        self.k_proj = TwiGLU(dim, initial_eye=True, initial_scale=0.01)
        self.out_proj = LearnableEyeLayer(dim, init_scale=1.0)

        # Physics correction operator
        rate = 16
        latent_rank = (H * W * dim) // rate
        self.delta_operator = TwiGLU(dim, initial_eye=True, initial_scale=0.01)
        
        # DeltaOperatorMultiChannel(H=H, W=W, C=dim, latent_rank=latent_rank, gate_init_scale=0.01)

    def forward_full(self, x, yp):
        """
        Single full attention pass that attends to all tokens at once.
        x: [B, H, W, C]
        yp: [B, H, W, C]  (physical state)
        """
        B, H, W, C = x.shape
        
        # Flatten to sequence: [B, H*W, C]
        _x = x.reshape(B, H*W, C)
        _yp = yp.reshape(B, H*W, C)
        seq_len = H * W
        
        # Layer norms
        _x = self.norm_x(_x)
        _yp = self.norm_p(_yp)
        
        # ---- Q, K ----
        q = self.q_proj(_x) * (self.head_dim ** -0.5)
        k = self.k_proj(_x)
        
        # ---- Reshape to multi-head ----
        q = q.reshape(B, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(B, seq_len, self.n_heads, self.head_dim)
        
        # ---- V: physics with delta correction ----
        p_corr = _yp + self.delta_operator(_yp.reshape(B, H, W, C)).reshape(B, seq_len, C)
        v = p_corr.reshape(B, seq_len, self.n_heads, self.head_dim)
        
        # Match dtypes for FlashAttention
        if q.dtype != v.dtype:
            v = v.to(q.dtype)
        
        # ---- FlashAttention ----
        att_out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        
        # Merge heads
        att_out = att_out.reshape(B, seq_len, C)
        
        # Output projection
        out = self.out_proj(att_out)
        
        # Restore original shape
        out = out.reshape(B, H, W, C)
        
        return out

    def forward(self, x):
        # Single full attention pass instead of separate axial passes
        return self.forward_full(x, x)

# ------------------- Test Code -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W, C, n_heads = 1, 180, 360, 12, 4
    loops = 5

    x = torch.randn(B, H, W, C, device=device)
    target = torch.randn_like(x)
    codec = AxialFlashAttention(dim=C, H=H, W=W, n_heads=n_heads).to(device)

    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x_bf16 = x.bfloat16()
    target_bf16 = target.bfloat16()

    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for _ in range(loops):
                optimizer.zero_grad(set_to_none=True)
                output = codec(x_bf16)
                loss = criterion(output, target_bf16)
                loss.backward()
                optimizer.step()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    else:
        import time
        start_time = time.perf_counter()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            for _ in range(loops):
                optimizer.zero_grad(set_to_none=True)
                output = codec(x_bf16)
                loss = criterion(output, target_bf16)
                loss.backward()
                optimizer.step()
        elapsed_time = time.perf_counter() - start_time

    print("Output shape:", output.shape)
    print(f"Elapsed time for full step: {elapsed_time:.4f} sec")
    print(f"Sequence length: {H*W} = {H}×{W}")
    print(f"Attention complexity: O((H×W)²) = O({(H*W)**2:,})")