import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
import math

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

        self.q_proj = TwiGLU(dim, initial_eye=True, initial_scale=1.)               #------------- let the scale large enough to undergo the non-linearity of tanh.
        self.k_proj = TwiGLU(dim, initial_eye=True, initial_scale=0.01)             #------------- scale las litttle meaning but make it small so that the softmax has small amplificattion
        self.out_proj = LearnableEyeLayer(dim, init_scale=1.0)

        # Physics correction operator
        rate = 16 # dim  # 50
        latent_rank = (H * W * dim) // rate
        self.delta_operator = TwiGLU(dim, initial_eye=True, initial_scale=0.01)     #------------- let the scale keep balance with p
        
        # DeltaOperatorMultiChannel(H=H, W=W, C=dim, latent_rank=latent_rank, gate_init_scale=0.01)

    def forward_axis(self, x, yp, axis):
        """
        Axial attention along one axis (height or width) with physics correction.
        x: [B, H, W, C]
        p: [B, H, W, C]  (physical state)
        axis: 1 for height, 2 for width
        
        Implementation: Process each row/column independently by expanding batch dimension.
        """
        B, H, W, C = x.shape

        if axis == 1:  # height attention (attend along columns)
            # [B, H, W, C] -> [B, W, H, C] -> [B*W, H, C]
            _x = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
            _yp = yp.permute(0, 2, 1, 3).reshape(B * W, H, C)
            seq_len = H
            batch_expanded = B * W
            reshape_back = lambda out: out.reshape(B, W, H, C).permute(0, 2, 1, 3)
        else:  # width attention (attend along rows)
            # [B, H, W, C] -> [B*H, W, C]
            _x = x.reshape(B * H, W, C)
            _yp = yp.reshape(B * H, W, C)
            seq_len = W
            batch_expanded = B * H
            reshape_back = lambda out: out.reshape(B, H, W, C)

        # Layer norms
        _x = self.norm_x(_x)
        _yp = self.norm_p(_yp)

        # ---- Q, K ----
        q = self.q_proj(_x) * (self.head_dim ** -0.5)
        k = self.k_proj(_x)

        # ---- Reshape to multi-head ----
        q = q.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)

        # ---- V: physics with delta correction ----
        # Need to reshape for delta_operator, then reshape back
        if axis == 1:
            # For height: _yp is [B*W, H, C], reshape to [B, W, H, C] for delta_operator
            yp_for_delta = _yp.reshape(B, W, H, C)
            p_corr = _yp + self.delta_operator(yp_for_delta).reshape(B*W, H, C)
        else:
            # For width: _yp is [B*H, W, C], reshape to [B, H, W, C] for delta_operator
            yp_for_delta = _yp.reshape(B, H, W, C)
            p_corr = _yp + self.delta_operator(yp_for_delta).reshape(B*H, W, C)
        
        v = p_corr.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)

        # Match dtypes for FlashAttention
        if q.dtype != v.dtype:
            v = v.to(q.dtype)

        # ---- FlashAttention ----
        att_out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # Merge heads (already done by flash_attn)
        att_out = att_out.reshape(batch_expanded, seq_len, C)

        # Output projection
        out = self.out_proj(att_out)

        # Restore original shape
        out = reshape_back(out)

        return out


    def forward(self, x):
        # Axial attention: height then width, repeated
        for _ in range(1):
            x = self.forward_axis(x, x, axis=1)
            x = self.forward_axis(x, x, axis=2)
        return x

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
    print(f"Sequence lengths: Height={H}, Width={W}")
    print(f"Memory usage comparison:")
    print(f"  Original: seq_len = {H*W} = {H*W}")
    print(f"  New: Height attention seq_len = {H}")
    print(f"  New: Width attention seq_len = {W}")