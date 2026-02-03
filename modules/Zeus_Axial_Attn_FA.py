import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func

class AxialFlashAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Separate Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward_axis(self, x, axis):
        B, H, W, C = x.shape

        # Flatten the axis to sequence
        if axis == 1:  # height
            x_flat = x.permute(0, 2, 1, 3).reshape(B, W*H, C)
            N = H
        else:  # width
            x_flat = x.reshape(B, H*W, C)
            N = W

        # Q/K/V projections
        q = self.q_proj(x_flat).view(B, -1, self.n_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B, -1, self.n_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B, -1, self.n_heads, self.head_dim)

        # FlashAttention
        out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # Merge heads safely
        out = out.reshape(B, x_flat.size(1), C)
        out = self.out_proj(out)

        # Restore original shape
        if axis == 1:
            out = out.reshape(B, W, H, C).permute(0, 2, 1, 3)
        else:
            out = out.reshape(B, H, W, C)

        return out

    def forward(self, x):
        # Axial attention: height then width
        for _ in range(2):
            x = self.forward_axis(x, axis=1)  # height
            x = self.forward_axis(x, axis=2)  # width
        return x


# ------------------- Test Code -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W, C, n_heads = 1, 180, 360, 12, 4
    loops = 1  # For full step timing, usually 1 is enough

    x = torch.randn(B, H, W, C, device=device)
    target = torch.randn_like(x)  # simple MSE target
    codec = AxialFlashAttention(dim=C, n_heads=n_heads).to(device)

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