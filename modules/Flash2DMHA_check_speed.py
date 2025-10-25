import torch
import torch.nn as nn
import time
from torch.amp import autocast, GradScaler

from Flash2DMHA import Flash2DMHA  # unchanged

# ----------------- settings -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 1024, 1024
nHeads = 8
steps = 10
torch.manual_seed(42)

# ----------------- input -----------------
x = torch.randn(B, L, D, device=device)
grad_output = torch.randn(B, L, D, device=device)

# ----------------- Flash2DMHA -----------------
flash_mha = Flash2DMHA(d_model=D, nHeads=nHeads,
                        dropout=0.0, causal=False,
                        tokens_per_chunk=None, heads_per_chunk=None,
                        kv_full_sequence=True).to(device)
flash_mha.eval()

# Warm-up
with autocast(device_type="cuda", dtype=torch.bfloat16):
    _ = flash_mha(x)
torch.cuda.synchronize() if device.startswith("cuda") else None

start = time.time()
for _ in range(steps):
    # ensure x is on the right device and requires grad
    x_flash = x.detach().clone().to(device).requires_grad_(True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        out = flash_mha(x_flash, train=False, modules_to_cpu=False)
    out.backward(grad_output.to(device))
    torch.cuda.synchronize() if device.startswith("cuda") else None
end = time.time()
flash_time = (end - start) / steps
print(f"Flash2DMHA, which shines with long sequences, avg step time: {flash_time:.6f} s")

# ----------------- nn.MultiheadAttention -----------------
mha = nn.MultiheadAttention(embed_dim=D, num_heads=nHeads, dropout=0.0, batch_first=True).to(device)
mha.eval()

# Warm-up
_ = mha(x, x, x)[0]
torch.cuda.synchronize() if device.startswith("cuda") else None

start = time.time()
for _ in range(steps):
    x_mha = x.detach().clone().to(device).requires_grad_(True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        out, _ = mha(x_mha, x_mha, x_mha)
    out.backward(grad_output.to(device))
    torch.cuda.synchronize() if device.startswith("cuda") else None
end = time.time()
mha_time = (end - start) / steps
print(f"nn.MultiheadAttention avg step time: {mha_time:.6f} s")

print(f"Speedup (MHA / Flash2DMHA): {mha_time / flash_time:.2f}x")
