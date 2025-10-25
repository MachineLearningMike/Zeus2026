import torch
import torch.nn as nn
import time
from Flash2DMHA import Flash2DMHA  # adjust import path if needed

# ----------------- settings -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 1024, 128  # moderate size for speed test
nHeads = 8
steps = 10
torch.manual_seed(42)

# ----------------- input -----------------
x = torch.randn(B, L, D, device=device, requires_grad=True)
grad_output = torch.randn(B, L, D, device=device)

# ----------------- Flash2DMHA -----------------
flash_mha = Flash2DMHA(d_model=D, nHeads=nHeads,
                        dropout=0.0, causal=False,
                        tokens_per_chunk=None, heads_per_chunk=None,
                        kv_full_sequence=True).to(device)
flash_mha.eval()

# Warm-up
_ = flash_mha(x)
torch.cuda.synchronize() if device.startswith("cuda") else None

start = time.time()
for _ in range(steps):
    x_flash = x.clone().detach().requires_grad_(True)
    out = flash_mha(x_flash)
    out.backward(grad_output)
    torch.cuda.synchronize() if device.startswith("cuda") else None
end = time.time()
flash_time = (end - start) / steps
print(f"Flash2DMHA avg step time: {flash_time:.6f} s")

# ----------------- nn.MultiheadAttention -----------------
mha = nn.MultiheadAttention(embed_dim=D, num_heads=nHeads, dropout=0.0, batch_first=True).to(device)
mha.eval()

# Optional: map Flash2DMHA weights if desired for fairness
# Skipping here; focus is on speed comparison

# Warm-up
_ = mha(x, x, x)[0]
torch.cuda.synchronize() if device.startswith("cuda") else None

start = time.time()
for _ in range(steps):
    x_mha = x.clone().detach().requires_grad_(True)
    out, _ = mha(x_mha, x_mha, x_mha)
    out.backward(grad_output)
    torch.cuda.synchronize() if device.startswith("cuda") else None
end = time.time()
mha_time = (end - start) / steps
print(f"nn.MultiheadAttention avg step time: {mha_time:.6f} s")

print(f"Speedup (MHA / Flash2DMHA): {mha_time / flash_time:.2f}x")
