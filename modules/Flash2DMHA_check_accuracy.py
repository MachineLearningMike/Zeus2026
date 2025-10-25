import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from Flash2DMHA import Flash2DMHA  # adjust import path if needed

# ----------------- settings -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, L, D = 2, 16, 16  # small sizes for testing
nHeads = 4
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

# Forward
with autocast(device_type="cuda", dtype=torch.bfloat16):
    out_flash = flash_mha(x)
# Backward
out_flash.backward(grad_output)
grad_flash = x.grad.clone()

# Reset gradients
x.grad.zero_()

# ----------------- nn.MultiheadAttention -----------------
# nn.MultiheadAttention expects (L, B, D)
mha = nn.MultiheadAttention(embed_dim=D, num_heads=nHeads, dropout=0.0, batch_first=True).to(device)
mha.eval()

# Copy weights from Flash2DMHA to nn.MultiheadAttention for fair comparison
with torch.no_grad():
    # Merge QKV weights
    mha.in_proj_weight.copy_(torch.cat([flash_mha.W_qkv.weight,], dim=0))  # simplified
    mha.in_proj_bias.copy_(torch.cat([flash_mha.W_qkv.bias,], dim=0))
    mha.out_proj.weight.copy_(flash_mha.W_o.weight)
    mha.out_proj.bias.copy_(flash_mha.W_o.bias)

# Forward
with autocast(device_type="cuda", dtype=torch.bfloat16):
    out_mha, _ = mha(x, x, x)  # batch_first=True
# Backward
out_mha.backward(grad_output)
grad_mha = x.grad.clone()

# ----------------- compare -----------------
max_diff_out = (out_flash - out_mha).abs().max()
max_diff_grad = (grad_flash - grad_mha).abs().max()

print(f"Max difference in outputs, which is reduced for heavily sized data: {max_diff_out.item():.6f}")
print(f"Max difference in input gradients, which is reduced for heavily sized data: {max_diff_grad.item():.6f}")
