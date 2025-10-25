import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from einops import rearrange
import time

# -----------------------------
# CylindricSpatialConvTD
# -----------------------------
class CylindricSpatialConvTD(nn.Module):
    """
    Input:  (B, T, N, D)
    Output: same shape
    Horizontal cylindrical convolution
    Vertical zero padding
    """
    def __init__(self, pm, kernel_size=3, device=None):
        super().__init__()
        in_channels_time = pm.T1
        in_channels_dim = pm.patch_dim
        self.H = pm.nPatches_h
        self.W = pm.nPatches_w
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Use circular padding for horizontal axis via PyTorch padding_mode
        self.conv2d_time = nn.Conv2d(
            in_channels_time,
            in_channels_time,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size//2, kernel_size//2),  # (H_pad, W_pad)
            padding_mode='circular',
            bias=True
        )
        
        # Use circular padding for horizontal axis via PyTorch padding_mode
        self.conv2d_dim = nn.Conv2d(
            in_channels_dim,
            in_channels_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size//2, kernel_size//2),  # (H_pad, W_pad)
            padding_mode='circular',
            bias=True
        )

    def save_cpu_activation(self, tensor, name):
        buf = getattr(self, name, None)
        if buf is None or buf.shape != tensor.shape:
            buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
            setattr(self, name, buf)
        buf.copy_(tensor, non_blocking=True)

    def forward(self, x, train=False, modules_to_cpu=True):   # (b t n d) -> (b t n d)
        B, T, N, D = x.shape
        H, W = self.H, self.W
        assert H*W == N, "N must equal H*W"

        if train:
            self.save_cpu_activation(x, "input_saved")

        self.conv2d_time.to(self.device)
        self.conv2d_dim.to(self.device)

        x = rearrange(x, 'b t (h w) d -> (b d) t h w', h=H, w=W).contiguous()
        x = self.conv2d_time(x)        
        x = rearrange(x, '(b d) t h w -> (b t) d h w', h=H, w=W, b=B).contiguous()
        x = self.conv2d_dim(x)
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B, t=T).contiguous()

        if modules_to_cpu:
            self.conv2d_time.to("cpu")
            self.conv2d_dim.to("cpu")

        return x

        # x = rearrange(x, 'b t n d -> (b d) t n')
        # x = rearrange(x, 'B t (h w) -> B t h w', h=self.pm.nPatches_h)
        # x = self.convolve(x)    # preserves shape
        # x = rearrange(x, '(b d) t h w -> b t (h w) d', d=self.pm.d_model)

    def backward(self, gradient):
        
        input = self.input_saved.to(self.device, non_blocking=True)        
        input.requires_grad_(True)
        # input.retain_grad()  # No need, it's a leaf tensor.
        self.to(self.device)
        gradient = gradient.to(self.device)
        # with torch.enable_grad():   # redundant for readability
        out = self(input, train=False, modules_to_cpu=False)
        out.backward(gradient, retain_graph=False)  # gradients are saved to out_proj_gpu
        gradient = input.grad
        self.to("cpu")
        
        return gradient

# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import sys
    from pathlib import Path
    # Get path to sibling folder
    sibling_path = Path(__file__).resolve().parent.parent / "models"
    sys.path.insert(0, str(sibling_path))
    from config import config
    from params import Params

    config['factor10'] = 2
    config['batch_size'] = 1
    config['T1'] = 24
    
    pm = Params(config)

    B, T, H, W, D = pm.batch_size, pm.T1, pm.nPatches_h, pm.nPatches_w, pm.patch_dim  # example dimensions
    N = H * W
    print("B, T, H, W, D, N", B, T, H, W, D, N)
    
    x_train = torch.randn(B, T, N, D, device=device)
    y_train = torch.randn(B, T, N, D, device=device)

    model = CylindricSpatialConvTD(pm, kernel_size=5).to(device)
    model.eval()

    steps = 100
    lr = 1e-3
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    chunkBackward = True

    start = time.time()
    print("start: ", start)
    for step in range(steps):
        optimizer.zero_grad()

        # model.to(device)
        # x_train.requires_grad_(True)   # it's False by default.

        # forward pass (offloads chunks to GPU internally)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.set_grad_enabled(not chunkBackward):
                out = model(x_train, train=chunkBackward, modules_to_cpu=chunkBackward)

        if chunkBackward:
            out.requires_grad_(chunkBackward)
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.enable_grad():   # redundant for readability
                loss = criterion(out, y_train.to(out.device))
            loss.backward(retain_graph=False)

        if chunkBackward:
            gradient = out.grad
            gradient = model.backward(gradient)
        
        optimizer.step()
        
        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")