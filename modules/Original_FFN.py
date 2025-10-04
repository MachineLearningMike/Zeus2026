import torch
from torch import nn
from typing import Optional, Callable, List
from timm.models.layers import to_2tuple
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

class Original_FFN(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            dim_feedforward: int = 2048,
            activation: Callable = F.relu,
            eps: float = 1e-6,
            device: Optional[str] = None,
    ):
        super().__init__()
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)
        
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)   
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, eps=eps)
        self.activation = activation
        
        self.x_saved = None
    
    def forward(self, x, train=False, modules_to_cpu=True):
        # return ff on self.gpu_device        
        # Move FFN weights to GPU temporarily
        if train:
            self.x_saved = x.to("cpu")
        
        x = x.to(self.gpu_device)
        linear1_gpu = self.linear1.to(self.gpu_device)
        linear2_gpu = self.linear2.to(self.gpu_device)
        ff = linear2_gpu(self.activation(linear1_gpu(x)))
        
        if modules_to_cpu:
            linear1_gpu.to("cpu")
            linear2_gpu.to("cpu")
        
        return ff
    
    def modules_to_cpu(self):
        self.linear1.to("cpu")
        self.linear2.to("cpu")
        return        

    def backward(self, gradient):
        x_saved = self.x_saved.to(self.gpu_device)
        x_saved.requires_grad_(True)
        x_saved.retain_grad()

        ff = self.forward(x_saved, train=True, modules_to_cpu=False)
        
        gradient = ff.backward(gradient)
        
        self.modules_to_cpu()

        return gradient

    
if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    factor10 = 2
    batch_size = 24         # it's also the data simplicity in this test.
    seq_len = round(1 * 72 * factor10 * 144 * factor10 /16/16)
    embed_dim = 1024
    num_heads = 8
    num_chunks = 4
    lr = 1e-3
    steps = 500

    model = Original_FFN(embed_dim=embed_dim, dim_feedforward=embed_dim * 4, activation=F.relu, device=device)

    #------------------- simple test -------------------
    x = torch.randn(batch_size, seq_len, embed_dim)  # (B, S, E)
    out = model.forward(x, train=True)
    print("Output:", out.shape, out.device)
        
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(batch_size, seq_len, embed_dim)
    y_train = torch.randn(batch_size, seq_len, embed_dim)

    for step in range(steps):
        optimizer.zero_grad()

        x_train.requires_grad_(True)
        x_train.retain_grad()
        with torch.no_grad():
            out = model.forward(x_train, train=True)
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with torch.enable_grad():   # redundant for readability
            loss = criterion(out, y_train.to(out.device))
            loss.backward()
        gradient = out.grad

        gradient = model.backward(gradient)

        optimizer.step()

        print(f"Step {step+1}/{steps} - Loss: {loss.item():.9f}")

    print("Training done.")