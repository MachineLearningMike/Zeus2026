import torch
from torch import nn
from typing import Optional, Callable, List
from timm.models.layers import to_2tuple
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
            device: Optional[str] = None,
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpu_device = torch.device(device)        

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])  
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)
    
    def forward(self, x, train=False, modules_to_cpu=True):
        if train:
            self.x_saved = x.to("cpu")
        
        fc1_g_gpu = self.fc1_g.to(self.gpu_device)
        fc1_x_gpu = self.fc1_x.to(self.gpu_device)
        norm_gpu = self.norm.to(self.gpu_device)
        fc2_gpu = self.fc2.to(self.gpu_device)
        
        x = x.to(self.gpu_device)
        
        x_gate = fc1_g_gpu(x)
        x = fc1_x_gpu(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = norm_gpu(x)
        x = fc2_gpu(x)
        x = self.drop2(x)
        del x_gate
        torch.cuda.empty_cache()
        
        if modules_to_cpu:
            fc1_g_gpu.to("cpu")
            fc1_x_gpu.to("cpu")
            norm_gpu.to("cpu")
            fc2_gpu.to("cpu")
        
        return x
    
    def modules_to_cpu(self):
        self.fc1_g.to("cpu")
        self.fc1_x.to("cpu")
        self.norm.to("cpu")
        self.fc2.to("cpu")
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

    model = SwiGLU(in_features=embed_dim, hidden_features=embed_dim * 4, out_features=embed_dim, act_layer=nn.SiLU, device=device)

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

        print(f"Step {step+1}/{steps} - Loss: {loss.item():.12f}")

    print("Training done.")
