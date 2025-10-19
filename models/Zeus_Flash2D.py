import torch
from torch import nn
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from modules.params import Params

from modules.Flash2DTransformer import Flash2DTransformer

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe    # (n_channels, dim)

class PatchEmbedWithPos(nn.Module):
    def __init__(self, to_patch_embedding, pos_embedding, noPE=False):
        super().__init__()
        self.noPE = noPE
        # Assign with the name you want in state_dict
        self.to_patch_embedding = to_patch_embedding  # nn.Linear
        if not noPE:
            self.pos_embedding = nn.Parameter(pos_embedding)    # trainable

    def forward(self, x):
        x = self.to_patch_embedding(x)
        if not self.noPE:
            x = x + self.pos_embedding
        return x

class ReshapeForMLP(nn.Module):
    # x = x.view(B, self.pm.nFramesOut, self.pm.nPatches_h, self.pm.nPatches_w, C, self.pm.patch_size, self.pm.patch_size)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.pm.nFramesOut, C, H, W)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # b m dim
        x = x.reshape(-1, self.dim)     # ((b t h w) (p1 p2 c))
        return x
    
class PatchReconstruct(nn.Module):
    # x = x.view(B, self.pm.nFramesOut, self.pm.nPatches_h, self.pm.nPatches_w, C, self.pm.patch_size, self.pm.patch_size)
    # x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.pm.nFramesOut, C, H, W)
    def __init__(self, nFramesOut, nPatches_h, nPatches_w, patch_size, C, H, W):
        super().__init__()
        self.nFramesOut = nFramesOut
        self.nPatches_h = nPatches_h
        self.nPatches_w = nPatches_w
        self.patch_size = patch_size
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        B = x.numel() // (self.nFramesOut * self.nPatches_h * self.nPatches_w * self.C * self.patch_size * self.patch_size)
        # ((b t h w) (p1 p2 c))
        x = rearrange(x, '(b t h w) (p1 p2 c) -> b t h w p1 p2 c', t=self.nFramesOut, h=self.nPatches_h, w=self.nPatches_w, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, 'b t h w p1 p2 c -> b t c (h p1) (w p2)')
        return x

class Zeus(nn.Module):
    def __init__(self, config, device = None):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)               
        self.pm = Params(config)
 
        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.dim),
            nn.Tanh(),
            nn.Linear(2 * self.pm.dim, self.pm.dim),
            )
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFramesId * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
        
        self.patch_module = PatchEmbedWithPos(to_patch_embedding, pos_embedding, self.pm.noPE)

        #-------------- transformer -------------------------
        self.transformer = Flash2DTransformer(
            num_layers = self.pm.attLayers,
            d_model = self.pm.dim, 
            num_heads = self.pm.nHeads, 
            dim_feedforward = self.pm.mlp_dim,
            dropout = 0.0,
            causal = False,
            tokens_per_chunk = self.pm.tokens_per_chunk,
            heads_per_chunk = self.pm.heads_per_chunk,
            ffn = self.pm.layerFFN, 
            device = self.device,       # used for forward / backward only
        )

        #--------------- ffn --------------------------------
        reshapeForMlp = ReshapeForMLP(self.pm.dim)
        reconstruct = PatchReconstruct(
            nFramesOut=self.pm.nFramesOut, nPatches_h=self.pm.nPatches_h, nPatches_w=self.pm.nPatches_w, patch_size=self.pm.patch_size,
            C=self.pm.nChannels, H=self.pm.image_height, W=self.pm.image_width
        )       
        
        self.mlp_head = nn.Sequential(
            reshapeForMlp,              # to ( (b t n), d )
            nn.LayerNorm(self.pm.dim),  # keep ( (b t n), d )
            nn.Linear(self.pm.dim, self.pm.nChannels * self.pm.patch_size ** 2),    # to ( (b t n), d)
            reconstruct     # to (b t c h w)
        )
        
        return
    
    def get_modules(self):
        return [self.patch_module, self.transformer, self.mlp_head]

    def forward(self, src, train=False, modules_to_cpu=True):        
        #----------------- patch --------------
        if train:
            self.src_saved = src.to("cpu")
        src = src.to(self.device)   # (b t c H W)
        patch_module = self.patch_module.to(self.device)
        src_patched = rearrange(patch_module(src), 'b t n d -> b (t n) d')  # (b m d)
        if modules_to_cpu:
            self.patch_module.to("cpu")
        
        #----------------- transformer ------------
        src_transformed = self.transformer(src_patched, train=train, modules_to_cpu=modules_to_cpu)  # (b m d)
        
        #---------------- mlp ---------------------
        if train:
            self.src_transformed_saved = src_transformed.to("cpu")
        mlp_head = self.mlp_head.to(self.device)
        out = mlp_head(src_transformed)
        if modules_to_cpu:
            self.mlp_head.to("cpu")

        return out    # (B,FG,C,H,W)
    
    def backward(self, gradient):
        #---------------- mlp ---------------------
        src_transformed = self.src_transformed_saved.to(self.device)
        src_transformed.requires_grad_(True)
        src_transformed.retain_grad()
        mlp_head = self.mlp_head.to(self.device)
        gradient = gradient.to(self.device)
        with torch.enable_grad():   # redundant for readability
            out = mlp_head(src_transformed)
            out.backward(gradient, retain_graph=False)  
        gradient = src_transformed.grad
        del src_transformed, out
        gc.collect()
        torch.cuda.empty_cache()
        self.mlp_head.to("cpu")

        #---------------- transformer -----------------
        gradient = self.transformer.backward(gradient)
        
        #---------------- patch -----------------------
        src = self.src_saved.to(self.device)
        src.requires_grad_(True)
        src.retain_grad()
        patch_module = self.patch_module.to(self.device)
        gradient = gradient.to(self.device)
        with torch.enable_grad():   # redundant for readability
            out = rearrange(patch_module(src), 'b t n d -> b (t n) d')  # (b m d)
            out.backward(gradient, retain_graph=False)  
        gradient = src.grad
        del src, out
        gc.collect()
        torch.cuda.empty_cache()
        self.patch_module.to("cpu")
        
        return gradient

    def sigle_GPU_forward_loss_backward(self, data, target, loss_fn, unnormal_fn, batch_diff, era5_data_vars):
        with torch.no_grad():
            pred = self(data, train=True, modules_to_cpu=True)
        pred = unnormal_fn(pred)   # torch.float32 !!!, cuda, True !!!
        target = unnormal_fn(target)   # torch.float32, cuda, False
        # target -> same device as out for loss
        pred.requires_grad_(True)
        # pred.retain_grad()    no need, pred is a leaf tensor.
        with torch.enable_grad():   # redundant for readability
            lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars)
            lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars) 
            loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics    
            loss.backward()
        gradient = pred.grad
        gradient = self.backward(gradient)

        return gradient, lossAndMetrics

if __name__ == "__main__":    
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 1e-3   # 1e-2: 1, 1e-3: 0.19, 1e-4: 
    steps = 100

    model = Zeus(config, device=device)

    #------------------- simple test -------------------
    x = torch.randn(config['batch_size'], config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])  # (B, S, E)
    with torch.no_grad():
        out = model(x, train=True, modules_to_cpu=True)
    print("Output:", out.shape, out.device)
    gradient = torch.ones_like(x)
    gradient = model.backward(gradient)
    print("Gradient:", gradient.shape, gradient.device)
        
    #------------------ training test --------------------
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(config['batch_size'], config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])
    y_train = torch.randn(config['batch_size'], config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])

    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()

        with torch.no_grad():
            out = model(x_train, train=True, modules_to_cpu=True)
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with torch.enable_grad():   # redundant for readability
            loss = criterion(out, y_train.to(out.device))
        loss.backward()
        gradient = out.grad
        gradient = model.backward(gradient)
        
        optimizer.step()

        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration / (step+1): .3f}")

    print("Training done.")