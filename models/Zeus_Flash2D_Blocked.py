import torch
from torch import nn
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.optim as optim
import time

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import config
from modules.params import Params

from modules.Flash2DZeusBlock import Flash2DZeusBlock

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe    # (n_channels, dim)

class PatchEmbedWithPos(nn.Module):
    def __init__(self, to_patch_embedding, pos_embedding, no_PE=False, learn_PE=False):
        super().__init__()
        self.no_PE = no_PE
        # Assign with the name you want in state_dict
        self.to_patch_embedding = to_patch_embedding  # nn.Linear
        if not no_PE:
            if learn_PE:
                self.pos_embedding = nn.Parameter(pos_embedding)    # trainable
            else:
                self.pos_embedding = pos_embedding

    def forward(self, x):
        x = self.to_patch_embedding(x)
        if not self.no_PE:
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
        # pe: (self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        if self.pm.PE == 'sinusodial':
            pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        elif self.pm.PE == 'random':
            pe = torch.randn(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.dim)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
        self.patch_module = PatchEmbedWithPos(to_patch_embedding, pos_embedding, no_PE=self.pm.no_PE, learn_PE=self.pm.learn_PE)

        #-------------- blocks -----------------------------
        blocks = []
        for _ in range(self.pm.zeusBlocks):
            blocks.append(
                Flash2DZeusBlock(
                    d_model=self.pm.dim,
                    num_heads=self.pm.nHeads,
                    num_layers=self.pm.attLayers,
                    dim_layerFFN=self.pm.dim_layerFFN,
                    dropout=self.pm.dropout,
                    causal=False,
                    tokens_per_chunk=self.pm.tokens_per_chunk,
                    heads_per_chunk=self.pm.heads_per_chunk,
                    kv_full_sequence=self.pm.kv_full_sequence,
                    ffn=self.pm.layerFFN,
                    device=device,
                    eps=1e-9,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        
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

        patch_module = self.patch_module.to(self.device)    # (b t n d)
        src_patched = patch_module(src)     # (b t n d)
        if modules_to_cpu:
            self.patch_module.to("cpu")
        
        #----------------- transformer ------------
        # start = time.time()
        for block in self.blocks:
            gc.collect()
            torch.cuda.empty_cache()
            src_blocked = block(src_patched, train=train, modules_to_cpu=modules_to_cpu)    # (b t n d)
        # end = time.time()
        # print("block-forward time {:.3f}".format(end-start))
                    
        #---------------- mlp ---------------------
        if train:
            self.src_blocked_saved = src_blocked.to("cpu")
        mlp_head = self.mlp_head.to(self.device)
        out = mlp_head(src_blocked)
        if modules_to_cpu:
            self.mlp_head.to("cpu")

        return out    # (B,FG,C,H,W)
    
    def backward(self, gradient):
        # #---------------- mlp ---------------------
        src_blocked = self.src_blocked_saved.to(self.device)
        src_blocked.requires_grad_(True)
        src_blocked.retain_grad()
        mlp_head = self.mlp_head.to(self.device)
        gradient = gradient.to(self.device)
        with torch.enable_grad():   # redundant for readability
            out = mlp_head(src_blocked)
        out.backward(gradient, retain_graph=False)  
        gradient = src_blocked.grad
        self.mlp_head.to("cpu")
        del src_blocked, out

        # start = time.time()
        #---------------- transformer -----------------
        for block in reversed(self.blocks):
            gc.collect()
            torch.cuda.empty_cache()
            gradient = block.backward(gradient)
        # end = time.time()
        # print("block-backward time {:.3f}".format(end-start))
        
        #---------------- patch -----------------------
        src = self.src_saved.to(self.device)
        src.requires_grad_(True)
        src.retain_grad()
        patch_module = self.patch_module.to(self.device)
        gradient = gradient.to(self.device)
        with torch.enable_grad():   # redundant for readability
            out = patch_module(src)  # (b t n d)
        out.backward(gradient, retain_graph=False)  
        gradient = src.grad
        self.patch_module.to("cpu")
        del src, out
        gc.collect()
        torch.cuda.empty_cache()
        
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

    lr = 1e-5   #   -5 
    steps = 30

    config['nHeads'] = 128
    config['dim_to_layerFFN_dim'] = 4
    config['tokens_per_chunk'] = None # round(24 * 72 * 2 * 144 * 2 /16/16)
    config['heads_per_chunk'] = None # 128
    
    config['zeusBlocks'] = 5
    config['attLayers'] = 4
    
    model = Zeus(config, device=device)

    #------------------- simple test -------------------
    x = torch.randn(1, config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])
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
    x_train = torch.randn(1, config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])
    y_train = torch.randn(1, config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])

    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()
                
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                out = model(x_train, train=True, modules_to_cpu=True)
        gc.collect()
        torch.cuda.empty_cache()
            
        # target -> same device as out for loss
        out.requires_grad_(True)
        out.retain_grad()
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.enable_grad():   # redundant for readability
                loss = criterion(out, y_train.to(out.device))
        loss.backward()
        gradient = out.grad

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            gradient = model.backward(gradient)
        gc.collect()
        torch.cuda.empty_cache()

        optimizer.step()

        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration / (step+1): .3f}")

    print("Training done.")