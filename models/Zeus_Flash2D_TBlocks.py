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

from modules.Flash2DZeusBlock import Flash2DZeusBlock

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe    # (n_channels, dim)

class PatchEmbedWithPos(nn.Module):
    def __init__(self, to_patch_embedding, pos_embedding, use_PE=True, learn_PE=False):
        super().__init__()
        self.use_PE = use_PE
        # Assign with the name you want in state_dict
        self.to_patch_embedding = to_patch_embedding  # nn.Linear
        if use_PE:
            if learn_PE:
                self.pos_embedding = nn.Parameter(pos_embedding)    # trainable
            else:
                self.pos_embedding = pos_embedding

    def forward(self, x):
        x = self.to_patch_embedding(x)
        if self.use_PE:
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
    def __init__(self, pm, device = None):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)               
        self.pm = pm
 
        #--------------- token embedding ---------------
        # We need to enable more complex non-linearity here.
        to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.pm.patch_size, p2=self.pm.patch_size),
            nn.Linear(self.pm.patch_dim, 2 * self.pm.d_model),
            nn.Tanh(),
            nn.Linear(2 * self.pm.d_model, self.pm.d_model),
            )
        #-------------- positional embedding ----------------
        # pe: (self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.d_model)
        if self.pm.PE == 'sinusodial':
            pe = sinusoidal_embedding(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.d_model)
        elif self.pm.PE == 'random':
            pe = torch.randn(self.pm.nFramesIn * self.pm.nPatches_h * self.pm.nPatches_w, self.pm.d_model)
        pos_embedding = rearrange(pe, '(t n) dim -> 1 t n dim', t=self.pm.nFramesIn)
        self.patch_module = PatchEmbedWithPos(to_patch_embedding, pos_embedding, use_PE=self.pm.use_PE, learn_PE=self.pm.learn_PE)

        #-------------- blocks -----------------------------
        blocks = []
        for _ in range(self.pm.zeusBlocks):
            blocks.append(
                Flash2DZeusBlock(self.pm, device=device)
            )
        self.blocks = nn.ModuleList(blocks)
        
        #--------------- layerFFN --------------------------------
        reshapeForMlp = ReshapeForMLP(self.pm.d_model)
        reconstruct = PatchReconstruct(
            nFramesOut=self.pm.nFramesOut, nPatches_h=self.pm.nPatches_h, nPatches_w=self.pm.nPatches_w, patch_size=self.pm.patch_size,
            C=self.pm.nVariables, H=self.pm.height, W=self.pm.width
        )       
        
        self.mlp_head = nn.Sequential(
            reshapeForMlp,              # to ( (b t n), d )
            nn.LayerNorm(self.pm.d_model),  # keep ( (b t n), d )
            nn.Linear(self.pm.d_model, self.pm.nVariables * self.pm.patch_size ** 2),    # to ( (b t n), d)
            reconstruct     # to (b t c h w)
        )
        return
    
    def save_cpu_activation(self, tensor, name):
        buf = getattr(self, name, None)
        if buf is None or buf.shape != tensor.shape:
            buf = torch.empty_like(tensor, device="cpu", pin_memory=True)
            setattr(self, name, buf)
        buf.copy_(tensor, non_blocking=True)

    def forward(self, src, train=False, modules_to_cpu=True):        
        #----------------- patch --------------
        if train:
            # self.src_saved = src.to("cpu")
            self.save_cpu_activation(src, "src_saved")
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
            self.save_cpu_activation(src_blocked, "src_blocked_saved")
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
   
    def sigle_GPU_forward_loss_backward(self, input, target, loss_fn, unnormal_fn, batch_diff, era5_data_vars, trainable=True):
        # The caller creates this context:
        # with autocast(device_type=self.device_type, dtype=torch.bfloat16):
        #     with torch.set_grad_enabled(trainable): 

        s = time.time()
        with torch.set_grad_enabled(False):
            pred = self(input, train=True, modules_to_cpu=True)
        e = time.time()
        # print(f"time1 {e-s:.5f}")

        s = time.time()
        if unnormal_fn is not None:
            pred = unnormal_fn(pred)   # torch.float32 !!!, cuda, True !!!
            target = unnormal_fn(target)   # torch.float32, cuda, False
        e = time.time()
        # print(f"time2 {e-s:.5f}")

        s = time.time()
        if trainable:
            pred.requires_grad_(trainable)
            pred.retain_grad()
        e = time.time()
        # print(f"time3 {e-s:.5f}")

        s = time.time()
        lossAndMetrics = loss_fn(target, pred, batch_diff, era5_data_vars)
        loss = lossAndMetrics[0] if isinstance(lossAndMetrics, (tuple, list)) else lossAndMetrics
        e = time.time()
        # print(f"time4 {e-s:.5f}")
        
        s = time.time()
        if trainable:
            loss.backward()
            gradient = pred.grad
            _ = self.backward(gradient)
        e = time.time()
        # print(f"time5 {e-s:.5f}")


        return lossAndMetrics



if __name__ == "__main__":    
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    lr = 5e-5   #   -5 
    steps = 30

    import time
    from config import config
    from modules.params import Params

    # config['factor10'] = 2
    # config['batch_size'] = 1
    # config['patch_size'] = 12
    # config['T1'] = 24
    # config['nLayers'] = 5
    # config['nHeads'] = 72
    # config['tokens_per_chunk'] = None
    # config['heads_per_chunk'] = None
    # config['kv_full_sequence'] = True
    # config['self.add_spatial_conv'] = False
    # config['layerFFN'] = "GLU"   

    config['zeusBlocks'] = 1

    pm = Params(config)
    batch_size, T1, N, d_model = pm.batch_size, pm.T1, pm.nPatches, pm.d_model

    lr = 1e-4               # -4
    steps = 100
    
    model = Zeus(pm, device=device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    #------------------- simple test -------------------
    x = torch.randn(1, config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])
    with torch.no_grad():
        out = model(x, train=True, modules_to_cpu=True)
    print("Output:", out.shape, out.device)
    gradient = torch.ones_like(x)
    gradient = model.backward(gradient)
    print("Gradient:", gradient.shape, gradient.device)

    #------------------ training test --------------------
    
    ERA5_DATA_VARS = {
        "2m_temperature": 0.3,
        "total_precipitation": 0.4,
        "100m_u_component_of_wind": 0.15,
        "100m_v_component_of_wind": 0.15,
    }
    era5_data_vars = torch.tensor([value for value in ERA5_DATA_VARS.values()])
    
    def loss_fun(target, pred, batch_diff, era5_data_vars):
        device = pred.device
        target = target.to(device)
        rmse = ((pred - target) ** 2).mean(dim=(0,1,3,4)).sqrt()    # (C,), 32, cuda, True !!!        
        era5_data_vars = era5_data_vars.to(device)
        reward = rmse * era5_data_vars   # (C,), 32, cuda, True !!!, the-higher-the-worse
        reward = reward.mean()  # (,)
        return reward, rmse # (,), (C,)     32, cuda, True
    
    # optimizer (params are on CPU at construction time)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # dummy data (CPU)
    x_train = torch.randn(config['batch_size'], config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])
    y_train = torch.randn(config['batch_size'], config['T1'], len(config['variables']), 72 * config['factor10'], 144 * config['factor10'])

    scaler = GradScaler()
    
    start = time.time()
    for step in range(steps):    
        
        optimizer.zero_grad()
        
        # with autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.set_grad_enabled(True):
            (loss, rmse) = model.sigle_GPU_forward_loss_backward(x_train, y_train, loss_fun, None, None, era5_data_vars, trainable=True)
                
        # with autocast(device_type="cuda", dtype=torch.bfloat16):
        #     with torch.no_grad():
        #         out = model(x_train, train=True, modules_to_cpu=True)
        # gc.collect()
        # torch.cuda.empty_cache()
            
        # # target -> same device as out for loss
        # out.requires_grad_(True)
        # out.retain_grad()
        # with autocast(device_type="cuda", dtype=torch.bfloat16):
        #     with torch.enable_grad():   # redundant for readability
        #         loss = criterion(out, y_train.to(out.device))
        #     loss.backward()
        # gradient = out.grad

        # with autocast(device_type="cuda", dtype=torch.bfloat16):
        #     gradient = model.backward(gradient)
        # gc.collect()
        # torch.cuda.empty_cache()

        s = time.time()
        if step % 10 == 0:
            optimizer.step()
        e = time.time()
        print(f"optimizer time: {e-s :.5f}")

        end = time.time()
        duration = end - start
        print(f"Step {step+1}/{steps}, Loss: {loss.item():.12f}, avg time: {duration/(step+1): .3f}")

    print("Training done.")