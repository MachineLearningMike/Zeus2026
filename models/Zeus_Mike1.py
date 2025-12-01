import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.zeus_patch_unpatch import PatchFeatureExtractor, UnpatchReconstructor
from modules.Zeus_Swin_Block_3D_CDHW import Zeus_SwinTransformerBlock3D
from modules.zeus_dataset_Snapshot import Snapshot


# -----------------------
# 4️⃣ Full Model
# -----------------------
class Zeus(nn.Module):
    # def __init__(self, pm, device='cpu'):
    #     patch_kernel = pm.patch_kernel
    #     patch_stride = pm.patch_stride
    #     swin_dims = pm.swin_dims
    #     window_size = pm.window_size
    #     num_swin_blocks = pm.num_swin_block
    
    def __init__(self, H, W, num_heads,
                num_steps, patch_kernel=8, patch_stride=8, swin_dims=16,
                window_size=(2,10,10), num_swin_blocks=30):
        super().__init__()
        
        #---------------- Patch extractor -----------------
        
        now = datetime.now(timezone.utc)
        ss1 = Snapshot(now.year, now.month, now.day, now.hour)
        then = now + timedelta(hours=6)
        ss2 = Snapshot(then.year, then.month, then.day, then.hour)
        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
        
        self.patch_kernel=(patch_kernel,patch_kernel)
        self.patch_stride=(patch_stride,patch_stride)
        
        self.patch_extractor = PatchFeatureExtractor(levels, surface, c_out=swin_dims, patch_kernel=self.patch_kernel, patch_stride=self.patch_stride)
        
        #---------------- Swin blocks ----------------------
        
        input_resolution = (self.patch_extractor.nTotalLevels, H//patch_stride, W//patch_stride) 
        
        self.swin_blocks = nn.ModuleList([
            Zeus_SwinTransformerBlock3D(
                dim=swin_dims,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if idx % 2 == 0 else (window_size[0]//2, window_size[1]//2, - window_size[2]//2),
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0
            )
            for idx in range(num_swin_blocks)  # reduced number for testing
        ])
        
        #--------------- Unpatch reconstructor        
        self.uppatch_reconstructor = UnpatchReconstructor(in_channels=self.patch_extractor.nTotalLevels * swin_dims, out_channels=Snapshot.meteo_dims, kernel_size=self.patch_kernel, stride=self.patch_stride)

    def forward(self, ss1, ss2):    # Not batch-processing, as hard drives are always bottlenecks.
        print(f"x befor patching: {ss1.meteo.shape, ss1.tempo.shape, ss1.geo.shape}")

        #----------- patch/extract level features ------------
        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
        s = time.time()
        x = self.patch_extractor(levels, surface)
        e = time.time()
        print(f"x after patching: {x.shape}, time: {e-s}")

        #----------- undergo swin blocks ----------------
        s = time.time()
        x = x.unsqueeze(0)
        B, C, D, H, W = x.shape
        x_flat = x.permute(0,2,3,4,1).reshape(B, D*H*W, C)

        for b in self.swin_blocks:
            x_flat = b(x_flat)

        x_out = x_flat.view(B, D, H, W, C).permute(0,4,1,2,3)
        x = x_out.squeeze(0)
        e = time.time()
        print(f"x after swin blocks: {x.shape}, time: {e-s}")
        
        #------------ unpatch/reconstruct --------------

        s = time.time()
        x = x.view(-1, H, W)
        x = x.unsqueeze(0)
        x = self.uppatch_reconstructor(x)
        x = x.squeeze(0)
        e = time.time()
        print(f"x after unpatching: {x.shape}, time: {e-s}")

        return x



# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    factor10 = 2 # 1, 2, 3, 4 ok on my notebook. 5: cuda out of memory

    swin_dims, num_heads = 16, 4    
    H, W = int(720/10*factor10), int(1440/10*factor10)
    window_size = (2, int(factor10), int(factor10))
    patch_size = 8
    num_steps = 5
    num_swin_blocks = 30
    
    model = Zeus(H, W, num_heads, num_steps, patch_kernel=patch_size, patch_stride=patch_size, swin_dims=swin_dims, window_size=window_size, num_swin_blocks=num_swin_blocks)
    model.to(device)
    
    Snapshot.device = device
    Snapshot.export_factor10 = factor10
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, device=device)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, device=device)
    
    s = time.time()
    x_out = model(ss1, ss2)
    e = time.time()
    
    print(x_out.shape, "time: ", e-s)
    
