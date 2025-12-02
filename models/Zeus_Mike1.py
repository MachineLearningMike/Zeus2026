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
    def __init__(self, pm, device='cpu'):
        super().__init__()
        
        # patch_kernel = pm.patch_kernel
        # patch_stride = pm.patch_stride
        # swin_dim = pm.swin_dim
        # window_size = pm.window_size
        # num_swin_blocks = pm.num_swin_block        

        self.factor10 = 2        
        swin_dim, num_heads = 16, 4    
        H, W = int(720/10*self.factor10), int(1440/10*self.factor10)
        window_size = (2, int(self.factor10), int(self.factor10))
        patch_kernel = 8
        patch_stride = 8
        num_swin_blocks = 30

        self.device = device
    
        #---------------- Patch extractor -----------------
        
        now = datetime.now(timezone.utc)
        ss1 = Snapshot(now.year, now.month, now.day, now.hour)
        then = now + timedelta(hours=6)
        ss2 = Snapshot(then.year, then.month, then.day, then.hour)
        levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
        
        self.patch_kernel=(patch_kernel,patch_kernel)
        self.patch_stride=(patch_stride,patch_stride)
        
        self.patch_extractor = PatchFeatureExtractor(levels, surface, c_out=swin_dim, patch_kernel=self.patch_kernel, patch_stride=self.patch_stride)
        
        #---------------- Swin blocks ----------------------
        
        input_resolution = (self.patch_extractor.nTotalLevels, H//patch_stride, W//patch_stride) 
        
        self.swin_blocks = nn.ModuleList([
            Zeus_SwinTransformerBlock3D(
                dim=swin_dim,
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
        self.uppatch_reconstructor = UnpatchReconstructor(in_channels=self.patch_extractor.nTotalLevels * swin_dim, out_channels=Snapshot.meteo_dims, kernel_size=self.patch_kernel, stride=self.patch_stride)

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

        x = x.view(-1, H, W)
        x = x.unsqueeze(0)
        s = time.time()
        x = self.uppatch_reconstructor(x)
        e = time.time()
        x = x.squeeze(0)
        print(f"x after unpatching: {x.shape}, time: {e-s}")

        return x



# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone
    import time
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    from models.config_Mike1 import config
    from modules.params_Mike1 import Params
    pm = Params(config)    
    
    model = Zeus(pm, device=device)
    model.to(device)

    Snapshot.device = device
    Snapshot.export_factor10 = model.factor10
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, device=device)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, device=device)
    
    s = time.time()
    x_out = model(ss1, ss2)
    e = time.time()
    
    print(x_out.shape, "time: ", e-s)
    
