import torch
import torch.nn as nn
from datetime import datetime, timezone

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Zeus_Swin_Block_3D_CDHW import Zeus_SwinTransformerBlock3D
from modules.zeus_dataset_Snapshot import Snapshot


# -----------------------
# 2️⃣ Swin Module
# -----------------------
class Mike1_Swin(nn.Module):
    def __init__(self, ch_input, D_down, H_down, W_down, num_heads, window_size=18, num_blocks=30):        
        super().__init__()
        self.ch_input, self.H_down, self.W_down, self.num_heads = ch_input, H_down, W_down, num_heads

        # We know the attention in the following transformer block cannot represent physical interaction between two geo-points,
        # even if we introduce position bias that is learnt from latitude-distortion.
        # Equal-area is necessary for simulating physical interaction, as the volume of grid cells are physical amounts.
        # We don't need positional encoding, as the token sequences are all of the same, full shape.

        # def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
        #             mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        #             act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):

        assert ch_input % num_heads == 0, f"ch_input {ch_input} is not divisible with num_heads {num_heads}"

        self.swin_blocks = nn.ModuleList([
            Zeus_SwinTransformerBlock3D(
                dim=ch_input,
                input_resolution=(H_down, W_down),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if idx % 2 == 0 else window_size//2,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0
            ) for idx in range(num_blocks)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        # permute and make contiguous before view
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, H*W, C)

        x_skip = x  # used in 6-hourly model only.

        for blk in self.swin_blocks:
            x = blk(x)      # (B, H*W, C)

        x = x + x_skip

        # reshape back to original (make contiguous for conv)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x    # (B, C, H, W)


# -----------------------
# 3️⃣ Postprocess (inverse of Preprocessor)
# -----------------------
class Reconstructor(nn.Module):
    def __init__(self, ch_input, ch_meteo, patch_kernel=8, patch_stride=8):
        super().__init__()

        self.convT_meteo = nn.ConvTranspose2d(
            in_channels=ch_input,
            out_channels=ch_meteo,
            kernel_size=patch_kernel,
            stride=patch_stride
        )

    def forward(self, x):   # (1, Batch_meteo*Channels_meteo + Channels_geo, down_H, down_W) -> (1, Channels_meteo, H, W)
        # ensure contiguous memory before ConvTranspose2d
        x = x.contiguous()
        x_m_up = self.convT_meteo(x)
        return x_m_up


# -----------------------
# 4️⃣ Full Model
# -----------------------
class Mike1(nn.Module):
    def __init__(self, ch_meteo, ch_meteo_tempo, ch_geo, H, W, num_heads,
                num_steps, patch_kernel=8, patch_stride=8, swin_dims=192,
                window_size=18, num_swin_blocks=30):
        super().__init__()

        self.preprocessor = Mike1_Preprocessor(ch_meteo_tempo, ch_geo, swin_dims, H, W, patch_kernel, patch_stride)

        h_down, w_down = H // patch_stride, W // patch_stride
        ch_total = ch_meteo_tempo * 2 + ch_geo

        self.swin = Mike1_Swin(swin_dims, h_down, w_down, 
                               num_heads=num_heads,
                               window_size=window_size,
                               num_blocks=num_swin_blocks,
                               )

        self.postprocess = Reconstructor(swin_dims, ch_meteo, patch_kernel, patch_stride)

    def forward(self, sp1, sp2):    # Not batch-processing, as hard drives are always bottlenecks.
        meteo_tempo1, meteo_tempo2, geo = sp1.meteo_tempo, sp2.meteo_tempo, sp1.geo
        x = torch.concat([meteo_tempo1, meteo_tempo2, geo], dim=0)
        x = x.unsqueeze(0)  # (1, C, H, W)
        x = self.preprocessor(x)
        # print("x after preprocessor: ", x.shape)
        x = self.swin(x)
        # print("x after swin: ", x.shape)
        x_meteo_recon = self.postprocess(x)
        # print("x after postprocessor: ", x_meteo_recon.shape)
        return x_meteo_recon.squeeze(0)


# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    from datetime import datetime, timedelta, timezone
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    now = datetime.now(timezone.utc)
    sp1 = Snapshot(now.year, now.month, now.day, now.hour)
    then = now + timedelta(hours=6)
    sp2 = Snapshot(then.year, then.month, then.day, then.hour)

    Snapshot.export_factor10 = 2
    Snapshot.device = device
    swin_dims = 192 # This is the nearest versatile number to the ugly 194.
    num_heads = 4

    ch_meteo, ch_meteo_tempo, ch_geo = Snapshot.meteo_dims, Snapshot.meteo_dims + Snapshot.tempo_dims, Snapshot.geo_dims
    H, W = Snapshot.get_export_size()
    num_steps = 6

    model = Mike1(ch_meteo, ch_meteo_tempo, ch_geo, H, W, num_heads, num_steps, swin_dims=192).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.1,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    import time
    s = time.time()
    loops = 3
    with torch.no_grad():
        for _ in range(loops):
            out = model(sp1, sp2)
            sp1 = sp2
            dt = datetime(sp2.year, sp2.month, sp2.day, sp2.hour) + timedelta(hours=6)
            # out = out.detach()
            # out.requires_grad = False
            loss = out.sum()
            # loss.backward()
            # optimizer.step()
        e = time.time()
    print("time:", (e-s)/loops)
    print("Output shape:", out.shape)

    s = time.time()
    loops = 3
    for _ in range(loops):
        out = model(sp1, sp2)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    e = time.time()
    print("time:", (e-s)/loops)
    print("Output shape:", out.shape)
