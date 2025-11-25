import torch
import torch.nn as nn

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Zeus_Swin_Block import Zeus_SwinTransformerBlock
from modules.zeus_dataset_Snapshot import Snapshot

# -----------------------
# 1️⃣ Preprocessing Module
# -----------------------
class Mike1_Preprocessor(nn.Module):
    def __init__(self, Batch_meteo, Channels_meteo, Channels_geo, H, W, patch_kernel=8, patch_stride=8):
        super().__init__()
        self.Batch_meteo, self.Channels_meteo = Batch_meteo, Channels_meteo
        self.Channels_geo = Channels_geo
        self.H, self.W = H, W

        self.conv_meteo = nn.Conv2d(Batch_meteo*Channels_meteo, Batch_meteo*Channels_meteo, kernel_size=patch_kernel, stride=patch_stride)
        self.conv_geo   = nn.Conv2d(Channels_geo, Channels_geo, kernel_size=patch_kernel, stride=patch_stride)
        self.Shock = nn.Parameter(torch.randn(H, W))    # Considering removing this.

    def forward(self, x_snapshot):  # (Batch_meteo, Channels_meteo, H, W), (1, Channels_geo, H, W)  -> (1, Batch_meteo*Channels_meteo + Channels_geo, down_H, down_W)
        x_meteo = x_meteo + self.Shock
        x_geo   = x_geo   + self.Shock

        # Ensure contiguous memory before reshape
        x_meteo = x_meteo.contiguous().reshape(1, self.Batch_meteo*self.Channels_meteo, self.H, self.W)
        x_meteo = self.conv_meteo(x_meteo)

        x_geo = x_geo.contiguous().reshape(1, self.Channels_geo, self.H, self.W)
        x_geo = self.conv_geo(x_geo)

        return torch.cat([x_meteo, x_geo], dim=1)


# -----------------------
# 2️⃣ Swin Module
# -----------------------
class Mike1_Swin(nn.Module):
    def __init__(self, C_total, H_down, W_down, window_size=18, num_blocks=30):
        super().__init__()
        self.C_total = C_total
        self.H_down, self.W_down = H_down, W_down

        # choose number of heads
        for nh in reversed(range(1, 9)):
            if C_total % nh == 0:
                self.num_heads = nh
                break
        print(f"[INFO] C_total={C_total}, using num_heads={self.num_heads}")

        # We know the attention in the following transformer block cannot represent physical interaction between two geo-points,
        # even if we introduce position bias that is learnt from latitude-distortion.
        # Equal-area is necessary for simulating physical interaction, as the volume of grid cells are physical amounts.
        # We don't need positional encoding, as the token sequences are all of the same, full shape.

        # def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
        #             mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        #             act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):

        num_heads = 89
        assert C_total % num_heads == 0, f"C_total {C_total} is not divisible with num_heads {num_heads}"

        self.swin_blocks = nn.ModuleList([
            Zeus_SwinTransformerBlock(
                dim=C_total,
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

    def forward(self, x):   # (1, Batch_meteo*Channels_meteo + Channels_geo, down_H, down_W) -> (1, Batch_meteo*Channels_meteo + Channels_geo, down_H, down_W)
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
class Mike1_Postprocess(nn.Module):
    def __init__(self, C_total, Channels_meteo, patch_kernel=8, patch_stride=8):
        super().__init__()

        self.convT_meteo = nn.ConvTranspose2d(
            in_channels=C_total,
            out_channels=Channels_meteo,
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
    def __init__(self, Batch_meteo, Channels_meteo, Channels_geo, H, W,
                 C_out, num_steps, patch_kernel=8, patch_stride=8,
                 window_size=18,
                 num_swin_blocks=30):
        super().__init__()

        self.preprocessor = Mike1_Preprocessor(Batch_meteo, Channels_meteo, Channels_geo, H, W, patch_kernel, patch_stride)

        H_down, W_down = H // patch_stride, W // patch_stride
        C_total = Batch_meteo*Channels_meteo + Channels_geo

        self.swin = Mike1_Swin(C_total, H_down, W_down,
                               window_size=window_size,
                               num_blocks=num_swin_blocks,
                               )

        self.postprocess = Mike1_Postprocess(C_total, Channels_meteo, patch_kernel, patch_stride)

    def forward(self, x_snapshot):
        # print(f"x_meteo: {x_meteo.shape}, x_geo: {x_geo.shape} before preprocessor")
        x = self.preprocessor(x_snapshot)
        # print("x after preprocessor: ", x.shape)
        x = self.swin(x)
        # print("x after swin: ", x.shape)
        x_meteo_recon = self.postprocess(x)
        # print("x after postprocessor: ", x_meteo_recon.shape)
        return x_meteo_recon


# -----------------------
# Demo
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Batch_meteo, Channels_meteo = 2, 88
    Channels_geo = 2
    Channels_tempo = 4
    factor10 = 2
    H, W = 72 * factor10, 144 * factor10
    C_out = Channels_meteo
    num_steps = 6

    model = Mike1(Batch_meteo, Channels_meteo, Channels_geo, H, W, C_out, num_steps).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.1,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    x_meteo = torch.randn(Batch_meteo, Channels_meteo, H, W, device=device)
    x_geo   = torch.randn(1, Channels_geo, H, W, device=device)
    x_tempo   = torch.randn(1, Channels_tempo, H, W, device=device)

    import time
    s = time.time()
    loops = 3
    for _ in range(loops):
        out = model(x_meteo, x_geo, x_tempo)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    e = time.time()

    print("time:", (e-s)/loops)
    print("Output shape:", out.shape)
