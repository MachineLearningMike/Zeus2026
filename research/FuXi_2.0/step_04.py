import torch
import torch.nn as nn
from swin_transformer_v2 import SwinTransformerBlock

# -----------------------
# 1️⃣ Preprocessing Module
# -----------------------
class FuXi2_Preprocessor(nn.Module):
    def __init__(self, B_m, C_m, C_g, H, W, patch_kernel=8, patch_stride=8):
        super().__init__()
        self.B_m, self.C_m = B_m, C_m
        self.C_g = C_g
        self.H, self.W = H, W

        self.conv_meteo = nn.Conv2d(B_m*C_m, B_m*C_m, kernel_size=patch_kernel, stride=patch_stride)
        self.conv_geo   = nn.Conv2d(C_g, C_g, kernel_size=patch_kernel, stride=patch_stride)
        self.Shock = nn.Parameter(torch.randn(H, W))

    def forward(self, x_meteo, x_geo):  # (B_m, C_m, H, W), (1, C_g, H, W)  -> (1, B_m*C_m + C_g, down_H, down_W)
        x_meteo = x_meteo + self.Shock
        x_geo   = x_geo   + self.Shock

        # Ensure contiguous memory before reshape
        x_meteo = x_meteo.contiguous().reshape(1, self.B_m*self.C_m, self.H, self.W)
        x_meteo = self.conv_meteo(x_meteo)

        x_geo = x_geo.contiguous().reshape(1, self.C_g, self.H, self.W)
        x_geo = self.conv_geo(x_geo)

        return torch.cat([x_meteo, x_geo], dim=1)


# -----------------------
# 2️⃣ Swin Module
# -----------------------
class FuXi2_Swin(nn.Module):
    def __init__(self, C_total, H_down, W_down, num_blocks=30, sWin=18):
        super().__init__()
        self.C_total = C_total
        self.H_down, self.W_down = H_down, W_down

        # choose number of heads
        for nh in reversed(range(1, 9)):
            if C_total % nh == 0:
                self.num_heads = nh
                break
        print(f"[INFO] C_total={C_total}, using num_heads={self.num_heads}")

        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=C_total,
                i_reso=(H_down, W_down),
                num_heads=self.num_heads,
                sWin=sWin
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):   # (1, B_m*C_m + C_g, down_H, down_W) -> (1, B_m*C_m + C_g, down_H, down_W)
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
class FuXi2_Postprocess(nn.Module):
    def __init__(self, C_total, C_m, patch_kernel=8, patch_stride=8):
        super().__init__()

        self.convT_meteo = nn.ConvTranspose2d(
            in_channels=C_total,
            out_channels=C_m,
            kernel_size=patch_kernel,
            stride=patch_stride
        )

    def forward(self, x):   # (1, B_m*C_m + C_g, down_H, down_W) -> (1, C_m, H, W)
        # ensure contiguous memory before ConvTranspose2d
        x = x.contiguous()
        x_m_up = self.convT_meteo(x)
        return x_m_up


# -----------------------
# 4️⃣ Full Model
# -----------------------
class FuXi2(nn.Module):
    def __init__(self, B_m, C_m, C_g, H, W,
                 C_out, num_steps, patch_kernel=8, patch_stride=8,
                 num_swin_blocks=30, sWin=18):
        super().__init__()

        self.preprocessor = FuXi2_Preprocessor(B_m, C_m, C_g, H, W, patch_kernel, patch_stride)

        H_down, W_down = H // patch_stride, W // patch_stride
        C_total = B_m*C_m + C_g

        self.swin = FuXi2_Swin(C_total, H_down, W_down,
                               num_blocks=num_swin_blocks,
                               sWin=sWin)

        self.postprocess = FuXi2_Postprocess(C_total, C_m, patch_kernel, patch_stride)

    def forward(self, x_meteo, x_geo):
        # print(f"x_meteo: {x_meteo.shape}, x_geo: {x_geo.shape} before preprocessor")
        x = self.preprocessor(x_meteo, x_geo)
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

    B_m, C_m = 2, 88
    C_g = 10
    factor10 = 4
    H, W = 72 * factor10, 144 * factor10
    C_out = 88
    num_steps = 6

    model = FuXi2(B_m, C_m, C_g, H, W, C_out, num_steps).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.1,
    )

    x_meteo = torch.randn(B_m, C_m, H, W, device=device)
    x_geo   = torch.randn(1, C_g, H, W, device=device)

    import time
    s = time.time()
    loops = 3
    for _ in range(loops):
        out = model(x_meteo, x_geo)
        loss = out.sum()
        loss.backward()
        optimizer.step()
    e = time.time()

    print("time:", (e-s)/loops)
    print("Output shape:", out.shape)
