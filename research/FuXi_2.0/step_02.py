import torch
import torch.nn as nn
from swin_transformer_v2 import SwinTransformerBlock

# -----------------------
# 1️⃣ Preprocessing Module
# -----------------------
class FuXi2_Preprocessor(nn.Module):
    def __init__(self, B_m, C_m, B_g, C_g, H, W, patch_kernel=8, patch_stride=8):
        super().__init__()
        self.B_m, self.C_m = B_m, C_m
        self.B_g, self.C_g = B_g, C_g
        self.H, self.W = H, W

        # Convolutions: preserve #channels
        self.conv_meteo = nn.Conv2d(B_m*C_m, B_m*C_m, kernel_size=patch_kernel, stride=patch_stride)
        self.conv_geo   = nn.Conv2d(B_g*C_g, B_g*C_g, kernel_size=patch_kernel, stride=patch_stride)

    def forward(self, x_meteo, x_geo):
        # x_meteo: (B_m, C_m, H, W)
        # x_geo:   (B_g, C_g, H, W)
        
        H, W = self.H, self.W
        
        # Merge batch × channel → channel, batch=1
        x_meteo = x_meteo.reshape(1, self.B_m*self.C_m, H, W)
        x_meteo = self.conv_meteo(x_meteo)  # (1, B_m*C_m, H/8, W/8)
        
        x_geo = x_geo.reshape(1, self.B_g*self.C_g, H, W)
        x_geo = self.conv_geo(x_geo)        # (1, B_g*C_g, H/8, W/8)
        
        # Concatenate along channel dim
        x = torch.cat([x_meteo, x_geo], dim=1)  # (1, C_total, H/8, W/8)
        return x


class FuXi2_Swin(nn.Module):
    def __init__(self, C_total, H_down, W_down, num_blocks=30, window_size=18):
        super().__init__()
        self.C_total = C_total
        self.H_down, self.W_down = H_down, W_down
        self.window_size = window_size

        # pick num_heads that divides C_total
        for nh in reversed(range(1, 9)):
            if C_total % nh == 0:
                self.num_heads = nh
                break
        print(f"[INFO] C_total={C_total}, using num_heads={self.num_heads}")

        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=C_total,
                input_resolution=(H_down, W_down),
                num_heads=self.num_heads,
                window_size=window_size
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x: (B, C_total, H_down, W_down)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H*W, C)                  # flatten → (B, L, C)
        
        for blk in self.swin_blocks:
            x = blk(x)  # Swin block expects (B, L, C)

        # reshape back to image grid
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x


# Demo
# -----------------------
B_m, C_m = 2, 88
B_g, C_g = 1, 10
H, W = 720, 1440
patch_stride = 8
H_down, W_down = H // patch_stride, W // patch_stride
C_total = B_m*C_m + B_g*C_g

x_meteo = torch.randn(B_m, C_m, H, W)
x_geo   = torch.randn(B_g, C_g, H, W)

# Step 1: Preprocessing
preprocessor = FuXi2_Preprocessor(B_m, C_m, B_g, C_g, H, W, patch_stride=patch_stride)
x = preprocessor(x_meteo, x_geo)
print("After preprocessing:", x.shape)  # (1, C_total, H/8, W/8)

x = torch.randn(1, C_total, H_down, W_down)
swin_model = FuXi2_Swin(C_total, H_down, W_down)
out = swin_model(x)
print("After Swin blocks:", out.shape)  # (1, C_total, H_down, W_down)
