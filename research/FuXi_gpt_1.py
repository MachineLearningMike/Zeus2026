import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from swin_transformer_v2 import SwinTransformerBlock, PatchMerging

# ---------------- Patch Embedding ----------------
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=(256,256), patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size, img_size[1]//patch_size)
        self.num_patches = self.grid_size[0]*self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x):
        x = self.proj(x)   # B,C,H,W -> B,embed,H',W'
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)  # B, num_patches, embed_dim
        x = self.norm(x)
        return x, (H,W)

# ---------------- FuXi 2.0 Multi-step ----------------
class FuXi2_MultiStep(nn.Module):
    def __init__(self,
                 img_size=(256,256),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2,2,6,2],
                 num_heads=[3,6,12,24],
                 window_size=7,
                 mlp_ratio=4.,
                 num_steps=6,  # predict t+1,...,t+6
                 num_vars=5   # variables per grid (e.g., temp, wind, etc.)
                 ):
        super().__init__()
        self.num_steps = num_steps
        self.num_vars = num_vars

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(0.)

        # build layers
        self.layers = nn.ModuleList()
        H,W = img_size[0]//patch_size, img_size[1]//patch_size
        dim = embed_dim
        for i_layer in range(len(depths)):
            layer_blocks = nn.ModuleList()
            for j in range(depths[i_layer]):
                shift = 0 if j%2==0 else window_size//2
                layer_blocks.append(SwinTransformerBlock(dim, (H,W), num_heads[i_layer],
                                                         window_size, shift, mlp_ratio=mlp_ratio))
            self.layers.append(nn.ModuleDict({
                "blocks": layer_blocks,
                "downsample": PatchMerging((H,W), dim) if (i_layer<len(depths)-1) else None
            }))
            if i_layer < len(depths)-1:
                dim *= 2
                H,W = H//2, W//2

        self.norm = nn.LayerNorm(dim)
        # Output head predicts all steps and variables per patch
        self.head = nn.Linear(dim, num_steps*num_vars)

    def forward(self,x):
        x, (H,W) = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            for blk in layer["blocks"]:
                x = blk(x)
            if layer["downsample"] is not None:
                x = layer["downsample"](x)
        x = self.norm(x)
        x = self.head(x)  # B, num_patches_final, num_steps*num_vars
        # reshape to grid
        B,N,C = x.shape
        H_final,W_final = int(N**0.5), int(N**0.5)
        x = x.view(B,H_final,W_final,self.num_steps,self.num_vars)
        return x

# ---------------- Demo ----------------
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FuXi2_MultiStep(img_size=(256,256), in_chans=3, num_steps=6, num_vars=5).to(device)
    dummy_input = torch.randn(2,3,256,256).to(device)
    output = model(dummy_input)
    print(output.shape)  # [2, H_final, W_final, num_steps, num_vars]
