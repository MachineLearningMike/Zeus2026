import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchFeatureExtractor_Level(nn.Module):
    """
    Stronger patch-based feature extractor:
    Conv → LN → GELU → MLP → LN
    """
    def __init__(self, C_in, C_out,
                 patch_kernel=(8,8), patch_stride=(8,8),
                 hidden_ratio=2,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU):
        
        super().__init__()

        # Patch embedding
        self.conv = nn.Conv2d(C_in, C_out,
                              kernel_size=patch_kernel,
                              stride=patch_stride)

        self.norm1 = norm_layer(C_out)
        self.act = act_layer()

        # Very lightweight MLP head
        hidden_dim = C_out * hidden_ratio
        self.mlp = nn.Sequential(
            nn.Linear(C_out, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, C_out),
        )

        self.norm2 = norm_layer(C_out)

    def forward(self, x):
        # 1) Patch conv
        x = self.conv(x)  # (1, C_out, nWh, nWw)
        B, C, nWh, nWw = x.shape

        # 2) LN + GELU
        x = x.permute(0,2,3,1)
        x = self.norm1(x)
        x = self.act(x)

        # 3) MLP
        x = self.mlp(x)

        # 4) LN again
        x = self.norm2(x)

        # Final shape (C_out, nWh, nWw)
        x = x.permute(0,3,1,2).contiguous()

        return x

import torch
import torch.nn as nn

class ConvStemResidualBlock(nn.Module):
    """Small residual block used in the stem with LayerNorm."""
    def __init__(self, channels, expansion=4, norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        hidden = channels // expansion if expansion > 1 else channels
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm1 = norm_layer(hidden)  # LayerNorm expects channels-last
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(channels)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        out = self.conv1(x)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = self.norm1(out)
        out = self.act(out)
        out = out.permute(0, 3, 1, 2)

        out = self.conv2(out)
        out = out.permute(0, 2, 3, 1)
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2)

        return self.act(out + x)  # residual


class PatchFeatureExtractor_Surface(nn.Module):
    """
    Resolution-preserving ConvStem using LayerNorm:
    - conv1: overlap but no downsample
    - conv2: refine but no downsample
    - res_blocks: same resolution
    - proj: the ONLY downsample, controlled by patch_stride
    """
    def __init__(self,
                 c_in,
                 c_out,
                 stem_channels=None,
                 num_res_blocks=2,
                 patch_kernel=(8,8),
                 patch_stride=(8,8),
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 use_overlap=True):
        super().__init__()
        if stem_channels is None:
            stem_channels = max(c_out // 2, 64)

        ph, pw = patch_kernel
        sh, sw = patch_stride

        k1 = 7 if use_overlap else 3
        p1 = k1 // 2

        # Initial convs: keep resolution
        self.conv1 = nn.Conv2d(c_in, stem_channels, kernel_size=k1, stride=1, padding=p1, bias=False)
        self.norm1 = norm_layer(stem_channels)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(stem_channels)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ConvStemResidualBlock(stem_channels,
                                   expansion=2,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer)
              for _ in range(num_res_blocks)]
        )

        # Final projection (downsample to patch grid)
        self.proj = nn.Conv2d(stem_channels, c_out, kernel_size=(ph, pw), stride=(sh, sw), bias=False)
        self.norm_out = norm_layer(c_out)

    def forward(self, x):
        # x: (B, C_in, H, W)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        x = self.res_blocks(x)

        # Final projection (downsample)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm_out(x)
        x = x.permute(0, 3, 1, 2)

        return x


class PatchFeatureExtractor(nn.Module):
    def __init__(self, sample_level_list, sample_surface, c_out, patch_kernel=(8,8), patch_stride=(8,8)):
        """
        Args:
            sample_level_list: list of example tensors per level (to get input channels)
            c_out: number of output channels per level
        """
        super().__init__()
        self.c_out = c_out
        self.channels = [level.shape[0] for level in sample_level_list]
        
        self.level_extractors = nn.ModuleList(
            [
                PatchFeatureExtractor_Level(
                    c_in, 
                    c_out, 
                    patch_kernel=patch_kernel, 
                    patch_stride=patch_stride
                    ) for c_in in self.channels
            ]
        )
        
        c_in = sample_surface.shape[0]
        
        self.surface_extractor = PatchFeatureExtractor_Surface(
            c_in,
            c_out,
            stem_channels=None,
            num_res_blocks=0,   #----------- save time ----------
            patch_kernel=(8,8),
            patch_stride=(8,8),
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_overlap=True
        )
        
        self.nTotalLevels = len(sample_level_list) + 1      # +1 for the surface

    def forward(self, level_list, surface):
        """
        Args:
            level_list: list of tensors [(C1,H,W), (C2,H,W), ...] length == self.nLevels
        Returns:
            Tensor: (c_out, nLevels, H//8, W//8)
        """
        levels_ext = []
        
        H_prev, W_prev = None, None
        for idx, level in enumerate(level_list):
            C, H, W = level.shape

            if H_prev is None: H_prev = H
            else: assert H_prev == H
            if W_prev is None: W_prev = W
            else: assert W_prev == W

            assert self.channels[idx] == C

            level = level.unsqueeze(0)
            level = self.level_extractors[idx](level)
            level = level.squeeze(0)
            levels_ext.append(level)    # (c_out, H//8, W//8)
        
        surface = surface.unsqueeze(0)
        surface_ext = self.surface_extractor(surface)
        surface_ext = surface_ext.squeeze(0)
        
        levels_ext.append(surface_ext)

        level_extraction = torch.stack(levels_ext, dim=1)  # (c_out, nLevels, H//8, W//8)
        
        return level_extraction
    

class UnpatchReconstructor(nn.Module):
    """
    Simple learnable reconstruction from 2D patches with horizontal wrap-around.
    Input: (C, H, W)
    Output: (out_channels, H*stride[0], W*stride[1])
    """
    def __init__(self, in_channels, out_channels, kernel_size=(8,8), stride=(8,8)):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (B, C, H, W) -> add batch dimension
        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        # Compute target output size
        H_target, W_target = H * self.stride[0], W * self.stride[1]

        # Slightly extra width to allow horizontal wrap
        extra_w = max(0, self.kernel_size[1] - self.stride[1])
        W_out = W_target + extra_w

        # Apply deconv with output_size
        out = self.deconv(x, output_size=(1, self.deconv.out_channels, H_target, W_out))
        
        # Horizontal cylindric wrap
        if extra_w > 0:
            out = out[..., :W_target] + out[..., W_target:W_out]

        return out

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from zeus_dataset_Snapshot import Snapshot
    from datetime import datetime, timedelta, timezone
    
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour)
    
    Snapshot.device = device
    levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    
    patch_kernel=(8,8)
    patch_stride=(8,8)
    latent_dim = 16
    
    patcher = PatchFeatureExtractor(levels, surface, c_out=latent_dim, patch_kernel=patch_kernel, patch_stride=patch_stride)
    patcher.to(device)
    
    import time
    loops = 3
    s = time.time()
    for _ in range(loops):
        patches = patcher(levels, surface)        # (c_out, 14, 90, 180)
    e = time.time()
    print("packined to: ", patches.shape, "time: ", (e-s)/loops) 
    
    patches = patches.view(-1, patches.shape[-2], patches.shape[-1])    # (C, H, W)
    print(patches.shape)
    
    unpatcher = UnpatchReconstructor(in_channels=patches.shape[0], out_channels=Snapshot.meteo_dims, kernel_size=patch_kernel, stride=patch_kernel)
    unpatcher.to(device)    
    
    patches = patches.unsqueeze(0)
    loops = 3
    s = time.time()
    for _ in range(loops):
        origin = unpatcher(patches)
    e = time.time()
    origin = origin.squeeze(0)
    print("unpacked to: ", origin.shape, "time: ", (e-s)/loops)
    
