import torch
import torch.nn as nn

class PatchFeatureExtractor_Level(nn.Module):
    def __init__(self, C_in, C_out, patch_kernel=(8,8), patch_stride=(8,8), hidden_ratio=2, act_layer=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=patch_kernel, stride=patch_stride)
        self.norm1 = nn.GroupNorm(1, C_out)
        self.act = act_layer()
        hidden_dim = C_out * hidden_ratio
        self.mlp = nn.Sequential(
            nn.Linear(C_out, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, C_out)
        )
        self.norm2 = nn.GroupNorm(1, C_out)

    def forward(self, x):
        # x: (1, C, H, W)
        x = self.conv(x)
        x = self.norm1(x)
        x = self.act(x)
        _, C, H, W = x.shape
        x_flat = x.permute(0,2,3,1).reshape(-1, C)
        x_flat = self.mlp(x_flat)
        x = x_flat.view(1, H, W, C).permute(0,3,1,2)
        x = self.norm2(x)
        return self.act(x)

class ConvStemResidualBlock(nn.Module):
    def __init__(self, channels, expansion=4, act_layer=nn.GELU):
        super().__init__()
        hidden = channels // expansion if expansion > 1 else channels
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.norm1 = nn.GroupNorm(1, hidden)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + x)

class PatchFeatureExtractor_Surface(nn.Module):
    def __init__(self, c_in, c_out, num_res_blocks=0, patch_kernel=(8,8), patch_stride=(8,8), act_layer=nn.GELU):
        super().__init__()
        stem_channels = max(c_out//2, 64)
        self.conv1 = nn.Conv2d(c_in, stem_channels, 7, stride=1, padding=3, bias=False)
        self.norm1 = nn.GroupNorm(1, stem_channels)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(stem_channels, stem_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(1, stem_channels)
        self.res_blocks = nn.Sequential(*[ConvStemResidualBlock(stem_channels) for _ in range(num_res_blocks)])
        self.proj = nn.Conv2d(stem_channels, c_out, kernel_size=patch_kernel, stride=patch_stride, bias=False)
        self.norm_out = nn.GroupNorm(1, c_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.res_blocks(x)
        x = self.proj(x)
        x = self.norm_out(x)
        return self.act(x)

class PatchFeatureExtractor(nn.Module):
    """Batch-free patch extractor for multiple levels + surface, with cascade_step"""
    def __init__(self, levels_list, surface, c_out, dim_for_cascade_embedding=1, max_cascades=10, patch_kernel=(8,8), patch_stride=(8,8)):
        super().__init__()
        # Channels
        self.level_channels_list = [level.shape[0] + dim_for_cascade_embedding for level in levels_list]
        self.surface_channels = surface.shape[0] + dim_for_cascade_embedding
        self.dim_for_cascade_embedding = dim_for_cascade_embedding
        self.max_cascades = max_cascades

        # Level extractors
        self.level_extractors = nn.ModuleList([
            PatchFeatureExtractor_Level(c_in, c_out, patch_kernel, patch_stride)
            for c_in in self.level_channels_list
        ])

        # Surface extractor
        self.surface_extractor = PatchFeatureExtractor_Surface(
            c_in=self.surface_channels,
            c_out=c_out,
            num_res_blocks=0,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride
        )

        # Cascade embeddings
        H, W = surface.shape[1], surface.shape[2]  # no batch dim
        cascade_shape = (dim_for_cascade_embedding, H, W)
        self.cascade_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(cascade_shape)) for _ in range(max_cascades)
        ])

    def forward(self, levels_list, surface, cascade_step=0):
        # Get cascade embedding
        cascade_embeds = self.cascade_embeds[min(cascade_step, self.max_cascades-1)]
        cascade_embeds = cascade_embeds.unsqueeze(0)  # add batch dim for conv: (1, dim, H, W)

        # Process levels
        levels_ext = []
        for i, level in enumerate(levels_list):
            level_cat = torch.cat([level.unsqueeze(0), cascade_embeds], dim=1)  # (1, C+dim, H, W)
            levels_ext.append(self.level_extractors[i](level_cat).squeeze(0))  # remove batch dim

        # Surface
        surface_cat = torch.cat([surface.unsqueeze(0), cascade_embeds], dim=1)
        surface_ext = self.surface_extractor(surface_cat).squeeze(0)

        levels_ext.append(surface_ext)

        # Stack along level dimension
        return torch.stack(levels_ext, dim=1)  # (C_out, nLevels+1, H_patch, W_patch)


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
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
    
    Snapshot.device = device
    levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    
    patch_kernel=(8,8)
    patch_stride=(8,8)
    latent_dim = 16
    
    patcher = PatchFeatureExtractor(levels, surface, c_out=latent_dim, patch_kernel=patch_kernel, patch_stride=patch_stride)
    patcher.to(device)
    
    import time
    loops = 5
    s = time.time()
    for step in range(loops):
        patches = patcher(levels, surface, cascade_step = step)        # (c_out, 14, 90, 180)
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
    
