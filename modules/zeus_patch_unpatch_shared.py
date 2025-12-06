import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedLevelExtractor(nn.Module):
    def __init__(self, c_in, c_out, n_levels, patch_kernel=(8,8), patch_stride=(8,8), dim_for_cascade_embed=1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.n_levels = n_levels
        self.dim_for_cascade_embed = dim_for_cascade_embed
        self.c_in = c_in
        self.c_out = c_out
        
        # level embeddings
        self.level_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(c_in,1,1)) for _ in range(n_levels)
        ])
        
        total_in_channels = c_in + c_in + dim_for_cascade_embed  # level + level_embed + cascade_embed
        self.conv = nn.Conv2d(total_in_channels, c_out, kernel_size=patch_kernel, stride=patch_stride)
        self.norm = norm_layer(c_out)
        self.act = act_layer()

    def forward(self, level_list, cascade_embed):
        level_features = []
        for idx, level in enumerate(level_list):
            x = level + self.level_embeds[idx] + cascade_embed
            level_features.append(x)
        stacked = torch.stack(level_features, dim=0)  # (n_levels, C, H, W)
        B, C, H, W = stacked.shape
        stacked = stacked.view(B, C, H, W)  # merge batch and level as needed
        out = self.conv(stacked)
        out = out.permute(0,2,3,1)
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0,3,1,2)
        return out

class SurfaceExtractor(nn.Module):
    def __init__(self, c_in, c_out, patch_kernel=(8,8), patch_stride=(8,8), dim_for_cascade_embed=1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_for_cascade_embed = dim_for_cascade_embed
        total_in_channels = c_in + dim_for_cascade_embed
        self.conv = nn.Conv2d(total_in_channels, c_out, kernel_size=patch_kernel, stride=patch_stride)
        self.norm = norm_layer(c_out)
        self.act = act_layer()

    def forward(self, surface, cascade_embed):
        x = surface + cascade_embed
        out = self.conv(x)
        out = out.permute(0,2,3,1)
        out = self.norm(out)
        out = self.act(out)
        out = out.permute(0,3,1,2)
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

    latent_dim = 16
    dim_for_cascade_embed = 1

    cascade_embed = torch.randn(dim_for_cascade_embed, levels[0].shape[1], levels[0].shape[2]).to(device)

    shared_extractor = SharedLevelExtractor(c_in=levels[0].shape[0], c_out=latent_dim, n_levels=len(levels), dim_for_cascade_embed=dim_for_cascade_embed).to(device)
    surface_extractor = SurfaceExtractor(c_in=surface.shape[0], c_out=latent_dim, dim_for_cascade_embed=dim_for_cascade_embed).to(device)

    levels = [l.to(device) for l in levels]
    surface = surface.to(device)

    import time
    loops = 5
    s = time.time()
    for _ in range(loops):
        level_features = shared_extractor(levels, cascade_embed)
        surface_feature = surface_extractor(surface, cascade_embed)
    e = time.time()
    print(f"Time per forward pass: {(e-s)/loops:.6f} seconds")
    print(f"Level features shape: {level_features.shape}")
    print(f"Surface feature shape: {surface_feature.shape}")