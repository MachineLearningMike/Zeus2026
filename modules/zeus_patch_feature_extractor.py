import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchFeatureExtractor(nn.Module):
    """
    Extracts patch-based features from a 2D tensor using convolution, activation, and normalization.
    Input: (C_in, H, W)
    Output: (C_out, nWh, nWw)
    """
    def __init__(self, C_in, C_out, patch_size=(4,4), norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        """
        Args:
            C_in: number of input channels
            C_out: number of output channels
            patch_size: (ph, pw) patch size
            norm_layer: normalization layer applied per patch
            act_layer: activation function
        """
        super().__init__()
        ph, pw = patch_size
        self.ph = ph
        self.pw = pw

        # Conv2d with kernel = patch size and stride = patch size → non-overlapping patches
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=(ph, pw), stride=(ph, pw))
        self.norm = nn.LayerNorm(C_out)
        self.act = act_layer()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (C_in, H, W)
        Returns:
            out: Tensor of shape (C_out, nWh, nWw)
        """
        x = self.conv(x)    # (1, C_out, nWh, nWw)
        
        # Permute to (nWh*nWw, C_out) for LayerNorm
        B, C, nWh, nWw = x.shape
        x_ln = x.permute(0,2,3,1).contiguous()  # (1, nWh, nWw, C)
        x_ln = self.norm(x_ln)
        x_ln = self.act(x_ln)
        # Permute back
        out = x_ln.permute(0,3,1,2).contiguous().squeeze(0)  # (C_out, nWh, nWw)
        return out


class LevelIntegrator(nn.Module):
    def __init__(self, sample_level_list, c_out, patch_size=(8,8)):
        """
        Args:
            sample_level_list: list of example tensors per level (to get input channels)
            c_out: number of output channels per level
        """
        super().__init__()
        self.c_out = c_out
        self.nLevels = len(sample_level_list)
        self.channels = [level.shape[0] for level in sample_level_list]
        
        self.extractors = nn.ModuleList(
            [PatchFeatureExtractor(c_in, c_out, patch_size=patch_size) for c_in in self.channels]
        )
        
    def forward(self, level_list):
        """
        Args:
            level_list: list of tensors [(C1,H,W), (C2,H,W), ...] length == self.nLevels
        Returns:
            Tensor: (c_out, nLevels, H//8, W//8)
        """
        norm_levels = []
        
        H_prev, W_prev = None, None
        for idx, level in enumerate(level_list):
            C, H, W = level.shape

            if H_prev is None: H_prev = H
            else: assert H_prev == H
            if W_prev is None: W_prev = W
            else: assert W_prev == W

            assert self.channels[idx] == C

            level = level.unsqueeze(0)
            a_norm_level = self.extractors[idx](level)
            norm_levels.append(a_norm_level)    # (c_out, H//8, W//8)
        
        normal_stack = torch.stack(norm_levels, dim=1)  # (c_out, nLevels, H//8, W//8)
        return normal_stack

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    x = torch.randn(2, 3, 16, 16)  # (C_in=3, H=16, W=16)
    patch_size = (4, 4)
    extractor = PatchFeatureExtractor(C_in=3, C_out=8, patch_size=patch_size)
    out = extractor(x)
    print(out.shape)  # expected: (8, 4, 4)

    from zeus_dataset_Snapshot import Snapshot
    from datetime import datetime, timedelta, timezone
    
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour)
    
    levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    for idx, lvl in enumerate(levels):
        print(idx, lvl.shape)
        break
    print(14, surface.shape)
    
    levels.append(surface)
    
    integrator = LevelIntegrator(levels, c_out=20)
    integral_tensor = integrator(levels)        # (c_out, 14, 90, 180)
    print(integral_tensor.shape)