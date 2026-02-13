import torch
import torch.nn as nn

import torch
import torch.nn as nn

class PatchMerging_4X_Dim(nn.Module):
    anchor = 2
    res_multiplier = (1/anchor, 1/anchor)
    dim_multiplier = anchor ** 2

    def __init__(self):
        super().__init__()        
        # Temporarily remove norm for exact inversion test
        # self.norm = norm_layer(4 * dim)

    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        x = x.view(B, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//2, W//2, 4 * C)
        return x

class PatchSplitting_4X_Dim(nn.Module):
    anchor = 2    
    res_multiplier = (anchor, anchor)
    dim_multiplier = 1 / anchor**2

    def __init__(self):
        super().__init__()
    
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Reverse the operations
        x = x.view(B, H, W, 2, 2, C//4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C//4)
        return x

if __name__ == "__main__":

    # Test with exact values
    def test_exact_inverse():
        B, H, W, C = 1, 4, 4, 1
        
        # Create tensor with unique values
        x = torch.arange(H * W).float().view(1, H, W, C)
        print("Original tensor:")
        print(x[0,:,:,0])
        
        # Merge
        merge = PatchMerging_4X_Dim()
        x_merged = merge(x)
        print(f"\nMerged shape: {x_merged.shape}")
        print("Merged values at position (0,0):", x_merged[0,0,0])
        
        # Split
        split = PatchSplitting_4X_Dim()
        x_restored = split(x_merged)
        print(f"\nRestored shape: {x_restored.shape}")
        print("Restored tensor:")
        print(x_restored[0,:,:,0])
        
        # Check exact equality
        print(f"\nExact match? {torch.allclose(x, x_restored)}")
        print(f"Max difference: {torch.max(torch.abs(x - x_restored))}")
        
        return x, x_merged, x_restored

    x, x_merged, x_restored = test_exact_inverse()