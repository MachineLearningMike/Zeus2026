import torch
import torch.nn as nn

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_HX_2X import Reshape_HX_2X, Reshape_2X_HX

class Reshape_QX_1X(nn.Module):     # Quarter, 1X
   
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res[0] % 4 == 0 and in_res[1] % 4 == 0, "Input must have resolution divisible by 4."
        out_res = tuple([in_res[i] // 4 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim  # Keep dimension constant
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        # Two-stage merging: first 2x, then another 2x, but keep dim constant
        # Stage1: dim -> dim (but with 2x spatial reduction)
        # Stage2: dim -> dim (but with another 2x spatial reduction)
        
        # For stage1: input dim, after merging 2x2 patches we get 4*dim, then reduce to dim
        self.stage1_reduction = nn.Linear(4 * dim, dim, bias=True)
        
        # For stage2: input dim, after merging 2x2 patches we get 4*dim, then reduce to dim
        self.stage2_reduction = nn.Linear(4 * dim, dim, bias=True)
        
        # Optional: Add normalization and activation between stages for better learning
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        assert C == self.dim, f"Input channels {C} must equal {self.dim}"
        assert H % 4 == 0 and W % 4 == 0, f"Input height {H} and width {W} must be divisible by 4."
        
        # Stage 1: (H, W, dim) -> (H//2, W//2, dim)
        # First merge 2x2 patches: (H, W, dim) -> (H//2, W//2, 4*dim)
        x = x.view(B, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//2, W//2, 4 * C)
        
        # Then reduce back to dim
        x = self.stage1_reduction(x)
        
        # Optional normalization and activation between stages
        x = self.norm(x)
        x = self.act(x)
        
        # Stage 2: (H//2, W//2, dim) -> (H//4, W//4, dim)
        # Merge 2x2 patches: (H//2, W//2, dim) -> (H//4, W//4, 4*dim)
        x = x.view(B, H//4, 2, W//4, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//4, W//4, 4 * C)
        
        # Reduce back to dim
        x = self.stage2_reduction(x)
        
        return x


class Reshape_4X_1X(nn.Module):

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = tuple([in_res[i] * 4 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim  # Keep dimension constant
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        # Two-stage splitting: first 2x, then another 2x, keep dim constant
        # Stage1: dim -> dim (but with 2x spatial expansion)
        # Stage2: dim -> dim (but with another 2x spatial expansion)
        
        # For stage1: expand from dim to 4*dim, then split into 2x2 patches
        self.stage1_expansion = nn.Linear(dim, 4 * dim, bias=True)
        
        # For stage2: expand from dim to 4*dim, then split into 2x2 patches
        self.stage2_expansion = nn.Linear(dim, 4 * dim, bias=True)
        
        # Optional: Add normalization and activation between stages
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        assert C == self.dim, f"Input channels {C} must equal {self.dim}"
        
        # Stage 1: (H, W, dim) -> (H*2, W*2, dim)
        # First expand channels: (H, W, dim) -> (H, W, 4*dim)
        x = self.stage1_expansion(x)
        
        # Then split into 2x2 patches: (H, W, 4*dim) -> (H*2, W*2, dim)
        x = x.view(B, H, W, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C)
        
        # Optional normalization and activation between stages
        x = self.norm(x)
        x = self.act(x)
        
        # Stage 2: (H*2, W*2, dim) -> (H*4, W*4, dim)
        # First expand channels: (H*2, W*2, dim) -> (H*2, W*2, 4*dim)
        x = self.stage2_expansion(x)
        
        # Then split into 2x2 patches: (H*2, W*2, 4*dim) -> (H*4, W*4, dim)
        x = x.view(B, H*2, W*2, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 4, W * 4, C)
        
        return x


if __name__ == "__main__":
    # Test to verify dimensions work correctly
    def test_simple_dimensions():
        print("\n\n=== Simple Dimension Test for 4X (Same Dim) ===")
        B, H, W, C = 2, 16, 16, 32  # H and W divisible by 4
        
        # Create random tensor
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        # Merge
        merge = Reshape_QX_1X(dim=C)
        x_merged = merge(x)
        print(f"After Reshape_QX_1X: {x_merged.shape}")
        print(f"  Spatial: {H}x{W} -> {H//4}x{W//4}")
        print(f"  Channels: {C} -> {x_merged.shape[-1]} (should be same)")
        
        # Split
        split = Reshape_4X_1X(dim=C)
        x_restored = split(x_merged)
        print(f"\nAfter Reshape_4X_1X: {x_restored.shape}")
        print(f"  Spatial: {H//4}x{W//4} -> {x_restored.shape[1]}x{x_restored.shape[2]}")
        print(f"  Channels: {x_merged.shape[-1]} -> {x_restored.shape[-1]} (should be same)")
        
        return x, x_merged, x_restored
    
    # Test intermediate shapes
    def test_intermediate_shapes():
        print("\n\n=== Intermediate Shape Test ===")
        B, H, W, C = 1, 16, 16, 16
        
        x = torch.randn(B, H, W, C)
        print(f"Input: {x.shape}")
        
        merge = Reshape_QX_1X(dim=C)
        split = Reshape_4X_1X(dim=C)
        
        # Manually compute intermediate results for merging
        # Stage1: merge 2x2 patches and reduce
        x_stage1_merged = x.view(B, H//2, 2, W//2, 2, C)
        x_stage1_merged = x_stage1_merged.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_stage1_merged = x_stage1_merged.view(B, H//2, W//2, 4*C)
        x_stage1_merged = merge.stage1_reduction(x_stage1_merged)
        print(f"After stage1 merge: {x_stage1_merged.shape} (should be 8x8x16)")
        
        # Apply norm and act
        x_stage1_merged = merge.norm(x_stage1_merged)
        x_stage1_merged = merge.act(x_stage1_merged)
        
        # Stage2
        x_stage2_merged = x_stage1_merged.view(B, H//4, 2, W//4, 2, C)
        x_stage2_merged = x_stage2_merged.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_stage2_merged = x_stage2_merged.view(B, H//4, W//4, 4*C)
        x_stage2_merged = merge.stage2_reduction(x_stage2_merged)
        print(f"After stage2 merge: {x_stage2_merged.shape} (should be 4x4x16)")
        
        # Test splitting
        # Stage1 split: expand and split
        x_stage1_split = split.stage1_expansion(x_stage2_merged)
        x_stage1_split = x_stage1_split.view(B, H//4, W//4, 2, 2, C)
        x_stage1_split = x_stage1_split.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_stage1_split = x_stage1_split.view(B, H//2, W//2, C)
        print(f"\nAfter stage1 split: {x_stage1_split.shape} (should be 8x8x16)")
        
        # Apply norm and act
        x_stage1_split = split.norm(x_stage1_split)
        x_stage1_split = split.act(x_stage1_split)
        
        # Stage2 split
        x_stage2_split = split.stage2_expansion(x_stage1_split)
        x_stage2_split = x_stage2_split.view(B, H//2, W//2, 2, 2, C)
        x_stage2_split = x_stage2_split.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_stage2_split = x_stage2_split.view(B, H, W, C)
        print(f"After stage2 split: {x_stage2_split.shape} (should be 16x16x16)")
    
    # Test with identity matrices
    def test_with_identity():
        print("\n\n=== Identity Test (Same Dim) ===")
        B, H, W, C = 1, 8, 8, 8
        
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        merge = Reshape_QX_1X(dim=C)
        split = Reshape_4X_1X(dim=C)
        
        # Replace norm and act with identity for testing
        merge.norm = nn.Identity()
        merge.act = nn.Identity()
        split.norm = nn.Identity()
        split.act = nn.Identity()
        
        # Set weights to identity-like matrices
        with torch.no_grad():
            # Stage1 reduction: 4C -> C
            weight_stage1_reduce = torch.zeros(C, 4*C)
            for i in range(C):
                weight_stage1_reduce[i, i] = 1.0
            merge.stage1_reduction.weight.data = weight_stage1_reduce
            
            # Stage2 reduction: 4C -> C
            weight_stage2_reduce = torch.zeros(C, 4*C)
            for i in range(C):
                weight_stage2_reduce[i, i] = 1.0
            merge.stage2_reduction.weight.data = weight_stage2_reduce
            
            # Stage1 expansion: C -> 4C
            weight_stage1_expand = torch.zeros(4*C, C)
            for i in range(C):
                weight_stage1_expand[i, i] = 1.0
            split.stage1_expansion.weight.data = weight_stage1_expand
            
            # Stage2 expansion: C -> 4C
            weight_stage2_expand = torch.zeros(4*C, C)
            for i in range(C):
                weight_stage2_expand[i, i] = 1.0
            split.stage2_expansion.weight.data = weight_stage2_expand
            
            print(f"merge.stage1_reduction.weight.shape: {merge.stage1_reduction.weight.shape}")  # [8, 32]
            print(f"merge.stage2_reduction.weight.shape: {merge.stage2_reduction.weight.shape}")  # [8, 32]
            print(f"split.stage1_expansion.weight.shape: {split.stage1_expansion.weight.shape}")  # [32, 8]
            print(f"split.stage2_expansion.weight.shape: {split.stage2_expansion.weight.shape}")  # [32, 8]
        
        x_merged = merge(x)
        print(f"Merged shape: {x_merged.shape}")  # [1, 2, 2, 8]
        
        x_restored = split(x_merged)
        print(f"Restored shape: {x_restored.shape}")  # [1, 8, 8, 8]
        
        # Check if shapes match
        print(f"Input shape: {x.shape}")
        print(f"Restored shape: {x_restored.shape}")
        
        # Calculate difference
        diff = (x - x_restored).abs().max()
        print(f"Max absolute difference: {diff:.6f}")
    
    # Run tests
    print("Running tests for Same Dim version...")
    
    try:
        x1, x_merged1, x_restored1 = test_simple_dimensions()
    except Exception as e:
        print(f"Error in test_simple_dimensions: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_intermediate_shapes()
    except Exception as e:
        print(f"Error in test_intermediate_shapes: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_with_identity()
    except Exception as e:
        print(f"Error in test_with_identity: {e}")
        import traceback
        traceback.print_exc()