import torch
import torch.nn as nn

from pathlib import Path
import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_HX_2X import Reshape_HX_2X, Reshape_2X_HX

class Reshape_QX_1X_Direct(nn.Module):        
   
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res[0] % 4 == 0 and in_res[1] % 4 == 0, "Input must have resolution divisible by 4."
        out_res = tuple([in_res[i] // 4 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim * 4
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        # Two-stage merging: first 2x, then another 2x
        self.stage1 = Reshape_HX_2X(dim)
        self.stage2 = Reshape_HX_2X(dim * 2)
        
        # Optional: Add normalization and activation between stages for better learning
        self.norm = nn.LayerNorm(dim * 2)
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        assert H % 4 == 0 and W % 4 == 0, f"Input height {H} and width {W} must be divisible by 4."
        
        # Stage 1: (H, W, dim) -> (H//2, W//2, 2*dim)
        x = self.stage1(x, ss)
        
        # Optional normalization and activation between stages
        x = self.norm(x)
        x = self.act(x)
        
        # Stage 2: (H//2, W//2, 2*dim) -> (H//4, W//4, 4*dim)
        x = self.stage2(x, ss)
        
        return x


class Reshape_4X_1X_Direct(nn.Module):

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_dim % 4 == 0, "Input must have dimensions divisible by 4."
        out_res = tuple([in_res[i] * 4 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim // 4
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        # Two-stage splitting: first 2x, then another 2x
        # Input has dim = 4 * original_dim
        # Stage1: (H, W, 4*orig) -> (H*2, W*2, 2*orig) = (H*2, W*2, dim//2)
        # Stage2: (H*2, W*2, 2*orig) -> (H*4, W*4, orig) = (H*4, W*4, dim//4)
        self.stage1 = Reshape_2X_HX(dim)  # Input: 4*orig, Output: 2*orig
        self.stage2 = Reshape_2X_HX(dim // 2)  # Input: 2*orig, Output: orig
        
        # Optional: Add normalization and activation between stages
        self.norm = nn.LayerNorm(dim // 2)  # 2*orig
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        assert C == self.dim, f"Input channels {C} must equal {self.dim}"
        
        # Stage 1: (H, W, 4*orig) -> (H*2, W*2, 2*orig)
        x = self.stage1(x, ss)
        
        # Optional normalization and activation between stages
        x = self.norm(x)
        x = self.act(x)
        
        # Stage 2: (H*2, W*2, 2*orig) -> (H*4, W*4, orig)
        x = self.stage2(x, ss)
        
        return x


if __name__ == "__main__":
    # Test to verify dimensions work correctly
    def test_simple_dimensions():
        print("\n\n=== Simple Dimension Test for 4X (Two-Stage) ===")
        B, H, W, C = 2, 16, 16, 8  # H and W divisible by 4
        
        # Create random tensor
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        # Merge
        merge = Reshape_QX_1X_Direct(dim=C)
        x_merged = merge(x)
        print(f"After Reshape_QX_1X_Direct: {x_merged.shape}")
        print(f"  Spatial: {H}x{W} -> {H//4}x{W//4}")
        print(f"  Channels: {C} -> {x_merged.shape[-1]}")
        
        # Split
        split = Reshape_4X_1X_Direct(dim=4*C)
        x_restored = split(x_merged)
        print(f"\nAfter Reshape_4X_1X_Direct: {x_restored.shape}")
        print(f"  Spatial: {H//4}x{W//4} -> {x_restored.shape[1]}x{x_restored.shape[2]}")
        print(f"  Channels: {x_merged.shape[-1]} -> {x_restored.shape[-1]}")
        
        return x, x_merged, x_restored
    
    # Test intermediate shapes
    def test_intermediate_shapes():
        print("\n\n=== Intermediate Shape Test ===")
        B, H, W, C = 1, 16, 16, 4
        
        x = torch.randn(B, H, W, C)
        print(f"Input: {x.shape}")
        
        merge = Reshape_QX_1X_Direct(dim=C)
        
        # Manually compute intermediate results
        stage1_out = merge.stage1(x)
        print(f"After stage1: {stage1_out.shape} (should be 8x8x8)")
        
        # Apply norm and act manually to match merge's forward
        stage1_out = merge.norm(stage1_out)
        stage1_out = merge.act(stage1_out)
        
        stage2_out = merge.stage2(stage1_out)
        print(f"After stage2: {stage2_out.shape} (should be 4x4x16)")
        
        # Test splitting
        split = Reshape_4X_1X_Direct(dim=16)
        
        split_stage1_out = split.stage1(stage2_out)
        print(f"\nSplit stage1: {split_stage1_out.shape} (should be 8x8x8)")
        
        # Apply norm and act manually to match split's forward
        split_stage1_out = split.norm(split_stage1_out)
        split_stage1_out = split.act(split_stage1_out)
        
        split_stage2_out = split.stage2(split_stage1_out)
        print(f"Split stage2: {split_stage2_out.shape} (should be 16x16x4)")
    
    # Test with identity matrices to verify invertibility
    def test_with_identity():
        print("\n\n=== Identity Test (Two-Stage) ===")
        B, H, W, C = 1, 8, 8, 2
        
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        merge = Reshape_QX_1X_Direct(dim=C)
        split = Reshape_4X_1X_Direct(dim=4*C)
        
        # Replace norm and act with identity for testing
        merge.norm = nn.Identity()
        merge.act = nn.Identity()
        split.norm = nn.Identity()
        split.act = nn.Identity()
        
        # Make the two-stage process approximately invertible by setting weights
        with torch.no_grad():
            # For C=2:
            # Stage1 reduction: 4C=8 -> 2C=4
            weight_stage1_reduce = torch.zeros(2*C, 4*C)  # [4, 8]
            for i in range(2*C):
                weight_stage1_reduce[i, i] = 1.0
            merge.stage1.reduction.weight.data = weight_stage1_reduce
            
            # Stage2 reduction: input to stage2 has 2C=4 channels, so after merging patches: 4*(2C)=8C=16 -> 4C=8
            weight_stage2_reduce = torch.zeros(4*C, 8*C)  # [8, 16]
            for i in range(4*C):
                weight_stage2_reduce[i, i] = 1.0
            merge.stage2.reduction.weight.data = weight_stage2_reduce
            
            # For splitting:
            # Stage2 expansion (in split) reverses stage1 reduction: 2C=4 -> 4C=8
            weight_stage2_expand = torch.zeros(4*C, 2*C)  # [8, 4]
            for i in range(2*C):
                weight_stage2_expand[i, i] = 1.0
            split.stage2.expansion.weight.data = weight_stage2_expand
            
            # Stage1 expansion (in split) reverses stage2 reduction: 4C=8 -> 8C=16
            weight_stage1_expand = torch.zeros(8*C, 4*C)  # [16, 8]
            for i in range(4*C):
                weight_stage1_expand[i, i] = 1.0
            split.stage1.expansion.weight.data = weight_stage1_expand
            
            # Print shapes for verification
            print(f"merge.stage1.reduction.weight.shape: {merge.stage1.reduction.weight.shape}")  # [4, 8]
            print(f"merge.stage2.reduction.weight.shape: {merge.stage2.reduction.weight.shape}")  # [8, 16]
            print(f"split.stage2.expansion.weight.shape: {split.stage2.expansion.weight.shape}")  # [8, 4]
            print(f"split.stage1.expansion.weight.shape: {split.stage1.expansion.weight.shape}")  # [16, 8]
        
        x_merged = merge(x)
        print(f"Merged shape: {x_merged.shape}")  # [1, 2, 2, 8]
        
        x_restored = split(x_merged)
        print(f"Restored shape: {x_restored.shape}")  # [1, 8, 8, 2]
        
        # Check if shapes match
        print(f"Input shape: {x.shape}")
        print(f"Restored shape: {x_restored.shape}")
        
        # Calculate difference to see how well we preserved information
        diff = (x - x_restored).abs().max()
        print(f"Max absolute difference: {diff:.6f}")
        print(f"\nNote: With identity transformations, we can only preserve information")
        print(f"if the linear layers are exact inverses (which requires square matrices).")
        print(f"Since we're reducing dimensions, exact reconstruction is impossible.")
    
    # Run tests
    print("Running tests...")
    
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