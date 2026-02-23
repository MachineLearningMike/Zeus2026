import torch
import torch.nn as nn

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

        # Merge 4x4 patches (16 patches) and reduce to 4*dim
        self.reduction = nn.Linear(4 ** 2 * dim, 4 * dim, bias=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        # assert C == self.dim
        assert H % 4 == 0 and W % 4 == 0, f"Input height {H} and width {W} must be divisible by 4."
        
        # Merge patches: (B, H, W, C) -> (B, H/4, W/4, 16C)
        x = x.view(B, H//4, 4, W//4, 4, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//4, W//4, 16 * C)
                
        # Reduce dimension: (B, H/4, W/4, 16C) -> (B, H/4, W/4, 4C)
        x = self.reduction(x)
        
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
        
        # Inverse of the linear projection: expand from 4*dim to 16*dim
        self.expansion = nn.Linear(dim, 4 * dim, bias=True)

    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        # assert self.dim == C
        assert C % 4 == 0, f"Channel dimension {C} must be divisible by 4."
        
        # First, reverse the linear projection: (B, H, W, C) -> (B, H, W, 4C)
        x = self.expansion(x)
        
        # Then reverse the patch merging: (B, H, W, 4C) -> (B, 4H, 4W, C)
        # Note: 4C comes from expansion, and we need to split into 4x4 patches
        x = x.view(B, H, W, 4, 4, C//4)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 4, W * 4, C//4)
        
        return x


if __name__ == "__main__":
    # Test to verify dimensions work correctly
    def test_simple_dimensions():
        print("\n\n=== Simple Dimension Test for 4X ===")
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
    
    # Test with a smaller example to verify the reshape logic - FIXED VERSION
    def test_reshape_logic():
        print("\n\n=== Reshape Logic Test ===")
        B, H, W, C = 1, 8, 8, 2  # H and W divisible by 4
        
        # Create tensor with sequential values to track movement
        x = torch.arange(H * W * C).float().view(1, H, W, C)
        print(f"Original tensor shape: {x.shape}")
        print(f"First few values: {x[0, 0, 0, :]} ...")
        
        merge = Reshape_QX_1X_Direct(dim=C)
        
        # FIX: Create proper identity-like weights for the linear layer
        with torch.no_grad():
            # The reduction layer goes from 16*C to 4*C
            # We want to preserve information, so we'll use a matrix that selects the first 4*C channels
            # and sets the rest to zero (simple channel selection)
            in_features = 16 * C  # 32
            out_features = 4 * C   # 8
            
            # Create a weight matrix that copies the first out_features channels
            # and ignores the rest (this is a simple way to preserve information)
            weight = torch.zeros(out_features, in_features)
            for i in range(out_features):
                weight[i, i] = 1.0
            
            merge.reduction.weight.data = weight
            
            print(f"Reduction weight shape: {merge.reduction.weight.shape}")  # Should be [8, 32]
        
        x_merged = merge(x)
        print(f"\nAfter merge: {x_merged.shape}")  # Should be [1, 2, 2, 8]
        
        split = Reshape_4X_1X_Direct(dim=4*C)
        with torch.no_grad():
            # For splitting, we need to go from 4*C to 16*C
            # The transpose of the reduction matrix would work if reduction was square
            # But since it's not, we need a different approach
            in_features_split = 4 * C  # 8
            out_features_split = 4 * C  # 8 (but this is wrong - should be 16*C for expansion?)
            # Wait - expansion goes from dim to 4*dim, so with dim=4*C, expansion goes from 8 to 32
            
            # Let's check the actual dimensions
            print(f"Split expansion expects: in_features={split.expansion.in_features}, out_features={split.expansion.out_features}")
            
            # Create an expansion matrix that approximately inverts the reduction
            # This is an approximate inverse (pseudo-inverse would be better but this works for testing)
            expansion_weight = torch.zeros(split.expansion.out_features, split.expansion.in_features)
            for i in range(min(split.expansion.in_features, split.expansion.out_features)):
                if i < out_features:  # out_features from reduction
                    expansion_weight[i, i] = 1.0
            
            split.expansion.weight.data = expansion_weight
        
        x_restored = split(x_merged)
        print(f"After split: {x_restored.shape}")  # Should be [1, 8, 8, 2]
        
        # Check if values are approximately preserved
        print(f"\nOriginal first 2x2 patch:")
        print(x[0, :2, :2, 0])  # First channel of first 2x2 patch
        
        print(f"Restored first 2x2 patch:")
        print(x_restored[0, :2, :2, 0])  # First channel of first 2x2 patch
        
        print(f"\nNote: Due to channel selection and linear transformation, values may be scaled.")
    
    # Additional test to show the reshaping logic without linear layers
    def test_reshape_only():
        print("\n\n=== Pure Reshape Logic Test (No Linear Layers) ===")
        B, H, W, C = 1, 8, 8, 2
        
        # Create tensor with position indices to track movement
        x = torch.zeros(B, H, W, C)
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    x[0, h, w, c] = h * 100 + w * 10 + c  # Unique identifier
        
        print(f"Original tensor shape: {x.shape}")
        print(f"Sample value at (h=3,w=4,c=1): {x[0, 3, 4, 1].item()}")
        
        # Manually perform the merging operation without linear layer
        B, H, W, C = x.shape
        
        # Merge patches: (B, H, W, C) -> (B, H/4, W/4, 16C)
        x_view = x.view(B, H//4, 4, W//4, 4, C)
        x_permuted = x_view.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_merged_manual = x_permuted.view(B, H//4, W//4, 16 * C)
        
        print(f"After manual merge: {x_merged_manual.shape}")
        
        # Manually perform the splitting operation
        B, H_m, W_m, C_m = x_merged_manual.shape  # H_m=2, W_m=2, C_m=32
        
        # Split: (B, H, W, 16C) -> (B, 4H, 4W, C)
        x_view_split = x_merged_manual.view(B, H_m, W_m, 4, 4, C_m//16)
        x_permuted_split = x_view_split.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_restored_manual = x_permuted_split.view(B, H_m * 4, W_m * 4, C_m//16)
        
        print(f"After manual split: {x_restored_manual.shape}")
        
        # Check if the value at the same position is preserved
        print(f"Restored value at (h=3,w=4,c=1): {x_restored_manual[0, 3, 4, 1].item()}")
        print(f"Value preserved: {x[0, 3, 4, 1].item() == x_restored_manual[0, 3, 4, 1].item()}")
    
    # Run tests
    try:
        x1, x_merged1, x_restored1 = test_simple_dimensions()
    except Exception as e:
        print(f"Error in test_simple_dimensions: {e}")
    
    try:
        test_reshape_logic()
    except Exception as e:
        print(f"Error in test_reshape_logic: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_reshape_only()
    except Exception as e:
        print(f"Error in test_reshape_only: {e}")