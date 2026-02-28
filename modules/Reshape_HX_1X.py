import torch
import torch.nn as nn

class Reshape_HX_1X(nn.Module):     # Half, 1X
   
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res[0] % 2 == 0 and in_res[1] % 2 == 0, "Input must have even resolution."
        out_res = tuple([in_res[i] // 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim  # Keep dimension constant
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Linear(2 ** 2 * dim, dim, bias=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        # assert C == self.dim
        
        x = x.view(B, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//2, W//2, 4 * C)
                
        x = self.reduction(x)
        
        return x


class Reshape_2X_1X(nn.Module):

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = tuple([in_res[i] * 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim  # Keep dimension constant
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        self.expansion = nn.Linear(dim, 4 * dim, bias=True)

    def forward(self, x, ss=None):
        B, H, W, C = x.shape       
        # assert self.dim == C
        
        x = self.expansion(x)
        
        x = x.view(B, H, W, 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C)
        
        return x


if __name__ == "__main__":
    # Test with exact values
    def test_inverse():
        print("\n\n=== Test Inverse (Same Dim) ===")
        B, H, W, C = 1, 4, 4, 4
        
        # Create tensor with unique values
        x = torch.arange(H * W * C).float().view(1, H, W, C)
        print(f"Original tensor shape: {x.shape}")
        
        # Merge
        merge = Reshape_HX_1X(dim=C)
        x_merged = merge(x)
        print(f"\nMerged shape: {x_merged.shape}")
        
        # Split
        split = Reshape_2X_1X(dim=C)
        x_restored = split(x_merged)
        print(f"\nRestored shape: {x_restored.shape}")
        
        print(f"\nWith randomly initialized linear layers, these are not exact inverses.")
        
        return x, x_merged, x_restored

    # Test with identity matrices to verify the operations are correct
    def test_with_identity_matrices():
        print("\n\n=== Test with Identity Matrices (Same Dim) ===")
        B, H, W, C = 1, 4, 4, 4
        
        # Create tensor with unique values
        x = torch.arange(H * W * C).float().view(1, H, W, C)
        print(f"Original tensor shape: {x.shape}")
        
        # Create merge with identity reduction
        merge = Reshape_HX_1X(dim=C)
        # Set reduction to select first C channels from 4C
        identity_matrix = torch.eye(4*C, C)
        merge.reduction.weight.data = identity_matrix
        
        x_merged = merge(x)
        print(f"Merged shape: {x_merged.shape}")
        
        # Create split with identity expansion
        split = Reshape_2X_1X(dim=C)
        # Set expansion to transpose of reduction
        split.expansion.weight.data = identity_matrix.transpose(0, 1)
        
        x_restored = split(x_merged)
        print(f"Restored shape: {x_restored.shape}")
        
        print(f"\nChecking if dimensions make sense:")
        print(f"Original: {x.shape}")
        print(f"After merge: {x_merged.shape}")
        print(f"After split: {x_restored.shape}")
        
        return x, x_merged, x_restored

    # Test to verify dimensions work correctly
    def test_simple_dimensions():
        print("\n\n=== Simple Dimension Test (Same Dim) ===")
        B, H, W, C = 2, 8, 8, 32
        
        # Create random tensor
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        # Merge
        merge = Reshape_HX_1X(dim=C)
        x_merged = merge(x)
        print(f"After Reshape_HX_1X: {x_merged.shape}")
        print(f"  Spatial: {H}x{W} -> {H//2}x{W//2}")
        print(f"  Channels: {C} -> {x_merged.shape[-1]} (should be same)")
        
        # Split
        split = Reshape_2X_1X(dim=C)
        x_restored = split(x_merged)
        print(f"\nAfter Reshape_2X_1X: {x_restored.shape}")
        print(f"  Spatial: {H//2}x{W//2} -> {x_restored.shape[1]}x{x_restored.shape[2]}")
        print(f"  Channels: {x_merged.shape[-1]} -> {x_restored.shape[-1]} (should be same)")
        
        return x, x_merged, x_restored

    # Test to verify the reshape logic is correct
    def test_reshape_logic():
        print("\n\n=== Reshape Logic Test (Same Dim) ===")
        B, H, W, C = 1, 4, 4, 2
        
        # Create tensor with position indices
        x = torch.zeros(B, H, W, C)
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    x[0, h, w, c] = h * 100 + w * 10 + c
        
        print(f"Original tensor shape: {x.shape}")
        print(f"Sample value at (h=2,w=3,c=1): {x[0, 2, 3, 1].item()}")
        
        # Manually perform merging without linear layer
        x_view = x.view(B, H//2, 2, W//2, 2, C)
        x_permuted = x_view.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_merged_manual = x_permuted.view(B, H//2, W//2, 4*C)
        
        print(f"After manual merge (before reduction): {x_merged_manual.shape}")
        
        # Manually perform splitting without linear layer
        x_view_split = x_merged_manual.view(B, H//2, W//2, 2, 2, C)
        x_permuted_split = x_view_split.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_restored_manual = x_permuted_split.view(B, H, W, C)
        
        print(f"After manual split: {x_restored_manual.shape}")
        print(f"Restored value at (h=2,w=3,c=1): {x_restored_manual[0, 2, 3, 1].item()}")
        print(f"Value preserved: {x[0, 2, 3, 1].item() == x_restored_manual[0, 2, 3, 1].item()}")

    try:
        x1, x_merged1, x_restored1 = test_inverse()
    except Exception as e:
        print(f"Error in test_inverse: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        x2, x_merged2, x_restored2 = test_with_identity_matrices()
    except Exception as e:
        print(f"Error in test_with_identity_matrices: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        x3, x_merged3, x_restored3 = test_simple_dimensions()
    except Exception as e:
        print(f"Error in test_simple_dimensions: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_reshape_logic()
    except Exception as e:
        print(f"Error in test_reshape_logic: {e}")
        import traceback
        traceback.print_exc()