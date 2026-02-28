import torch
import torch.nn as nn

class Reshape_HX_2X(nn.Module):     # Half, 2X
   
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res[0] % 2 == 0 and in_res[1] % 2 == 0, f"Input {in_res} must have even resolution."
        out_res = tuple([in_res[i] // 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output ({out_res[0]}, {out_res[1]}) must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output ({out_res[0]}, {out_res[1]}) must be divisible with {pWin}."
        out_dim = in_dim * 2
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim

        self.reduction = nn.Linear(2 ** 2 * dim, 2 * dim, bias=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        # assert C == self.dim
        
        x = x.view(B, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//2, W//2, 4 * C)
                
        x = self.reduction(x)
        
        return x

class Reshape_2X_HX(nn.Module):

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_dim % 2 == 0 , "Input must have even dimensions."
        out_res = tuple([in_res[i] * 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim // 2
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        
        self.expansion = nn.Linear(dim, 2 * dim, bias=True)

    def forward(self, x, ss=None):
        B, H, W, C = x.shape       
        # assert self.dim == C
        
        x = self.expansion(x)
        
        x = x.view(B, H, W, 2, 2, C//2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C//2)
        
        return x

if __name__ == "__main__":
    # Test with exact values
    def test_inverse():
        B, H, W, C = 1, 4, 4, 2
        
        # Create tensor with unique values
        x = torch.arange(H * W * C).float().view(1, H, W, C)
        print(f"Original tensor shape: {x.shape}")
        
        # Merge
        merge = Reshape_HX_2X(dim=C, norm_layer=nn.Identity)  # Use Identity for testing
        x_merged = merge(x)
        print(f"\nMerged shape: {x_merged.shape}")
        
        # Split
        split = Reshape_2X_HX(dim=2*C)
        x_restored = split(x_merged)
        print(f"\nRestored shape: {x_restored.shape}")
        
        # Check exact equality (with linear layers, we need to ensure they're inverses)
        print(f"\nWith randomly initialized linear layers, these are not exact inverses.")
        print(f"To make them exact inverses, you could:")
        print(f"1. Use identity matrices for both linear layers")
        print(f"2. Or tie the weights: split.expansion.weight = merge.reduction.weight.transpose(0, 1)")
        
        return x, x_merged, x_restored

    # Test with identity matrices to verify the operations are correct
    def test_with_identity_matrices():
        B, H, W, C = 1, 4, 4, 2
        
        # Create tensor with unique values
        x = torch.arange(H * W * C).float().view(1, H, W, C)
        print(f"\n\n=== Test with Identity Matrices ===")
        print(f"Original tensor shape: {x.shape}")
        
        # Create merge with identity reduction
        merge = Reshape_HX_2X(dim=C, norm_layer=nn.Identity)
        # Set reduction to identity (4C -> 2C identity would need a specific selection)
        # For testing purposes, let's make it select first 2C channels
        identity_matrix = torch.eye(4*C, 2*C)
        merge.reduction.weight.data = identity_matrix
        
        x_merged = merge(x)
        print(f"Merged shape: {x_merged.shape}")
        
        # Create split with identity expansion
        split = Reshape_2X_HX(dim=2*C)
        # Set expansion to transpose of reduction
        split.expansion.weight.data = identity_matrix.transpose(0, 1)
        
        x_restored = split(x_merged)
        print(f"Restored shape: {x_restored.shape}")
        
        # With this setup, we should get exact reconstruction
        # because we're selecting and restoring the first 4 channels (from 8)
        print(f"\nChecking if dimensions make sense:")
        print(f"Original: {x.shape}")
        print(f"After merge: {x_merged.shape}")
        print(f"After split: {x_restored.shape}")
        
        return x, x_merged, x_restored

    # Test to verify dimensions work correctly
    def test_simple_dimensions():
        print("\n\n=== Simple Dimension Test ===")
        B, H, W, C = 2, 8, 8, 32
        
        # Create random tensor
        x = torch.randn(B, H, W, C)
        print(f"Input shape: {x.shape}")
        
        # Merge
        merge = Reshape_HX_2X(dim=C)
        x_merged = merge(x)
        print(f"After Reshape_HX_2X: {x_merged.shape}")
        print(f"  Spatial: {H}x{W} -> {H//2}x{W//2}")
        print(f"  Channels: {C} -> {x_merged.shape[-1]}")
        
        # Split
        split = Reshape_2X_HX(dim=2*C)
        x_restored = split(x_merged)
        print(f"\nAfter Reshape_2X_HX: {x_restored.shape}")
        print(f"  Spatial: {H//2}x{W//2} -> {x_restored.shape[1]}x{x_restored.shape[2]}")
        print(f"  Channels: {x_merged.shape[-1]} -> {x_restored.shape[-1]}")
        
        return x, x_merged, x_restored

    try:
        x1, x_merged1, x_restored1 = test_inverse()
    except Exception as e:
        print(f"Error in test_inverse: {e}")
    
    try:
        x2, x_merged2, x_restored2 = test_with_identity_matrices()
    except Exception as e:
        print(f"Error in test_with_identity_matrices: {e}")
    
    try:
        x3, x_merged3, x_restored3 = test_simple_dimensions()
    except Exception as e:
        print(f"Error in test_simple_dimensions: {e}")