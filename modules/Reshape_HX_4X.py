import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Reshape_HX_4X(nn.Module):     # Half, 4X
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_res[0] % 2 == 0 and in_res[1] % 2 == 0, "Input must have even resolution."
        out_res = tuple([in_res[i] // 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim * 4
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()

    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        x = x.view(B, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H//2, W//2, 4 * C)
        return x

class Reshape_2X_QX(nn.Module):

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        assert in_dim % 2 == 0 , "Input must have even dimensions."
        out_res = tuple([in_res[i] * 2 for i in range(2)])
        assert out_res[0] % nWin[0] == 0 and out_res[1] % nWin[1] == 0, f"Output must have exactly {nWin} windows."
        assert out_res[0] % pWin == 0 and out_res[1] % pWin == 0, f"Output must be divisible with {pWin}."
        out_dim = in_dim // 4
        return (out_res, out_dim)

    def __init__(self, dim, output_shape=None, pad_dims=0):
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
        merge = Reshape_HX_4X()
        x_merged = merge(x)
        print(f"\nMerged shape: {x_merged.shape}")
        print("Merged values at position (0,0):", x_merged[0,0,0])
        
        # Split
        split = Reshape_2X_QX()
        x_restored = split(x_merged)
        print(f"\nRestored shape: {x_restored.shape}")
        print("Restored tensor:")
        print(x_restored[0,:,:,0])
        
        # Check exact equality
        print(f"\nExact match? {torch.allclose(x, x_restored)}")
        print(f"Max difference: {torch.max(torch.abs(x - x_restored))}")
        
        return x, x_merged, x_restored

    x, x_merged, x_restored = test_exact_inverse()