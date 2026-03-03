import torch
import torch.nn as nn

import torch
import torch.nn as nn

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Zeus_Flash_MHA_Embedded_Query import Seq_Len_Changing_MHA
from modules.utils import resize_continuous_bilinear_channel_wise_cylindric

# SwiGLU (unchanged)
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)
        return swiglu

# ---------------------------
# MLP (unchanged)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
#----------------------------------------------------------------------------
# Horizontal-only circular padding (for longitude wrap, not latitude)

import torch.nn.functional as F

class HorizontalCircularPad(nn.Module):
    """Pad only width dimension with circular padding (cylindric horizontally)"""
    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # x: (B, C, H, W)
        # Pad left and right with circular, top and bottom with zero
        return F.pad(x, (self.padding, self.padding, 0, 0), mode='circular')

class Reshape_IX_Add(nn.Module):     # Half, 4X
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        return (in_res, target_dim)

    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()

        self.output_shape = output_shape
        self.pad_dims = pad_dims
        (self.out_H, self.out_W), self.out_dim = output_shape
        
        # Giving up this block, due to GPU limit.
        # num_heads = 1   # versatile
        # self.seq_len_changing_mha = Seq_Len_Changing_MHA(d_model=pad_dims, num_heads=num_heads, target_seq_len = self.out_H * self.out_W, use_flash=True)        

        # Instead of the above block...
        self.latent_of_pad = nn.Linear(pad_dims, pad_dims, bias=True)

        self.expansion = nn.Linear(dim + pad_dims, self.out_dim, bias=True)
        # self.expansion = MLP(dim + pad_dims, hidden_features=4*(dim + pad_dims), out_features=self.out_dim, act_layer=nn.Sigmoid, drop=0.)

    def forward(self, x, ss=None):
        x_main, x_pad = x

        #----------------------------- Resize x_pad to (self.out_H, self.out_W) keeping dim

        B, H, W, C = x_pad.shape

        # Giving up this block, due tot GPU limits
        # x_pad = x_pad.view(B, H*W, C)
        # x_pad = self.seq_len_changing_mha(x_pad)         # 
        # x_pad = x_pad.view(B, self.out_H, self.out_W, C)

        # Instead of the above block...
        x_pad = self.latent_of_pad(x_pad)   # (B, H, W, C//2)
        x_pad = x_pad.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        x_pad = resize_continuous_bilinear_channel_wise_cylindric(x_pad, (self.out_H, self.out_W), mode='bilinear')
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # (B, out_H, out_W, C//2)

        #---------------------------- Concat m_main and x_pad at dim=-1
        x = torch.concat([x_main, x_pad], dim=-1)   # (B, out_H, out_W, dim + C//2)

        #--------------------------- Redim x to self.out_dim
        x = self.expansion(x)

        return x

# class Reshape_IX_Add(nn.Module):     # Half, 4X
    
#     @staticmethod
#     def get_output_shape(input_shape, nWin, pWin, target_dim=0):
#         in_res, in_dim = input_shape
#         return (in_res, target_dim)

#     def __init__(self, dim, output_shape, pad_dims=15):
#         super().__init__()

#         self.output_shape = output_shape
#         self.pad_dims = pad_dims
#         (self.out_H, self.out_W), self.out_dim = output_shape  # (360, 720), 24
        
#         # Horizontal padding layer (width only)
#         self.pad_width = HorizontalCircularPad(padding=1)
        
        
#         # Downsample conv with kernel_size=3, stride=2, padding=0 (we pad manually)
#         self.downsample_conv = nn.Conv2d(
#             in_channels=pad_dims,           # 15
#             out_channels=pad_dims,           # 15
#             kernel_size=3,
#             stride=2,
#             padding=0,
#             bias=True
#         )
        
#         # Expansion conv
#         self.expansion_conv = nn.Conv2d(
#             in_channels=dim + pad_dims,      # dim + 15
#             out_channels=self.out_dim,        # 24
#             kernel_size=3,
#             padding=1,
#             padding_mode='zeros',
#             bias=True
#         )

#     def forward(self, x, ss=None):
#         x_main, x_pad = x
        
#         # Process x_pad
#         x_pad = x_pad.permute(0, 3, 1, 2)
        
#         x_pad = self.pad_width(x_pad)
        
#         x_pad = F.pad(x_pad, (0, 0, 1, 1), mode='constant', value=0)
        
#         x_pad = self.downsample_conv(x_pad)
        
#         # Process x_main
#         x_main = x_main.permute(0, 3, 1, 2)
        
#         # Check shapes before cat
        
#         x = torch.cat([x_main, x_pad], dim=1)
        
#         x = self.expansion_conv(x)
        
#         x = x.permute(0, 2, 3, 1)
        
#         return x

class Reshape_IX_Remove(nn.Module):

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        self.output_shape = output_shape
        _, out_dim = output_shape

        self.reduction = nn.Linear(dim, out_dim, bias=True)
            
    def forward(self, x, ss=None):

        x = self.reduction(x)

        return x
    
# class Reshape_IX_Remove(nn.Module):

#     def __init__(self, dim, output_shape=None, pad_dims=0):
#         super().__init__()
#         self.dim = dim
#         self.output_shape = output_shape
#         (self.out_H, self.out_W), self.out_dim = output_shape  # (360, 720), out_dim
        
#         # Horizontal padding layer
#         self.pad = HorizontalCircularPad(padding=1)
        
#         # 3x3 conv with no padding
#         self.reduction_conv = nn.Conv2d(
#             in_channels=dim,
#             out_channels=self.out_dim,
#             kernel_size=3,
#             padding=0,
#             bias=True
#         )
            
#     def forward(self, x, ss=None):
#         # Permute to (B, C, H, W)
#         x = x.permute(0, 3, 1, 2).contiguous()
        
#         # Pad height with zeros to preserve size
#         x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        
#         # Horizontal circular padding
#         x = self.pad(x)
        
#         # 3x3 conv
#         x = self.reduction_conv(x)
        
#         # Permute back
#         x = x.permute(0, 2, 3, 1).contiguous()
        
#         return x

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