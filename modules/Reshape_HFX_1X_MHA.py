import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly
from modules.Zeus_Flash_MHA_Embedded_Query import Seq_Len_Changing_MHA

class Reshape_HFX_1X_MHA(nn.Module):    # Half-Flex, 1X
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W
        
        num_heads = 8
        self.eqMHA = Seq_Len_Changing_MHA(d_model=d_model, num_heads=num_heads, target_seq_len = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.eqMHA(x)
        x = x.view(B, self.out_H, self.out_W, C)
        
        return x

class Reshape_2FX_1X_MHA(nn.Module):

    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W

        num_heads = 8
        self.eqMHA = Seq_Len_Changing_MHA(d_model=d_model, num_heads=num_heads, target_seq_len = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.eqMHA(x)
        x = x.view(B, self.out_H, self.out_W, C)
        
        return x
