import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly
from modules.Zeus_Flash_MHA_Embedded_Query import EmbeddedQueryMultiHeadAttention

class EmbeddedQuery_MHA_Down_1X_Dim(nn.Module):
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W
        
        num_heads = 8
        self.eqMHA = EmbeddedQueryMultiHeadAttention(d_model=d_model, num_heads=num_heads, num_queries = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.eqMHA(x)
        x = x.view(B, self.out_H, self.out_W, C)
        
        return x

class EmbeddedQuery_MHA_Up_1X_Dim(nn.Module):

    def __init__(self, dim, output_shape):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W

        num_heads = 8
        self.eqMHA = EmbeddedQueryMultiHeadAttention(d_model=d_model, num_heads=num_heads, num_queries = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.eqMHA(x)
        x = x.view(B, self.out_H, self.out_W, C)
        
        return x
