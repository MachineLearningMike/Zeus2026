import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly
from modules.Zeus_Flash_MHA_Embedded_Query import EmbeddedQueryMultiHeadAttention

class Reshape_HFX_2X(nn.Module):
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = 2 * in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W
        
        self.dim = dim  # debug
        
        self.expansion = nn.Linear(dim, 2 * dim, bias=False)
        
        num_heads = 8
        self.eqMHA = EmbeddedQueryMultiHeadAttention(d_model=2*dim, num_heads=num_heads, num_queries = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.expansion(x)
        x = self.eqMHA(x)
        x = x.view(B, self.out_H, self.out_W, 2*C)
        
        return x

class Reshape_2FX_HX(nn.Module):

    def __init__(self, dim, output_shape):
        super().__init__()
        (out_H, out_W), d_model = output_shape
        self.out_H = out_H
        self.out_W = out_W

        self.reduction = nn.Linear(dim, dim//2, bias=False)

        num_heads = 8
        self.eqMHA = EmbeddedQueryMultiHeadAttention(d_model=dim, num_heads=num_heads, num_queries = out_H * out_W, use_flash=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.view(B, H*W, C)
        x = self.eqMHA(x)
        x = self.reduction(x)
        x = x.view(B, self.out_H, self.out_W, C//2)
        
        return x
