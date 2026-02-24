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
        self.latent_of_pad = nn.Linear(pad_dims, pad_dims//2, bias=True)

        self.expansion = nn.Linear(dim + pad_dims//2, self.out_dim, bias=True)

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

