import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import copy

from enum import Enum

import sys
from pathlib import Path
sys.path

class Conn(Enum):
    Nothig = 0
    Global = 1
    Axial = 3
    Shift = 4
    
class Flash(Enum):
    Nothing = 0
    Flash = 1
    NonFlash = 2
    
class PEnc(Enum):
    Nothing = 0
    Learnable = 1
    Sinusodial = 2
    Linear = 3
    
class PBias(Enum):
    Nothing = 0
    Learnable = 1

class uSquare(nn.Module):
    def __init__(self, dict: dict) -> None:
        super().__init__()
        
        # top_res should be a multiple of win_size.        
        win_size = dict.get('prime_size', [1, 1])     #
        self.padding_dim = [(win_size[i] - dict['res'][i] % win_size[i]) % win_size[i] for i in range(2)]
        self.internal_res = [dict['res'][i] + self.padding_dim[i] for i in range(2)]
        
        attn = dict['attn']
        conn = attn.get('conn', Conn.Axial)
        penc = attn.get('penc', PEnc.Learnable)
        flash = attn.get('flash', Flash.Flash)
        pbias = attn.get('pbias', PBias.Nothing)

        if conn == Conn.Axial:
            assert pbias == PBias.Nothing 
            
            pass
        elif conn == Conn.Shift:
            assert flash != Flash.Flash     # shifted attention requires attn_mask, which is not supported by FlashAttention
            
            
            pass
        elif conn == Conn.Global:
            assert flash != Flash.Flash     # Global attention is best with attn_mask, which is not supported by FlashAttention
        
        else: assert False
        

    def forward(self, x):
        print(f"uSquare forwarding... {x.shape} with padding {self.padding_dim}")
        return x
