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
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Transformer_Modes import Conn, Flash, PEnc, PBias

class uSquare(nn.Module):
    def __init__(self, dict: dict) -> None:
        super().__init__()
        
        # top_res should be a multiple of win_size.        
        win_size = dict.get('prime_size', [1, 1])     #
        self.padding_dim = [(win_size[i] - dict['res'][i] % win_size[i]) % win_size[i] for i in range(2)]
        self.internal_res = [dict['res'][i] + self.padding_dim[i] for i in range(2)]
        
        res = dict['res']
        dim = dict['dim']
        prime_size = dict['prime_size']
        n_layers = dict['n_layers']
        transForm = dict['transForm']
        # attnconn, peconn, petype, flash, pbias = transForm.attnconn, transForm.peconn, transForm.petype, transForm.flash, transForm.pbias        
     
     
        
        model = Axial_Flash_Attention(H=H, W=W, d_model=C, num_heads=n_heads, transFrom=tf).to(device)
        

    def forward(self, x):
        print(f"uSquare forwarding... {x.shape} with padding {self.padding_dim}")
        return x
