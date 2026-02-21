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

from modules.Transformer_Modes import AttnConn, PEConn, PEType, PBias, TransForm
from modules.Zeus_Versatile_Encoder_Layer import Flash_Versatile_Encoder_Layer

class uSquare(nn.Module):
    def __init__(self, dict: dict) -> None:
        super().__init__()
                
        (H, W) = dict['res']
        d_model = dict['dim']
        num_heads = dict['num_heads']
        nWin = dict.get('nWin', (1, 1))
        n_layers = dict['nLayers']
        transForm = dict['transForm']
        # attnconn, peconn, petype, flash, pbias = transForm.attnconn, transForm.peconn, transForm.petype, transForm.flash, transForm.pbias
     
        self.layers = nn.ModuleList()
        
        for idx in range(n_layers):
            layer = Flash_Versatile_Encoder_Layer(H=H, W=W, d_model=d_model, num_heads=num_heads, nWin=nWin, transFrom=transForm)
            self.layers.append(layer)

    def forward(self, x, ss=None):

        for layer in self.layers:
            x = layer(x, ss=ss)
            pass
        
        return x
