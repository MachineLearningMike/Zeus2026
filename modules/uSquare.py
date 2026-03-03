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

from modules.util_snapshots_auto import save, ctx

from modules.Transformer_Modes import AttnConn, PEConn, PEType, PBias, TransForm
from modules.Zeus_Versatile_Encoder_Layer import Flash_Versatile_Encoder_Layer

class uSquare(nn.Module):
    def __init__(self, is_enc: bool, dict: dict) -> None:
        super().__init__()
        
        self.is_enc = is_enc
                
        (H, W) = dict['res']
        d_model = dict['dim']
        num_heads = dict['num_heads']
        nWin = dict.get('nWin', (1, 1))
        n_layers = dict['nLayers']
        transForm = dict['transForm']
        act_layer = dict['act_layer']
        skip_learn = dict.get('skip_learn', False)
        self.n_layers = n_layers
        
        # attnconn, peconn, petype, flash, pbias = transForm.attnconn, transForm.peconn, transForm.petype, transForm.flash, transForm.pbias
     
        if act_layer == 'gelu':
            act_layer = nn.GELU
        elif act_layer == 'sigmoid':
            act_layer = nn.Sigmoid
        else: assert False
     
        self.layers = nn.ModuleList()
        
        for idx in range(n_layers):
            layer = Flash_Versatile_Encoder_Layer(H=H, W=W, d_model=d_model, num_heads=num_heads, nWin=nWin, transFrom=transForm, act_layer=act_layer, skip_learn=skip_learn)
            self.layers.append(layer)

    def forward(self, x, ss=None):

        for idx, layer in enumerate(self.layers):
            layer_id = idx if self.is_enc else self.n_layers - idx - 1

            with ctx(f'L{layer_id}', layer):
                x = layer(x, ss = ss)
                save(ss, x, 'vers_out')  
        return x
