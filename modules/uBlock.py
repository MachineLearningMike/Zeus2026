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

from modules.uSide import uSide

from modules.util_snapshots_auto import save, ctx

class uBlock(nn.Module):
    
    def __init__(self, dict: dict, inherited: dict, id: int) -> None:
        super().__init__()
        
        self.id = id
        side_dict = dict
        
        self.skip_learner = None
        if dict.get('skip_learn', False) is True:
            dim = dict['top_dim']
            self.skip_learner = nn.Linear(dim, dim, bias=True)
            torch.nn.init.eye_(self.skip_learner.weight)
            torch.nn.init.zeros_(self.skip_learner.bias)
        
        self.enc = uSide(is_enc=True, dict=side_dict, inherited=inherited)
        self.dec = uSide(is_enc=False, dict=side_dict, inherited=inherited)

        assert self.enc.down_res == self.dec.down_res
        assert self.enc.down_dim == self.dec.down_dim
        
        self.down_res = self.enc.down_res
        self.down_dim = self.enc.down_dim
        
    def inherit(self):
        return self.enc.inherit()
    
    def is_padding(self):
        return self.enc.is_padding()
        
    def forward(self, x, is_enc=True, ss=None):
        if is_enc:
            with ctx('e', self.enc):
                x = self.enc(x, ss=ss)
        else:
            with ctx('d', self.dec):
                x = self.dec(x, ss=ss)
                
        if self.skip_learner is not None:
            save(ss, self.skip_learner, 'skip_learner')
                
        return x
