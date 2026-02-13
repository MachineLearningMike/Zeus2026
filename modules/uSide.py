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

from modules.uSquare import uSquare
from modules.PatchMergingtPreserveInfo import PatchMerging_4X_Dim, PatchSplitting_4X_Dim
from modules.PatchMergingReduceInto import PatchMerging_2X_Dim, PatchSplitting_2X_Dim

class Merge4(Enum):
    Nothing = 0
    Dim4 = 1
    Dim2 = 3

class uSide(nn.Module):
    """
    This class can decide to add pads to top_res.
    """
    def __init__(self, is_enc: bool, dict: dict) -> None:
        super().__init__()
        self.dict = dict
        self.is_enc = is_enc
        
        self.input_res, self.input_dim = dict['top_res'], dict['top_dim']

        #----------------------- Track and modify res and dim -------------------------------
        
        res_enc_top, dim_enc_top = self.input_res, self.input_dim
       
        #--------------- Modify res_enc_top ----------
        res = res_enc_top
        
        res_divisor = [1, 1]
        if dict['top_mersplit'] == Merge4.Dim2:
            res_divisor = [round(res_divisor[i] * 1 / PatchMerging_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['top_mersplit'] == Merge4.Dim4:
            res_divisor = [round(res_divisor[i] * 1 / PatchMerging_4X_Dim.res_multiplier[i]) for i in range(2)]

        if dict['down_mersplit'] == Merge4.Dim2:
            res_divisor = [round(res_divisor[i] * 1 / PatchMerging_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['down_mersplit'] == Merge4.Dim4:
            res_divisor = [round(res_divisor[i] * 1 / PatchMerging_4X_Dim.res_multiplier[i]) for i in range(2)]

        self.padding = [0 if res[i] % res_divisor[i] == 0 else res_divisor[i] - res[i] % res_divisor[i] for i in range(2)]
        res_enc_top = [res_enc_top[i] + self.padding[i] for i in range(2)]
        
        for i in range(2):
            if self.padding[i] != 0: print(f"------ Resolution {self.input_res[i]} is padded with {self.padding[i]}")
                       
        #---------------- plan res and dim with modified res_enc_top and dim_enc_top -----------------
        dim_enc_down, res_enc_down = dim_enc_top, res_enc_top
        if dict['top_mersplit'] == Merge4.Dim2:
            dim_enc_down = round(dim_enc_down * PatchMerging_2X_Dim.dim_multiplier)
            res_enc_down = [round(res_enc_down[i] * PatchMerging_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['top_mersplit'] == Merge4.Dim4:
            dim_enc_down = round(dim_enc_down * PatchMerging_4X_Dim.dim_multiplier)
            res_enc_down = [round(res_enc_down[i] * PatchMerging_4X_Dim.res_multiplier[i]) for i in range(2)]

            
        dim_dec_down, res_dec_down = dim_enc_down, res_enc_down
        if dict['down_mersplit'] == Merge4.Dim2:
            dim_dec_down = round(dim_dec_down * PatchMerging_2X_Dim.dim_multiplier)
            res_dec_down = [round(res_dec_down[i] * PatchMerging_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['down_mersplit'] == Merge4.Dim4:
            dim_dec_down = round(dim_dec_down * PatchMerging_4X_Dim.dim_multiplier)
            res_dec_down = [round(res_dec_down[i] * PatchMerging_4X_Dim.res_multiplier[i]) for i in range(2)]

        dim_dec_top, res_dec_top = dim_dec_down, res_dec_down
        if dict['down_mersplit'] == Merge4.Dim2:
            dim_dec_top = round(dim_dec_top * PatchSplitting_2X_Dim.dim_multiplier)
            res_dec_top = [round(res_dec_top[i] * PatchSplitting_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['down_mersplit'] == Merge4.Dim4:
            dim_dec_top = round(dim_dec_top * PatchSplitting_4X_Dim.dim_multiplier)
            res_dec_top = [round(res_dec_top[i] * PatchSplitting_4X_Dim.res_multiplier[i]) for i in range(2)]

        dim_enc_top_back, res_enc_top_back = dim_dec_top, res_dec_top
        if dict['top_mersplit'] == Merge4.Dim2:
            dim_enc_top_back = round(dim_enc_top_back * PatchSplitting_2X_Dim.dim_multiplier)
            res_enc_top_back = [round(res_enc_top_back[i] * PatchSplitting_2X_Dim.res_multiplier[i]) for i in range(2)]
        elif dict['top_mersplit'] == Merge4.Dim4:
            dim_enc_top_back = round(dim_enc_top_back * PatchSplitting_4X_Dim.dim_multiplier)
            res_enc_top_back = [round(res_enc_top_back[i] * PatchSplitting_4X_Dim.res_multiplier[i]) for i in range(2)]
            
        assert dim_enc_top == dim_enc_top_back
        assert res_enc_top[0] == res_enc_top_back[0] and res_enc_top[1] == res_enc_top_back[1]      # Fails if dict['top_res'] has an odd number.

        #--------------------------- top_mersplit. Lucky, it's resolution-agnostic
        
        self.top_mersplit = None
        if dict['top_mersplit'] == Merge4.Dim2:
            self.top_mersplit = PatchMerging_2X_Dim(dim_enc_top) if is_enc else PatchSplitting_2X_Dim(dim_dec_top)
        elif dict['top_mersplit'] == Merge4.Dim4:
            self.top_mersplit = PatchMerging_4X_Dim() if is_enc else PatchSplitting_4X_Dim()
        else:
            pass
        
        #-------------------------- down_mersplit, Luck, it's resolution-agnostic
        self.down_mersplit = None
        if dict['down_mersplit'] == Merge4.Dim2:
            self.down_mersplit = PatchMerging_2X_Dim(dim_enc_down) if is_enc else PatchSplitting_2X_Dim(dim_dec_down)
        elif dict['down_mersplit'] == Merge4.Dim4:
            self.down_mersplit = PatchMerging_4X_Dim() if is_enc else PatchSplitting_4X_Dim()
        else:
            pass

        #-------------------------- track res & dim  ------------------------------------

        # get the from top, not down. down is not defined yet.
        res_for_square = res_enc_down if is_enc else res_dec_top
        dim_for_square = dim_enc_down if is_enc else dim_dec_top

        self.down_res =  res_dec_down
        self.down_dim =  dim_dec_down

        #------------------------- Implement padding (in enc) and depadding (in dec)



        #-------------------------- square: (res, dim) to (res, dim)
        self.square = None
        if dict['square'] is not None:
            square_dict = dict['square']
            square_dict['res'] = res_for_square
            square_dict['dim'] = dim_for_square
            square_dict['padding'] = self.padding
            self.square = uSquare(square_dict)
            
        #------------------------- elementary functions

        self.funs = [
            lambda x, ss: self.top_mersplit(x, ss=ss) if self.top_mersplit is not None else x,
            lambda x, ss: self.square(x, ss=ss) if self.square is not None else x,
            lambda x, ss: self.down_mersplit(x, ss=ss) if self.down_mersplit is not None else x
        ]

        
        
        

    def forward(self, x, ss=None):
        if self.is_enc:
            for fun in self.funs:
                x = fun(x, ss=ss)
        else:
            for fun in reversed(self.funs):
                x = fun(x, ss=ss)                    
        return x
