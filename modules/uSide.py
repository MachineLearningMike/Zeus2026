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
from modules.Reshape_R_Half_D_4X import PatchMerging_4X_Dim, PatchSplitting_4X_Dim
from modules.Reshape_R_Half_D_2X import PatchMerging_2X_Dim, PatchSplitting_2X_Dim
from modules.Resize_R_Half_Flex_D_1X_Interpol import Interpolate_Down_1X_Dim, Interpolate_Up_1X_Dim
from modules.Resize_R_Half_Flex_D_1X_MHA import EmbeddedQuery_MHA_Down_1X_Dim, EmbeddedQuery_MHA_Up_1X_Dim

class Reshape(Enum):
    Nothing = 0
    R_2X_D_4X = 1
    R_2X_D_2X = 3
    R_Any_D_1X = 4
    R_Any_MHA = 5

class uSide(nn.Module):
    """
    This class can decide to add pads to top_res.
    """
    def __init__(self, is_enc: bool, dict: dict) -> None:
        super().__init__()
        self.dict = dict
        self.is_enc = is_enc
        
        self.input_res, self.input_dim = dict['top_res'], dict['top_dim']
       
        #------------------------ create shortcuts ---
        enc_top_mersplit_class = None
        if dict['top_mersplit'] == Reshape.R_2X_D_2X:
            enc_top_mersplit_class = PatchMerging_2X_Dim
        elif dict['top_mersplit'] == Reshape.R_2X_D_4X:
            enc_top_mersplit_class = PatchMerging_4X_Dim
        elif dict['top_mersplit'] == Reshape.R_Any_D_1X:
            enc_top_mersplit_class = Interpolate_Down_1X_Dim
        elif dict['top_mersplit'] == Reshape.R_Any_MHA:
            enc_top_mersplit_class = EmbeddedQuery_MHA_Down_1X_Dim

        enc_down_mersplit_class = None
        if dict['down_mersplit'] == Reshape.R_2X_D_2X:
            enc_down_mersplit_class = PatchMerging_2X_Dim
        elif dict['down_mersplit'] == Reshape.R_2X_D_4X:
            enc_down_mersplit_class = PatchMerging_4X_Dim
        elif dict['down_mersplit'] == Reshape.R_Any_D_1X:
            enc_down_mersplit_class = Interpolate_Down_1X_Dim
        elif dict['down_mersplit'] == Reshape.R_Any_MHA:
            enc_down_mersplit_class = EmbeddedQuery_MHA_Down_1X_Dim

        dec_top_mersplit_class = None
        if dict['top_mersplit'] == Reshape.R_2X_D_2X:
            dec_top_mersplit_class = PatchSplitting_2X_Dim
        elif dict['top_mersplit'] == Reshape.R_2X_D_4X:
            dec_top_mersplit_class = PatchSplitting_4X_Dim
        elif dict['top_mersplit'] == Reshape.R_Any_D_1X:
            dec_top_mersplit_class = Interpolate_Up_1X_Dim
        elif dict['top_mersplit'] == Reshape.R_Any_MHA:
            dec_top_mersplit_class = EmbeddedQuery_MHA_Up_1X_Dim

        dec_down_mersplit_class = None
        if dict['down_mersplit'] == Reshape.R_2X_D_2X:
            dec_down_mersplit_class = PatchSplitting_2X_Dim
        elif dict['down_mersplit'] == Reshape.R_2X_D_4X:
            dec_down_mersplit_class = PatchSplitting_4X_Dim
        elif dict['down_mersplit'] == Reshape.R_Any_D_1X:
            dec_down_mersplit_class = Interpolate_Up_1X_Dim
        elif dict['down_mersplit'] == Reshape.R_Any_MHA:
            dec_down_mersplit_class = EmbeddedQuery_MHA_Up_1X_Dim

        #--------------- Planning for res and dim ----------
        enc_top_input_shape = (dict['top_res'], dict['top_dim'])
        nWin = dict.get('top_nWin', [1, 1])
        pWin = dict.get('top_pWin', 1)
        enc_top_output_shape = enc_top_mersplit_class.get_output_shape(enc_top_input_shape, nWin, pWin) if enc_top_mersplit_class is not None else enc_top_input_shape

        enc_down_input_shape = enc_top_output_shape
        nWin = dict.get('down_nWin', [1, 1])
        pWin = dict.get('down_pWin', 1)        
        enc_down_output_shape = enc_down_mersplit_class.get_output_shape(enc_down_input_shape, nWin, pWin) if enc_down_mersplit_class is not None else enc_down_input_shape

        #-------------- We replace dec shapes with enc shapes, which is requiredd if dec shape is not uniquely calculated ---

        # dec_down_input_shape = enc_down_output_shape
        # nWin = dict.get('down_nWin', [1, 1])
        # pWin = dict.get('down_pWin', 1)
        # dec_down_output_shape = dec_down_mersplit_class.get_output_shape(dec_down_input_shape, nWin, pWin) if dec_down_mersplit_class is not None else dec_down_input_shape

        # dec_top_input_shape = dec_down_output_shape
        # nWin = dict.get('top_nWin', [1, 1])
        # pWin = dict.get('top_pWin', 1)
        # dec_top_output_shape = dec_top_mersplit_class.get_output_shape(dec_top_input_shape, nWin, pWin) if dec_top_mersplit_class is not None else dec_top_input_shape

        # assert dec_top_output_shape == enc_top_input_shape
        
        # assert dec_top_input_shape == enc_top_output_shape
        # assert dec_down_output_shape == enc_top_output_shape
        # assert dec_down_input_shape == enc_down_output_shape


        self.top_mersplit = None
        if enc_top_mersplit_class is not None:
            self.top_mersplit = enc_top_mersplit_class(enc_top_input_shape[1], output_shape=enc_top_output_shape) if is_enc else dec_top_mersplit_class(enc_top_output_shape[1], output_shape=enc_top_input_shape) # == dec_top_mersplit_class(dec_top_input_shape[1])

        self.down_mersplit = None
        if enc_down_mersplit_class is not None:
            self.down_mersplit = enc_down_mersplit_class(enc_down_input_shape[1], output_shape=enc_down_output_shape) if is_enc else dec_down_mersplit_class(enc_down_output_shape[1], output_shape=enc_top_output_shape) # == dec_down_mersplit_class(dec_down_input_shape[1])


        res_for_square, dim_for_square = enc_top_output_shape if is_enc else enc_top_output_shape # == dec_down_output_shape

        self.down_res, self.down_dim =  enc_down_output_shape


        #-------------------------- square: (res, dim) to (res, dim)
        self.square = None
        if dict.get('square', None) is not None:
            square_dict = dict['square']
            square_dict['res'] = res_for_square
            square_dict['dim'] = dim_for_square
            square_dict['padding'] = 0          #----------------
            self.square = uSquare(square_dict)
            
        #------------------------- elementary functions

        self.funs = [
            # implement self.padding here
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
