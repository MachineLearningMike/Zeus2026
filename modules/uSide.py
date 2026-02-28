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

from modules.uSquare import uSquare
from modules.Reshape_HX_4X import Reshape_HX_4X, Reshape_2X_QX
from modules.Reshape_HX_2X import Reshape_HX_2X, Reshape_2X_HX
from modules.Reshape_HX_1X import Reshape_HX_1X, Reshape_2X_1X
from modules.Reshape_HFX_1X_MHA import Reshape_HFX_1X_MHA, Reshape_2FX_1X_MHA
from modules.Reshape_HFX_2X_MHA import Reshape_HFX_2X_MHA, Reshape_2FX_HX_MHA
from modules.Reshape_IX_Add import Reshape_IX_Add, Reshape_IX_Remove

class Reshape(Enum):
    Nothing = 0
    HX_4X = 1
    HX_2X = 3
    HX_1X = 4
    HFX_1X_MHA = 5
    HFX_2X_MHA = 6
    IX_ADD = 7

class uSide(nn.Module):
    """
    This class can decide to add pads to top_res.
    """
    def __init__(self, is_enc: bool, dict: dict, inherited: dict = None) -> None:
        super().__init__()
        self.dict = dict
        self.is_enc = is_enc
        
        self.input_res, self.input_dim = dict['top_res'], dict['top_dim']
       
        #------------------------ create shortcuts ---
        top_mersplit = dict.get('top_mersplit', Reshape.Nothing)
        down_mersplit = dict.get('down_mersplit', Reshape.Nothing)
        
        enc_mersplit_dict = {
            Reshape.HX_2X: Reshape_HX_2X,
            Reshape.HX_4X: Reshape_HX_4X,
            Reshape.HX_1X: Reshape_HX_1X,
            Reshape.HFX_1X_MHA: Reshape_HFX_1X_MHA,
            Reshape.HFX_2X_MHA: Reshape_HFX_2X_MHA,
            Reshape.IX_ADD: Reshape_IX_Add,
        }
        
        dec_mersplit_dict = {
            Reshape.HX_2X: Reshape_2X_HX,
            Reshape.HX_4X: Reshape_2X_QX,
            Reshape.HX_1X: Reshape_2X_1X,
            Reshape.HFX_1X_MHA: Reshape_2FX_1X_MHA,
            Reshape.HFX_2X_MHA: Reshape_2FX_HX_MHA,
            Reshape.IX_ADD: Reshape_IX_Remove,
        }
                    
        enc_top_mersplit_class = enc_mersplit_dict.get(top_mersplit, None)

        enc_down_mersplit_class = enc_mersplit_dict.get(down_mersplit, None)
            
        dec_top_mersplit_class = dec_mersplit_dict.get(top_mersplit, None)

        dec_down_mersplit_class = dec_mersplit_dict.get(down_mersplit, None)

        #--------------- Planning for res and dim ----------
        pad_dims = dict.get('pad_dims', 0)
        target_dims = dict.get('target_dims', 0)
        
        enc_top_input_shape = (dict['top_res'], dict['top_dim'])
        nWin = dict.get('top_nWin', [1, 1] if inherited is None else inherited.get('nWin', [1,1]))
        pWin = dict.get('top_pWin', 1 if inherited is None else inherited.get('pWin', 1) )
        enc_top_output_shape = enc_top_mersplit_class.get_output_shape(enc_top_input_shape, nWin, pWin, target_dims) if enc_top_mersplit_class is not None else enc_top_input_shape
        
        nWin_for_square = nWin
        pWin_for_square = pWin

        enc_down_input_shape = enc_top_output_shape
        nWin = dict.get('down_nWin', [1, 1])
        pWin = dict.get('down_pWin', 1)        
        enc_down_output_shape = enc_down_mersplit_class.get_output_shape(enc_down_input_shape, nWin, pWin, target_dims) if enc_down_mersplit_class is not None else enc_down_input_shape
        
        self.nWin_to_inherit = nWin
        self.pWin_to_inherit = pWin

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


        # self.top_mersplit = None
        # if enc_top_mersplit_class is not None:
        #     self.top_mersplit = enc_top_mersplit_class(enc_top_input_shape[1], output_shape=enc_top_output_shape, pad_dims=pad_dims) if is_enc \
        #         else dec_top_mersplit_class(enc_top_output_shape[1], output_shape=enc_top_input_shape, pad_dims=pad_dims) # == dec_top_mersplit_class(dec_top_input_shape[1])

        # self.down_mersplit = None
        # if enc_down_mersplit_class is not None:
        #     self.down_mersplit = enc_down_mersplit_class(enc_down_input_shape[1], output_shape=enc_down_output_shape, pad_dims=pad_dims) if is_enc \
        #         else dec_down_mersplit_class(enc_down_output_shape[1], output_shape=enc_top_output_shape, pad_dims=pad_dims) # == dec_down_mersplit_class(dec_down_input_shape[1])


        self.top_mersplit = None
        self.top_mersplit = (enc_top_mersplit_class(enc_top_input_shape[1], output_shape=enc_top_output_shape, pad_dims=pad_dims) if enc_top_mersplit_class is not None else None ) if is_enc \
            else (dec_top_mersplit_class(enc_top_output_shape[1], output_shape=enc_top_input_shape, pad_dims=pad_dims) if dec_top_mersplit_class is not None else None)

        self.down_mersplit = None
        self.down_mersplit = (enc_down_mersplit_class(enc_down_input_shape[1], output_shape=enc_down_output_shape, pad_dims=pad_dims) if enc_down_mersplit_class is not None else None ) if is_enc \
            else (dec_down_mersplit_class(enc_down_output_shape[1], output_shape=enc_top_output_shape, pad_dims=pad_dims) if dec_down_mersplit_class is not None else None )

        res_for_square, dim_for_square = enc_top_output_shape if is_enc else enc_top_output_shape # == dec_down_output_shape

        self.down_res, self.down_dim =  enc_down_output_shape


        #-------------------------- square: (res, dim) to (res, dim)
        self.square = None
        if dict.get('square', None) is not None:
            square_dict = dict['square']
            square_dict['res'] = res_for_square
            square_dict['dim'] = dim_for_square
            square_dict['padding'] = 0          #----------------
            square_dict['nWin'] = nWin_for_square
            square_dict['pWin'] = pWin_for_square
            self.square = uSquare(is_enc, square_dict)
            
        #------------------------- elementary functions

        self.funs = [
            # implement self.padding here
            lambda x, ss: self.top_mersplit(x, ss=ss) if self.top_mersplit is not None else x,
            lambda x, ss: self.square(x, ss=ss) if self.square is not None else x,
            lambda x, ss: self.down_mersplit(x, ss=ss) if self.down_mersplit is not None else x
        ]
        self.phases = ['top', 'square', 'down']
        
    def inherit(self):
        return {'nWin': self.nWin_to_inherit, 'pWin': self.pWin_to_inherit}
        
    def is_padding(self):
        return self.dict.get('pad_dims', None) is not None
    
    # inherited = {'nWin': (1, 1), 'pWin': 1}

    def forward(self, x, ss=None):
        
        if self.is_enc:
            if self.top_mersplit is not None:
                x = self.top_mersplit(x)
            if self.square is not None:
                x = self.square(x, ss=ss)
            if self.down_mersplit is not None:
                x = self.down_mersplit(x)
        else:
            if self.down_mersplit is not None:
                x = self.down_mersplit(x)
            if self.square is not None:
                x = self.square(x, ss=ss)
            if self.top_mersplit is not None:
                x = self.top_mersplit(x)
        return x
