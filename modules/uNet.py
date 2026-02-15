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

from modules.uBlock import uBlock

class uNet(nn.Module):
    def __init__(self, H, W, C, block_dicts) -> None:
        super().__init__()

        self.depth = len(block_dicts)
        
        self.uBlocks = nn.ModuleList()

        res, dim = (H, W), C    
        for i in range(self.depth):
            block_dict = block_dicts[i]

            block_dict['top_dim'] = dim
            block_dict['top_res'] = res     # top_res might be modified internally.

            print(f"Creating block with resolution {res} and dim {dim}...")
            block = uBlock(block_dict)
            self.uBlocks.append(block)

            res = block.down_res    # it might not simply a half of the previous block's res.
            dim = block.down_dim
            
    def forward(self, x, ss=None):
        for ub in self.uBlocks:
            ub.skip = x
            x = ub(x, is_enc=True, ss=ss)
        for ub in reversed(self.uBlocks):
            x = ub(x, is_enc=False, ss=ss)
            x = x + ub.skip
        return x



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    H, W, C = 180, 360, 40
    
    from modules.uSide import Reshape
    from modules.Transformer_Modes import AttnConn, PEConn, PEType, Flash, PBias, TransForm
    
    block_dicts = [
        # block and side
        {
            # managed: 'top_res', 'top_dim'
            'top_mersplit': Reshape.Nothing,
            'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 2], 'down_pWin': 1,
            'square' : {
                # managed: 'res', 'dim'
                'num_heads' : 8,
                'prime_size': [1, 1],         # (2, 2)
                'nLayers' : 3,
                'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
            },
        },
        {
            # managed: 'top_res', 'top_dim'
            'top_mersplit': Reshape.Nothing,
            'down_mersplit': Reshape.R_2X_D_2X, 'down_nWin': [1, 2], 'down_pWin': 1,
            'square' : {
                # managed: 'res', 'dim'
                'num_heads' : 8,
                'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                'nLayers' : 3,
                'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),
            },
        },
        {
            # managed: 'top_res', 'top_dim'
            'top_mersplit': Reshape.Nothing,
            'down_mersplit': Reshape.Nothing,
            'square' : {
                # managed: 'res', 'dim'
                'num_heads' : 8,
                'prime_size': [1, 1],         # (1, 1) means no windows, unless 'connection' == shift
                'nLayers' : 4,
                'transForm': TransForm(AttnConn.Axial, PEConn.Axial, PEType.Learnable, Flash.Yes),     # Flash.No: 2.72 s, Flash.No 1.99 s
            },
        },
    ]
    
    x = torch.randn(1, H, W, C).to(device=device, dtype=dtype)
    
    net = uNet(H, W, C, block_dicts).to(device=device, dtype=dtype)
    
    
    loops = 5
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.autocast(device_type='cuda', dtype=dtype):
        for _ in range(loops):
            out = net(x)
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
    
    print(f"input {x.shape}, output {out.shape}, time {elapsed_time}")
    
    