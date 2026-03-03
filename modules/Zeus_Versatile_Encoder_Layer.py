import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
import math

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.util_snapshots_auto import save, ctx

# -------------------- Utility Modules --------------------
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        return swish * self.linear_2(x)

class TwiGLU(nn.Module):
    def __init__(self, dimension, initial_eye=False, initial_scale=1.0):
        super().__init__()
        self.dimension = dimension
        self.linear_1 = nn.Linear(dimension, dimension)
        self.linear_2 = nn.Linear(dimension, dimension)
        
        # Initialize the linear layers
        self._init_weights(initial_eye, initial_scale)
    
    def _init_weights(self, initial_eye, initial_scale):
        """Initialize weights based on the given parameters."""
        if initial_eye:
            # Initialize as scaled identity matrix: eye * initial_scale
            scaled_eye = torch.eye(self.dimension) * initial_scale
            with torch.no_grad():
                self.linear_1.weight.data.copy_(scaled_eye)
                self.linear_2.weight.data.copy_(scaled_eye)
                
                # Initialize biases to zero for scaled identity transformation
                if self.linear_1.bias is not None:
                    self.linear_1.bias.data.zero_()
                if self.linear_2.bias is not None:
                    self.linear_2.bias.data.zero_()
        else:
            # Initialize with normal distribution with specified std (initial_scale)
            for linear in [self.linear_1, self.linear_2]:
                nn.init.normal_(linear.weight, mean=0.0, std=initial_scale)
                if linear.bias is not None:
                    nn.init.zeros_(linear.bias)
    
    def forward(self, x):
        output = self.linear_1(x)
        twish = output * torch.tanh(output)  # Swish-like activation
        return twish * self.linear_2(x)
    
class PositionwiseFeedForward(nn.Module):
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
        return x

class LearnableEyeLayer(nn.Module):
    def __init__(self, dim, init_scale=0.01):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        with torch.no_grad():
            self.weight.data *= init_scale
            self.bias.data *= init_scale

    def forward(self, x):
        return x @ self.weight.T + self.bias


#--------------- Positional embeddings ----------------------

class PositionalEncoding_Axial_Axial(nn.Module):
    # Goal: Create a 1D, axial position embedding from 1D, axial learnables or constants.

    def __init__(self, max_len, dim, pe_type="learnable"):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        
        if pe_type == "learnable":
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.normal_(self.pe, mean=0.0, std=0.02)

        elif pe_type == "sinusoidal":
            # Sinusoidal positional encoding (like Transformer)
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, dim]
            self.register_buffer('pe', pe)

        elif pe_type == "linear":
            # Linear positional encoding (simple)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) / max_len
            pe = pe.expand(max_len, dim)
            pe = pe.unsqueeze(0)  # [1, max_len, dim]
            self.register_buffer('pe', pe)

        else:
            self.pe = None
    
    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]            
        x = x + pe.to(x.device).to(x.dtype)

        return x

class PositionalEncoding_Global_Axial(nn.Module):
    # Goal: Create a full 2D, global position embedding from 1D, axial learnables or constants.
    
    def __init__(self, max_height, max_width, dim, pe_type="learnable"):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.dim = dim
        self.pe_type = pe_type
        self.faster = True
        self.pe = None
        
        if pe_type == "learnable":
            assert dim % 2 == 0
            self.pe_h = nn.Parameter(torch.randn(max_height, dim // 2) * 0.02)
            self.pe_w = nn.Parameter(torch.randn(max_width, dim // 2) * 0.02)
            
        elif pe_type == "sinusoidal":
            assert dim % 4 == 0
            pe_h = self._get_1d_sinusoidal_pe(max_height, dim // 2)
            pe_w = self._get_1d_sinusoidal_pe(max_width, dim // 2)
            self.register_buffer('pe_h', pe_h)
            self.register_buffer('pe_w', pe_w)
            
        elif pe_type == "linear":
            assert dim % 2 == 0
            pe_h = torch.arange(max_height, dtype=torch.float).unsqueeze(1) / max_height
            pe_h = pe_h.expand(-1, dim // 2)
            pe_w = torch.arange(max_width, dtype=torch.float).unsqueeze(1) / max_width
            pe_w = pe_w.expand(-1, dim // 2)
            self.register_buffer('pe_h', pe_h)
            self.register_buffer('pe_w', pe_w)
            
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")

    def _get_1d_sinusoidal_pe(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        B, H, W, C = x.shape
        # assert C == self.dim, f"Input dim {C} != model dim {self.dim}"
        # assert H <= self.max_height, f"Height {H} > max {self.max_height}"
        # assert W <= self.max_width, f"Width {W} > max {self.max_width}"

        pe_h = self.pe_h[:H, :].unsqueeze(1).expand(-1, W, -1)
        pe_w = self.pe_w[:W, :].unsqueeze(0).expand(H, -1, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1).unsqueeze(0)
            
        return x + pe.to(device=x.device, dtype=x.dtype)

class PositionalEncoding_Global_Global(nn.Module):
    # Goal: Create a 2D, full position embedding from 2D, global learning or constants.
    
    def __init__(self, max_height, max_width, dim, pe_type="learnable"):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.dim = dim
        self.pe_type = pe_type
        
        if pe_type == "learnable":
            # Learnable full 2D - TRUE 2D (learns any pattern)
            self.pe = nn.Parameter(torch.randn(1, max_height, max_width, dim) * 0.02)
            
        elif pe_type == "sinusoidal_2d_true":
            assert dim % 2 == 0
            
            # Step 1: Create UNIQUE 1D positions for each cell in the grid
            n_positions = max_height * max_width
            position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
            
            # Step 2: Apply 1D sinusoidal encoding
            div_term = torch.exp(
                torch.arange(0, dim, 2).float() * 
                (-math.log(10000.0) / dim)
            )
            
            pe = torch.zeros(n_positions, dim)
            pe[:, 0::2] = torch.sin(position_1d * div_term)
            pe[:, 1::2] = torch.cos(position_1d * div_term)
            
            # Step 3: Reshape into 2D grid - THIS CREATES TRUE 2D STRUCTURE!
            pe = pe.reshape(max_height, max_width, dim)
            
            self.register_buffer('pe', pe.unsqueeze(0))
                        
        elif pe_type == "linear_2d_true":
            assert dim % 2 == 0
            
            n_positions = max_height * max_width
            position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1) / n_positions
            pe_1d = position_1d.expand(-1, dim)
            pe = pe_1d.reshape(max_height, max_width, dim)
            
            self.register_buffer('pe', pe.unsqueeze(0))
            
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")
    
    def _get_1d_sinusoidal_pe(self, max_len, dim):
        """1D sinusoidal helper - only for axial/legacy."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        """Add positional encoding to input tensor."""
        B, H, W, C = x.shape
        # assert C == self.dim, f"Input dim {C} != model dim {self.dim}"
        # assert H <= self.max_height, f"Height {H} > max {self.max_height}"
        # assert W <= self.max_width, f"Width {W} > max {self.max_width}"
        
        pe = self.pe[:, :H, :W, :]
        return x + pe.to(device=x.device, dtype=x.dtype)
    
    def extra_repr(self):
        return f"pe_type={self.pe_type}, dim={self.dim}, max=({self.max_height}, {self.max_width})"
    
    
class PositionalEncoding_Axial_Global(PositionalEncoding_Global_Global):
    # Original Goal: Create a 1D, axial position embedding from 2D, global learnables or constants.
    # Modified Goal: Create a 2D, full position embedding from 2D, global learning or constants, as PositionalEncoding_Global_Global does.
    pass


class PositionalEncoding_Shift_Global(PositionalEncoding_Global_Global):
    # Original Goal: Create a 1D, axial position embedding from 2D, global learnables or constants.
    # Modified Goal: Create a 2D, full position embedding from 2D, global learning or constants, as PositionalEncoding_Global_Global does.
    pass

class PositionalEncoding_Shift_Axial(PositionalEncoding_Global_Axial):
    # Original Goal: Create a 1D, axial position embedding from 2D, global learnables or constants.
    # Modified Goal: Create a 2D, full position embedding from 2D, global learning or constants, as PositionalEncoding_Global_Global does.
    pass


# -------------------- 2D Positional Bias --------------------

class PositionalBias_er_Full(nn.Module):
    # Goal: Create a 2D, full position embedding from 2D, global learning or constants.
    
    def __init__(self, num_heads, seq_len, pb_type="learnable"):
        super().__init__()
        
        if pb_type == "learnable":
            # Learnable full 2D - TRUE 2D (learns any pattern)
            self.pb = nn.Parameter(torch.randn(1, num_heads, seq_len, seq_len) * 0.02 ) # 0.00002)    # (batch, num_heads, seq_len, seq_len)
                        
        elif pb_type == "sinusoidal_2d_true":
            pass    # not implemented

            # assert dim % 2 == 0
            
            # # Step 1: Create UNIQUE 1D positions for each cell in the grid
            # n_positions = max_height * max_width
            # position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
            
            # # Step 2: Apply 1D sinusoidal encoding
            # div_term = torch.exp(
            #     torch.arange(0, dim, 2).float() * 
            #     (-math.log(10000.0) / dim)
            # )
            
            # pe = torch.zeros(n_positions, dim)
            # pe[:, 0::2] = torch.sin(position_1d * div_term)
            # pe[:, 1::2] = torch.cos(position_1d * div_term)
            
            # # Step 3: Reshape into 2D grid - THIS CREATES TRUE 2D STRUCTURE!
            # pe = pe.reshape(max_height, max_width, dim)
            
            # self.register_buffer('pe', pe.unsqueeze(0))
                        
        elif pb_type == "linear_2d_true":
            pass    # not implemented

            # assert dim % 2 == 0
            
            # n_positions = max_height * max_width
            # position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1) / n_positions
            # pe_1d = position_1d.expand(-1, dim)
            # pe = pe_1d.reshape(max_height, max_width, dim)
            
            # self.register_buffer('pe', pe.unsqueeze(0))
            
        else:
            raise ValueError(f"Unknown pb_type: {pb_type}")
    
    
    def forward(self, scores):
        # scores: (batch, num_heads, seq_len, seq_len)
        scores = scores + self.pb.to(device=scores.device, dtype=scores.dtype)
        return scores
    
    
class PositionalBias_er_Bandwise(nn.Module):
    # Goal: Create a 2D, full position embedding from 2D, global learning or constants.
    
    def __init__(self, H, W, num_heads, sWin, pb_type="learnable"):
        super().__init__()
        
        self.H, self.W = H, W
        self.sWin_H, self.sWin_W = sWin
        self.nBands_horizonal = H // self.sWin_H
        self.nBands_vertical = W // self.sWin_W
        self.num_heads = num_heads
        
        if pb_type == "learnable":

            sWin_H, sWin_W = sWin
            # (n_windows on a lon band of windows, win_vol, win_vol, num_heads)
            self.pb = nn.Parameter(torch.empty(self.nBands_horizonal, sWin_H * sWin_H, 2 * sWin_W - 1, self.num_heads))  
            # (nBands_horizonal, h**2, 2w-1, num_heads) Much less than H*W*H*W * num_heads

            nn.init.trunc_normal_(self.pb, std=0.02)  # 0.02
            
        elif pb_type == "sinusoidal_2d_true":
            pass    # not implemented

            # assert dim % 2 == 0
            
            # # Step 1: Create UNIQUE 1D positions for each cell in the grid
            # n_positions = max_height * max_width
            # position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
            
            # # Step 2: Apply 1D sinusoidal encoding
            # div_term = torch.exp(
            #     torch.arange(0, dim, 2).float() * 
            #     (-math.log(10000.0) / dim)
            # )
            
            # pe = torch.zeros(n_positions, dim)
            # pe[:, 0::2] = torch.sin(position_1d * div_term)
            # pe[:, 1::2] = torch.cos(position_1d * div_term)
            
            # # Step 3: Reshape into 2D grid - THIS CREATES TRUE 2D STRUCTURE!
            # pe = pe.reshape(max_height, max_width, dim)
            
            # self.register_buffer('pe', pe.unsqueeze(0))
                        
        elif pb_type == "linear_2d_true":
            pass    # not implemented

            # assert dim % 2 == 0
            
            # n_positions = max_height * max_width
            # position_1d = torch.arange(n_positions, dtype=torch.float).unsqueeze(1) / n_positions
            # pe_1d = position_1d.expand(-1, dim)
            # pe = pe_1d.reshape(max_height, max_width, dim)
            
            # self.register_buffer('pe', pe.unsqueeze(0))
            
        else:
            raise ValueError(f"Unknown pb_type: {pb_type}")
    
    def forward(self, scores):
        
        device = self.pb.device
        G, _, _, num_heads = self.pb.shape
        N = self.sWin_H * self.sWin_W

        # --- Token coordinates inside a window
        h = torch.arange(self.sWin_H, device=device)
        w = torch.arange(self.sWin_W, device=device)
        coords = torch.stack(torch.meshgrid(h, w, indexing="ij")).view(2, -1)  # (2, N)
        h1, h2 = coords[0][:, None], coords[0][None, :]     # (N,), (N,)
        w1, w2 = coords[1][:, None], coords[1][None, :]     # (N,), (N,)

        idx_h = h1 * self.sWin_H + h2   # (N, N)
        idx_w = (w1 - w2 + (self.sWin_W - 1)) % ( 2 * self.sWin_W - 1)  # (N, N)

        # Expand along longitude
        band_idx_horizontal_shaped = torch.arange(self.nBands_horizonal, device=device)[:, None, None, None]    # (nBands_horizonal,)
        band_idx_vertical_shapedd = torch.arange(self.nBands_vertical, device=device)[None, :, None, None]  # (nBands_vertical,)

        idx_h_shaped = idx_h[None, None, :, :]  # (N, N)
        idx_w_shaped = idx_w[None, None, :, :]  # (N, N)

        # repeat learnable_pair_types along longitude. # self.pb: (nBands_horizonal, h**2, 2w-1, num_heads)
        pos_bias = self.pb[:, None, :, :, :].repeat(1, self.nBands_vertical, 1, 1, 1)  # (nBands_horizonal, nBands_vertical, h**2, 2w-1, num_heads)

        # advanced indexing to select correct pair types
        pos_bias = pos_bias[
            band_idx_horizontal_shaped, band_idx_vertical_shapedd, idx_h_shaped, idx_w_shaped
        ]  # (nBands_horizonal, nBands_vertical, N, N, num_heads)

        # merge G and longitude dims
        pos_bias = pos_bias.view(self.nBands_horizonal * self.nBands_vertical, N, N, num_heads).permute(0, 3, 1, 2).contiguous()     # (total_nWin, N, N, num_heads)
        return scores + pos_bias    
    
# -------------------- Axial FlashAttention --------------------

from modules.Transformer_Modes import AttnConn, PEConn, PEType, Flash, PBias, TransForm
from modules.Zeus_Flash_Encoder_Layer import FlashEncoderLayer
from modules.Zeus_Shift_Encoder_Layer import ShiftEncoderLayer

class Flash_Versatile_Encoder_Layer(nn.Module):         # The lowest ranking module that knows H and W.
    def __init__(self, H, W, d_model, num_heads, nWin, dropout=0, act_layer=nn.Sigmoid, eps=1e-9,
            transFrom=TransForm(), skip_learn=False
    ):
        super().__init__()
        self.H, self.W = H, W
        self.d_model = d_model
        self.num_heads = num_heads
        self.transForm = transFrom
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        self.encoder, self.encoder_h, self.encoder_w, self.encoder_normal, self.encoder_shifted = None, None, None, None, None
        self.pos_encoding, self.pos_encoding_h, self.pos_encoding_w = None, None, None
        self.pos_bias_er, self.pos_bias_er_h, self.pos_bias_er_w, self.pos_bias_er_normal, self.pos_bias_er_shifted = None, None, None, None, None

        if transFrom.attnconn == AttnConn.Global:
            self.encoder = FlashEncoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout, use_flash=transFrom.flash.value, act_layer=act_layer, eps=eps, skip_learn=skip_learn)
            
            if transFrom.peconn == PEConn.Global:   # output: global, intermal: global
                self.pos_encoding = PositionalEncoding_Global_Global(max_height=H, max_width=W, dim=d_model, pe_type=transFrom.petype.value)

            elif transFrom.peconn == PEConn.Axial:  # output: global, internal: axial
                self.pos_encoding = PositionalEncoding_Global_Axial(max_height=H, max_width=W, dim=d_model, pe_type=transFrom.petype.value)

            if transFrom.flash == Flash.No and transFrom.pbias == PBias.Learnable:
                # self.pos_bias_er = PositionalBias_er_Full(num_heads, seq_len=H*W, pb_type="learnable")    # Used for versions older than "Feb21"
                self.pos_bias_er = PositionalBias_er_Bandwise(H, W, num_heads, sWin=(H, W), pb_type="learnable")
                
            # consume: self.encoder, self.pos_encoding, self.pos_bias_er
            
        elif transFrom.attnconn == AttnConn.Axial:
            self.encoder_h = FlashEncoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout, use_flash=transFrom.flash.value, act_layer=act_layer, eps=eps, skip_learn=skip_learn)
            self.encoder_w = FlashEncoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout, use_flash=transFrom.flash.value, act_layer=act_layer, eps=eps, skip_learn=skip_learn)

            self.pos_encoding, self.pos_encoding_h, self.pos_encoding_w = None, None, None
            if transFrom.peconn == PEConn.Global:
                self.pos_encoding = PositionalEncoding_Axial_Global(max_height=H, max_width=W, dim=d_model, pe_type=transFrom.petype.value)
                
            elif transFrom.peconn == PEConn.Axial:
                self.pos_encoding_h = PositionalEncoding_Axial_Axial(max_len=H, dim=d_model, pe_type=transFrom.petype.value)
                self.pos_encoding_w = PositionalEncoding_Axial_Axial(max_len=W, dim=d_model, pe_type=transFrom.petype.value)

            if transFrom.pbias == PBias.Learnable and transFrom.flash == Flash.No:
                self.pos_bias_er_h = PositionalBias_er_Full(num_heads, seq_len=H, pb_type="learnable")  
                self.pos_bias_er_w = PositionalBias_er_Full(num_heads, seq_len=W, pb_type="learnable")  

            # consume: self.encoder_h/w, self.pos_encoding, self.pos_encodding_h/w, self_pos_pias_er_h/w
            
        elif transFrom.attnconn == AttnConn.Shift:

            self.encoder_normal = ShiftEncoderLayer(H, W, d_model=d_model, num_heads=num_heads, nWin=nWin, shift=False, act_layer=act_layer, skip_learn=skip_learn)
            self.encoder_shifted = ShiftEncoderLayer(H, W, d_model=d_model, num_heads=num_heads, nWin=nWin, shift=True, act_layer=act_layer, skip_learn=skip_learn)
 
            self.pos_encoding, self.pos_encoding_normal, self.pos_encoding_shifted = None, None, None
            if transFrom.peconn == PEConn.Global:
                self.pos_encoding = PositionalEncoding_Shift_Global(max_height=H, max_width=W, dim=d_model, pe_type=transFrom.petype.value)
            elif transFrom.peconn == PEConn.Axial:
                assert False, "AttnConn.Shift cannot have PEConn.Axial"
            
            if transFrom.pbias == PBias.Learnable and transFrom.flash == Flash.No:
                sWin = (H//nWin[0], W//nWin[1])
                self.pos_bias_er_normal = PositionalBias_er_Bandwise(H, W, num_heads, sWin, pb_type="learnable")
                self.pos_bias_er_shifted = PositionalBias_er_Bandwise(H, W, num_heads, sWin, pb_type="learnable")

            # consume: self.encoder_normal/shifted, self.pos_encoding, self.pos_bias_er_normal/shifted
            
        else: pass
        

    def forward(self, x, mask=None, ss=False):

        B, H, W, C = x.shape
        assert H == self.H
        assert W == self.W
        assert C == self.d_model
        
        if self.transForm.attnconn == AttnConn.Global:
            # consume: self.encoder, self.pos_encoding, self.pos_bias_er
            
            if self.pos_encoding is not None:   x = self.pos_encoding(x)
            
            x = x.view(B, H*W, C)
            if self.encoder is not None:
                x = self.encoder(x, mask=mask, pos_bias_er=self.pos_bias_er, ss=ss)
            x = x.view(B, H, W, C)
                        
        elif self.transForm.attnconn == AttnConn.Axial:
            # consume: self.encoder_h/w, self.pos_encoding, self.pos_encodding_h/w, self_pos_pias_er_h/w
            
            if self.pos_encoding is not None: x = self.pos_encoding(x)
            x = self.forward_axial(x, 'height', mask=mask, ss=ss)
            x = self.forward_axial(x, 'width', mask=mask, ss=ss)
            
        elif self.transForm.attnconn == AttnConn.Shift:
            # consume: self.encoder_normal/shifted, self.pos_encoding, self.pos_bias_er_normal/shifted
            
            if self.pos_encoding is not None: x = self.pos_encoding(x)
            with ctx('ns', self.encoder_normal):
                x = self.encoder_normal(x, pos_bias_er = self.pos_bias_er_normal, ss=ss)
            with ctx('sh', self.encoder_shifted):
                x = self.encoder_shifted(x, pos_bias_er = self.pos_bias_er_shifted, ss=ss)

        return x

    def forward_axial(self, x, axis, mask=None, ss=None):
        B, H, W, C = x.shape

        # Take the standard shape
        if axis == 'height':
            # (B, H, W, C) -> (B, W, H, C) -> (B*W, H, C)
            x = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
            reshape_back = lambda out: out.reshape(B, W, H, C).permute(0, 2, 1, 3)
        elif axis == 'width':  # width attention (attend along rows)
            # (B, H, W, C) -> (B*H, W, C)
            x = x.reshape(B * H, W, C)
            reshape_back = lambda out: out.reshape(B, H, W, C)
        else: assert False

        # Transformer.Encode, with possible positional encoding.
        if axis == 'height':
            if self.pos_encoding_h is not None: x = self.pos_encoding_h(x)
            if self.encoder_h is not None: 
                with ctx('h', self.encoder_h):
                    x = self.encoder_h(x, mask=mask, pos_bias_er=self.pos_bias_er_h, ss=ss)

        elif axis == 'width':  
            if self.pos_encoding_w is not None: x = self.pos_encoding_w(x)
            if self.encoder_w is not None: 
                with ctx('w', self.encoder_w):
                    x = self.encoder_w(x, mask=mask, pos_bias_er=self.pos_bias_er_w, ss=ss)
    
        # Restore original shape
        x = reshape_back(x)
                    
        return x  # (B, H, W, C)

        
# ------------------- Test Code -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # B, H, W, C, num_heads = 1, 720, 1440, 20, 4
    # B, H, W, C, num_heads = 1, 360, 720,  20, 4
    # B, H, W, C, num_heads = 1, 180, 360,  40, 8
    B, H, W, C, num_heads = 1, 96,  192,  80, 8
    # B, H, W, C, num_heads = 1, 48,  96,  160, 8

    nWin = (12, 12)   # (2, 2): OOM
    
    loops = 3

    # Test different positional encoding types
    for petype in [PEType.Nothing]: #[PEType.Learnable, PEType.Sinusoidal, PEType.Linear, PEType.Nothing]:
        print(f"\n{'='*60}")
        print(f"Testing with positional encoding: {petype}")
        print('='*60)
        
        use_pe = (petype != PEType.Nothing)
        x = torch.randn(B, H, W, C, device=device)
        target = torch.randn_like(x)
        
        tf = TransForm(AttnConn.Shift, PEConn.Global, PEType.Learnable, Flash.No, PBias.Learnable)
        
        model = Flash_Versatile_Encoder_Layer(H=H, W=W, d_model=C, num_heads=num_heads, nWin=nWin, transFrom=tf).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Use bfloat16 for training
        x_bf16 = x.to(torch.bfloat16)
        target_bf16 = target.to(torch.bfloat16)

        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

            # Use autocast for mixed precision training
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for _ in range(loops):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x_bf16,)
                    loss = criterion(output, target_bf16)
                    loss.backward()
                    optimizer.step()

            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            import time
            start_time = time.perf_counter()
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                for _ in range(loops):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x_bf16)
                    loss = criterion(output, target_bf16)
                    loss.backward()
                    optimizer.step()
            elapsed_time = time.perf_counter() - start_time

        print(f"Output shape: {output.shape}")
        print(f"Elapsed time: {elapsed_time:.4f} sec")
        print(f"With positional encoding: {use_pe}")
        if use_pe:
            print(f"Positional encoding type: {petype}")
            
            # Show positional encoding statistics
            if petype == 'learnable' and model.pos_encoding_h is not None:
                height_pe = model.pos_encoding_h.pe
                width_pe = model.pos_encoding_w.pe
                print(f"Height PE stats: mean={height_pe.mean():.4f}, std={height_pe.std():.4f}")
                print(f"Width PE stats: mean={width_pe.mean():.4f}, std={width_pe.std():.4f}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"Sequence lengths: Height={H}, Width={W}")
    print(f"Positional encoding added: BEFORE LayerNorm")
    print(f"FlashAttention requires fp16/bf16 - added explicit casting")
    print('='*60)