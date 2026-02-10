import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

# Check if flash attention is available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("FlashAttention not available. Install with: pip install flash-attn")

# ==================== TModes Definition ====================

class Conn(Enum):
    Nothing = 0
    Global = 1
    Axial = 3
    Shift = 4
    
class Flash(Enum):
    Nothing = 0
    UseFlash = 1
    NoFlash = 2
    
class PEnc(Enum):
    Nothing = 0
    Learnable = 1
    Sinusoidal = 2
    Linear = 3

class PShare(Enum):
    Global = 1
    Axial = 2
    
class PBias(Enum):
    Nothing = 0
    Learnable = 1

class TModes:
    def __init__(self, conn=Conn.Nothing, 
                 flash=Flash.Nothing, 
                 penc=PEnc.Nothing, 
                 pshare=PShare.Global, 
                 pbias=PBias.Nothing) -> None:
        
        assert self.is_valid(conn, flash, penc, pshare, pbias), "Invalid mode"
        
        self.conn = conn
        self.flash = flash
        self.penc = penc
        self.pshare = pshare
        self.pbias = pbias
        
    
    def is_valid(self, conn, flash, penc, pshare, pbias):
        
        invalid_modes = [
            (Flash.UseFlash, Conn.Shift),
            (Flash.UseFlash, PBias.Learnable)
        ]
        
        valid = True
        for im in invalid_modes:
            sum = int(conn in im) + int(flash in im) + int(penc in im) + int(pshare in im) + int(pbias in im)
            valid = valid and (sum < 2)
            
        return valid

# ==================== Positional Encoding Classes ====================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for spatial data"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        if d_model % 4 != 0:
            d_model = ((d_model + 3) // 4) * 4  # Ensure divisible by 4
        
        pe_h = self._get_1d_encoding(max_h, d_model // 2)
        pe_w = self._get_1d_encoding(max_w, d_model // 2)
        
        # Create 2D grid
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Fill with height encodings (first half of channels)
        pe[:, :, :d_model//2] = pe_h.unsqueeze(1).expand(-1, max_w, -1)
        
        # Fill with width encodings (second half of channels)
        pe[:, :, d_model//2:] = pe_w.unsqueeze(0).expand(max_h, -1, -1)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def _get_1d_encoding(self, max_len, dim):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        B, H, W, C = x.shape
        return x + self.pe[:, :H, :W, :]

class LearnedPositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_h, max_w, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        return x + self.pos_embedding[:, :H, :W, :]

class AxialPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        if d_model % 2 != 0:
            d_model += 1  # Ensure even dimension
        
        self.d_model = d_model
        self.row_encoding = nn.Parameter(torch.randn(1, max_h, 1, d_model // 2))
        self.col_encoding = nn.Parameter(torch.randn(1, 1, max_w, d_model // 2))
        nn.init.normal_(self.row_encoding, mean=0.0, std=0.02)
        nn.init.normal_(self.col_encoding, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        row_enc = self.row_encoding[:, :H, :, :]
        col_enc = self.col_encoding[:, :, :W, :]
        pos_enc = torch.cat([
            row_enc.expand(B, H, W, self.d_model // 2),
            col_enc.expand(B, H, W, self.d_model // 2)
        ], dim=-1)
        return x + pos_enc

class GlobalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)

class LinearPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Linear(1, d_model, bias=False)
        self.positions = torch.arange(max_len).float().unsqueeze(1)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos = self.positions[:seq_len].unsqueeze(0).to(x.device)
        pos_enc = self.pe(pos)
        return x + pos_enc

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x[..., :self.dim], x[..., self.dim:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

# ==================== Time Encoding Classes ====================

class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal encoding for time features"""
    def __init__(self, d_model, max_period=10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        
    def forward(self, time_features):
        """
        Args:
            time_features: (B, T, features) or (B, features)
                           Features can include: hour_of_day, day_of_year, etc.
        """
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(1)  # (B, 1, features)
        
        B, T, F = time_features.shape
        device = time_features.device
        
        # Create sinusoidal encodings
        positions = time_features.unsqueeze(-1)  # (B, T, F, 1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() *
            (-math.log(self.max_period) / self.d_model)
        )  # (d_model/2,)
        
        # Expand dimensions for broadcasting
        div_term = div_term.view(1, 1, 1, -1)  # (1, 1, 1, d_model/2)
        
        # Compute sine and cosine
        sin_enc = torch.sin(positions * div_term)  # (B, T, F, d_model/2)
        cos_enc = torch.cos(positions * div_term)  # (B, T, F, d_model/2)
        
        # Interleave sin and cos
        time_enc = torch.stack([sin_enc, cos_enc], dim=-1)  # (B, T, F, d_model/2, 2)
        time_enc = time_enc.view(B, T, F, self.d_model)  # (B, T, F, d_model)
        
        # Sum over feature dimension and average over feature count
        time_enc = time_enc.sum(dim=2) / math.sqrt(F)  # (B, T, d_model)
        
        return time_enc

# ==================== Weather Forecasting Transformer ====================

class WeatherForecastingTransformer(nn.Module):
    """
    Transformer for weather forecasting with continuous 2D spatial data.
    
    Input: (B, H, W, C) - continuous weather variables
    Output: (B, H, W, C) or (B, H, W, C_out) - predicted weather variables
    
    Usage:
        config = {
            'd_model': 512,
            'tmodes': TModes(conn=Conn.Axial, flash=Flash.NoFlash)
        }
        model = WeatherForecastingTransformer(input_dim=10, output_dim=10, capacity_config=config)
    """
    
    def __init__(self, input_dim, output_dim=None, capacity_config=None):
        super().__init__()
        
        # Extract TModes from config if present
        self.tmodes = None
        if capacity_config and 'tmodes' in capacity_config:
            self.tmodes = capacity_config['tmodes']
            # Create a copy of config without tmodes for the rest of initialization
            config_for_init = {k: v for k, v in capacity_config.items() if k != 'tmodes'}
        else:
            config_for_init = capacity_config or {}
        
        # Default configuration for weather forecasting
        self.config = {
            # Architecture
            'd_model': 512,
            'num_layers_enc': 6,
            'num_layers_dec': 6,
            'num_heads': 8,
            'd_ff_multiplier': 4,
            'use_bias': True,
            
            # Input/Output
            'input_dim': input_dim,
            'output_dim': output_dim if output_dim is not None else input_dim,
            'input_format': '2d',  # Always 2D for weather data
            'patch_size': 1,  # No patching by default
            
            # Attention
            'attention_dropout': 0.1,
            'use_flash_attention': False,
            'use_causal_mask': False,  # Weather forecasting doesn't use causal mask in spatial domain
            'attention_window': None,
            
            # Regularization
            'dropout': 0.1,
            'layer_dropout': 0.0,
            'residual_dropout': 0.1,
            
            # Normalization
            'norm_type': 'layernorm',
            'norm_eps': 1e-6,
            'pre_norm': True,
            
            # Positional Encoding
            'pos_encoding': '2d_sinusoidal',  # Better for spatial data
            'height': 32,  # Default spatial dimensions
            'width': 32,
            'scale_embeddings': True,
            'embedding_dropout': 0.1,
            
            # Feed-Forward
            'activation': 'gelu',
            'glu': False,
            'dropout_ffn': 0.1,
            'ffn_bottleneck': None,
            
            # Initialization
            'init_range': 0.02,
            'init_method': 'normal',
            'bias_init': 'zero',
            
            # Advanced
            'use_checkpointing': False,
            'gradient_checkpointing': False,
            'mixed_precision': False,
            'dtype': torch.float32,  # Default dtype
            
            # Weather-specific
            'use_lead_time_embedding': True,
            'use_time_encoding': True,
            'num_forecast_steps': 24,  # Default to 24 hours (1 day)
            
            # TModes will be applied after initialization
            'tmodes_applied': False,
        }
        
        # Update with user config (without tmodes)
        if config_for_init:
            self.config.update(config_for_init)
        
        # Apply TModes if provided
        if self.tmodes:
            self._apply_tmodes_config()
        
        # Set derived values
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']
        
        # Initialize components
        self._initialize_input_projection()
        self._initialize_positional_encoding()
        self._initialize_time_encodings()
        self._initialize_encoder()
        self._initialize_decoder()
        self._initialize_output_layer()
        
        # Initialize weights
        self._initialize_weights()
        
    def _apply_tmodes_config(self):
        """Apply TModes configuration to the transformer config"""
        if not self.tmodes:
            return
        
        # Connection type affects attention pattern
        if self.tmodes.conn == Conn.Global:
            self.config['attention_window'] = None  # Full attention
        elif self.tmodes.conn == Conn.Axial:
            self.config['attention_window'] = 'axial'  # Special axial attention
            self.config['pos_encoding'] = 'axial'  # Use axial positional encoding
        elif self.tmodes.conn == Conn.Shift:
            self.config['attention_window'] = 3  # Local window
        
        # Flash attention
        if self.tmodes.flash == Flash.UseFlash:
            self.config['use_flash_attention'] = True
        elif self.tmodes.flash == Flash.NoFlash:
            self.config['use_flash_attention'] = False
        
        # Positional encoding type
        if self.tmodes.penc == PEnc.Learnable:
            self.config['pos_encoding'] = 'learned_2d'
        elif self.tmodes.penc == PEnc.Sinusoidal:
            self.config['pos_encoding'] = '2d_sinusoidal'
        elif self.tmodes.penc == PEnc.Linear:
            self.config['pos_encoding'] = 'linear'
        
        # Positional encoding sharing
        if self.tmodes.pshare == PShare.Global:
            self.config['pos_encoding'] = 'global_shared'
        elif self.tmodes.pshare == PShare.Axial:
            self.config['pos_encoding'] = 'axial'
        
        # Bias configuration
        self.config['use_bias'] = True  # Always use bias by default
        if self.tmodes.pbias == PBias.Learnable:
            self.config['attention_bias'] = True
            self.config['ffn_bias'] = True
        else:
            self.config['attention_bias'] = False
            self.config['ffn_bias'] = False
        
        self.config['tmodes_applied'] = True
    
    def _initialize_input_projection(self):
        """Initialize input projection from weather variables to model dimension"""
        if self.config.get('patch_size', 1) > 1:
            # Use patch embedding
            patch_size = self.config['patch_size']
            patch_dim = self.input_dim * patch_size * patch_size
            self.patch_embedding = nn.Conv2d(
                self.input_dim,
                self.config['d_model'],
                kernel_size=patch_size,
                stride=patch_size
            )
            self.input_proj = nn.Identity()
        else:
            # Linear projection
            self.patch_embedding = None
            self.input_proj = nn.Linear(
                self.input_dim,
                self.config['d_model'],
                bias=self.config['use_bias'] and self.config.get('ffn_bias', True)
            )
        
        self.embedding_dropout = nn.Dropout(self.config['embedding_dropout'])
    
    def _initialize_positional_encoding(self):
        """Initialize positional encoding for spatial data"""
        pos_encoding_type = self.config['pos_encoding']
        
        if pos_encoding_type == '2d_sinusoidal':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            self.pos_encoding = PositionalEncoding2D(
                self.config['d_model'],
                max_h=h,
                max_w=w
            )
        elif pos_encoding_type == 'learned_2d':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            self.pos_encoding = LearnedPositionalEncoding2D(
                self.config['d_model'],
                max_h=h,
                max_w=w
            )
        elif pos_encoding_type == 'axial':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            self.pos_encoding = AxialPositionalEncoding(
                self.config['d_model'],
                max_h=h,
                max_w=w
            )
        elif pos_encoding_type == 'global_shared':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            max_len = h * w
            self.pos_encoding = GlobalPositionalEncoding(
                self.config['d_model'],
                max_len
            )
        elif pos_encoding_type == 'rotary':
            self.pos_encoding = RotaryPositionalEncoding(
                self.config['d_model'] // self.config['num_heads']
            )
        elif pos_encoding_type == 'sinusoidal':
            max_len = self.config.get('height', 32) * self.config.get('width', 32)
            self.pos_encoding = SinusoidalPositionalEncoding(
                self.config['d_model'],
                max_len
            )
        elif pos_encoding_type == 'learned':
            max_len = self.config.get('height', 32) * self.config.get('width', 32)
            self.pos_encoding = LearnedPositionalEncoding(
                self.config['d_model'],
                max_len
            )
        elif pos_encoding_type == 'linear':
            max_len = self.config.get('height', 32) * self.config.get('width', 32)
            self.pos_encoding = LinearPositionalEncoding(
                self.config['d_model'],
                max_len
            )
        else:
            self.pos_encoding = None
    
    def _initialize_time_encodings(self):
        """Initialize time-related embeddings for weather forecasting"""
        self.use_lead_time_embedding = self.config.get('use_lead_time_embedding', True)
        self.use_time_encoding = self.config.get('use_time_encoding', True)
        self.num_forecast_steps = self.config.get('num_forecast_steps', 24)
        
        if self.use_lead_time_embedding:
            self.lead_time_embedding = nn.Embedding(
                self.num_forecast_steps + 1,  # +1 for zero lead time
                self.config['d_model']
            )
        
        if self.use_time_encoding:
            # Sinusoidal encoding for time of day, day of year, etc.
            self.time_encoding = SinusoidalTimeEncoding(
                self.config['d_model']
            )
    
    def _initialize_encoder(self):
        """Initialize encoder layers"""
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                d_ff=self.config['d_model'] * self.config['d_ff_multiplier'],
                dropout=self.config['dropout'],
                attention_dropout=self.config['attention_dropout'],
                activation=self.config['activation'],
                norm_type=self.config['norm_type'],
                norm_eps=self.config['norm_eps'],
                pre_norm=self.config['pre_norm'],
                use_bias=self.config['use_bias'] and self.config.get('ffn_bias', True),
                attention_bias=self.config.get('attention_bias', True),
                ffn_bias=self.config.get('ffn_bias', True),
                ffn_bottleneck=self.config['ffn_bottleneck'],
                glu=self.config['glu'],
                residual_dropout=self.config['residual_dropout'],
                layer_dropout=self.config['layer_dropout'],
                attention_window=self.config['attention_window'],
                use_flash_attention=self.config['use_flash_attention'],
                flash_attn_available=FLASH_ATTN_AVAILABLE,
                dtype=self.config.get('dtype', torch.float32)
            )
            for _ in range(self.config['num_layers_enc'])
        ])
        
        # Encoder final norm
        self.encoder_norm = self._create_norm_layer()
    
    def _initialize_decoder(self):
        """Initialize decoder layers"""
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=self.config['d_model'],
                num_heads=self.config['num_heads'],
                d_ff=self.config['d_model'] * self.config['d_ff_multiplier'],
                dropout=self.config['dropout'],
                attention_dropout=self.config['attention_dropout'],
                activation=self.config['activation'],
                norm_type=self.config['norm_type'],
                norm_eps=self.config['norm_eps'],
                pre_norm=self.config['pre_norm'],
                use_bias=self.config['use_bias'] and self.config.get('ffn_bias', True),
                attention_bias=self.config.get('attention_bias', True),
                ffn_bias=self.config.get('ffn_bias', True),
                ffn_bottleneck=self.config['ffn_bottleneck'],
                glu=self.config['glu'],
                residual_dropout=self.config['residual_dropout'],
                layer_dropout=self.config['layer_dropout'],
                attention_window=self.config['attention_window'],
                use_flash_attention=self.config['use_flash_attention'],
                use_causal_mask=self.config['use_causal_mask'],
                flash_attn_available=FLASH_ATTN_AVAILABLE,
                dtype=self.config.get('dtype', torch.float32)
            )
            for _ in range(self.config['num_layers_dec'])
        ])
        
        # Decoder final norm
        self.decoder_norm = self._create_norm_layer()
    
    def _initialize_output_layer(self):
        """Initialize output projection back to weather variables"""
        if self.config.get('patch_size', 1) > 1:
            # Use patch unembedding
            patch_size = self.config['patch_size']
            self.patch_unembedding = nn.ConvTranspose2d(
                self.config['d_model'],
                self.output_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            self.output_proj = nn.Identity()
        else:
            # Linear projection
            self.patch_unembedding = None
            self.output_proj = nn.Linear(
                self.config['d_model'],
                self.output_dim,
                bias=self.config['use_bias'] and self.config.get('ffn_bias', True)
            )
    
    def _create_norm_layer(self):
        """Create normalization layer based on config"""
        if self.config['norm_type'] == 'layernorm':
            return nn.LayerNorm(self.config['d_model'], eps=self.config['norm_eps'])
        elif self.config['norm_type'] == 'rmsnorm':
            return RMSNorm(self.config['d_model'], eps=self.config['norm_eps'])
        elif self.config['norm_type'] == 'batchnorm':
            return nn.BatchNorm1d(self.config['d_model'])
        else:
            raise ValueError(f"Unknown norm type: {self.config['norm_type']}")
    
    def _initialize_weights(self):
        """Initialize weights based on config"""
        init_range = self.config['init_range']
        init_method = self.config['init_method']
        
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_method == 'normal':
                    nn.init.normal_(param, mean=0.0, std=init_range)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    raise ValueError(f"Unknown init method: {init_method}")
            
            elif 'bias' in name:
                if self.config['bias_init'] == 'zero':
                    nn.init.zeros_(param)
                elif self.config['bias_init'] == 'normal':
                    nn.init.normal_(param, mean=0.0, std=init_range)
    
    def forward(self, x, lead_time=None, time_features=None, src_mask=None, tgt_mask=None):
        """
        Forward pass for weather forecasting.
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, T, H, W, C) for multiple time steps
            lead_time: Lead time indices for forecasting (B,) or (B, T). Values should be in range [0, num_forecast_steps]
            time_features: Additional time features (B, time_dim) or (B, T, time_dim)
            src_mask: Source mask for encoder
            tgt_mask: Target mask for decoder
            
        Returns:
            Output tensor of shape (B, H, W, C_out) or (B, T, H, W, C_out)
        """
        B = x.shape[0]
        
        # Handle multi-step input
        if x.dim() == 5:
            # (B, T, H, W, C) -> Process each time step
            T = x.shape[1]
            outputs = []
            for t in range(T):
                # Handle lead_time indexing
                if lead_time is not None:
                    lt_t = lead_time[:, t] if lead_time.dim() > 1 else lead_time
                    # Clamp lead time to valid range
                    lt_t = torch.clamp(lt_t, 0, self.num_forecast_steps)
                else:
                    lt_t = None
                
                # Handle time_features indexing
                if time_features is not None:
                    tf_t = time_features[:, t] if time_features.dim() > 2 else time_features
                else:
                    tf_t = None
                
                output_t = self._forward_single_step(x[:, t], 
                                                    lt_t,
                                                    tf_t,
                                                    src_mask, tgt_mask)
                outputs.append(output_t)
            return torch.stack(outputs, dim=1)
        else:
            # Single time step
            # Clamp lead time to valid range if provided
            if lead_time is not None:
                lead_time = torch.clamp(lead_time, 0, self.num_forecast_steps)
            return self._forward_single_step(x, lead_time, time_features, src_mask, tgt_mask)
    
    def _forward_single_step(self, x, lead_time, time_features, src_mask, tgt_mask):
        """Forward pass for a single time step"""
        B, H, W, C = x.shape
        
        # Update config with actual spatial dimensions
        self.config['height'] = H
        self.config['width'] = W
        
        # Project input to model dimension
        if self.patch_embedding is not None:
            # Patch embedding (B, C, H, W) -> (B, d_model, H/patch, W/patch)
            x_2d = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x_embedded = self.patch_embedding(x_2d)  # (B, d_model, H', W')
            H_patch = x_embedded.shape[2]
            W_patch = x_embedded.shape[3]
            x = x_embedded.permute(0, 2, 3, 1)  # (B, H', W', d_model)
            x = x.reshape(B, H_patch * W_patch, -1)  # (B, L, d_model)
            # Set encoding dimensions for shift connections
            H_enc = H_patch
            W_enc = W_patch
        else:
            # Linear projection
            x = x.reshape(B, H * W, C)  # (B, L, C)
            x = self.input_proj(x)  # (B, L, d_model)
            # Set encoding dimensions for shift connections (default to original dimensions)
            H_enc = H
            W_enc = W
        
        # Apply positional encoding
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (PositionalEncoding2D, AxialPositionalEncoding, LearnedPositionalEncoding2D)):
                # Reshape to 2D for spatial positional encoding
                L = x.shape[1]
                if H_enc * W_enc == L:
                    x = x.reshape(B, H_enc, W_enc, -1)
                    x = self.pos_encoding(x)
                    x = x.reshape(B, L, -1)
            else:
                x = self.pos_encoding(x)
        
        # Add time encodings if provided
        if self.use_lead_time_embedding and lead_time is not None:
            # Ensure lead_time is within valid range
            lead_time = torch.clamp(lead_time, 0, self.num_forecast_steps)
            lead_time_emb = self.lead_time_embedding(lead_time)  # (B, d_model) or (B, L, d_model)
            if lead_time_emb.dim() == 2:
                lead_time_emb = lead_time_emb.unsqueeze(1)  # (B, 1, d_model)
            x = x + lead_time_emb
        
        if self.use_time_encoding and time_features is not None:
            time_emb = self.time_encoding(time_features)  # (B, d_model) or (B, L, d_model)
            if time_emb.dim() == 2:
                time_emb = time_emb.unsqueeze(1)  # (B, 1, d_model)
            x = x + time_emb
        
        # Scale embeddings
        if self.config['scale_embeddings']:
            x = x * math.sqrt(self.config['d_model'])
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        # Apply shift connection if specified by TModes
        if self.tmodes and self.tmodes.conn == Conn.Shift:
            # Use H_enc and W_enc which are defined before the positional encoding check
            x = self._apply_shift_connection(x, H_enc, W_enc)
        
        # Encoder
        encoder_output = self._encode(x, src_mask)
        
        # For auto-regressive forecasting, we might use decoder
        # For now, using encoder-only architecture for simplicity
        # You can modify this for encoder-decoder architecture
        
        if self.config['num_layers_dec'] > 0:
            # Use decoder for auto-regressive prediction
            # Here we use the same input as memory (could be previous predictions)
            decoder_output = self._decode(encoder_output, encoder_output, tgt_mask, None)
            x = decoder_output
        else:
            x = encoder_output
        
        if self.config['pre_norm']:
            x = self.decoder_norm if self.config['num_layers_dec'] > 0 else self.encoder_norm(x)
        
        # Project back to output dimension
        if self.patch_unembedding is not None:
            # Unpatch
            x = x.reshape(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)  # (B, d_model, H', W')
            x = self.patch_unembedding(x)  # (B, C_out, H, W)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C_out)
        else:
            # Linear unprojection
            x = self.output_proj(x)  # (B, L, C_out)
            x = x.reshape(B, H, W, -1)  # (B, H, W, C_out)
        
        return x
    
    def _apply_shift_connection(self, x, H, W):
        """Apply shift operation for shift connections"""
        # x is (B, L, d_model) where L = H*W
        B, L, C = x.shape
        
        if H * W == L:
            # Reshape to 2D
            x_2d = x.reshape(B, H, W, C)
            
            # Create shifted versions
            x_shifted_down = torch.zeros_like(x_2d)
            x_shifted_right = torch.zeros_like(x_2d)
            
            # Shift down (vertical shift)
            x_shifted_down[:, 1:, :, :] = x_2d[:, :-1, :, :]
            
            # Shift right (horizontal shift)
            x_shifted_right[:, :, 1:, :] = x_2d[:, :, :-1, :]
            
            # Combine shifts (you can use either or combine them)
            # Here we'll use average of both shifts
            x_combined = (x_shifted_down + x_shifted_right) / 2
            
            # Handle boundaries - copy original values at boundaries
            # For first row, use original values (no shift down available)
            x_combined[:, 0, :, :] = x_2d[:, 0, :, :]
            # For first column, use original values (no shift right available)
            x_combined[:, :, 0, :] = x_2d[:, :, 0, :]
            
            return x_combined.reshape(B, L, C)
        return x
    
    def _encode(self, x, mask=None):
        """Encoder forward pass"""
        # Apply axial attention if specified by TModes
        if self.config.get('attention_window') == 'axial':
            x = self._apply_axial_attention(x, mask)
        else:
            for layer in self.encoder_layers:
                if self.config.get('use_checkpointing', False) and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, mask, use_reentrant=False
                    )
                else:
                    x = layer(x, mask=mask)
        
        if self.config['pre_norm']:
            x = self.encoder_norm(x)
        
        return x
    
    def _apply_axial_attention(self, x, mask):
        """Apply axial attention (row-wise then column-wise)"""
        B, L, C = x.shape
        H = self.config.get('height', int(math.sqrt(L)))
        W = L // H
        
        if H * W == L:
            # Reshape to 2D
            x = x.reshape(B, H, W, C)
            
            # Row-wise attention
            x_rows = x.reshape(B * H, W, C)
            for layer in self.encoder_layers:
                x_rows = layer(x_rows)
            x_rows = x_rows.reshape(B, H, W, C)
            
            # Column-wise attention
            x_cols = x_rows.permute(0, 2, 1, 3).reshape(B * W, H, C)
            for layer in self.encoder_layers:
                x_cols = layer(x_cols)
            x_cols = x_cols.reshape(B, W, H, C).permute(0, 2, 1, 3)
            
            return x_cols.reshape(B, L, C)
        
        return x
    
    def _decode(self, x, memory, tgt_mask=None, memory_mask=None):
        """Decoder forward pass"""
        for layer in self.decoder_layers:
            if self.config.get('use_checkpointing', False) and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, memory, tgt_mask, memory_mask, use_reentrant=False
                )
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        if self.config['pre_norm']:
            x = self.decoder_norm(x)
        
        return x
    
    def get_tmodes_info(self):
        """Get TModes information"""
        if not self.tmodes:
            return "No TModes configured"
        
        return {
            'conn': self.tmodes.conn.name,
            'flash': self.tmodes.flash.name,
            'penc': self.tmodes.penc.name,
            'pshare': self.tmodes.pshare.name,
            'pbias': self.tmodes.pbias.name,
            'applied': self.config['tmodes_applied']
        }
    
    def get_config_summary(self):
        """Get configuration summary including TModes effects"""
        summary = {}
        
        # Add basic config
        for key in ['d_model', 'num_layers_enc', 'num_layers_dec', 'num_heads', 
                   'd_ff_multiplier', 'pos_encoding', 'attention_window',
                   'input_dim', 'output_dim', 'height', 'width', 'num_forecast_steps',
                   'dtype']:
            if key in self.config:
                summary[key] = self.config[key]
        
        # Add TModes info
        if self.tmodes:
            summary['tmodes'] = self.get_tmodes_info()
            summary['use_flash_attention'] = self.config['use_flash_attention']
            summary['attention_bias'] = self.config.get('attention_bias', True)
            summary['ffn_bias'] = self.config.get('ffn_bias', True)
        
        return summary

# ==================== Supporting Classes ====================

class TransformerEncoderLayer(nn.Module):
    """Encoder layer with TModes support"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, 
                 attention_dropout=0.1, activation='gelu', norm_type='layernorm',
                 norm_eps=1e-6, pre_norm=True, use_bias=True, attention_bias=True,
                 ffn_bias=True, ffn_bottleneck=None, glu=False, residual_dropout=0.1,
                 layer_dropout=0.0, attention_window=None, use_flash_attention=False,
                 flash_attn_available=False, dtype=torch.float32):
        super().__init__()
        
        self.layer_dropout = layer_dropout
        self.use_flash_attention = use_flash_attention and flash_attn_available
        self.dtype = dtype
        
        # Self-attention with TModes-controlled bias
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=self.use_flash_attention,
            causal=False,
            dtype=dtype
        )
        
        # FFN with TModes-controlled bias
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias and ffn_bias,
            bottleneck=ffn_bottleneck,
            glu=glu,
            dtype=dtype
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps) if norm_type == 'layernorm' else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps) if norm_type == 'layernorm' else nn.Identity()
        
        # Dropout
        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        
        self.pre_norm = pre_norm
    
    def forward(self, x, mask=None):
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                return x
        
        residual = x
        
        if self.pre_norm:
            x = self.norm1(x)
            x_attn = self.self_attn(x, x, x, mask=mask)
            x = residual + self.dropout1(x_attn)
        else:
            x_attn = self.self_attn(x, x, x, mask=mask)
            x = self.norm1(residual + self.dropout1(x_attn))
        
        residual = x
        
        if self.pre_norm:
            x = self.norm2(x)
            x_ffn = self.ffn(x)
            x = residual + self.dropout2(x_ffn)
        else:
            x_ffn = self.ffn(x)
            x = self.norm2(residual + self.dropout2(x_ffn))
        
        return x

class TransformerDecoderLayer(nn.Module):
    """Decoder layer with TModes support"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, 
                 attention_dropout=0.1, activation='gelu', norm_type='layernorm',
                 norm_eps=1e-6, pre_norm=True, use_bias=True, attention_bias=True,
                 ffn_bias=True, ffn_bottleneck=None, glu=False, residual_dropout=0.1,
                 layer_dropout=0.0, attention_window=None, use_flash_attention=False,
                 use_causal_mask=True, flash_attn_available=False, dtype=torch.float32):
        super().__init__()
        
        self.layer_dropout = layer_dropout
        self.use_flash_attention = use_flash_attention and flash_attn_available
        self.dtype = dtype
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=self.use_flash_attention,
            causal=use_causal_mask,
            dtype=dtype
        )
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=self.use_flash_attention,
            causal=False,
            dtype=dtype
        )
        
        # FFN
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias and ffn_bias,
            bottleneck=ffn_bottleneck,
            glu=glu,
            dtype=dtype
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps) if norm_type == 'layernorm' else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps) if norm_type == 'layernorm' else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model, eps=norm_eps) if norm_type == 'layernorm' else nn.Identity()
        
        # Dropout
        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)
        
        self.pre_norm = pre_norm
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                return x
        
        residual = x
        
        if self.pre_norm:
            x = self.norm1(x)
            x_attn = self.self_attn(x, x, x, mask=tgt_mask)
            x = residual + self.dropout1(x_attn)
        else:
            x_attn = self.self_attn(x, x, x, mask=tgt_mask)
            x = self.norm1(residual + self.dropout1(x_attn))
        
        residual = x
        
        if self.pre_norm:
            x = self.norm2(x)
            x_cross = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = residual + self.dropout2(x_cross)
        else:
            x_cross = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = self.norm2(residual + self.dropout2(x_cross))
        
        residual = x
        
        if self.pre_norm:
            x = self.norm3(x)
            x_ffn = self.ffn(x)
            x = residual + self.dropout3(x_ffn)
        else:
            x_ffn = self.ffn(x)
            x = self.norm3(residual + self.dropout3(x_ffn))
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 attention_window=None, use_flash_attention=False, causal=False,
                 dtype=torch.float32):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.causal = causal
        self.attention_window = attention_window
        self.dtype = dtype
        
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for FlashAttention or standard attention
        Q = Q.view(B, L, self.num_heads, self.d_k)
        K = K.view(B, L, self.num_heads, self.d_k)
        V = V.view(B, L, self.num_heads, self.d_k)
        
        # For standard attention, we need to transpose
        Q_standard = Q.transpose(1, 2)
        K_standard = K.transpose(1, 2)
        V_standard = V.transpose(1, 2)
        
        if isinstance(self.attention_window, int) and self.attention_window > 0:
            mask = self._create_local_mask(L, mask)
        
        if self.causal and mask is None:
            mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
        
        # Check if mask is not None and has correct shape
        if mask is not None:
            # Ensure mask has correct shape for broadcasting
            if mask.dim() == 2:  # (L, L) mask
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            elif mask.dim() == 3:  # (B, L, L) mask
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            # mask should now be (B, 1, L, L) or (1, 1, L, L)
        
        if self.use_flash_attention:
            # FlashAttention doesn't support arbitrary masks, only causal mask
            # We need to handle other masks by falling back to standard attention
            if mask is not None and not (self.causal and mask.equal(torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool())):
                # Fall back to standard attention for non-causal masks
                attn_scores = torch.matmul(Q_standard, K_standard.transpose(-2, -1)) / math.sqrt(self.d_k)
                
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(mask, -1e9)
                
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                x = torch.matmul(attn_weights, V_standard)
                x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
            else:
                # Use FlashAttention for causal attention or no mask
                try:
                    # Convert to supported dtype for FlashAttention if needed
                    original_dtype = Q.dtype
                    
                    # Check if we need to convert for FlashAttention
                    if original_dtype not in [torch.float16, torch.bfloat16]:
                        # Convert to bfloat16 if available, otherwise float16
                        target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        Q_fa = Q.to(target_dtype)
                        K_fa = K.to(target_dtype)
                        V_fa = V.to(target_dtype)
                    else:
                        Q_fa = Q
                        K_fa = K
                        V_fa = V
                    
                    # FlashAttention expects (batch_size, seqlen, nheads, headdim)
                    # and returns (batch_size, seqlen, nheads, headdim)
                    x = flash_attn_func(
                        Q_fa, K_fa, V_fa,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        causal=self.causal,
                        softmax_scale=1.0 / math.sqrt(self.d_k)
                    )
                    
                    # Convert back to original dtype
                    if x.dtype != original_dtype:
                        x = x.to(original_dtype)
                    
                    # Reshape back to (batch_size, seqlen, d_model)
                    x = x.reshape(B, L, self.d_model)
                except Exception as e:
                    print(f"FlashAttention failed: {e}. Falling back to standard attention.")
                    # Fall back to standard attention
                    attn_scores = torch.matmul(Q_standard, K_standard.transpose(-2, -1)) / math.sqrt(self.d_k)
                    
                    if mask is not None:
                        attn_scores = attn_scores.masked_fill(mask, -1e9)
                    
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    x = torch.matmul(attn_weights, V_standard)
                    x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        else:
            # Standard attention
            attn_scores = torch.matmul(Q_standard, K_standard.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            x = torch.matmul(attn_weights, V_standard)
            x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        x = self.W_o(x)
        
        return x
    
    def _create_local_mask(self, seq_len, existing_mask):
        """Create local attention mask for windowed attention"""
        if self.attention_window is None:
            return existing_mask
        
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.W_q.weight.device)
        for i in range(seq_len):
            start = max(0, i - self.attention_window)
            end = min(seq_len, i + self.attention_window + 1)
            local_mask[i, :start] = False
            local_mask[i, end:] = False
        
        if existing_mask is not None:
            local_mask = local_mask & existing_mask
        
        return local_mask

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1,
                 use_bias=True, bottleneck=None, glu=False, dtype=torch.float32):
        super().__init__()
        
        self.glu = glu
        self.bottleneck = bottleneck
        self.dtype = dtype
        
        if bottleneck is not None:
            self.fc1 = nn.Linear(d_model, bottleneck, bias=use_bias)
            self.fc2 = nn.Linear(bottleneck, d_ff, bias=use_bias)
            self.fc3 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.act1 = self._get_activation(activation)
            self.act2 = self._get_activation(activation)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        elif glu:
            self.fc1 = nn.Linear(d_model, d_ff * 2, bias=use_bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.activation = self._get_activation(activation)
            self.dropout = nn.Dropout(dropout)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.activation = self._get_activation(activation)
            self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        if self.bottleneck is not None:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
        elif self.glu:
            x = self.fc1(x)
            x, gate = x.chunk(2, dim=-1)
            x = x * self.activation(gate)
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.fc2(x)
        
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight

# ==================== Testing Weather Forecasting Transformer ====================

if __name__ == "__main__":
    print("Testing Weather Forecasting Transformer")
    print(f"FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    print("=" * 60)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 2
    height, width = 32, 32
    input_dim = 10  # 10 weather variables (temperature, pressure, humidity, etc.)
    output_dim = 10  # Predict same 10 variables
    
    # Test 1: Basic weather transformer
    print("\n1. Basic Weather Transformer:")
    basic_config = {
        'd_model': 128,
        'num_layers_enc': 4,
        'num_layers_dec': 0,  # Encoder-only for simplicity
        'num_heads': 8,
        'height': height,
        'width': width,
        'num_forecast_steps': 24,
    }
    
    basic_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=basic_config
    ).to(device)
    
    # Create random weather data
    weather_data = torch.randn(batch_size, height, width, input_dim).to(device)
    
    with torch.no_grad():
        output = basic_model(weather_data)
        print(f"  Input shape: {weather_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, {height}, {width}, {output_dim})")
    
    # Test 2: Transformer with TModes (Axial attention)
    print("\n2. Weather Transformer with Axial Attention:")
    axial_config = {
        'd_model': 192,
        'num_layers_enc': 6,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'num_forecast_steps': 24,
        'tmodes': TModes(
            conn=Conn.Axial,
            flash=Flash.NoFlash,
            penc=PEnc.Sinusoidal,
            pshare=PShare.Axial,
            pbias=PBias.Learnable
        )
    }
    
    axial_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=axial_config
    ).to(device)
    
    with torch.no_grad():
        output = axial_model(weather_data)
        print(f"  Output shape: {output.shape}")
        print(f"  Using axial attention: {axial_model.config['attention_window'] == 'axial'}")
    
    # Test 3: Transformer with Shift connections
    print("\n3. Weather Transformer with Shift Connections:")
    shift_config = {
        'd_model': 128,
        'num_layers_enc': 4,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'num_forecast_steps': 24,
        'tmodes': TModes(
            conn=Conn.Shift,
            flash=Flash.NoFlash,
            penc=PEnc.Learnable,
            pshare=PShare.Global,
            pbias=PBias.Nothing
        )
    }
    
    shift_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=shift_config
    ).to(device)
    
    with torch.no_grad():
        output = shift_model(weather_data)
        print(f"  Output shape: {output.shape}")
        print(f"  Using shift connections: {shift_model.tmodes.conn == Conn.Shift}")
    
    # Test 4: With lead time and time features
    print("\n4. Weather Transformer with Time Encodings:")
    time_config = {
        'd_model': 128,
        'num_layers_enc': 4,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'use_lead_time_embedding': True,
        'use_time_encoding': True,
        'num_forecast_steps': 24,
    }
    
    time_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=time_config
    ).to(device)
    
    # Create time features - use valid lead times (0-23)
    lead_time = torch.randint(0, 24, (batch_size,)).to(device)  # Lead time in hours (0-23)
    time_features = torch.randn(batch_size, 3).to(device)  # e.g., hour_of_day, day_of_year, month
    
    with torch.no_grad():
        output = time_model(weather_data, lead_time=lead_time, time_features=time_features)
        print(f"  Output shape: {output.shape}")
        print(f"  Using lead time embedding: {time_model.use_lead_time_embedding}")
        print(f"  Using time encoding: {time_model.use_time_encoding}")
    
    # Test 5: Multi-time-step input
    print("\n5. Weather Transformer with Multi-step Input:")
    multi_step_config = {
        'd_model': 128,
        'num_layers_enc': 4,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'num_forecast_steps': 24,  # Important: Set this to match lead times
    }
    
    multi_step_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=multi_step_config
    ).to(device)
    
    # Create multi-step weather data (e.g., past 6 hours)
    T = 6  # 6 time steps
    multi_step_data = torch.randn(batch_size, T, height, width, input_dim).to(device)
    # Use valid lead times (0-5 for 6 time steps, all within 0-23 range)
    multi_step_lead_time = torch.randint(0, 6, (batch_size, T)).to(device)  # Reduced range
    multi_step_time_features = torch.randn(batch_size, T, 3).to(device)
    
    with torch.no_grad():
        output = multi_step_model(
            multi_step_data, 
            lead_time=multi_step_lead_time,
            time_features=multi_step_time_features
        )
        print(f"  Input shape: {multi_step_data.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: ({batch_size}, {T}, {height}, {width}, {output_dim})")
        print(f"  Lead time range: 0-{multi_step_model.num_forecast_steps}")
    
    # Test 6: With patching
    print("\n6. Weather Transformer with Patching:")
    patch_config = {
        'd_model': 256,
        'num_layers_enc': 4,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'patch_size': 4,  # 4x4 patches
        'num_forecast_steps': 24,
    }
    
    patch_model = WeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=patch_config
    ).to(device)
    
    with torch.no_grad():
        output = patch_model(weather_data)
        print(f"  Output shape: {output.shape}")
        print(f"  Patch size: {patch_model.config['patch_size']}")
        print(f"  Has patch embedding: {patch_model.patch_embedding is not None}")
    
    # Test 7: Performance test with different dtypes
    print("\n7. Performance Test with Different Dtypes:")
    
    test_cases = [
        ("Small spatial", 2, 32, 32, 10),
        ("Medium spatial", 2, 64, 64, 10),
        ("Large spatial", 1, 128, 128, 10),
    ]
    
    # Available dtypes based on device support
    available_dtypes = [("fp32", torch.float32)]
    
    if device.type == 'cuda':
        available_dtypes.append(("fp16", torch.float16))
        if torch.cuda.is_bf16_supported():
            available_dtypes.append(("bf16", torch.bfloat16))
    
    for name, b, h, w, c in test_cases:
        print(f"\n  {name} (batch={b}, h={h}, w={w}, c={c}):")
        
        for dtype_name, dtype in available_dtypes:
            # Only use FlashAttention if CUDA is available and dtype is half-precision
            can_use_flash = (FLASH_ATTN_AVAILABLE and device.type == 'cuda' and 
                           dtype in [torch.float16, torch.bfloat16])
            
            perf_config = {
                'd_model': 256,
                'num_layers_enc': 8,
                'num_layers_dec': 0,
                'num_heads': 8,
                'height': h,
                'width': w,
                'num_forecast_steps': 24,
                'dtype': dtype,
                'tmodes': TModes(
                    conn=Conn.Global,
                    flash=Flash.UseFlash if can_use_flash else Flash.NoFlash,
                    penc=PEnc.Sinusoidal,
                    pshare=PShare.Global,
                    pbias=PBias.Nothing
                )
            }
            
            perf_model = WeatherForecastingTransformer(
                input_dim=c,
                output_dim=c,
                capacity_config=perf_config
            ).to(device)
            
            # Cast model to appropriate dtype
            if dtype != torch.float32:
                perf_model = perf_model.to(dtype)
            
            test_data = torch.randn(b, h, w, c, dtype=dtype).to(device)
            
            # Warm-up
            with torch.no_grad():
                _ = perf_model(test_data)
            
            # Time the forward pass
            import time
            start_time = time.time()
            with torch.no_grad():
                output = perf_model(test_data)
            
            # Synchronize if using CUDA
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            
            print(f"    {dtype_name}:")
            print(f"      Time: {elapsed_time:.4f}s")
            print(f"      Memory: {output.element_size() * output.nelement() / 1024 / 1024:.2f} MB")
            print(f"      Using FlashAttention: {perf_model.config['use_flash_attention']}")
            print(f"      Output dtype: {output.dtype}")
    
    # Test 8: Verify TModes configuration
    print("\n8. Verifying TModes Configuration:")
    
    verify_config = {
        'd_model': 128,
        'height': 32,
        'width': 32,
        'num_forecast_steps': 24,
        'tmodes': TModes(
            conn=Conn.Axial,
            flash=Flash.NoFlash,
            penc=PEnc.Learnable,
            pshare=PShare.Axial,
            pbias=PBias.Learnable
        )
    }
    
    verify_model = WeatherForecastingTransformer(
        input_dim=10,
        output_dim=10,
        capacity_config=verify_config
    ).to(device)
    
    summary = verify_model.get_config_summary()
    print(f"  Config summary:")
    print(f"    d_model: {summary['d_model']}")
    print(f"    Input dim: {summary['input_dim']}")
    print(f"    Output dim: {summary['output_dim']}")
    print(f"    Height: {summary['height']}")
    print(f"    Width: {summary['width']}")
    print(f"    Pos encoding: {summary['pos_encoding']}")
    print(f"    Attention window: {summary['attention_window']}")
    print(f"    Flash attention: {summary['use_flash_attention']}")
    print(f"    Num forecast steps: {summary['num_forecast_steps']}")
    print(f"    dtype: {summary.get('dtype', 'torch.float32')}")
    
    # Test 9: Test with bfloat16
    print("\n9. Testing with BF16:")
    
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        bf16_config = {
            'd_model': 128,
            'num_layers_enc': 4,
            'num_layers_dec': 0,
            'num_heads': 8,
            'height': 32,
            'width': 32,
            'num_forecast_steps': 24,
            'dtype': torch.bfloat16,
            'tmodes': TModes(
                conn=Conn.Global,
                flash=Flash.UseFlash if FLASH_ATTN_AVAILABLE else Flash.NoFlash,
                penc=PEnc.Sinusoidal,
                pshare=PShare.Global,
                pbias=PBias.Nothing
            )
        }
        
        bf16_model = WeatherForecastingTransformer(
            input_dim=10,
            output_dim=10,
            capacity_config=bf16_config
        ).to(device).to(torch.bfloat16)  # Cast to bfloat16
        
        test_data_bf16 = torch.randn(2, 32, 32, 10, dtype=torch.bfloat16).to(device)
        
        with torch.no_grad():
            output_bf16 = bf16_model(test_data_bf16)
        
        print(f"  BF16 model output shape: {output_bf16.shape}")
        print(f"  BF16 model using FlashAttention: {bf16_model.config['use_flash_attention']}")
        print(f"  Output dtype: {output_bf16.dtype}")
        
        # Test 10: Mixed precision training simulation
        print("\n10. Mixed Precision Training Simulation:")
        
        # Create model in fp32
        mixed_model = WeatherForecastingTransformer(
            input_dim=10,
            output_dim=10,
            capacity_config={
                'd_model': 128,
                'num_layers_enc': 4,
                'num_layers_dec': 0,
                'num_heads': 8,
                'height': 32,
                'width': 32,
                'num_forecast_steps': 24,
                'dtype': torch.float32,
            }
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(mixed_model.parameters(), lr=1e-4)
        
        # Simulate mixed precision training
        test_input = torch.randn(2, 32, 32, 10, device=device)
        target = torch.randn(2, 32, 32, 10, device=device)
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output_mixed = mixed_model(test_input)
            loss = F.mse_loss(output_mixed, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Mixed precision training completed successfully")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Output shape: {output_mixed.shape}")
        print(f"  Output dtype: {output_mixed.dtype}")
        
    else:
        print("  Skipping BF16 tests (CUDA with BF16 support not available)")
    
    print("\n" + "=" * 60)
    print("All weather forecasting tests completed successfully!")
    print("=" * 60)