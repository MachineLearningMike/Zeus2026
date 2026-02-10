import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

# Import FlashAttention function
try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    print("Warning: FlashAttention not installed. Install via: pip install flash-attn")
    FLASH_ATTENTION_AVAILABLE = False

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

# ==================== FlashAttention Helper Functions ====================

class FlashAttentionHelper:
    @staticmethod
    def check_flash_compatibility(Q, K, V, mask=None):
        """Check if FlashAttention can be used with given tensors and mask"""
        if not FLASH_ATTENTION_AVAILABLE:
            return False
        
        # Check data types - FlashAttention requires float16 or bfloat16
        if Q.dtype not in [torch.float16, torch.bfloat16]:
            return False
        
        # Check if mask is FlashAttention-compatible
        if mask is not None:
            # FlashAttention only supports causal masks or no mask
            if mask.dtype == torch.bool:
                # Check if it's a causal mask (upper triangular)
                seq_len = mask.shape[-1]
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=mask.device, dtype=torch.bool))
                if not torch.all(mask == causal_mask):
                    return False
            else:
                # Float masks not supported by standard FlashAttention
                return False
        
        return True
    
    @staticmethod
    def apply_flash_attention(Q, K, V, mask=None, dropout_p=0.0, causal=False, 
                             window_size=(-1, -1), alibi_slopes=None):
        """Apply FlashAttention directly"""
        if not FLASH_ATTENTION_AVAILABLE:
            raise RuntimeError("FlashAttention not installed")
        
        # Convert mask to causal flag if applicable
        if mask is not None and mask.dtype == torch.bool:
            # Check if it's a causal mask
            seq_len = mask.shape[-1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=mask.device, dtype=torch.bool))
            if torch.all(mask == causal_mask):
                causal = True
                mask = None
            else:
                # Non-causal bool mask not supported by FlashAttention
                # Fall back to manual implementation
                return None
        
        # Reshape for FlashAttention: (batch, seqlen, nheads, headdim)
        batch_size, nheads, seqlen, head_dim = Q.shape
        
        # Transpose to FlashAttention format: (batch, seqlen, nheads, headdim)
        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        
        # Apply FlashAttention
        try:
            output = flash_attn_func(
                Q, K, V,
                dropout_p=dropout_p,
                softmax_scale=1.0 / math.sqrt(head_dim),
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=False
            )
            
            # Transpose back: (batch, nheads, seqlen, headdim)
            output = output.transpose(1, 2).contiguous()
            return output
            
        except RuntimeError as e:
            print(f"FlashAttention failed: {e}")
            return None

# ==================== Fixed Positional Encoding Classes ====================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Get sequence length
        seq_len = x.size(1)
        
        # Get positional encoding slice
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        pe_slice = self.pe[:, :seq_len, :]
        
        # Ensure positional encoding matches the dtype of x
        if pe_slice.dtype != x.dtype:
            pe_slice = pe_slice.to(x.dtype)
        
        # Ensure dimensions match
        if pe_slice.shape[-1] != x.shape[-1]:
            raise ValueError(f"Positional encoding dimension {pe_slice.shape[-1]} "
                           f"doesn't match input dimension {x.shape[-1]}. "
                           f"Expected d_model={self.d_model}, got input shape={x.shape}")
        
        return x + pe_slice

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pe(positions)
        
        # Ensure positional encoding matches the dtype of x
        if pos_emb.dtype != x.dtype:
            pos_emb = pos_emb.to(x.dtype)
        
        # Ensure dimensions match
        if pos_emb.shape[-1] != x.shape[-1]:
            raise ValueError(f"Positional encoding dimension {pos_emb.shape[-1]} "
                           f"doesn't match input dimension {x.shape[-1]}. "
                           f"Expected d_model={self.d_model}, got input shape={x.shape}")
        
        return x + pos_emb

class GlobalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        pe_slice = self.pe[:, :seq_len, :]
        
        # Ensure positional encoding matches the dtype of x
        if pe_slice.dtype != x.dtype:
            pe_slice = pe_slice.to(x.dtype)
        
        # Ensure dimensions match
        if pe_slice.shape[-1] != x.shape[-1]:
            raise ValueError(f"Positional encoding dimension {pe_slice.shape[-1]} "
                           f"doesn't match input dimension {x.shape[-1]}. "
                           f"Expected d_model={self.d_model}, got input shape={x.shape}")
        
        return x + pe_slice

class AxialPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.d_model = d_model
        self.row_encoding = nn.Parameter(torch.randn(max_h, d_model // 2))
        self.col_encoding = nn.Parameter(torch.randn(max_w, d_model // 2))
        nn.init.normal_(self.row_encoding, mean=0.0, std=0.02)
        nn.init.normal_(self.col_encoding, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        
        if C != self.d_model:
            raise ValueError(f"Input dimension {C} doesn't match axial PE dimension {self.d_model}")
        
        row_enc = self.row_encoding[:H].unsqueeze(1).unsqueeze(0)
        col_enc = self.col_encoding[:W].unsqueeze(0).unsqueeze(0)
        
        # Ensure encodings match the dtype of x
        if row_enc.dtype != x.dtype:
            row_enc = row_enc.to(x.dtype)
            col_enc = col_enc.to(x.dtype)
        
        pos_enc = torch.cat([
            row_enc.expand(B, H, W, C // 2),
            col_enc.expand(B, H, W, C // 2)
        ], dim=-1)
        return x + pos_enc

class LinearPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = nn.Linear(1, d_model, bias=False)
        self.positions = torch.arange(max_len).float().unsqueeze(1)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")
        
        pos = self.positions[:seq_len].unsqueeze(0).to(x.device)
        pos_enc = self.pe(pos)
        
        # Ensure positional encoding matches the dtype of x
        if pos_enc.dtype != x.dtype:
            pos_enc = pos_enc.to(x.dtype)
        
        # Ensure dimensions match
        if pos_enc.shape[-1] != x.shape[-1]:
            raise ValueError(f"Positional encoding dimension {pos_enc.shape[-1]} "
                           f"doesn't match input dimension {x.shape[-1]}. "
                           f"Expected d_model={self.d_model}, got input shape={x.shape}")
        
        return x + pos_enc

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.d_model = d_model
        pe_h = self._get_1d_encoding(max_h, d_model // 2)
        pe_w = self._get_1d_encoding(max_w, d_model // 2)
        pe = torch.zeros(1, max_h, max_w, d_model)
        pe[0, :, :, :d_model//2] = pe_h.unsqueeze(2)
        pe[0, :, :, d_model//2:] = pe_w.unsqueeze(1)
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
        
        if H > self.pe.shape[1] or W > self.pe.shape[2]:
            raise ValueError(f"Input size ({H}x{W}) exceeds max size "
                           f"({self.pe.shape[1]}x{self.pe.shape[2]})")
        
        if C != self.d_model:
            raise ValueError(f"Input dimension {C} doesn't match 2D PE dimension {self.d_model}")
        
        # Ensure positional encoding matches the dtype of x
        pe = self.pe[:, :H, :W, :]
        if pe.dtype != x.dtype:
            pe = pe.to(x.dtype)
        return x + pe

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
        
        # Ensure cos/sin match x dtype
        if cos.dtype != x.dtype:
            cos = cos.to(x.dtype)
            sin = sin.to(x.dtype)
            
        x1, x2 = x[..., :self.dim], x[..., self.dim:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight

# ==================== MultiHeadAttention with explicit FlashAttention ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 attention_window=None, use_flash_attention=False, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention and FLASH_ATTENTION_AVAILABLE
        self.causal = causal
        self.attention_window = attention_window
        
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Window attention parameters for FlashAttention
        self.window_size = (-1, -1)  # Default: no window
        
        if isinstance(attention_window, int) and attention_window > 0:
            # For local window attention
            self.window_size = (attention_window, attention_window)
    
    def forward(self, query, key, value, mask=None):
        # Check if query is 2D (B, L) or 3D (B, L, C)
        if query.dim() == 2:
            # Add embedding dimension
            query = query.unsqueeze(-1)
            key = key.unsqueeze(-1)
            value = value.unsqueeze(-1)
        
        B, L, _ = query.shape
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for attention: (B, nheads, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Store original dtype for fallback
        original_dtype = Q.dtype
        
        # Create local mask if using windowed attention
        if isinstance(self.attention_window, int) and self.attention_window > 0:
            mask = self._create_local_mask(L, mask)
        
        # Create causal mask if needed
        if self.causal and mask is None:
            # Create causal mask: (L, L)
            mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
        
        # Try to use FlashAttention if enabled
        if self.use_flash_attention:
            # Check if FlashAttention can handle this case
            if FlashAttentionHelper.check_flash_compatibility(Q, K, V, mask):
                output = FlashAttentionHelper.apply_flash_attention(
                    Q, K, V,
                    mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    causal=self.causal and mask is None,
                    window_size=self.window_size
                )
                
                if output is not None:
                    # FlashAttention succeeded
                    x = output
                else:
                    # FlashAttention failed, fall back to manual
                    x = self._manual_attention(Q, K, V, mask)
            else:
                # Not compatible with FlashAttention, use manual
                x = self._manual_attention(Q, K, V, mask)
        else:
            # Manual attention
            x = self._manual_attention(Q, K, V, mask)
        
        # Ensure output dtype matches input dtype
        if x.dtype != original_dtype:
            x = x.to(original_dtype)
        
        # Reshape back: (B, L, d_model)
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        x = self.W_o(x)
        
        return x
    
    def _manual_attention(self, Q, K, V, mask):
        """Manual attention implementation (fallback)"""
        B, nheads, L, d_k = Q.shape
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Choose mask value based on dtype to avoid overflow in float16
            if attn_scores.dtype == torch.float16:
                mask_value = -1e4  # Smaller value for float16 to avoid overflow
            else:
                mask_value = -1e9  # Standard value for float32
            
            # Handle mask based on its dtype
            if mask.dtype == torch.bool:
                # For boolean masks, ensure proper shape
                if mask.dim() == 2:
                    # (L, L) -> (1, 1, L, L) -> (B, nheads, L, L)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask = mask.expand(B, nheads, L, L)
                elif mask.dim() == 3:
                    # (B, L, L) -> (B, 1, L, L) -> (B, nheads, L, L)
                    mask = mask.unsqueeze(1)
                    mask = mask.expand(B, nheads, L, L)
                
                # Apply boolean mask directly
                attn_scores = attn_scores.masked_fill(mask, mask_value)
            else:
                # For float masks, ensure same dtype and proper shape
                mask = mask.to(attn_scores.dtype)
                if mask.dim() == 2:
                    # (L, L) -> (1, 1, L, L) -> (B, nheads, L, L)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask = mask.expand(B, nheads, L, L)
                elif mask.dim() == 3:
                    # (B, L, L) -> (B, 1, L, L) -> (B, nheads, L, L)
                    mask = mask.unsqueeze(1)
                    mask = mask.expand(B, nheads, L, L)
                
                # Add mask (should contain mask_value where attention should be masked)
                attn_scores = attn_scores + mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        x = torch.matmul(attn_weights, V)
        
        return x
    
    def _create_local_mask(self, seq_len, existing_mask):
        """Create local attention mask"""
        if self.attention_window is None:
            return existing_mask
        
        device = existing_mask.device if existing_mask is not None else 'cpu'
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            start = max(0, i - self.attention_window)
            end = min(seq_len, i + self.attention_window + 1)
            local_mask[i, :start] = False
            local_mask[i, end:] = False
        
        if existing_mask is not None:
            if existing_mask.dtype == torch.bool:
                local_mask = local_mask & existing_mask
            else:
                # If existing mask is float, we need to handle it differently
                # For now, just use the local mask
                pass
        
        return local_mask

# ==================== CapacityControlTransformer with explicit FlashAttention ====================

class CapacityControlTransformer(nn.Module):
    """
    Transformer with TModes integrated directly into config.
    
    Usage:
        config = {
            'd_model': 512,
            'tmodes': TModes(conn=Conn.Global, flash=Flash.UseFlash)
        }
        model = CapacityControlTransformer(vocab_size=1000, capacity_config=config)
    """
    
    def __init__(self, vocab_size, capacity_config=None):
        super().__init__()
        
        # Extract TModes from config if present
        self.tmodes = None
        if capacity_config and 'tmodes' in capacity_config:
            self.tmodes = capacity_config['tmodes']
            # Create a copy of config without tmodes for the rest of initialization
            config_for_init = {k: v for k, v in capacity_config.items() if k != 'tmodes'}
        else:
            config_for_init = capacity_config or {}
        
        # Default capacity configuration
        self.config = {
            # Architecture
            'd_model': 512,
            'num_layers_enc': 6,
            'num_layers_dec': 6,
            'num_heads': 8,
            'd_ff_multiplier': 4,
            'use_bias': True,
            
            # Attention
            'attention_dropout': 0.1,
            'use_flash_attention': False,
            'use_causal_mask': True,
            'attention_window': None,
            
            # Regularization
            'dropout': 0.1,
            'layer_dropout': 0.0,
            'residual_dropout': 0.1,
            
            # Normalization
            'norm_type': 'layernorm',
            'norm_eps': 1e-6,
            'pre_norm': True,
            
            # Embedding
            'embedding_dim': 512,  # Default to d_model
            'pos_encoding': 'learned',
            'max_seq_len': 1024,
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
            
            # 2D Spatial Specific
            'input_format': 'sequence',  # Default to sequence
            'patch_size': 1,
            'use_2d_pos_encoding': True,
            'height': 32,  # Default values
            'width': 32,   # Default values
            
            # TModes will be applied after initialization
            'tmodes_applied': False,
        }
        
        # Update with user config (without tmodes)
        if config_for_init:
            self.config.update(config_for_init)
        
        # Check if FlashAttention is available
        self.flash_attention_available = FLASH_ATTENTION_AVAILABLE
        
        # Apply TModes if provided
        if self.tmodes:
            self._apply_tmodes_config()
        
        # Set derived values
        self.vocab_size = vocab_size
        
        # Ensure embedding_dim is set properly
        if self.config['embedding_dim'] is None:
            self.config['embedding_dim'] = self.config['d_model']
        
        # Debug: Print final config
        print(f"Final model config:")
        print(f"  d_model: {self.config['d_model']}")
        print(f"  embedding_dim: {self.config['embedding_dim']}")
        print(f"  pos_encoding: {self.config['pos_encoding']}")
        print(f"  use_flash_attention: {self.config['use_flash_attention']}")
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_encoder()
        self._initialize_decoder()
        self._initialize_output_layer()
        
        # Initialize weights
        self._initialize_weights()
        
    def _apply_tmodes_config(self):
        """Apply TModes configuration to the transformer config"""
        if not self.tmodes:
            return
        
        print(f"Applying TModes:")
        print(f"  conn: {self.tmodes.conn.name}")
        print(f"  flash: {self.tmodes.flash.name}")
        print(f"  penc: {self.tmodes.penc.name}")
        print(f"  pshare: {self.tmodes.pshare.name}")
        print(f"  pbias: {self.tmodes.pbias.name}")
        
        # Connection type affects attention pattern
        if self.tmodes.conn == Conn.Global:
            self.config['attention_window'] = None  # Full attention
        elif self.tmodes.conn == Conn.Axial:
            self.config['attention_window'] = 'axial'  # Special axial attention
            if self.config['input_format'] == '2d':
                self.config['pos_encoding'] = 'axial'  # Use axial positional encoding
        elif self.tmodes.conn == Conn.Shift:
            self.config['attention_window'] = 3  # Local window
        
        # Flash attention
        if self.tmodes.flash == Flash.UseFlash:
            if self.flash_attention_available:
                self.config['use_flash_attention'] = True
                print("  FlashAttention: Enabled")
            else:
                print("  FlashAttention: Not available (library not installed)")
                self.config['use_flash_attention'] = False
        elif self.tmodes.flash == Flash.NoFlash:
            self.config['use_flash_attention'] = False
            print("  FlashAttention: Disabled")
        
        # Positional encoding type - penc takes precedence
        if self.tmodes.penc != PEnc.Nothing:
            if self.tmodes.penc == PEnc.Learnable:
                self.config['pos_encoding'] = 'learned'
            elif self.tmodes.penc == PEnc.Sinusoidal:
                self.config['pos_encoding'] = 'sinusoidal'
            elif self.tmodes.penc == PEnc.Linear:
                self.config['pos_encoding'] = 'linear'
            print(f"  Positional encoding (from penc): {self.config['pos_encoding']}")
        
        # Positional encoding sharing - only for 2D with axial
        if self.tmodes.pshare == PShare.Axial and self.config['input_format'] == '2d':
            self.config['pos_encoding'] = 'axial'
            print(f"  Positional encoding (from pshare): {self.config['pos_encoding']}")
        
        # Bias configuration
        self.config['use_bias'] = True  # Always use bias by default
        if self.tmodes.pbias == PBias.Learnable:
            self.config['attention_bias'] = True
            self.config['ffn_bias'] = True
            print("  Learnable bias: Enabled")
        else:
            self.config['attention_bias'] = False
            self.config['ffn_bias'] = False
            print("  Learnable bias: Disabled")
        
        self.config['tmodes_applied'] = True

    def _initialize_embeddings(self):
        """Initialize token and positional embeddings"""
        d_model = self.config['d_model']
        embedding_dim = self.config['embedding_dim']
        
        print(f"Initializing embeddings:")
        print(f"  Token embedding: vocab_size={self.vocab_size} -> embedding_dim={embedding_dim}")
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            self.vocab_size, 
            embedding_dim
        )
        
        # Positional encoding based on config (which may have been modified by TModes)
        pos_encoding_type = self.config['pos_encoding']
        max_seq_len = self.config['max_seq_len']
        
        print(f"  Positional encoding: type={pos_encoding_type}, d_model={d_model}")
        
        # Positional encoding uses d_model, not embedding_dim
        if pos_encoding_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_len=max_seq_len
            )
        elif pos_encoding_type == 'learned':
            # Calculate appropriate max_len based on input format
            if self.config['input_format'] == '2d':
                h = self.config.get('height', 32)
                w = self.config.get('width', 32)
                max_len = h * w
            else:
                max_len = max_seq_len
            self.pos_encoding = LearnedPositionalEncoding(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_len=max_len
            )
        elif pos_encoding_type == 'global_shared':
            self.pos_encoding = GlobalPositionalEncoding(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_len=max_seq_len
            )
        elif pos_encoding_type == 'axial':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            self.pos_encoding = AxialPositionalEncoding(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_h=h,
                max_w=w
            )
        elif pos_encoding_type == 'linear':
            self.pos_encoding = LinearPositionalEncoding(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_len=max_seq_len
            )
        elif pos_encoding_type == '2d_sinusoidal':
            h = self.config.get('height', 32)
            w = self.config.get('width', 32)
            self.pos_encoding = PositionalEncoding2D(
                d_model=d_model,  # Use d_model, not embedding_dim
                max_h=h,
                max_w=w
            )
        elif pos_encoding_type == 'rotary':
            head_dim = d_model // self.config['num_heads']
            self.pos_encoding = RotaryPositionalEncoding(
                dim=head_dim
            )
        else:
            self.pos_encoding = None
            print(f"  No positional encoding")
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(self.config['embedding_dropout'])
        
        # Projection if embedding_dim != d_model
        if embedding_dim != d_model:
            print(f"  Embedding projection: {embedding_dim} -> {d_model}")
            self.embedding_proj = nn.Linear(
                embedding_dim,
                d_model,
                bias=self.config['use_bias'] and self.config.get('ffn_bias', True)
            )
        else:
            print(f"  No embedding projection needed (embedding_dim == d_model)")
            self.embedding_proj = nn.Identity()
    
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
                use_flash_attention=self.config['use_flash_attention']
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
                use_causal_mask=self.config['use_causal_mask']
            )
            for _ in range(self.config['num_layers_dec'])
        ])
        
        # Decoder final norm
        self.decoder_norm = self._create_norm_layer()
    
    def _initialize_output_layer(self):
        """Initialize output projection"""
        self.output_proj = nn.Linear(
            self.config['d_model'],
            self.vocab_size,
            bias=self.config['use_bias'] and self.config.get('ffn_bias', True)
        )
        
        # Optional: tie embeddings
        if self.config['embedding_dim'] == self.config['d_model']:
            self.output_proj.weight = self.token_embedding.weight
    
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
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass with TModes awareness
        """
        # Handle 2D input
        if self.config['input_format'] == '2d':
            # For 2D input, we expect (B, H, W, C) or (B, H, W) for token indices
            if src.dim() == 3:  # (B, H, W) - token indices
                B, H, W = src.shape
                src = src.reshape(B, H*W)
            elif src.dim() == 4:  # (B, H, W, C) - already embedded
                B, H, W, C = src.shape
                src = src.reshape(B, H*W, C)
            
            # Apply shift connection if specified by TModes (only for 2D)
            if self.tmodes and self.tmodes.conn == Conn.Shift and src.dim() == 3:
                src = self._apply_shift_connection(src, H, W)
        else:
            # For sequence input, ensure it's the right shape
            if src.dim() == 2:  # (B, L) - token indices
                # This is correct for sequence input
                pass
            elif src.dim() == 3:  # (B, L, C) - already embedded
                # This is also valid
                pass
        
        # Encoder
        encoder_output = self._encode(src, src_mask)
        
        # Decoder (if target provided)
        if tgt is not None:
            output = self._decode(tgt, encoder_output, tgt_mask, memory_mask)
            return output, encoder_output
        else:
            return encoder_output
    
    def _apply_shift_connection(self, x, H, W):
        """Apply shift operation for shift connections - only for 2D token indices"""
        # x is (B, L) where L = H*W
        B, L = x.shape
        
        if H * W == L:  # Valid 2D shape
            # Reshape to 2D
            x_2d = x.reshape(B, H, W)
            
            # Shift features in spatial dimensions
            x_shifted = torch.zeros_like(x_2d)
            x_shifted[:, :-1, :] = x_2d[:, 1:, :]  # Shift down
            x_shifted[:, :, :-1] = x_2d[:, :, 1:]  # Shift right
            
            return x_shifted.reshape(B, L)
        return x
    
    def _encode(self, src, src_mask=None):
        """Encoder forward pass"""
        x = self._embed_tokens(src)
        
        # Apply axial attention if specified by TModes
        if self.config.get('attention_window') == 'axial':
            x = self._apply_axial_attention(x, src_mask)
        else:
            for layer in self.encoder_layers:
                if self.config.get('use_checkpointing', False) and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, src_mask, use_reentrant=False
                    )
                else:
                    x = layer(x, mask=src_mask)
        
        if self.config['pre_norm']:
            x = self.encoder_norm(x)
        
        return x
    
    def _apply_axial_attention(self, x, mask):
        """Apply axial attention (row-wise then column-wise)"""
        B, L, C = x.shape
        H = int(math.sqrt(L))
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
    
    def _decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decoder forward pass"""
        x = self._embed_tokens(tgt)
        
        for layer in self.decoder_layers:
            if self.config.get('use_checkpointing', False) and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, memory, tgt_mask, memory_mask, use_reentrant=False
                )
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        if self.config['pre_norm']:
            x = self.decoder_norm(x)
        
        output = self.output_proj(x)
        return output
    
    def _embed_tokens(self, tokens):
        """Embed tokens with positional encoding"""
        if tokens.dtype == torch.long:
            # Get token embeddings (always returns float32 from nn.Embedding)
            x = self.token_embedding(tokens)
            
            # Convert to the same dtype as the model parameters
            # This is CRITICAL for half precision models
            model_dtype = next(self.parameters()).dtype
            if x.dtype != model_dtype:
                x = x.to(model_dtype)
        else:
            x = tokens
        
        # Scale embeddings
        if self.config['scale_embeddings']:
            x = x * math.sqrt(self.config['d_model'])
        
        # Project if needed (BEFORE positional encoding!)
        x = self.embedding_proj(x)
        
        # Apply positional encoding (AFTER projection)
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (PositionalEncoding2D, AxialPositionalEncoding)):
                B, L, C = x.shape
                H = int(math.sqrt(L))
                W = L // H
                if H * W == L:
                    x = x.reshape(B, H, W, C)
                    x = self.pos_encoding(x)
                    x = x.reshape(B, L, C)
            else:
                x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
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
            'applied': self.config['tmodes_applied'],
            'flash_attention_actually_available': self.flash_attention_available
        }
    
    def get_config_summary(self):
        """Get configuration summary including TModes effects"""
        summary = {}
        
        # Add basic config
        for key in ['d_model', 'num_layers_enc', 'num_layers_dec', 'num_heads', 
                   'd_ff_multiplier', 'pos_encoding', 'attention_window']:
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
                 layer_dropout=0.0, attention_window=None, use_flash_attention=False):
        super().__init__()
        
        self.layer_dropout = layer_dropout
        
        # Self-attention with TModes-controlled bias
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=False
        )
        
        # FFN with TModes-controlled bias
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias and ffn_bias,
            bottleneck=ffn_bottleneck,
            glu=glu
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
                 use_causal_mask=True):
        super().__init__()
        
        self.layer_dropout = layer_dropout
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=use_causal_mask
        )
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=False
        )
        
        # FFN
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias and ffn_bias,
            bottleneck=ffn_bottleneck,
            glu=glu
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

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1,
                 use_bias=True, bottleneck=None, glu=False):
        super().__init__()
        
        self.glu = glu
        self.bottleneck = bottleneck
        
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

# ==================== Working Testing Code ====================

if __name__ == "__main__":
    print("Testing FlashAttention Integration - WORKING VERSION")
    print("=" * 60)
    print(f"FlashAttention available: {FLASH_ATTENTION_AVAILABLE}")
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test 1: Basic test without FlashAttention in float32
    print("\n1. Basic test in float32 (no FlashAttention):")
    
    basic_config = {
        'd_model': 256,
        'embedding_dim': 256,
        'num_layers_enc': 2,
        'num_layers_dec': 2,
        'num_heads': 8,
        'input_format': 'sequence',
        'max_seq_len': 1024,
        'use_flash_attention': False,
        'tmodes': TModes(
            conn=Conn.Global,
            flash=Flash.NoFlash,
            penc=PEnc.Learnable,
            pshare=PShare.Global,
            pbias=PBias.Nothing
        )
    }
    
    # Create model
    basic_model = CapacityControlTransformer(
        vocab_size=10000,
        capacity_config=basic_config
    ).to(device)
    
    # Test forward pass
    src = torch.randint(0, 10000, (2, 128), device=device)
    tgt = torch.randint(0, 10000, (2, 128), device=device)
    
    with torch.no_grad():
        output, memory = basic_model(src, tgt)
        print(f"  ✓ Basic float32 test passed")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
    
    # Test 2: With FlashAttention if available
    if FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available():
        print("\n2. Testing with FlashAttention (float16):")
        
        flash_config = {
            'd_model': 256,
            'embedding_dim': 256,
            'num_layers_enc': 2,
            'num_layers_dec': 2,
            'num_heads': 8,
            'input_format': 'sequence',
            'max_seq_len': 1024,
            'tmodes': TModes(
                conn=Conn.Global,
                flash=Flash.UseFlash,
                penc=PEnc.Learnable,
                pshare=PShare.Global,
                pbias=PBias.Nothing
            )
        }
        
        # Create model in float32 first
        flash_model = CapacityControlTransformer(
            vocab_size=10000,
            capacity_config=flash_config
        ).to(device)
        
        # Convert to half precision
        flash_model = flash_model.half()
        
        # Create inputs - ensure they're on the right device and dtype
        src_flash = torch.randint(0, 10000, (2, 128), device=device)
        tgt_flash = torch.randint(0, 10000, (2, 128), device=device)
        
        print(f"  Model dtype: {next(flash_model.parameters()).dtype}")
        
        with torch.no_grad():
            # Forward pass with half precision
            output_flash, memory_flash = flash_model(src_flash, tgt_flash)
            print(f"  ✓ FlashAttention test passed")
            print(f"  Output shape: {output_flash.shape}")
            print(f"  Output dtype: {output_flash.dtype}")
    else:
        print("\n2. Skipping FlashAttention test (not available)")
    
    # Test 3: Test different connection types (simpler test)
    print("\n3. Testing Global connection type (basic):")
    
    config = {
        'd_model': 128,
        'embedding_dim': 128,
        'num_layers_enc': 1,
        'num_layers_dec': 1,
        'num_heads': 4,
        'input_format': 'sequence',
        'max_seq_len': 64,
        'use_flash_attention': False,
        'tmodes': TModes(
            conn=Conn.Global,
            flash=Flash.NoFlash,
            penc=PEnc.Learnable,
            pshare=PShare.Global,
            pbias=PBias.Nothing
        )
    }
    
    model = CapacityControlTransformer(
        vocab_size=1000,
        capacity_config=config
    ).to(device)
    
    # Prepare input
    src = torch.randint(0, 1000, (2, 64), device=device)
    tgt = torch.randint(0, 1000, (2, 64), device=device)
    
    with torch.no_grad():
        output, memory = model(src, tgt)
        print(f"    ✓ Global connection test passed")
        print(f"    Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")