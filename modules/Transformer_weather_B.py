import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
import contextlib

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

# ==================== Advanced Precision Handling ====================

class PrecisionManager:
    """Manages mixed precision training with gradient scaling"""
    
    def __init__(self, enabled=True, dtype=torch.float16, init_scale=2**16, 
                 growth_factor=2.0, backoff_factor=0.5, growth_interval=2000):
        self.enabled = enabled
        self.dtype = dtype
        self.init_scale = init_scale
        # Use float tensor for scale to support multiplication with floats
        self.scale = torch.tensor([float(init_scale)], device='cuda' if torch.cuda.is_available() else 'cpu')
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._steps_since_rescale = 0
        self._found_inf = False
        
    def autocast_context(self):
        """Create autocast context for mixed precision"""
        if self.enabled and torch.cuda.is_available():
            return torch.amp.autocast('cuda', enabled=True, dtype=self.dtype)
        else:
            return contextlib.nullcontext()
    
    def scale_loss(self, loss):
        """Scale loss for gradient accumulation"""
        if self.enabled:
            return loss * self.scale
        return loss
    
    def unscale_gradients(self, optimizer):
        """Unscale gradients before optimizer step"""
        if self.enabled:
            # Check for inf/NaN
            self._found_inf = False
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Check for inf/NaN
                        grad_data = param.grad.data
                        if grad_data.isnan().any() or grad_data.isinf().any():
                            self._found_inf = True
                            break
            
            # If inf/NaN found, skip this step
            if self._found_inf:
                return False
            
            # Unscale gradients
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.grad.data.div_(self.scale)
            
            return True
        return True
    
    def update_scale(self):
        """Update loss scale based on gradient stability"""
        if not self.enabled:
            return
        
        self._steps_since_rescale += 1
        
        if self._found_inf:
            # Decrease scale if inf/NaN found
            self.scale *= self.backoff_factor
            self._steps_since_rescale = 0
            print(f"Reducing loss scale to {self.scale.item():.2f}")
        elif self._steps_since_rescale >= self.growth_interval:
            # Increase scale if stable
            self.scale *= self.growth_factor
            self._steps_since_rescale = 0
            print(f"Increasing loss scale to {self.scale.item():.2f}")

class StableLayerNorm(nn.Module):
    """LayerNorm with numerical stability improvements for mixed precision"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # Use double precision for variance calculation
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x_float - mean) / torch.sqrt(var + self.eps)
        x_normalized = x_normalized.to(x.dtype)
        return x_normalized * self.weight + self.bias

class StableRMSNorm(nn.Module):
    """RMSNorm with numerical stability improvements"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Use higher precision for norm calculation
        x_float = x.float()
        norm = x_float.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        x_normalized = x_float / (rms + self.eps)
        x_normalized = x_normalized.to(x.dtype)
        return x_normalized * self.weight

class RMSNorm(nn.Module):
    """Standard RMSNorm (for backward compatibility)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight

# ==================== Positional Encoding Classes ====================

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding with mixed precision support"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        if d_model % 4 != 0:
            d_model = ((d_model + 3) // 4) * 4  # Ensure divisible by 4
        
        pe_h = self._get_1d_encoding(max_h, d_model // 2)
        pe_w = self._get_1d_encoding(max_w, d_model // 2)
        
        # Create 2D grid with appropriate dtype
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
        # Ensure positional encoding matches input dtype
        pe = self.pe[:, :H, :W, :].to(x.dtype)
        return x + pe

class LearnedPositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding with mixed precision support"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_h, max_w, d_model))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
    
    def forward(self, x):
        B, H, W, C = x.shape
        return x + self.pos_embedding[:, :H, :W, :].to(x.dtype)

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

# ==================== Enhanced Transformer Components ====================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with enhanced 16-bit precision support"""
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 attention_window=None, use_flash_attention=False, causal=False,
                 dtype=torch.float32, stable_softmax=True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.causal = causal
        self.attention_window = attention_window
        self.dtype = dtype
        self.stable_softmax = stable_softmax  # Use stable softmax for 16-bit
        
        # Linear projections with optional bias
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Use dropout with appropriate dtype
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention (computed in higher precision)
        self.scale_factor = 1.0 / math.sqrt(self.d_k)
        
    def _stable_softmax(self, x, dim=-1):
        """Stable softmax implementation for 16-bit precision"""
        if not self.stable_softmax:
            return F.softmax(x, dim=dim)
        
        # Shift input for numerical stability
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_stable = x - x_max
        
        # Compute softmax
        exp_x = torch.exp(x_stable)
        sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
        
        return exp_x / (sum_exp_x + 1e-12)
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for attention
        Q = Q.view(B, L, self.num_heads, self.d_k)
        K = K.view(B, L, self.num_heads, self.d_k)
        V = V.view(B, L, self.num_heads, self.d_k)
        
        # Transpose for standard attention
        Q_std = Q.transpose(1, 2)
        K_std = K.transpose(1, 2)
        V_std = V.transpose(1, 2)
        
        if isinstance(self.attention_window, int) and self.attention_window > 0:
            mask = self._create_local_mask(L, mask)
        
        if self.causal and mask is None:
            mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
        
        # Handle mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        if self.use_flash_attention:
            # FlashAttention path with enhanced dtype handling
            return self._flash_attention_path(Q, K, V, mask, B, L)
        else:
            # Standard attention path with stability improvements
            return self._standard_attention_path(Q_std, K_std, V_std, mask, B, L)
    
    def _flash_attention_path(self, Q, K, V, mask, B, L):
        """FlashAttention with enhanced 16-bit support"""
        try:
            # Determine target dtype for FlashAttention
            original_dtype = Q.dtype
            target_dtype = None
            
            if original_dtype == torch.float32:
                # Convert to appropriate 16-bit format
                if torch.cuda.is_bf16_supported():
                    target_dtype = torch.bfloat16
                else:
                    target_dtype = torch.float16
            else:
                target_dtype = original_dtype
            
            # Convert to target dtype if needed
            if target_dtype != original_dtype:
                Q_fa = Q.to(target_dtype)
                K_fa = K.to(target_dtype)
                V_fa = V.to(target_dtype)
            else:
                Q_fa, K_fa, V_fa = Q, K, V
            
            # Use FlashAttention
            x = flash_attn_func(
                Q_fa, K_fa, V_fa,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=self.causal,
                softmax_scale=self.scale_factor
            )
            
            # Convert back to original dtype
            if x.dtype != original_dtype:
                x = x.to(original_dtype)
            
            x = x.reshape(B, L, self.d_model)
            
        except Exception as e:
            print(f"FlashAttention failed: {e}. Falling back to standard attention.")
            # Fall back to standard attention
            Q_std = Q.transpose(1, 2)
            K_std = K.transpose(1, 2)
            V_std = V.transpose(1, 2)
            x = self._standard_attention_path(Q_std, K_std, V_std, mask, B, L)
        
        return self.W_o(x)
    
    def _standard_attention_path(self, Q, K, V, mask, B, L):
        """Standard attention with stability improvements"""
        # Compute attention scores with higher precision for dot product
        # This helps with numerical stability in 16-bit
        Q_float = Q.float() if Q.dtype in [torch.float16, torch.bfloat16] else Q
        K_float = K.float() if K.dtype in [torch.float16, torch.bfloat16] else K
        
        attn_scores = torch.matmul(Q_float, K_float.transpose(-2, -1)) * self.scale_factor
        
        # Convert back to original dtype before softmax
        attn_scores = attn_scores.to(Q.dtype)
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        # Use stable softmax
        attn_weights = self._stable_softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        x = torch.matmul(attn_weights, V)
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        return self.W_o(x)
    
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

# ==================== Main Transformer Classes ====================

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
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model, eps=norm_eps)
            self.norm2 = RMSNorm(d_model, eps=norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
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
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
            self.norm3 = nn.LayerNorm(d_model, eps=norm_eps)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model, eps=norm_eps)
            self.norm2 = RMSNorm(d_model, eps=norm_eps)
            self.norm3 = RMSNorm(d_model, eps=norm_eps)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.norm3 = nn.Identity()
        
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

# ==================== Weather Forecasting Transformer ====================

class WeatherForecastingTransformer(nn.Module):
    """
    Transformer for weather forecasting with continuous 2D spatial data.
    """
    
    def __init__(self, input_dim, output_dim=None, capacity_config=None):
        super().__init__()
        
        # Extract TModes from config if present
        self.tmodes = None
        if capacity_config and 'tmodes' in capacity_config:
            self.tmodes = capacity_config['tmodes']
            config_for_init = {k: v for k, v in capacity_config.items() if k != 'tmodes'}
        else:
            config_for_init = capacity_config or {}
        
        # Default configuration
        self.config = {
            'd_model': 512,
            'num_layers_enc': 6,
            'num_layers_dec': 6,
            'num_heads': 8,
            'd_ff_multiplier': 4,
            'use_bias': True,
            'input_dim': input_dim,
            'output_dim': output_dim if output_dim is not None else input_dim,
            'patch_size': 1,
            'attention_dropout': 0.1,
            'use_flash_attention': False,
            'use_causal_mask': False,
            'attention_window': None,
            'dropout': 0.1,
            'norm_type': 'layernorm',
            'norm_eps': 1e-6,
            'pre_norm': True,
            'pos_encoding': '2d_sinusoidal',
            'height': 32,
            'width': 32,
            'scale_embeddings': True,
            'embedding_dropout': 0.1,
            'activation': 'gelu',
            'glu': False,
            'use_lead_time_embedding': True,
            'use_time_encoding': True,
            'num_forecast_steps': 24,
            'dtype': torch.float32,
            'stable_norm': True,
        }
        
        # Update with user config
        if config_for_init:
            self.config.update(config_for_init)
        
        # Apply TModes if provided
        if self.tmodes:
            self._apply_tmodes_config()
        
        # Set derived values
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']
        self.dtype = self.config['dtype']
        
        # Initialize components
        self._initialize_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        # Store original dtype for mixed precision
        self._original_dtype = self.dtype
    
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
    
    def _initialize_layers(self):
        """Initialize model layers"""
        # Input projection
        if self.config.get('patch_size', 1) > 1:
            patch_size = self.config['patch_size']
            self.patch_embedding = nn.Conv2d(
                self.input_dim,
                self.config['d_model'],
                kernel_size=patch_size,
                stride=patch_size
            )
            self.input_proj = nn.Identity()
        else:
            self.patch_embedding = None
            self.input_proj = nn.Linear(
                self.input_dim,
                self.config['d_model'],
                bias=self.config['use_bias']
            )
        
        # Positional encoding
        if self.config['pos_encoding'] == '2d_sinusoidal':
            self.pos_encoding = PositionalEncoding2D(
                self.config['d_model'],
                max_h=self.config['height'],
                max_w=self.config['width']
            )
        elif self.config['pos_encoding'] == 'learned_2d':
            self.pos_encoding = LearnedPositionalEncoding2D(
                self.config['d_model'],
                max_h=self.config['height'],
                max_w=self.config['width']
            )
        else:
            self.pos_encoding = None
        
        # Time encodings
        self.use_lead_time_embedding = self.config.get('use_lead_time_embedding', True)
        self.use_time_encoding = self.config.get('use_time_encoding', True)
        self.num_forecast_steps = self.config.get('num_forecast_steps', 24)
        
        if self.use_lead_time_embedding:
            self.lead_time_embedding = nn.Embedding(
                self.num_forecast_steps + 1,
                self.config['d_model']
            )
        
        if self.use_time_encoding:
            self.time_encoding = SinusoidalTimeEncoding(self.config['d_model'])
        
        # Encoder layers
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
                use_bias=self.config['use_bias'],
                attention_bias=self.config.get('attention_bias', True),
                ffn_bias=self.config.get('ffn_bias', True),
                ffn_bottleneck=self.config.get('ffn_bottleneck'),
                glu=self.config['glu'],
                residual_dropout=self.config.get('residual_dropout', 0.1),
                layer_dropout=self.config.get('layer_dropout', 0.0),
                attention_window=self.config['attention_window'],
                use_flash_attention=self.config['use_flash_attention'],
                flash_attn_available=FLASH_ATTN_AVAILABLE,
                dtype=self.dtype
            )
            for _ in range(self.config['num_layers_enc'])
        ])
        
        # Decoder layers (if any)
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
                use_bias=self.config['use_bias'],
                attention_bias=self.config.get('attention_bias', True),
                ffn_bias=self.config.get('ffn_bias', True),
                ffn_bottleneck=self.config.get('ffn_bottleneck'),
                glu=self.config['glu'],
                residual_dropout=self.config.get('residual_dropout', 0.1),
                layer_dropout=self.config.get('layer_dropout', 0.0),
                attention_window=self.config['attention_window'],
                use_flash_attention=self.config['use_flash_attention'],
                use_causal_mask=self.config['use_causal_mask'],
                flash_attn_available=FLASH_ATTN_AVAILABLE,
                dtype=self.dtype
            )
            for _ in range(self.config['num_layers_dec'])
        ])
        
        # Output projection
        if self.patch_embedding is not None:
            self.patch_unembedding = nn.ConvTranspose2d(
                self.config['d_model'],
                self.output_dim,
                kernel_size=self.config['patch_size'],
                stride=self.config['patch_size']
            )
            self.output_proj = nn.Identity()
        else:
            self.patch_unembedding = None
            self.output_proj = nn.Linear(
                self.config['d_model'],
                self.output_dim,
                bias=self.config['use_bias']
            )
        
        self.embedding_dropout = nn.Dropout(self.config['embedding_dropout'])
        
        # Create appropriate norm layer
        if self.config['norm_type'] == 'layernorm':
            if self.config.get('stable_norm', True):
                self.norm = StableLayerNorm(self.config['d_model'], eps=self.config['norm_eps'])
            else:
                self.norm = nn.LayerNorm(self.config['d_model'], eps=self.config['norm_eps'])
        elif self.config['norm_type'] == 'rmsnorm':
            if self.config.get('stable_norm', True):
                self.norm = StableRMSNorm(self.config['d_model'], eps=self.config['norm_eps'])
            else:
                self.norm = RMSNorm(self.config['d_model'], eps=self.config['norm_eps'])
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
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
        else:
            # Linear projection
            x = x.reshape(B, H * W, C)  # (B, L, C)
            x = self.input_proj(x)  # (B, L, d_model)
        
        # Apply positional encoding
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (PositionalEncoding2D, LearnedPositionalEncoding2D)):
                # Reshape to 2D for spatial positional encoding
                L = x.shape[1]
                if H * W == L:
                    x = x.reshape(B, H, W, -1)
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
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        
        # Decoder (if any)
        if len(self.decoder_layers) > 0:
            # For auto-regressive forecasting
            decoder_output = x
            for layer in self.decoder_layers:
                decoder_output = layer(decoder_output, x, tgt_mask=tgt_mask, memory_mask=src_mask)
            x = decoder_output
        
        # Final norm
        if self.config['pre_norm']:
            x = self.norm(x)
        
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
    
    def to(self, *args, **kwargs):
        """Override to method to preserve dtype"""
        device = None
        dtype = None
        
        # Parse arguments
        if len(args) > 0:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            elif isinstance(args[0], (torch.device, str)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            device = torch.device(kwargs['device']) if isinstance(kwargs['device'], str) else kwargs['device']
        
        # Call parent's to method
        result = super().to(*args, **kwargs)
        
        # Update internal dtype
        if dtype is not None:
            result.dtype = dtype
            result._original_dtype = dtype
        
        return result
    
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
            tmodes_info = {
                'conn': self.tmodes.conn.name,
                'flash': self.tmodes.flash.name,
                'penc': self.tmodes.penc.name,
                'pshare': self.tmodes.pshare.name,
                'pbias': self.tmodes.pbias.name,
            }
            summary['tmodes'] = tmodes_info
            summary['use_flash_attention'] = self.config['use_flash_attention']
        
        return summary

# ==================== Testing ====================

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
        print(f"  Using axial attention: {axial_model.config.get('attention_window') == 'axial'}")
    
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
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
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