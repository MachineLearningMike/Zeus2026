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

# ==================== Axial Positional Encoding ====================

class AxialPositionalEncoding2D(nn.Module):
    """Axial positional encoding for 2D data"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
        # Split dimensions for height and width encodings
        self.d_model = d_model
        self.d_h = d_model // 2
        self.d_w = d_model - self.d_h
        
        # Height encoding (applied column-wise)
        self.pe_h = self._get_1d_encoding(max_h, self.d_h)
        # Width encoding (applied row-wise)
        self.pe_w = self._get_1d_encoding(max_w, self.d_w)
        
        # Register buffers
        self.register_buffer('pe_h_buffer', self.pe_h)
        self.register_buffer('pe_w_buffer', self.pe_w)
    
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
        device = x.device
        
        # Create height encodings (applied to each column)
        pe_h = self.pe_h_buffer[:H, :self.d_h].to(x.dtype).to(device)
        pe_h = pe_h.unsqueeze(1).expand(-1, W, -1)  # (H, W, d_h)
        
        # Create width encodings (applied to each row)
        pe_w = self.pe_w_buffer[:W, :self.d_w].to(x.dtype).to(device)
        pe_w = pe_w.unsqueeze(0).expand(H, -1, -1)  # (H, W, d_w)
        
        # Combine height and width encodings
        pe = torch.cat([pe_h, pe_w], dim=-1)  # (H, W, d_model)
        pe = pe.unsqueeze(0)  # (1, H, W, d_model)
        
        return x + pe

# ==================== Enhanced Positional Encoding Classes ====================

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

# ==================== Axial Attention Implementation ====================

class AxialMultiHeadAttention(nn.Module):
    """Axial multi-head attention for 2D data"""
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 dtype=torch.float32, stable_softmax=True):
        super().__init__()
        assert d_model % (num_heads * 2) == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // (num_heads * 2)  # Split for height and width attention
        self.dtype = dtype
        self.stable_softmax = stable_softmax
        
        # Height attention (vertical)
        self.W_q_h = nn.Linear(d_model, d_model // 2, bias=use_bias)
        self.W_k_h = nn.Linear(d_model, d_model // 2, bias=use_bias)
        self.W_v_h = nn.Linear(d_model, d_model // 2, bias=use_bias)
        
        # Width attention (horizontal)
        self.W_q_w = nn.Linear(d_model, d_model // 2, bias=use_bias)
        self.W_k_w = nn.Linear(d_model, d_model // 2, bias=use_bias)
        self.W_v_w = nn.Linear(d_model, d_model // 2, bias=use_bias)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
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
    
    def forward(self, x, mask=None):
        B, H, W, C = x.shape
        
        # Reshape for axial attention
        x_flat = x.view(B, H * W, C)
        
        # Height attention (attend along columns)
        Q_h = self.W_q_h(x_flat).view(B, H, W, self.num_heads, self.d_k)
        K_h = self.W_k_h(x_flat).view(B, H, W, self.num_heads, self.d_k)
        V_h = self.W_v_h(x_flat).view(B, H, W, self.num_heads, self.d_k)
        
        # Width attention (attend along rows)
        Q_w = self.W_q_w(x_flat).view(B, H, W, self.num_heads, self.d_k)
        K_w = self.W_k_w(x_flat).view(B, H, W, self.num_heads, self.d_k)
        V_w = self.W_v_w(x_flat).view(B, H, W, self.num_heads, self.d_k)
        
        # Height attention computation
        Q_h = Q_h.permute(0, 3, 2, 1, 4).contiguous().view(B * self.num_heads * W, H, self.d_k)
        K_h = K_h.permute(0, 3, 2, 1, 4).contiguous().view(B * self.num_heads * W, H, self.d_k)
        V_h = V_h.permute(0, 3, 2, 1, 4).contiguous().view(B * self.num_heads * W, H, self.d_k)
        
        # Width attention computation
        Q_w = Q_w.permute(0, 3, 1, 2, 4).contiguous().view(B * self.num_heads * H, W, self.d_k)
        K_w = K_w.permute(0, 3, 1, 2, 4).contiguous().view(B * self.num_heads * H, W, self.d_k)
        V_w = V_w.permute(0, 3, 1, 2, 4).contiguous().view(B * self.num_heads * H, W, self.d_k)
        
        # Compute attention scores with higher precision for stability
        Q_h_float = Q_h.float() if Q_h.dtype in [torch.float16, torch.bfloat16] else Q_h
        K_h_float = K_h.float() if K_h.dtype in [torch.float16, torch.bfloat16] else K_h
        
        attn_h = torch.bmm(Q_h_float, K_h_float.transpose(1, 2)) * self.scale_factor
        attn_h = attn_h.to(Q_h.dtype)
        
        Q_w_float = Q_w.float() if Q_w.dtype in [torch.float16, torch.bfloat16] else Q_w
        K_w_float = K_w.float() if K_w.dtype in [torch.float16, torch.bfloat16] else K_w
        
        attn_w = torch.bmm(Q_w_float, K_w_float.transpose(1, 2)) * self.scale_factor
        attn_w = attn_w.to(Q_w.dtype)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 4:  # (B, H, W, H, W)
                mask_h = mask.view(B, H, W, H, W)
                mask_h = mask_h.permute(0, 2, 1, 3).contiguous().view(B * W, H, H)
                mask_h = mask_h.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                mask_h = mask_h.reshape(B * self.num_heads * W, H, H)
                
                mask_w = mask.view(B, H, W, H, W)
                mask_w = mask_w.permute(0, 1, 2, 3).contiguous().view(B * H, W, W)
                mask_w = mask_w.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                mask_w = mask_w.reshape(B * self.num_heads * H, W, W)
                
                attn_h = attn_h.masked_fill(mask_h, -1e9)
                attn_w = attn_w.masked_fill(mask_w, -1e9)
        
        # Apply softmax
        attn_weights_h = self._stable_softmax(attn_h, dim=-1)
        attn_weights_w = self._stable_softmax(attn_w, dim=-1)
        
        attn_weights_h = self.dropout(attn_weights_h)
        attn_weights_w = self.dropout(attn_weights_w)
        
        # Apply attention to values
        out_h = torch.bmm(attn_weights_h, V_h)
        out_w = torch.bmm(attn_weights_w, V_w)
        
        # Reshape back
        out_h = out_h.view(B, self.num_heads, W, H, self.d_k)
        out_h = out_h.permute(0, 3, 2, 1, 4).contiguous().view(B, H, W, self.d_model // 2)
        
        out_w = out_w.view(B, self.num_heads, H, W, self.d_k)
        out_w = out_w.permute(0, 2, 1, 3, 4).contiguous().view(B, H, W, self.d_model // 2)
        
        # Combine height and width attention outputs
        out = torch.cat([out_h, out_w], dim=-1)
        out = self.W_o(out.view(B, H * W, -1))
        
        return out.view(B, H, W, -1)

# ==================== Enhanced Transformer Components ====================

class MultiHeadAttention(nn.Module):
    """Multi-head attention with enhanced 16-bit precision support"""
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 attention_window=None, use_flash_attention=False, causal=False,
                 dtype=torch.float32, stable_softmax=True, attention_type='global'):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE
        self.causal = causal
        self.attention_window = attention_window
        self.dtype = dtype
        self.stable_softmax = stable_softmax
        self.attention_type = attention_type
        
        # For axial attention, use specialized module
        if attention_type == 'axial':
            self.axial_attention = AxialMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_bias=use_bias,
                dtype=dtype,
                stable_softmax=stable_softmax
            )
            self.is_axial = True
            return
        
        self.is_axial = False
        
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
        # Handle axial attention separately
        if self.is_axial:
            B, L, C = query.shape
            H = int(math.sqrt(L)) if math.sqrt(L).is_integer() else self.attention_window or 32
            W = L // H
            query_2d = query.view(B, H, W, C)
            return self.axial_attention(query_2d, mask).view(B, L, C)
        
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
                 flash_attn_available=False, dtype=torch.float32, attention_type='global'):
        super().__init__()
        
        self.layer_dropout = layer_dropout
        self.use_flash_attention = use_flash_attention and flash_attn_available
        self.dtype = dtype
        self.attention_type = attention_type
        
        # Self-attention with TModes-controlled bias
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias and attention_bias,
            attention_window=attention_window,
            use_flash_attention=self.use_flash_attention,
            causal=False,
            dtype=dtype,
            stable_softmax=True if dtype in [torch.float16, torch.bfloat16] else False,
            attention_type=attention_type
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
        
        # Normalization - use stable versions for mixed precision
        if norm_type == 'layernorm':
            if dtype in [torch.float16, torch.bfloat16]:
                self.norm1 = StableLayerNorm(d_model, eps=norm_eps)
                self.norm2 = StableLayerNorm(d_model, eps=norm_eps)
            else:
                self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)
                self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)
        elif norm_type == 'rmsnorm':
            if dtype in [torch.float16, torch.bfloat16]:
                self.norm1 = StableRMSNorm(d_model, eps=norm_eps)
                self.norm2 = StableRMSNorm(d_model, eps=norm_eps)
            else:
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

# ==================== Enhanced Weather Forecasting Transformer ====================

class EnhancedWeatherForecastingTransformer(nn.Module):
    """
    Enhanced Transformer for weather forecasting with continuous 2D spatial data.
    Combines Version B's weather-specific features with Version A's advanced mixed precision.
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
        
        # Default configuration with enhanced mixed precision support
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
            'mixed_precision': False,  # Enhanced: Enable for mixed precision
            'precision_dtype': torch.float16,  # Enhanced: Default precision
            'gradient_checkpointing': False,  # Enhanced: For memory efficiency
            'stable_norm': True,  # Enhanced: Use stable normalization
            'use_precision_manager': True,  # Enhanced: Use advanced precision manager
            'gradient_clip_norm': 1.0,  # Enhanced: Gradient clipping
            'attention_type': 'global',  # Enhanced: Add attention type
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
        self.dtype = self.config.get('dtype', torch.float32)
        
        # Mixed precision tracking
        self._mixed_precision_enabled = self.config['mixed_precision']
        self._target_dtype = self.config['precision_dtype'] if self._mixed_precision_enabled else None
        
        # Initialize components
        self._initialize_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize precision manager if needed
        if self.config['use_precision_manager'] and self._mixed_precision_enabled:
            self.precision_manager = PrecisionManager(
                enabled=True,
                dtype=self._target_dtype,
                init_scale=2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
        else:
            self.precision_manager = None
        
        # Store original dtype for mixed precision
        self._original_dtype = self.dtype
    
    def _apply_tmodes_config(self):
        """Apply TModes configuration to the transformer config"""
        if not self.tmodes:
            return
        
        # Connection type affects attention pattern
        if self.tmodes.conn == Conn.Global:
            self.config['attention_window'] = None  # Full attention
            self.config['attention_type'] = 'global'
        elif self.tmodes.conn == Conn.Axial:
            self.config['attention_window'] = None  # Axial attention
            self.config['attention_type'] = 'axial'
            self.config['pos_encoding'] = 'axial'  # Use axial positional encoding
        elif self.tmodes.conn == Conn.Shift:
            self.config['attention_window'] = 3  # Local window
            self.config['attention_type'] = 'global'
        
        # Flash attention
        if self.tmodes.flash == Flash.UseFlash:
            self.config['use_flash_attention'] = True
        elif self.tmodes.flash == Flash.NoFlash:
            self.config['use_flash_attention'] = False
        
        # Positional encoding type
        if self.tmodes.penc == PEnc.Learnable:
            self.config['pos_encoding'] = 'learned_2d'
        elif self.tmodes.penc == PEnc.Sinusoidal:
            if self.tmodes.conn == Conn.Axial:
                self.config['pos_encoding'] = 'axial'
            else:
                self.config['pos_encoding'] = '2d_sinusoidal'
        elif self.tmodes.penc == PEnc.Linear:
            self.config['pos_encoding'] = 'linear'
        
        # Positional encoding sharing
        if self.tmodes.pshare == PShare.Axial and self.tmodes.conn == Conn.Axial:
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
        """Initialize model layers with mixed precision awareness"""
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
        
        # Positional encoding - ensure they match the model dtype
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
        elif self.config['pos_encoding'] == 'axial':
            self.pos_encoding = AxialPositionalEncoding2D(
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
        
        # Encoder layers with appropriate dtype and attention type
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
                dtype=self.dtype,
                attention_type=self.config['attention_type']
            )
            for _ in range(self.config['num_layers_enc'])
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
        
        # Create appropriate norm layer with stability for mixed precision
        if self.config['norm_type'] == 'layernorm':
            if self.config.get('stable_norm', True) or self._mixed_precision_enabled:
                self.norm = StableLayerNorm(self.config['d_model'], eps=self.config['norm_eps'])
            else:
                self.norm = nn.LayerNorm(self.config['d_model'], eps=self.config['norm_eps'])
        elif self.config['norm_type'] == 'rmsnorm':
            if self.config.get('stable_norm', True) or self._mixed_precision_enabled:
                self.norm = StableRMSNorm(self.config['d_model'], eps=self.config['norm_eps'])
            else:
                self.norm = RMSNorm(self.config['d_model'], eps=self.config['norm_eps'])
        
        # Gradient checkpointing if enabled
        if self.config['gradient_checkpointing']:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        for i, layer in enumerate(self.encoder_layers):
            layer.checkpoint = True
    
    def _initialize_weights(self):
        """Initialize weights with appropriate scaling for mixed precision"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Scale initialization for better mixed precision stability
                if self._mixed_precision_enabled and 'weight' in name:
                    # Smaller initialization for mixed precision
                    gain = 0.5 if param.dtype in [torch.float16, torch.bfloat16] else 1.0
                    nn.init.xavier_uniform_(param, gain=gain)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, lead_time=None, time_features=None, src_mask=None):
        """
        Forward pass for weather forecasting with mixed precision awareness.
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, T, H, W, C) for multiple time steps
            lead_time: Lead time indices for forecasting (B,) or (B, T). Values should be in range [0, num_forecast_steps]
            time_features: Additional time features (B, time_dim) or (B, T, time_dim)
            src_mask: Source mask for encoder
            
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
                                                    src_mask)
                outputs.append(output_t)
            return torch.stack(outputs, dim=1)
        else:
            # Single time step
            # Clamp lead time to valid range if provided
            if lead_time is not None:
                lead_time = torch.clamp(lead_time, 0, self.num_forecast_steps)
            return self._forward_single_step(x, lead_time, time_features, src_mask)
    
    def _forward_single_step(self, x, lead_time, time_features, src_mask):
        """Forward pass for a single time step with mixed precision handling"""
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
        
        # Apply positional encoding with dtype matching
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (PositionalEncoding2D, LearnedPositionalEncoding2D, AxialPositionalEncoding2D)):
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
            x = x + lead_time_emb.to(x.dtype)  # Ensure dtype match
        
        if self.use_time_encoding and time_features is not None:
            time_emb = self.time_encoding(time_features)  # (B, d_model) or (B, L, d_model)
            if time_emb.dim() == 2:
                time_emb = time_emb.unsqueeze(1)  # (B, 1, d_model)
            x = x + time_emb.to(x.dtype)  # Ensure dtype match
        
        # Scale embeddings
        if self.config['scale_embeddings']:
            x = x * math.sqrt(self.config['d_model'])
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        # Encoder with gradient checkpointing if enabled
        for layer in self.encoder_layers:
            if self.config['gradient_checkpointing'] and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_mask, 
                    use_reentrant=False
                )
            else:
                x = layer(x, mask=src_mask)
        
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
    
    def training_step(self, batch, optimizer, criterion, clip_grad_norm=True):
        """
        Perform a training step with sophisticated mixed precision support.
        
        Args:
            batch: Dictionary with 'input', 'target', and optionally 'lead_time', 'time_features'
            optimizer: PyTorch optimizer
            criterion: Loss function
            clip_grad_norm: Whether to clip gradients
            
        Returns:
            loss: The computed loss
        """
        # Handle mixed precision
        if not self._mixed_precision_enabled or not torch.cuda.is_available():
            # Standard training step
            optimizer.zero_grad()
            output = self(
                batch['input'],
                lead_time=batch.get('lead_time'),
                time_features=batch.get('time_features')
            )
            loss = criterion(output, batch['target'])
            loss.backward()
            
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 
                    max_norm=self.config['gradient_clip_norm']
                )
            
            optimizer.step()
            return loss
        
        # Mixed precision training step with precision manager
        optimizer.zero_grad()
        
        # Forward pass with autocast
        with torch.amp.autocast('cuda', enabled=True, dtype=self._target_dtype):
            output = self(
                batch['input'],
                lead_time=batch.get('lead_time'),
                time_features=batch.get('time_features')
            )
            loss = criterion(output, batch['target'])
        
        # Scale loss if using precision manager
        if self.precision_manager:
            scaled_loss = self.precision_manager.scale_loss(loss)
        else:
            scaled_loss = loss
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients and check for inf/NaN
        if self.precision_manager and self.precision_manager.unscale_gradients(optimizer):
            # Clip gradients for stability
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 
                    max_norm=self.config['gradient_clip_norm']
                )
            
            # Optimizer step
            optimizer.step()
        elif not self.precision_manager:
            # No precision manager, just clip and step
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 
                    max_norm=self.config['gradient_clip_norm']
                )
            optimizer.step()
        
        # Update loss scale if using precision manager
        if self.precision_manager:
            self.precision_manager.update_scale()
        
        return loss
    
    def enable_mixed_precision(self, dtype=torch.float16):
        """Enable mixed precision training with the specified dtype"""
        self.config['mixed_precision'] = True
        self.config['precision_dtype'] = dtype
        self._mixed_precision_enabled = True
        self._target_dtype = dtype
        
        # Update config for all components
        self.config['stable_norm'] = True  # Always use stable norms for mixed precision
        
        # Create or update precision manager
        if self.config['use_precision_manager']:
            self.precision_manager = PrecisionManager(
                enabled=True,
                dtype=dtype,
                init_scale=2**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000
            )
        
        print(f"Mixed precision enabled with dtype: {dtype}")
        print(f"Using precision manager: {self.config['use_precision_manager']}")
        print(f"Using stable normalization: {self.config['stable_norm']}")
    
    def disable_mixed_precision(self):
        """Disable mixed precision training"""
        self.config['mixed_precision'] = False
        self._mixed_precision_enabled = False
        self._target_dtype = None
        
        if hasattr(self, 'precision_manager'):
            delattr(self, 'precision_manager')
        
        print("Mixed precision disabled")
    
    def get_mixed_precision_info(self):
        """Get information about mixed precision configuration"""
        if not self._mixed_precision_enabled:
            return {
                'enabled': False,
                'dtype': str(self.dtype),
                'message': 'Mixed precision not enabled'
            }
        
        info = {
            'enabled': True,
            'dtype': str(self._target_dtype),
            'stable_norm': self.config.get('stable_norm', True),
            'use_precision_manager': self.config.get('use_precision_manager', False),
            'gradient_clip_norm': self.config.get('gradient_clip_norm', 1.0),
        }
        
        if hasattr(self, 'precision_manager') and self.precision_manager:
            info['loss_scale'] = self.precision_manager.scale.item()
            info['steps_since_rescale'] = self.precision_manager._steps_since_rescale
        
        return info
    
    def to(self, *args, **kwargs):
        """Override to method to preserve dtype and update precision manager"""
        device = None
        dtype = None
        
        # Parse arguments
        if len(args) > 0:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
                # Auto-enable mixed precision if casting to half-precision dtype
                if dtype in [torch.float16, torch.bfloat16] and not self._mixed_precision_enabled:
                    print(f"Auto-enabling mixed precision for dtype: {dtype}")
                    self.enable_mixed_precision(dtype)
            elif isinstance(args[0], (torch.device, str)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
        
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
            # Auto-enable mixed precision if casting to half-precision dtype
            if dtype in [torch.float16, torch.bfloat16] and not self._mixed_precision_enabled:
                print(f"Auto-enabling mixed precision for dtype: {dtype}")
                self.enable_mixed_precision(dtype)
        if 'device' in kwargs:
            device = torch.device(kwargs['device']) if isinstance(kwargs['device'], str) else kwargs['device']
        
        # Call parent's to method
        result = super().to(*args, **kwargs)
        
        # Update internal dtype
        if dtype is not None:
            result.dtype = dtype
            result._original_dtype = dtype
        
        # Update precision manager device if it exists
        if hasattr(result, 'precision_manager') and result.precision_manager and device is not None:
            result.precision_manager.scale = result.precision_manager.scale.to(device)
        
        return result
    
    def get_config_summary(self):
        """Get configuration summary including TModes effects and mixed precision info"""
        summary = {}
        
        # Add basic config
        for key in ['d_model', 'num_layers_enc', 'num_layers_dec', 'num_heads', 
                   'd_ff_multiplier', 'pos_encoding', 'attention_window',
                   'input_dim', 'output_dim', 'height', 'width', 'num_forecast_steps',
                   'dtype', 'mixed_precision', 'precision_dtype', 'gradient_checkpointing',
                   'attention_type']:
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
        
        # Add mixed precision info
        mp_info = self.get_mixed_precision_info()
        summary['mixed_precision_info'] = mp_info
        
        return summary


# ==================== Testing ====================

if __name__ == "__main__":
    print("Testing Enhanced Weather Forecasting Transformer with Axial Attention")
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
    input_dim = 10
    output_dim = 10
    
    # Test 1: Axial attention with mixed precision
    print("\n1. Testing Axial Attention with Mixed Precision:")
    axial_config = {
        'd_model': 128,
        'num_layers_enc': 4,
        'num_layers_dec': 0,
        'num_heads': 8,
        'height': height,
        'width': width,
        'num_forecast_steps': 24,
        'mixed_precision': True,
        'precision_dtype': torch.float16,
        'use_precision_manager': True,
        'attention_type': 'axial',
        'pos_encoding': 'axial',
        'tmodes': TModes(
            conn=Conn.Axial,
            flash=Flash.NoFlash,  # FlashAttention not compatible with axial
            penc=PEnc.Sinusoidal,
            pshare=PShare.Axial,
            pbias=PBias.Learnable
        )
    }
    
    axial_model = EnhancedWeatherForecastingTransformer(
        input_dim=input_dim,
        output_dim=output_dim,
        capacity_config=axial_config
    ).to(device)
    
    # Enable mixed precision
    axial_model.enable_mixed_precision(dtype=torch.float16)
    
    # Test data
    weather_data = torch.randn(batch_size, height, width, input_dim).to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        output = axial_model(weather_data)
    
    print(f"  Input shape: {weather_data.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Using axial attention: {axial_model.config['attention_type'] == 'axial'}")
    print(f"  Using axial positional encoding: {axial_model.config['pos_encoding'] == 'axial'}")
    print(f"  Mixed precision info: {axial_model.get_mixed_precision_info()}")
    
    # Test 2: Performance comparison
    print("\n2. Performance Comparison (32x32 grid):")
    
    test_cases = [
        ("FP32 - Global", {'attention_type': 'global', 'mixed_precision': False, 'use_flash_attention': False}),
        ("FP16 - Global + Flash", {'attention_type': 'global', 'mixed_precision': True, 'use_flash_attention': True}),
        ("FP16 - Axial", {'attention_type': 'axial', 'mixed_precision': True, 'use_flash_attention': False}),
    ]
    
    if torch.cuda.is_bf16_supported():
        test_cases.append(("BF16 - Global + Flash", {'attention_type': 'global', 'mixed_precision': True, 'use_flash_attention': True, 'precision_dtype': torch.bfloat16}))
    
    for name, config in test_cases:
        print(f"\n  Testing {name}:")
        
        perf_config = {
            'd_model': 128,
            'num_layers_enc': 4,
            'num_layers_dec': 0,
            'num_heads': 8,
            'height': height,
            'width': width,
            'num_forecast_steps': 24,
            'gradient_checkpointing': False,
        }
        perf_config.update(config)
        
        if config.get('attention_type') == 'axial':
            perf_config['pos_encoding'] = 'axial'
        
        model = EnhancedWeatherForecastingTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            capacity_config=perf_config
        ).to(device)
        
        if config.get('mixed_precision', False):
            dtype = config.get('precision_dtype', torch.float16)
            model.enable_mixed_precision(dtype=dtype)
        
        test_data = torch.randn(batch_size, height, width, input_dim).to(device)
        
        # Warm-up
        with torch.no_grad():
            if config.get('mixed_precision', False):
                with torch.amp.autocast('cuda', enabled=True, dtype=config.get('precision_dtype', torch.float16)):
                    _ = model(test_data)
            else:
                _ = model(test_data)
        
        # Time forward pass
        import time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            if config.get('mixed_precision', False):
                with torch.amp.autocast('cuda', enabled=True, dtype=config.get('precision_dtype', torch.float16)):
                    output = model(test_data)
            else:
                output = model(test_data)
        
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        # Measure memory
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == 'cuda' else 0
        
        print(f"    Time: {elapsed_time:.4f}s")
        print(f"    Memory: {memory_used:.2f} MB")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Attention type: {config['attention_type']}")
        print(f"    Using FlashAttention: {config.get('use_flash_attention', False)}")
    
    # Test 3: Mixed precision training
    print("\n3. Mixed Precision Training Simulation:")
    
    if device.type == 'cuda':
        train_config = {
            'd_model': 128,
            'num_layers_enc': 4,
            'num_layers_dec': 0,
            'num_heads': 8,
            'height': height,
            'width': width,
            'num_forecast_steps': 24,
            'mixed_precision': True,
            'precision_dtype': torch.float16,
            'use_precision_manager': True,
            'gradient_clip_norm': 1.0,
            'attention_type': 'global',
            'use_flash_attention': True,
        }
        
        train_model = EnhancedWeatherForecastingTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            capacity_config=train_config
        ).to(device)
        
        train_model.enable_mixed_precision(dtype=torch.float16)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Training batch
        batch = {
            'input': torch.randn(batch_size, height, width, input_dim, device=device),
            'target': torch.randn(batch_size, height, width, output_dim, device=device),
            'lead_time': torch.randint(0, 24, (batch_size,), device=device),
            'time_features': torch.randn(batch_size, 3, device=device),
        }
        
        # Training step
        loss = train_model.training_step(batch, optimizer, criterion)
        
        print(f"  Training step completed successfully")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Mixed precision info: {train_model.get_mixed_precision_info()}")
        
    # Test 4: Different grid sizes
    print("\n4. Testing Different Grid Sizes with Axial Attention:")
    
    grid_sizes = [
        ("Small", 16, 16),
        ("Medium", 32, 32),
        ("Large", 64, 64),
    ]
    
    for name, h, w in grid_sizes:
        print(f"\n  {name} grid ({h}x{w}):")
        
        grid_config = {
            'd_model': 128,
            'num_layers_enc': 4,
            'num_layers_dec': 0,
            'num_heads': 8,
            'height': h,
            'width': w,
            'num_forecast_steps': 24,
            'mixed_precision': True,
            'precision_dtype': torch.float16,
            'use_precision_manager': True,
            'attention_type': 'axial',
            'pos_encoding': 'axial',
        }
        
        grid_model = EnhancedWeatherForecastingTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            capacity_config=grid_config
        ).to(device)
        
        grid_model.enable_mixed_precision(dtype=torch.float16)
        
        test_grid = torch.randn(1, h, w, input_dim).to(device)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            output = grid_model(test_grid)
        
        print(f"    Input shape: {test_grid.shape}")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")
        print(f"    Using axial attention: {grid_model.config['attention_type'] == 'axial'}")
    
    print("\n" + "=" * 60)
    print("Enhanced Weather Forecasting Transformer testing completed!")
    print("=" * 60)