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

# ==================== Simplified Weather Forecasting Transformer ====================

class WeatherForecastingTransformer(nn.Module):
    """
    Transformer for weather forecasting with advanced mixed precision support.
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
            'mixed_precision': False,  # Enable for best 16-bit performance
            'precision_dtype': torch.float16,  # Default to fp16
            'gradient_checkpointing': False,
            'stable_norm': True,  # Use stable normalization for mixed precision
        }
        
        # Update with user config
        if config_for_init:
            self.config.update(config_for_init)
        
        # Set derived values
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']
        
        # Initialize components (simplified for brevity)
        self._initialize_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        # Track if we're in mixed precision mode
        self._mixed_precision_enabled = False
        self._target_dtype = None
    
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
        
        # Simple transformer layers (for demonstration)
        self.encoder_layers = nn.ModuleList([
            self._create_transformer_layer()
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
    
    def _create_transformer_layer(self):
        """Create a transformer layer that works with mixed precision"""
        return nn.TransformerEncoderLayer(
            d_model=self.config['d_model'],
            nhead=self.config['num_heads'],
            dim_feedforward=self.config['d_model'] * self.config['d_ff_multiplier'],
            dropout=self.config['dropout'],
            activation=self.config['activation'],
            batch_first=True,
            norm_first=self.config['pre_norm']
        )
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, lead_time=None, time_features=None):
        """Forward pass with mixed precision support"""
        B, H, W, C = x.shape
        
        # Input projection
        if self.patch_embedding is not None:
            x_2d = x.permute(0, 3, 1, 2)
            x_embedded = self.patch_embedding(x_2d)
            H_patch = x_embedded.shape[2]
            W_patch = x_embedded.shape[3]
            x = x_embedded.permute(0, 2, 3, 1)
            x = x.reshape(B, H_patch * W_patch, -1)
        else:
            x = x.reshape(B, H * W, C)
            x = self.input_proj(x)
        
        # Positional encoding
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, (PositionalEncoding2D, LearnedPositionalEncoding2D)):
                L = x.shape[1]
                if H * W == L:  # Original spatial dimensions
                    x = x.reshape(B, H, W, -1)
                    x = self.pos_encoding(x)
                    x = x.reshape(B, L, -1)
            else:
                x = self.pos_encoding(x)
        
        # Scale embeddings
        if self.config['scale_embeddings']:
            x = x * math.sqrt(self.config['d_model'])
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        # Apply norm if pre_norm
        if self.config['pre_norm']:
            x = self.norm(x)
        
        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Final norm if not pre_norm
        if not self.config['pre_norm']:
            x = self.norm(x)
        
        # Output projection
        if self.patch_unembedding is not None:
            x = x.reshape(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)
            x = self.patch_unembedding(x)
            x = x.permute(0, 2, 3, 1)
        else:
            x = self.output_proj(x)
            x = x.reshape(B, H, W, -1)
        
        return x
    
    def enable_mixed_precision(self, dtype=torch.float16):
        """Enable mixed precision training"""
        self.config['mixed_precision'] = True
        self.config['precision_dtype'] = dtype
        self._mixed_precision_enabled = True
        self._target_dtype = dtype
        
        # Create precision manager
        self.precision_manager = PrecisionManager(
            enabled=True,
            dtype=dtype
        )
        
        print(f"Mixed precision enabled with dtype: {dtype}")
    
    def disable_mixed_precision(self):
        """Disable mixed precision training"""
        self.config['mixed_precision'] = False
        self._mixed_precision_enabled = False
        self._target_dtype = None
        if hasattr(self, 'precision_manager'):
            delattr(self, 'precision_manager')
        print("Mixed precision disabled")
    
    def mixed_precision_forward(self, x, lead_time=None, time_features=None):
        """Forward pass specifically for mixed precision without autocast"""
        # This is a simplified forward that doesn't rely on autocast
        # For production, you should use autocast as shown in training_step
        
        # Cast model parameters to target dtype if needed
        if self._mixed_precision_enabled and self._target_dtype is not None:
            # We'll let autocast handle the dtype conversions in the actual forward pass
            # For this method, we just use the regular forward
            return self.forward(x, lead_time, time_features)
        else:
            return self.forward(x, lead_time, time_features)
    
    def training_step(self, batch, optimizer, criterion):
        """Perform a training step with mixed precision support"""
        if not self.config.get('mixed_precision', False) or not torch.cuda.is_available():
            # Standard training step
            optimizer.zero_grad()
            output = self(batch['input'])
            loss = criterion(output, batch['target'])
            loss.backward()
            optimizer.step()
            return loss
        
        # Mixed precision training step
        optimizer.zero_grad()
        
        # Forward pass with autocast (using new API)
        with torch.amp.autocast('cuda', enabled=True, dtype=self.config['precision_dtype']):
            output = self(batch['input'])
            loss = criterion(output, batch['target'])
        
        # Scale loss
        scaled_loss = self.precision_manager.scale_loss(loss)
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients and check for inf/NaN
        if self.precision_manager.unscale_gradients(optimizer):
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
        
        # Update loss scale
        self.precision_manager.update_scale()
        
        return loss
    
    def get_mixed_precision_info(self):
        """Get information about mixed precision configuration"""
        if not self.config.get('mixed_precision', False):
            return "Mixed precision not enabled"
        
        info = {
            'enabled': True,
            'dtype': str(self.config['precision_dtype']),
            'stable_norm': self.config.get('stable_norm', True),
        }
        
        if hasattr(self, 'precision_manager'):
            info['loss_scale'] = self.precision_manager.scale.item()
            info['steps_since_rescale'] = self.precision_manager._steps_since_rescale
        
        return info

# ==================== Testing ====================

if __name__ == "__main__":
    print("Testing Weather Forecasting Transformer with Advanced Precision Handling")
    print(f"FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test 1: Basic model
    print("\n1. Testing basic model:")
    model = WeatherForecastingTransformer(
        input_dim=10,
        output_dim=10,
        capacity_config={
            'd_model': 128,
            'num_layers_enc': 4,
            'num_layers_dec': 0,
            'num_heads': 8,
            'height': 32,
            'width': 32,
        }
    ).to(device)
    
    test_input = torch.randn(2, 32, 32, 10, device=device)
    output = model(test_input)
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    
    # Test 2: Enable mixed precision
    if device.type == 'cuda':
        print("\n2. Testing mixed precision with autocast:")
        model_mp = WeatherForecastingTransformer(
            input_dim=10,
            output_dim=10,
            capacity_config={
                'd_model': 128,
                'num_layers_enc': 4,
                'num_layers_dec': 0,
                'num_heads': 8,
                'height': 32,
                'width': 32,
                'mixed_precision': True,
                'precision_dtype': torch.float16,
                'stable_norm': True,
            }
        ).to(device)
        
        model_mp.enable_mixed_precision(dtype=torch.float16)
        
        # Test forward pass WITH autocast
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            output_mp = model_mp(test_input)
        
        print(f"  Output shape: {output_mp.shape}")
        print(f"  Output dtype: {output_mp.dtype}")
        
        # Test 3: Test with BF16 if supported
        if torch.cuda.is_bf16_supported():
            print("\n3. Testing BF16 precision with autocast:")
            model_bf16 = WeatherForecastingTransformer(
                input_dim=10,
                output_dim=10,
                capacity_config={
                    'd_model': 128,
                    'num_layers_enc': 4,
                    'num_layers_dec': 0,
                    'num_heads': 8,
                    'height': 32,
                    'width': 32,
                    'mixed_precision': True,
                    'precision_dtype': torch.bfloat16,
                    'stable_norm': True,
                }
            ).to(device)
            
            model_bf16.enable_mixed_precision(dtype=torch.bfloat16)
            
            test_input_bf16 = torch.randn(2, 32, 32, 10, dtype=torch.float32, device=device)
            
            # Use autocast to handle dtype conversions
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                output_bf16 = model_bf16(test_input_bf16)
            
            print(f"  Output shape: {output_bf16.shape}")
            print(f"  Output dtype: {output_bf16.dtype}")
            print(f"  Memory usage: {output_bf16.element_size() * output_bf16.nelement() / 1024 / 1024:.2f} MB")
            
            # Test 4: Test manual dtype conversion (without autocast)
            print("\n4. Testing manual BF16 conversion:")
            # Cast model parameters to bfloat16
            model_bf16_manual = WeatherForecastingTransformer(
                input_dim=10,
                output_dim=10,
                capacity_config={
                    'd_model': 128,
                    'num_layers_enc': 4,
                    'num_layers_dec': 0,
                    'num_heads': 8,
                    'height': 32,
                    'width': 32,
                }
            ).to(device)
            
            # Manually cast the entire model to bfloat16
            model_bf16_manual = model_bf16_manual.to(torch.bfloat16)
            
            test_input_manual = torch.randn(2, 32, 32, 10, dtype=torch.bfloat16, device=device)
            
            with torch.no_grad():
                output_manual = model_bf16_manual(test_input_manual)
            
            print(f"  Output shape: {output_manual.shape}")
            print(f"  Output dtype: {output_manual.dtype}")
            print(f"  Model dtype: {next(model_bf16_manual.parameters()).dtype}")
    
    # Test 5: Precision comparison
    print("\n5. Precision comparison with autocast:")
    
    if device.type == 'cuda':
        test_cases = [
            ("FP32", torch.float32, False),
            ("FP16 with autocast", torch.float16, True),
        ]
        
        if torch.cuda.is_bf16_supported():
            test_cases.append(("BF16 with autocast", torch.bfloat16, True))
        
        for name, dtype, use_mp in test_cases:
            print(f"\n  Testing {name}:")
            
            model_test = WeatherForecastingTransformer(
                input_dim=10,
                output_dim=10,
                capacity_config={
                    'd_model': 128,
                    'num_layers_enc': 4,
                    'num_layers_dec': 0,
                    'num_heads': 8,
                    'height': 32,
                    'width': 32,
                    'mixed_precision': use_mp,
                    'precision_dtype': dtype,
                    'stable_norm': True,
                }
            ).to(device)
            
            if use_mp:
                model_test.enable_mixed_precision(dtype=dtype)
            
            test_input_dtype = torch.randn(2, 32, 32, 10, dtype=torch.float32, device=device)
            
            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            
            # Use autocast for mixed precision, no autocast for FP32
            if use_mp:
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=dtype):
                    output_test = model_test(test_input_dtype)
            else:
                with torch.no_grad():
                    output_test = model_test(test_input_dtype)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            print(f"    Output dtype: {output_test.dtype}")
            print(f"    Peak memory: {memory_used:.2f} MB")
            
            # Check for numerical issues
            has_nan = torch.isnan(output_test).any().item()
            has_inf = torch.isinf(output_test).any().item()
            print(f"    Has NaN: {has_nan}")
            print(f"    Has Inf: {has_inf}")
            
            if has_nan or has_inf:
                print(f"    ⚠️  Numerical instability detected in {name}")
    
    print("\n" + "=" * 60)
    print("Advanced precision testing completed!")
    print("=" * 60)