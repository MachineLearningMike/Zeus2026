import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CapacityControlTransformer(nn.Module):
    """
    Transformer with detailed capacity control knobs.
    
    CAPACITY KNOBS:
    ==============
    
    1. ARCHITECTURE KNOBS:
    ----------------------
    - d_model: Base embedding dimension (512-2048)
    - num_layers_enc: Encoder depth (6-24)
    - num_layers_dec: Decoder depth (6-24)
    - num_heads: Attention heads (8-16)
    - d_ff_multiplier: FFN dimension multiplier (1-8)
    - use_bias: Enable biases in linear layers
    - activation: FFN activation function
    
    2. ATTENTION KNOBS:
    -------------------
    - attention_dropout: Attention dropout rate (0.0-0.3)
    - use_flash_attention: Use FlashAttention if available
    - use_causal_mask: Causal masking in decoder
    - attention_window: Local attention window size
    
    3. REGULARIZATION KNOBS:
    -----------------------
    - dropout: General dropout rate (0.0-0.3)
    - layer_dropout: Layer dropout probability (0.0-0.2)
    - attention_dropout: Attention dropout (0.0-0.3)
    - weight_decay: L2 regularization
    - label_smoothing: Label smoothing epsilon
    
    4. NORMALIZATION KNOBS:
    ----------------------
    - norm_type: LayerNorm, RMSNorm, or BatchNorm
    - norm_eps: Normalization epsilon
    - pre_norm: Apply norm before or after attention
    
    5. EMBEDDING KNOBS:
    ------------------
    - embedding_dim: Token embedding dimension
    - pos_encoding: Positional encoding type
    - max_seq_len: Maximum sequence length
    - scale_embeddings: Scale embeddings by sqrt(d_model)
    
    6. FEED-FORWARD KNOBS:
    ---------------------
    - d_ff_multiplier: Hidden dimension multiplier
    - activation: Activation function
    - glu: Use GLU variants
    - dropout_ffn: FFN-specific dropout
    
    7. INITIALIZATION KNOBS:
    -----------------------
    - init_range: Weight initialization range
    - init_method: xavier, normal, or kaiming
    - bias_init: Bias initialization
    
    USAGE:
    ------
    model = CapacityControlTransformer(
        vocab_size=32000,
        capacity_config={
            'd_model': 768,
            'num_layers_enc': 12,
            'num_layers_dec': 12,
            'num_heads': 12,
            'd_ff_multiplier': 4,
            'use_flash_attention': True,
            ...
        }
    )
    """
    
    def __init__(self, vocab_size, capacity_config=None):
        super().__init__()
        
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
            'attention_dropout': 0.1,
            'residual_dropout': 0.1,
            
            # Normalization
            'norm_type': 'layernorm',  # 'layernorm', 'rmsnorm', 'batchnorm'
            'norm_eps': 1e-6,
            'pre_norm': True,  # Pre-norm or post-norm
            
            # Embedding
            'embedding_dim': None,  # Defaults to d_model if None
            'pos_encoding': 'learned',  # 'sinusoidal', 'learned', 'relative', 'rotary'
            'max_seq_len': 512,
            'scale_embeddings': True,
            'embedding_dropout': 0.1,
            
            # Feed-Forward
            'activation': 'gelu',  # 'relu', 'gelu', 'silu', 'swish'
            'glu': False,  # Use Gated Linear Units
            'dropout_ffn': 0.1,
            'ffn_bottleneck': None,  # Bottleneck dimension if not None
            
            # Initialization
            'init_range': 0.02,
            'init_method': 'normal',  # 'xavier', 'normal', 'kaiming'
            'bias_init': 'zero',
            
            # Advanced
            'use_checkpointing': False,
            'gradient_checkpointing': False,
            'mixed_precision': False,
            'attention_bias': True,  # Add bias to attention projection
            
            # 2D Spatial Specific
            'input_format': '2d',  # '2d' or 'flattened'
            'patch_size': 1,  # For patch embedding
            'use_2d_pos_encoding': True,
            'height': None,
            'width': None,
        }
        
        # Update with user config
        if capacity_config:
            self.config.update(capacity_config)
        
        # Set derived values
        self.vocab_size = vocab_size
        if self.config['embedding_dim'] is None:
            self.config['embedding_dim'] = self.config['d_model']
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_encoder()
        self._initialize_decoder()
        self._initialize_output_layer()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_embeddings(self):
        """Initialize token and positional embeddings"""
        # Token embedding
        self.token_embedding = nn.Embedding(
            self.vocab_size, 
            self.config['embedding_dim']
        )
        
        # Positional encoding
        if self.config['pos_encoding'] == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(
                self.config['embedding_dim'],
                self.config['max_seq_len']
            )
        elif self.config['pos_encoding'] == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(
                self.config['embedding_dim'],
                self.config['max_seq_len']
            )
        elif self.config['pos_encoding'] == '2d_sinusoidal':
            self.pos_encoding = PositionalEncoding2D(
                self.config['embedding_dim'],
                max_h=self.config.get('height', 32),
                max_w=self.config.get('width', 32)
            )
        elif self.config['pos_encoding'] == 'rotary':
            self.pos_encoding = RotaryPositionalEncoding(
                self.config['embedding_dim'] // self.config['num_heads']
            )
        else:
            self.pos_encoding = None
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(self.config['embedding_dropout'])
        
        # Projection if embedding_dim != d_model
        if self.config['embedding_dim'] != self.config['d_model']:
            self.embedding_proj = nn.Linear(
                self.config['embedding_dim'],
                self.config['d_model'],
                bias=self.config['use_bias']
            )
        else:
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
                use_bias=self.config['use_bias'],
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
                use_bias=self.config['use_bias'],
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
        # Project back to vocabulary size
        self.output_proj = nn.Linear(
            self.config['d_model'],
            self.vocab_size,
            bias=self.config['use_bias']
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
            if param.dim() > 1:  # Weight matrices
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(param)
                elif init_method == 'normal':
                    nn.init.normal_(param, mean=0.0, std=init_range)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    raise ValueError(f"Unknown init method: {init_method}")
            
            elif 'bias' in name:  # Bias terms
                if self.config['bias_init'] == 'zero':
                    nn.init.zeros_(param)
                elif self.config['bias_init'] == 'normal':
                    nn.init.normal_(param, mean=0.0, std=init_range)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass
        
        Args:
            src: Source tokens (B, H, W, dim) or (B, seq_len)
            tgt: Target tokens (B, seq_len) for decoder
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            memory_mask: Cross-attention mask
        
        Returns:
            output: Decoder output
            encoder_output: Encoder output (for attention visualization)
        """
        # Handle 2D input
        if self.config['input_format'] == '2d':
            B, H, W, C = src.shape
            src = src.reshape(B, H*W, C)  # Flatten spatial dimensions
        
        # Encoder
        encoder_output = self._encode(src, src_mask)
        
        # Decoder (if target provided)
        if tgt is not None:
            output = self._decode(tgt, encoder_output, tgt_mask, memory_mask)
            return output, encoder_output
        else:
            return encoder_output
    
    def _encode(self, src, src_mask=None):
        """Encoder forward pass"""
        # Embed source tokens
        x = self._embed_tokens(src)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            if self.config['use_checkpointing'] and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_mask, use_reentrant=False
                )
            else:
                x = layer(x, mask=src_mask)
        
        # Final normalization
        if self.config['pre_norm']:
            x = self.encoder_norm(x)
        
        return x
    
    def _decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Decoder forward pass"""
        # Embed target tokens
        x = self._embed_tokens(tgt)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            if self.config['use_checkpointing'] and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, memory, tgt_mask, memory_mask, use_reentrant=False
                )
            else:
                x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Final normalization
        if self.config['pre_norm']:
            x = self.decoder_norm(x)
        
        # Output projection
        output = self.output_proj(x)
        
        return output
    
    def _embed_tokens(self, tokens):
        """Embed tokens with positional encoding"""
        if tokens.dtype == torch.long:  # Token indices
            x = self.token_embedding(tokens)
        else:  # Already embedded
            x = tokens
        
        # Apply positional encoding
        if self.pos_encoding is not None:
            if isinstance(self.pos_encoding, PositionalEncoding2D):
                # Reshape for 2D positional encoding
                B, L, C = x.shape
                H = int(math.sqrt(L))
                W = L // H
                x = x.reshape(B, H, W, C)
                x = self.pos_encoding(x)
                x = x.reshape(B, L, C)
            else:
                x = self.pos_encoding(x)
        
        # Scale embeddings
        if self.config['scale_embeddings']:
            x = x * math.sqrt(self.config['d_model'])
        
        # Project if needed
        x = self.embedding_proj(x)
        
        # Apply dropout
        x = self.embedding_dropout(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer with capacity control"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, 
                 attention_dropout=0.1, activation='gelu', norm_type='layernorm',
                 norm_eps=1e-6, pre_norm=True, use_bias=True, ffn_bottleneck=None,
                 glu=False, residual_dropout=0.1, layer_dropout=0.0,
                 attention_window=None, use_flash_attention=False):
        super().__init__()
        
        # Layer dropout
        self.layer_dropout = layer_dropout
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=False
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias,
            bottleneck=ffn_bottleneck,
            glu=glu
        )
        
        # Normalization layers
        self.norm1 = self._create_norm_layer(d_model, norm_type, norm_eps)
        self.norm2 = self._create_norm_layer(d_model, norm_type, norm_eps)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        
        # Configuration
        self.pre_norm = pre_norm
        self.d_model = d_model
    
    def _create_norm_layer(self, d_model, norm_type, eps):
        if norm_type == 'layernorm':
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == 'rmsnorm':
            return RMSNorm(d_model, eps=eps)
        else:
            return nn.Identity()
    
    def forward(self, x, mask=None):
        # Apply layer dropout during training
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                return x
        
        # Self-attention with residual
        residual = x
        
        if self.pre_norm:
            x = self.norm1(x)
            x_attn = self.self_attn(x, x, x, mask=mask)
            x = residual + self.dropout1(x_attn)
        else:
            x_attn = self.self_attn(x, x, x, mask=mask)
            x = self.norm1(residual + self.dropout1(x_attn))
        
        # Feed-forward with residual
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
    """Decoder layer with capacity control"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, 
                 attention_dropout=0.1, activation='gelu', norm_type='layernorm',
                 norm_eps=1e-6, pre_norm=True, use_bias=True, ffn_bottleneck=None,
                 glu=False, residual_dropout=0.1, layer_dropout=0.0,
                 attention_window=None, use_flash_attention=False,
                 use_causal_mask=True):
        super().__init__()
        
        # Layer dropout
        self.layer_dropout = layer_dropout
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=use_causal_mask
        )
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_bias=use_bias,
            attention_window=attention_window,
            use_flash_attention=use_flash_attention,
            causal=False
        )
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_bias=use_bias,
            bottleneck=ffn_bottleneck,
            glu=glu
        )
        
        # Normalization layers
        self.norm1 = self._create_norm_layer(d_model, norm_type, norm_eps)
        self.norm2 = self._create_norm_layer(d_model, norm_type, norm_eps)
        self.norm3 = self._create_norm_layer(d_model, norm_type, norm_eps)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(residual_dropout)
        self.dropout2 = nn.Dropout(residual_dropout)
        self.dropout3 = nn.Dropout(residual_dropout)
        
        # Configuration
        self.pre_norm = pre_norm
        self.d_model = d_model
    
    def _create_norm_layer(self, d_model, norm_type, eps):
        if norm_type == 'layernorm':
            return nn.LayerNorm(d_model, eps=eps)
        elif norm_type == 'rmsnorm':
            return RMSNorm(d_model, eps=eps)
        else:
            return nn.Identity()
    
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Apply layer dropout during training
        if self.training and self.layer_dropout > 0:
            if torch.rand(1).item() < self.layer_dropout:
                return x
        
        # Self-attention
        residual = x
        
        if self.pre_norm:
            x = self.norm1(x)
            x_attn = self.self_attn(x, x, x, mask=tgt_mask)
            x = residual + self.dropout1(x_attn)
        else:
            x_attn = self.self_attn(x, x, x, mask=tgt_mask)
            x = self.norm1(residual + self.dropout1(x_attn))
        
        # Cross-attention
        residual = x
        
        if self.pre_norm:
            x = self.norm2(x)
            x_cross = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = residual + self.dropout2(x_cross)
        else:
            x_cross = self.cross_attn(x, memory, memory, mask=memory_mask)
            x = self.norm2(residual + self.dropout2(x_cross))
        
        # Feed-forward
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
    """Multi-head attention with various optimizations"""
    def __init__(self, d_model, num_heads, dropout=0.1, use_bias=True,
                 attention_window=None, use_flash_attention=False, causal=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        self.causal = causal
        self.attention_window = attention_window
        
        # Projection layers
        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Cache for attention patterns (for visualization)
        self.attention_patterns = None
    
    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape
        
        # Linear projections
        Q = self.W_q(query)  # (B, L, d_model)
        K = self.W_k(key)    # (B, L, d_model)
        V = self.W_v(value)  # (B, L, d_model)
        
        # Reshape for multi-head
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention window if specified
        if self.attention_window is not None:
            mask = self._create_local_mask(L, mask)
        
        # Apply causal mask if needed
        if self.causal and mask is None:
            mask = torch.triu(torch.ones(L, L, device=query.device), diagonal=1).bool()
        
        # Use optimized attention if available
        if self.use_flash_attention:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                x = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
            attn_weights = None  # Not computed in flash attention
        else:
            # Standard attention
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            x = torch.matmul(attn_weights, V)
            
            # Store attention patterns for visualization
            if self.training:
                self.attention_patterns = attn_weights.detach()
        
        # Reshape back
        x = x.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        # Output projection
        x = self.W_o(x)
        
        return x
    
    def _create_local_mask(self, seq_len, existing_mask):
        """Create local attention mask"""
        if self.attention_window is None:
            return existing_mask
        
        local_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.attention_window)
            end = min(seq_len, i + self.attention_window + 1)
            local_mask[i, :start] = False
            local_mask[i, end:] = False
        
        if existing_mask is not None:
            local_mask = local_mask & existing_mask
        
        return local_mask


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with various options"""
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1,
                 use_bias=True, bottleneck=None, glu=False):
        super().__init__()
        
        self.glu = glu
        self.bottleneck = bottleneck
        
        if bottleneck is not None:
            # Bottleneck architecture
            self.fc1 = nn.Linear(d_model, bottleneck, bias=use_bias)
            self.fc2 = nn.Linear(bottleneck, d_ff, bias=use_bias)
            self.fc3 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.act1 = self._get_activation(activation)
            self.act2 = self._get_activation(activation)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        elif glu:
            # Gated Linear Unit variant
            self.fc1 = nn.Linear(d_model, d_ff * 2, bias=use_bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.activation = self._get_activation(activation)
            self.dropout = nn.Dropout(dropout)
        else:
            # Standard FFN
            self.fc1 = nn.Linear(d_model, d_ff, bias=use_bias)
            self.fc2 = nn.Linear(d_ff, d_model, bias=use_bias)
            self.activation = self._get_activation(activation)
            self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu' or name == 'swish':
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


# Helper classes
class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps) * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class RotaryPositionalEncoding(nn.Module):
    """Rotary positional encoding (RoPE)"""
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
        
        # Reshape to match input
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        # Apply rotary encoding
        x1, x2 = x[..., :self.dim], x[..., self.dim:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial data"""
    def __init__(self, d_model, max_h=100, max_w=100):
        super().__init__()
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
        return x + self.pe[:, :H, :W, :]


# Example usage with capacity control
if __name__ == "__main__":
    # Small model configuration
    small_config = {
        'd_model': 256,
        'num_layers_enc': 4,
        'num_layers_dec': 4,
        'num_heads': 4,
        'd_ff_multiplier': 2,
        'dropout': 0.1,
        'use_flash_attention': True,
        'activation': 'gelu',
        'norm_type': 'layernorm',
        'pre_norm': True,
        'input_format': '2d',
        'height': 32,
        'width': 32,
        'pos_encoding': '2d_sinusoidal'
    }
    
    # Large model configuration
    large_config = {
        'd_model': 1024,
        'num_layers_enc': 24,
        'num_layers_dec': 24,
        'num_heads': 16,
        'd_ff_multiplier': 4,
        'dropout': 0.2,
        'layer_dropout': 0.1,
        'use_flash_attention': True,
        'activation': 'swish',
        'norm_type': 'rmsnorm',
        'pre_norm': True,
        'ffn_bottleneck': 256,
        'glu': True,
        'attention_window': 512,
        'input_format': '2d',
        'height': 64,
        'width': 64,
        'pos_encoding': 'rotary'
    }
    
    # Create small model
    small_model = CapacityControlTransformer(
        vocab_size=1000,
        capacity_config=small_config
    )
    
    # Create large model
    large_model = CapacityControlTransformer(
        vocab_size=1000,
        capacity_config=large_config
    )
    
    # Test forward pass
    batch_size = 4
    height, width, channels = 32, 32, 256
    
    # 2D input
    input_2d = torch.randn(batch_size, height, width, channels)
    
    # For small model
    with torch.no_grad():
        small_output = small_model(input_2d)
        print(f"Small model output shape: {small_output.shape}")
        print(f"Small model parameters: {sum(p.numel() for p in small_model.parameters()):,}")
    
    # For large model
    with torch.no_grad():
        large_output = large_model(input_2d)
        print(f"Large model output shape: {large_output.shape}")
        print(f"Large model parameters: {sum(p.numel() for p in large_model.parameters()):,}")
    
    # Calculate and print capacity metrics
    def print_capacity_metrics(model, config):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{'='*50}")
        print(f"Model Capacity Metrics:")
        print(f"{'='*50}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"d_model: {config['d_model']}")
        print(f"Encoder layers: {config['num_layers_enc']}")
        print(f"Decoder layers: {config['num_layers_dec']}")
        print(f"Heads: {config['num_heads']}")
        print(f"FFN multiplier: {config['d_ff_multiplier']}")
        print(f"Activation: {config['activation']}")
        print(f"Normalization: {config['norm_type']}")
        print(f"Attention type: {'FlashAttention' if config.get('use_flash_attention', False) else 'Standard'}")
        print(f"Positional encoding: {config['pos_encoding']}")
        
        # Estimate memory usage
        param_memory = total_params * 4 / (1024**3)  # GB for float32
        activation_memory = estimate_activation_memory(model, config)
        print(f"Parameter memory: {param_memory:.2f} GB")
        print(f"Estimated activation memory: {activation_memory:.2f} GB")
    
    def estimate_activation_memory(model, config):
        # Rough estimation of activation memory
        seq_len = config.get('height', 32) * config.get('width', 32)
        batch_size = 4
        
        # Attention activations
        attn_memory = batch_size * seq_len * seq_len * config['num_heads'] * 4 / (1024**3)
        
        # FFN activations
        ffn_memory = batch_size * seq_len * config['d_model'] * config['d_ff_multiplier'] * 4 / (1024**3)
        
        # Total per layer
        per_layer = (attn_memory + ffn_memory) * 2  # Encoder + Decoder
        
        # Multiply by number of layers
        total_layers = config['num_layers_enc'] + config['num_layers_dec']
        return per_layer * total_layers
    
    print_capacity_metrics(small_model, small_config)
    print_capacity_metrics(large_model, large_config)