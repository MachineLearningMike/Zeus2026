import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
import math

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
    
class Mlp(nn.Module):
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
        x = self.drop(x)
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

# -------------------- 1D Positional Encoding --------------------
class AxialPositionalEncoding1D(nn.Module):
    """
    1D Positional Encoding for axial attention.
    Adds position information to each row/column.
    """
    def __init__(self, max_len, dim, dropout=0.0, pos_type="learnable"):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        
        if pos_type == "learnable":
            # Learnable positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
            nn.init.normal_(self.pe, mean=0.0, std=0.02)
        elif pos_type == "sinusoidal":
            # Sinusoidal positional encoding (like Transformer)
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, dim]
            self.register_buffer('pe', pe)
        elif pos_type == "linear":
            # Linear positional encoding (simple)
            pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) / max_len
            pe = pe.expand(max_len, dim)
            pe = pe.unsqueeze(0)  # [1, max_len, dim]
            self.register_buffer('pe', pe)
        else:
            self.register_buffer('pe', None)
    
    def forward(self, x, positions=None):
        """
        x: [batch, seq_len, dim]
        positions: optional positions to use (for variable length)
        """
        seq_len = x.size(1)
        
        if self.pe is not None:
            if positions is None:
                # Use first seq_len positions
                pe = self.pe[:, :seq_len, :]
            else:
                # Use specified positions
                pe = self.pe[:, positions, :]
            
            # Add positional encoding (cast to x's dtype)
            x = x + pe.to(x.device).to(x.dtype)
        
        x = self.dropout(x)
        return x

# -------------------- Axial FlashAttention --------------------
class AxialFlashAttention(nn.Module):
    def __init__(self, dim, H, W, n_heads=8, eps=1e-5, 
                use_pos_encoding=True, pos_type='learnable',
                axial_loops=1, pos_dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.H, self.W = H, W
        self.axial_loops = axial_loops

        self.norm_x = nn.LayerNorm(dim, eps=eps)        
        self.norm_p = nn.LayerNorm(dim, eps=eps)

        self.q_proj =   nn.Linear(dim, dim) # nn.Linear(dim, dim) # LearnableEyeLayer(dim, init_scale=1.0)
        self.k_proj =   nn.Linear(dim, dim) # nn.Linear(dim, dim) # LearnableEyeLayer(dim, init_scale=1.0)
        self.out_proj = nn.Linear(dim, dim) # nn.Linear(dim, dim) # TwiGLU(dim, initial_eye=True, initial_scale=0.01) # LearnableEyeLayer(dim, init_scale=1.0)
        self.delta_operator = nn.Linear(dim, dim) # TwiGLU(dim, initial_eye=True, initial_scale=0.01) # LearnableEyeLayer(dim, init_scale=0.1)
        self.mlp = Mlp(in_features = dim, hidden_features = 4 * dim, out_features = dim, act_layer = nn.Tanh, drop = 0)

        #-------------------- Test Report ------------------
        # BF16. scaler disabled as it doesn't support BF16.
        # (q_proj, k_proj, out_proj, delta_operator, mlp, dim, heads, attns, lr)
        # (L    L    L    T.01   N    40      10  5e-4):      loss 0.6    after 11%
        # (Is1  Is1  T1   T.01   N    40      10  5e-4):      loss 0.9    after 31%   Is: I1 + N(0,1)
        # (L    L    T1   T.01   N    40      10  5-4 ):      loss 0.6    after 40%
        
        # (L    L    L    T.01   N    12      40  5e-4):      loss 0.7    after 40%,  2 hrs
        # (L    L    L    T.01   N    40      10  10 5e-4):   loss 0.8    after 12%,  0:23 hrs
        # (L    L    T1   T.01   N    40   5  10 5e-4):       loss 0.64   after 13%,  0:23 hrs
        
        # (L    L    L    T.01   N    40   5  10 5e-4):       loss 0.6    after 20%   0:21 hrs    1.70 it/s
        # (L    L    L    L      N    40   5  10 5e-4):       loss 0.6    after 14%   0:14 hrs    1.78 it/s
        # (L    L    L    T.01   N    40   4  10 5e-4):       loss 0.6    after 10%   0:10 hrs    1.60 it/s
        
        # (L    L    L    T.01   N    40   2  10 5e-4):       loss 0.6    after 10%   0:10 hrs    1.65 it/s
        # (L    L    L    T.01   N    40   8  10 5e-4):       loss 0.6    after 10%   0:12 hrs    1.42 it/s            
        # (L    L    L    T.01   N    40   5  20 5e-4):       loss 0.6    after 7%    0:10 hrs    1.20 it/s
        # (I1   I1   T1   T.01   N    40   5  20 5e-4):       loss 0.7    atter 6%    0:10 hrs    1.13 it/s

        # (L    L    L    L      M    40   5  20 5e-4):       loss 0.8    after 11%   0:15 hrs    1.14 it/s
        # (L    L    L    L      M    40   5  20 5e-4) + 2 skips: loss 0.32     after 7% ):10 hrs   1.16 it/s   -------- Only skips cross the limit.

                
        self.height_pos_encoding, self.width_pos_encoding = None, None
        
        # Positional encodings for height and width axes
        if use_pos_encoding:
            self.height_pos_encoding = AxialPositionalEncoding1D(
                max_len=H, dim=dim, dropout=pos_dropout,
                pos_type=pos_type,
            )
            self.width_pos_encoding = AxialPositionalEncoding1D(
                max_len=W, dim=dim, dropout=pos_dropout,
                pos_type=pos_type,
            )
        
        # DeltaOperatorMultiChannel(H=H, W=W, C=dim, latent_rank=latent_rank, gate_init_scale=0.01)

    def add_positional_encoding(self, x, axis):
        """Add 1D positional encoding to the input tensor."""
        # if not self.use_pos_encoding:
        #     return x
        
        if axis == 1 and self.height_pos_encoding is not None:  # Height axis
            # x: [B*W, H, C] - each sequence is a column (height positions)
            x = self.height_pos_encoding(x)
        elif axis == 2 and self.width_pos_encoding is not None:  # Width axis  
            # x: [B*H, W, C] - each sequence is a row (width positions)
            x = self.width_pos_encoding(x)
        
        return x

    def forward_axis(self, x, yp, axis):
        """
        Axial attention along one axis (height or width) with physics correction.
        x: [B, H, W, C]
        p: [B, H, W, C]  (physical state)
        axis: 1 for height, 2 for width
        """
        B, H, W, C = x.shape

        if axis == 1:  # height attention (attend along columns)
            # [B, H, W, C] -> [B, W, H, C] -> [B*W, H, C]
            _x = x.permute(0, 2, 1, 3).reshape(B * W, H, C)
            _yp = yp.permute(0, 2, 1, 3).reshape(B * W, H, C)
            seq_len = H
            batch_expanded = B * W
            reshape_back = lambda out: out.reshape(B, W, H, C).permute(0, 2, 1, 3)
        else:  # width attention (attend along rows)
            # [B, H, W, C] -> [B*H, W, C]
            _x = x.reshape(B * H, W, C)
            _yp = yp.reshape(B * H, W, C)
            seq_len = W
            batch_expanded = B * H
            reshape_back = lambda out: out.reshape(B, H, W, C)

        # Add positional encoding
        _x = self.add_positional_encoding(_x, axis)
        _yp = self.add_positional_encoding(_yp, axis)
        
        # Layer norms (after positional encoding)
        dtype = _x.dtype
        _x = self.norm_x(_x).to(dtype=dtype)    # layernorm convers everythig to f32
        dtype = _yp.dtype
        _yp = self.norm_p(_yp).to(dtype=dtype)  # layernorm convers everythig to f32

        # ---- Q, K ----
        q = self.q_proj(_x) * (self.head_dim ** -0.5)
        k = self.k_proj(_x)

        # ---- Reshape to multi-head ----
        q = q.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)

        # ---- V: physics with delta correction ----
        # Need to reshape for delta_operator, then reshape back
        if axis == 1:
            # For height: _yp is [B*W, H, C], reshape to [B, W, H, C] for delta_operator
            yp_for_delta = _yp.reshape(B, W, H, C)
            p_corr = _yp + self.delta_operator(yp_for_delta).reshape(B*W, H, C)
        else:
            # For width: _yp is [B*H, W, C], reshape to [B, H, W, C] for delta_operator
            yp_for_delta = _yp.reshape(B, H, W, C)
            p_corr = _yp + self.delta_operator(yp_for_delta).reshape(B*H, W, C)
        
        v = p_corr.reshape(batch_expanded, seq_len, self.n_heads, self.head_dim)

        # ---- CRITICAL: Ensure all tensors are in bfloat16 for FlashAttention ----
        # FlashAttention only supports fp16 and bf16
        # if q.dtype not in [torch.bfloat16]:
        #     # Cast to bfloat16 if needed
        #     q = q.to(torch.bfloat16)
        #     k = k.to(torch.bfloat16)
        #     v = v.to(torch.bfloat16)
        
        # ---- FlashAttention ----
        att_out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)

        # Cast back to original dtype if needed
        original_dtype = _x.dtype
        if att_out.dtype != original_dtype:
            att_out = att_out.to(original_dtype)
        
        # Merge heads (already done by flash_attn)
        att_out = att_out.reshape(batch_expanded, seq_len, C)

        # Output projection
        out = self.out_proj(att_out)

        # Restore original shape
        out = reshape_back(out)
        
        # print(f"q: {q.mean().item():.3f}, {q.std().item():.3f} / "
        #     f"k: {k.mean().item():.3f}, {k.std().item():.3f} / "
        #     f"v: {v.mean().item():.3f}, {v.std().item():.3f} / "
        #     f"att: {att_out.mean().item():.3f}, {att_out.std().item():.3f} / "
        #     f"out: {out.mean().item():.3f}, {out.std().item():.3f}")
                
        return out

    def forward(self, x, yp):
        # Axial attention: height then width, repeated
        x_org = x                
        for _ in range(self.axial_loops):
            x = self.forward_axis(x, yp, axis=1)
            x = self.forward_axis(x, yp, axis=2)
        x = x + x_org

        x_org = x            
        if self.mlp is not None:
            x = self.mlp(x)
        x = x + x_org
        
        return x

# ------------------- Test Code -------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W, C, n_heads = 1, 180, 360, 12, 4
    loops = 5

    # Test different positional encoding types
    for pos_type in ['learnable', 'sinusoidal', 'linear', 'none']:
        print(f"\n{'='*60}")
        print(f"Testing with positional encoding: {pos_type}")
        print('='*60)
        
        use_pe = (pos_type != 'none')
        x = torch.randn(B, H, W, C, device=device)
        yp = torch.randn_like(x)
        target = torch.randn_like(x)
        
        codec = AxialFlashAttention(
            dim=C, H=H, W=W, n_heads=n_heads, 
            use_pos_encoding=use_pe,
            pos_type=pos_type if use_pe else 'none',
            pos_dropout=0.1 if use_pe else 0.0
        ).to(device)

        optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Use bfloat16 for training
        x_bf16 = x.to(torch.bfloat16)
        yp_bf16 = yp.to(torch.bfloat16)
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
                    output = codec(x_bf16, yp_bf16)
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
                    output = codec(x_bf16, yp_bf16, add_pe=use_pe)
                    loss = criterion(output, target_bf16)
                    loss.backward()
                    optimizer.step()
            elapsed_time = time.perf_counter() - start_time

        print(f"Output shape: {output.shape}")
        print(f"Elapsed time: {elapsed_time:.4f} sec")
        print(f"With positional encoding: {use_pe}")
        if use_pe:
            print(f"Positional encoding type: {pos_type}")
            
            # Show positional encoding statistics
            if pos_type == 'learnable':
                height_pe = codec.height_pos_encoding.pe
                width_pe = codec.width_pos_encoding.pe
                print(f"Height PE stats: mean={height_pe.mean():.4f}, std={height_pe.std():.4f}")
                print(f"Width PE stats: mean={width_pe.mean():.4f}, std={width_pe.std():.4f}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"Sequence lengths: Height={H}, Width={W}")
    print(f"Positional encoding added: BEFORE LayerNorm")
    print(f"FlashAttention requires fp16/bf16 - added explicit casting")
    print('='*60)