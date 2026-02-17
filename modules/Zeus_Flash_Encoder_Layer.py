import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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


from modules.Zeus_Flash_MHA_Standard import FlashMultiHeadAttention

class FlashEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0, use_flash=False, eps=1e-9):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.use_flash = use_flash
        
        self.attention = FlashMultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, use_flash=self.use_flash)
        self.feed_forward = PositionwiseFeedForward(in_features=self.d_model, hidden_features=4*self.d_model, out_features=self.d_model, act_layer=nn.GELU, drop=dropout)

        self.norm_after_attn = nn.LayerNorm(d_model, eps=eps)
        self.norm_after_feed_forward = nn.LayerNorm(d_model, eps=eps)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x, mask=None, pos_bias_er=None):
        # x: (B, seq_len, d_model)        

        x_skip = x
        x = self.attention(x, x, x, mask=mask, pos_bias_er=pos_bias_er)
        x = x + x_skip
        dtype = x.dtype
        x = self.norm_after_attn(x).to(dtype=dtype)
        
        x_skip = x
        x = self.feed_forward(x)
        x = x + x_skip
        dtype = x.dtype
        x = self.norm_after_feed_forward(x).to(dtype=dtype)
        
        return x
                
        