import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flash_attn.flash_attn_interface import flash_attn_func

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.util_snapshots_auto import save, ctx

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_flash=False):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.use_flash = use_flash
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.stable_softmax = True
        
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
    
    def forward(self, Q, K, V, mask=None, pos_bias_er=None, ss=None):    
        # Q, K, V: (B, seq_len, d_model)
        
        batch_size, seq_len, dim = Q.shape

        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k)  # (B, seq_len, num_heads, head_dim)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k)  # (B, seq_len, num_heads, head_dim)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k)  # (B, seq_len, num_heads, head_dim)
        
        save(ss, self.W_q, 'wq')
        save(ss, self.W_k, 'wk')
        save(ss, self.W_v, 'wv')

        softmax_scale = 1 / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if self.use_flash and mask is None and pos_bias_er is None:
            assert mask is None
            assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
            
            att_out = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=softmax_scale, causal=False)    # (B, seq_len, num_heads, head_dim)
            if not att_out.is_contiguous(): att_out = att_out.contiguous()
            
        else:
            # 
            Q = Q.transpose(1, 2)   # (B, num_heads, seq_len, head_dim)
            K = K.transpose(1, 2)   # (B, num_heads, seq_len, head_dim)
            V = V.transpose(1, 2)   # (B, num_heads, seq_len, head_dim)
        
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale   # (B, num_heads, seq_len, seq_len)
            save(ss, scores, "sc_r")
            
            if pos_bias_er is not None:         # pos_bias before masking
                scores_org = scores
                scores = pos_bias_er(scores)
                pb = scores-scores_org
                save(ss, torch.abs(pb)/(torch.abs(scores)+1e-9), "rate")
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)                # (B, num_heads, seq_len, seq_len)
                save(ss, scores, "sc_m")     

            # attn = F.softmax(scores, dim=-1)                                # (B, num_heads, seq_len, seq_len)
            attn = self._stable_softmax(scores, dim=-1)

            att_out = torch.matmul(attn, V)                                 # (B, num_heads, seq_len, head_dim)
            att_out = att_out.transpose(1, 2).contiguous()  # (B, seq_len, num_heads, head_dim)

        # Concatenate heads and apply final linear
        att_out = att_out.view(batch_size, seq_len, self.num_heads * self.d_k)   # (B, seq_len, d_model)
        return self.W_o(att_out)
        