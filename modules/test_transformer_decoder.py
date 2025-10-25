# transformer_decoder_from_scratch.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# --------------------------
# Utility: causal mask
# --------------------------
# def causal_mask(seq_len: int, device=None, dtype=torch.bool):
#     # mask[i, j] = True if j > i (future) -> should be masked
#     mask = torch.triu(torch.ones((seq_len, seq_len), dtype=dtype, device=device), diagonal=1)
#     return mask  # shape (L, L)

def causal_mask(seq_len: int, batch_size: int = 1, n_head: int = 1, device=None, dtype=torch.bool):
    """
    Returns a mask of shape (B, n_head, L, L) where True = masked (future positions)
    seq_len: target sequence length
    batch_size: number of sequences in batch
    n_head: number of attention heads
    """
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=dtype, device=device), diagonal=1)
    # (1,1,L,L) -> broadcastable to (B, n_head, L, L)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, n_head, seq_len, seq_len)
    return mask

# --------------------------
# Scaled dot-product attention
# --------------------------
def scaled_dot_product_attention(q, k, v, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Module] = None):
    """
    q, k, v: (B, n_heads, L, head_dim)
    mask: broadcastable to (B, n_heads, L_q, L_k) where mask==True means masked-out # No, there is no head dimension.
    returns: attn_output (B, n_heads, L_q, head_dim), attn_weights (B, n_heads, L_q, L_k)
    """
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)  # (B, h, Lq, Lk)
    if mask is not None:
        # mask True -> set to -inf so softmax -> 0
        scores = scores.masked_fill(mask.unsqueeze(1).to(torch.bool), float("-1e9"))
    attn = F.softmax(scores, dim=-1)  # (B, h, Lq, Lk)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)  # (B, h, Lq, head_dim)
    return out, attn

# --------------------------
# Multi-head attention
# --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask: Optional[torch.Tensor] = None):
        """
        query: (B, Lq, d_model)
        key/value: (B, Lk, d_model)
        attn_mask: (Lq, Lk) or (B, Lq, Lk) boolean mask where True means masked
        """
        B = query.size(0)

        # linear projections
        q = self.q_lin(query)  # (B, Lq, d_model)
        k = self.k_lin(key)
        v = self.v_lin(value)

        # split heads
        def split_heads(x):
            # (B, L, d_model) -> (B, n_head, L, head_dim)
            return x.view(B, x.size(1), self.n_head, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # attn
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask=attn_mask, dropout=self.dropout)
        # concat heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B, Lq, d_model)
        out = self.out_lin(attn_out)
        return out, attn_weights

# --------------------------
# Feed-forward
# --------------------------
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --------------------------
# Decoder layer
# --------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None):
        """
        x: (B, Lt, d_model)  - target embeddings / states
        memory: (B, Lm, d_model) - encoder outputs
        self_mask: (Lt, Lt) or (B, Lt, Lt) - causal mask for self-attn
        memory_mask: (Lt, Lm) or (B, Lt, Lm) - optional mask for cross-attn
        """
        # Self-attention (masked)
        residual = x
        x_attn, _ = self.self_attn(x, x, x, attn_mask=self_mask)
        x = residual + self.dropout(x_attn)
        x = self.norm1(x)

        # Cross-attention (attend to encoder memory)
        residual = x
        x_attn2, _ = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = residual + self.dropout(x_attn2)
        x = self.norm2(x)

        # FFN
        residual = x
        x_ffn = self.ffn(x)
        x = residual + self.dropout(x_ffn)
        x = self.norm3(x)

        return x

# --------------------------
# Full Decoder
# --------------------------
class TransformerDecoderFromScratch(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)])
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None):
        """
        tgt: (B, Lt, d_model)
        memory: (B, Lm, d_model)
        tgt_mask: (Lt, Lt) or (B, Lt, Lt)
        memory_mask: (Lt, Lm) or (B, Lt, Lm)
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, self_mask=tgt_mask, memory_mask=memory_mask)
        return x  # (B, Lt, d_model)


if __name__ == "__main__":
    B = 2
    Lm = 7   # encoder length
    Lt = 5   # target length
    d_model = 64
    n_head = 8
    d_ff = 256
    num_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # toy encoder memory (like encoder outputs)
    memory = torch.randn(B, Lm, d_model, device=device)

    # toy target embeddings (teacher-forcing)
    tgt = torch.randn(B, Lt, d_model, device=device)

    # build decoder
    decoder = TransformerDecoderFromScratch(num_layers, d_model, n_head, d_ff, dropout=0.1).to(device)

    # causal mask for tgt self-attention
    mask = causal_mask(Lt, device=device, dtype=torch.bool)  # shape (Lt, Lt) True means masked
    # The MultiHeadAttention expects mask where True = masked. It broadcasts internally.

    out = decoder(tgt, memory, tgt_mask=mask)
    print("Decoder output shape:", out.shape)  # (B, Lt, d_model)

    # -------------------------
    # Greedy autoregressive generation (inefficient)
    # -------------------------
    # Suppose we have an embedding table and output linear layer to logits over vocab
    vocab_size = 1000
    embed = nn.Embedding(vocab_size, d_model).to(device)
    out_proj = nn.Linear(d_model, vocab_size).to(device)

    # Start tokens (batch)
    start_tokens = torch.tensor([2, 2], device=device).unsqueeze(1)  # shape (B, 1)
    generated = start_tokens  # token ids

    max_len = 8
    for step in range(max_len - 1):
        # embed all generated tokens so far
        tgt_emb = embed(generated)  # (B, cur_len, d_model)
        cur_len = tgt_emb.size(1)
        # causal mask for current length
        m = causal_mask(cur_len, device=device, dtype=torch.bool)
        dec_out = decoder(tgt_emb, memory, tgt_mask=m)  # (B, cur_len, d_model)
        last = dec_out[:, -1, :]  # (B, d_model)
        logits = out_proj(last)   # (B, vocab)
        next_tok = logits.argmax(dim=-1, keepdim=True)  # greedy
        generated = torch.cat([generated, next_tok], dim=1)

    print("Generated token ids shape:", generated.shape)  # (B, max_len)
    print("Generated (example):", generated[0].tolist())
