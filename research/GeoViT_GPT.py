import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_func
import gc

# ------------------- Flash2DMHA -------------------
class Flash2DMHA(nn.Module):
    def __init__(self, d_model, nHeads, dropout=0.0, causal=False,
                 tokens_per_chunk=None, heads_per_chunk=None, kv_full_sequence=True):
        super().__init__()
        assert d_model % nHeads == 0
        if heads_per_chunk is not None:
            assert nHeads % heads_per_chunk == 0

        self.d_model = d_model
        self.nHeads = nHeads
        self.head_dim = d_model // nHeads
        self.dropout = dropout
        self.causal = causal

        self.tokens_per_chunk = tokens_per_chunk
        self.heads_per_chunk = heads_per_chunk or nHeads
        self.kv_full_sequence = kv_full_sequence

        self.W_qkv = nn.Linear(d_model, 3*d_model, dtype=torch.bfloat16)
        self.W_o = nn.Linear(d_model, d_model, dtype=torch.bfloat16)

    def forward_attention(self, q_h, k_h, v_h, s_start, s_end, attn_mask):
        q_seq = q_h[:, :, s_start:s_end, :]
        if self.kv_full_sequence:
            k_seq, v_seq = k_h, v_h
            mask_chunk = None
            if attn_mask is not None and attn_mask.dim() == 4:
                mask_chunk = attn_mask[:, :, s_start:s_end, :]
        else:
            k_seq, v_seq = q_seq, q_seq
            mask_chunk = None

        q_seq_for = q_seq.permute(0,2,1,3).contiguous()
        k_seq_for = k_seq.permute(0,2,1,3).contiguous()
        v_seq_for = v_seq.permute(0,2,1,3).contiguous()

        del q_seq, k_seq, v_seq
        if attn_mask is not None: del attn_mask
        gc.collect()
        torch.cuda.empty_cache()

        if mask_chunk is not None:
            out_seq = flash_attn_func(q_seq_for, k_seq_for, v_seq_for,
                                      attn_mask=mask_chunk,
                                      dropout_p=self.dropout,
                                      softmax_scale=None,
                                      causal=self.causal)
        else:
            out_seq = flash_attn_func(q_seq_for, k_seq_for, v_seq_for,
                                      dropout_p=self.dropout,
                                      softmax_scale=None,
                                      causal=self.causal)
        return out_seq

    def forward(self, input, attn_mask=None):
        B,L,D = input.shape
        input = input.to(torch.bfloat16)
        qkv = self.W_qkv(input).view(B,L,self.nHeads,3*self.head_dim).permute(0,2,1,3).contiguous()
        q,k,v = torch.chunk(qkv,3,dim=-1)

        heads_per_chunk = self.heads_per_chunk
        q_head_splits = list(q.split(heads_per_chunk, dim=1))
        k_head_splits = list(k.split(heads_per_chunk, dim=1))
        v_head_splits = list(v.split(heads_per_chunk, dim=1))

        seq_ranges = [(0,L)] if self.tokens_per_chunk is None else [(i,min(i+self.tokens_per_chunk,L)) for i in range(0,L,self.tokens_per_chunk)]
        out_head_chunks = []

        for q_h,k_h,v_h in zip(q_head_splits,k_head_splits,v_head_splits):
            seq_out_parts = []
            for s_start,s_end in seq_ranges:
                out_seq = self.forward_attention(q_h,k_h,v_h,s_start,s_end,attn_mask)
                seq_out_parts.append(out_seq)
            out_h = torch.cat(seq_out_parts,dim=1)
            out_head_chunks.append(out_h)

        out_all = torch.cat(out_head_chunks, dim=2)
        out_final = out_all.contiguous().view(B,L,D)
        out_proj = self.W_o(out_final)
        return out_proj

# ------------------- Embeddings -------------------
class LatLonPosEmbedding(nn.Module):
    def __init__(self, embed_dim,H,W):
        super().__init__()
        self.H,self.W = H,W
        self.lat_emb = nn.Parameter(torch.randn(H, embed_dim//2, dtype=torch.bfloat16))
        self.lon_emb = nn.Parameter(torch.randn(W, embed_dim//2, dtype=torch.bfloat16))

    def forward(self,x):
        # x: (B,T,H,W,C)
        device = x.device
        lat = self.lat_emb.unsqueeze(1).expand(-1,self.W,-1)
        lon = self.lon_emb.unsqueeze(0).expand(self.H,-1,-1)
        lat_lon = torch.cat([lat,lon],dim=-1).reshape(self.H*self.W,-1).to(device)
        B,T,H,W,C = x.shape
        x = x.reshape(B*T,H*W,C)
        x = x + lat_lon.unsqueeze(0)
        return x.reshape(B,T,H,W,C)

class TemporalEmbedding(nn.Module):
    def __init__(self, embed_dim,T):
        super().__init__()
        self.time_emb = nn.Parameter(torch.randn(T, embed_dim, dtype=torch.bfloat16))
    def forward(self,x):
        # x: (B,T,H,W,C)
        B,T,H,W,C = x.shape
        device = x.device
        return x + self.time_emb.unsqueeze(0).unsqueeze(2).unsqueeze(2).to(device)

# ------------------- SpatioTemporal Block -------------------
class SpatioTemporalBlock(nn.Module):
    def __init__(self, embed_dim,nHeads,mha_class,mha_kwargs=None):
        super().__init__()
        mha_kwargs = mha_kwargs or {}
        self.norm_spatial = nn.LayerNorm(embed_dim, dtype=torch.bfloat16)
        self.mha_spatial = mha_class(embed_dim,nHeads,**mha_kwargs)
        self.norm_temporal = nn.LayerNorm(embed_dim, dtype=torch.bfloat16)
        self.mha_temporal = mha_class(embed_dim,nHeads,**mha_kwargs)
        self.norm_ffn = nn.LayerNorm(embed_dim, dtype=torch.bfloat16)
        self.ffn = nn.Sequential(nn.Linear(embed_dim,embed_dim*4, dtype=torch.bfloat16),
                                 nn.GELU(),
                                 nn.Linear(embed_dim*4,embed_dim, dtype=torch.bfloat16))
    def forward(self,x,T,H,W):
        # x: (B,T,H,W,C)
        B,T,H,W,C = x.shape
        x_res = x

        # --- spatial ---
        x_spatial = x.reshape(B*T,H*W,C)
        x_spatial = self.norm_spatial(x_spatial)
        x_spatial = self.mha_spatial(x_spatial)
        x_spatial = x_spatial.reshape(B,T,H,W,C)
        x = x_res + x_spatial

        # --- temporal ---
        x_res = x
        x_temporal = x.permute(0,2,3,1,4).reshape(B*H*W,T,C)
        x_temporal = self.norm_temporal(x_temporal)
        x_temporal = self.mha_temporal(x_temporal)
        x_temporal = x_temporal.reshape(B,H,W,T,C).permute(0,3,1,2,4)
        x = x_res + x_temporal

        # --- FFN ---
        x_res = x
        x_ffn = self.norm_ffn(x.reshape(B*T*H*W,C))
        x_ffn = self.ffn(x_ffn).reshape(B,T,H,W,C)
        x = x_res + x_ffn
        return x

# ------------------- Patch Merge/Upsample -------------------
class PatchMerging(nn.Module):
    def __init__(self,in_dim,out_dim,H,W):
        super().__init__()
        self.H,self.W = H,W
        self.reduction = nn.Linear(4*in_dim,out_dim, dtype=torch.bfloat16)
    def forward(self,x):
        # x: (B,T,H,W,C)
        B,T,H,W,C = x.shape
        x0 = x[:,:,0::2,0::2,:]; x1 = x[:,:,0::2,1::2,:]
        x2 = x[:,:,1::2,0::2,:]; x3 = x[:,:,1::2,1::2,:]
        x_merged = torch.cat([x0,x1,x2,x3],dim=-1)
        return self.reduction(x_merged), H//2, W//2

class PatchUpsample(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim,out_dim*4, dtype=torch.bfloat16)
    def forward(self,x):
        B,T,H,W,D = x.shape
        x = self.proj(x).reshape(B,T,H,W,2,2,D)
        x = x.permute(0,1,2,4,3,5,6).reshape(B,T,H*2,W*2,D)
        return x

# ------------------- GeoViT -------------------
class GeoViTHierarchical(nn.Module):
    def __init__(self,C=6,T=24,H=721,W=1440,embed_dim=128,
                 nHeads=8,nLayers=[1,1,1],mha_class=Flash2DMHA,mha_kwargs=None):
        super().__init__()
        self.C,self.T,self.H,self.W,self.embed_dim = C,T,H,W,embed_dim
        self.input_proj = nn.Linear(C,embed_dim, dtype=torch.bfloat16)
        self.pos_emb = LatLonPosEmbedding(embed_dim,H,W)
        self.temp_emb = TemporalEmbedding(embed_dim,T)

        self.stages = nn.ModuleList()
        self.patch_merge_layers = nn.ModuleList()
        H_cur,W_cur = H,W
        for nl in nLayers:
            stage_blocks = nn.ModuleList([SpatioTemporalBlock(embed_dim,nHeads,mha_class,mha_kwargs) for _ in range(nl)])
            self.stages.append(stage_blocks)
            if H_cur>1 and W_cur>1:
                self.patch_merge_layers.append(PatchMerging(embed_dim,embed_dim,H_cur,W_cur))
                H_cur,W_cur = H_cur//2,W_cur//2
            else:
                self.patch_merge_layers.append(None)
        self.output_proj = nn.Linear(embed_dim,C, dtype=torch.bfloat16)
        self.upsample_layers = nn.ModuleList([PatchUpsample(embed_dim,embed_dim) if pm is not None else None
                                              for pm in self.patch_merge_layers[::-1]])

    def forward(self,x):
        # x: (B,T,C,H,W)
        device = x.device
        B,T,C,H,W = x.shape
        x = x.permute(0,1,3,4,2).to(torch.bfloat16)  # (B,T,H,W,C)
        x = self.input_proj(x)
        x = self.pos_emb(x)
        x = self.temp_emb(x)
        H_cur,W_cur = H,W
        skip_connections = []
        for stage_blocks, patch_merge in zip(self.stages,self.patch_merge_layers):
            for blk in stage_blocks:
                x = blk(x,T,H_cur,W_cur)
            skip_connections.append(x)
            if patch_merge is not None:
                x,H_cur,W_cur = patch_merge(x)
        for x_skip, up in zip(reversed(skip_connections), self.upsample_layers):
            if up is not None:
                x = up(x)
        x = self.output_proj(x)
        x = x.permute(0,1,4,2,3)  # (B,T,C,H,W)
        return x

# ------------------- Demo -------------------
if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    B,T,C,H,W = 1,24,6,72,144
    x = torch.randn(B,T,C,H,W,device=device)
    model = GeoViTHierarchical(C=C,T=T,H=H,W=W,embed_dim=128,nHeads=8,nLayers=[1,1],
                               mha_class=Flash2DMHA,mha_kwargs={"tokens_per_chunk":None,"heads_per_chunk":2,"kv_full_sequence":True})
    model.to(device)
    
    import time
    s = time.time()
    out = model(x)
    e = time.time()
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("time: {}".format(e-s))