import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for height/width split"
        self.dim = dim
        self.base = base
        
        self.h_dim = dim // 2
        self.w_dim = dim // 2
        
        self.rope_h = RotaryPositionalEmbedding1D(self.h_dim, base)
        self.rope_w = RotaryPositionalEmbedding1D(self.w_dim, base)
    
    def forward(self, x):
        batch_size, height, width, n_heads, head_dim = x.shape
        
        # Split into height and width components
        x_h = x[..., :self.h_dim]
        x_w = x[..., self.h_dim:]
        
        # Height RoPE
        x_h_reshaped = x_h.permute(0, 3, 2, 1, 4).contiguous()
        x_h_reshaped = x_h_reshaped.view(-1, height, self.h_dim)
        x_h_rot_reshaped = self.rope_h(x_h_reshaped, seq_dim=1)
        x_h_rot = x_h_rot_reshaped.view(batch_size, n_heads, width, height, self.h_dim)
        x_h_rot = x_h_rot.permute(0, 3, 2, 1, 4)
        
        # Width RoPE
        x_w_reshaped = x_w.permute(0, 3, 1, 2, 4).contiguous()
        x_w_reshaped = x_w_reshaped.view(-1, width, self.w_dim)
        x_w_rot_reshaped = self.rope_w(x_w_reshaped, seq_dim=1)
        x_w_rot = x_w_rot_reshaped.view(batch_size, n_heads, height, width, self.w_dim)
        x_w_rot = x_w_rot.permute(0, 2, 3, 1, 4)
        
        return torch.cat([x_h_rot, x_w_rot], dim=-1)


class RotaryPositionalEmbedding1D(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        
        view_shape = [1] * x.ndim
        view_shape[seq_dim] = seq_len
        view_shape[-1] = self.dim
        freqs = freqs.view(*view_shape)
        
        x_rot = torch.empty_like(x)
        cos_f = torch.cos(freqs)
        sin_f = torch.sin(freqs)
        
        x_rot[..., 0::2] = x[..., 0::2] * cos_f[..., 0::2] - x[..., 1::2] * sin_f[..., 0::2]
        x_rot[..., 1::2] = x[..., 1::2] * cos_f[..., 0::2] + x[..., 0::2] * sin_f[..., 0::2]
        
        return x_rot


def visualize_correct_rope_similarity():
    """CORRECT visualization showing Query-Key similarity matrix with diagonal stripes."""
    
    print("=" * 70)
    print("CORRECT RoPE Similarity Visualization")
    print("=" * 70)
    print("\nKey Concept: Similarity = ⟨RoPE(Q, pos_q), RoPE(K, pos_k)⟩")
    print("This depends only on relative position: pos_q - pos_k")
    print("Creates DIAGONAL STRIPES in similarity matrix\n")
    
    # Setup for 1D RoPE first
    dim = 64
    seq_len = 32
    
    rope_1d = RotaryPositionalEmbedding1D(dim)
    
    # Create all query and key vectors
    # All start as same unit vector: [1, 0, 0, ...]
    Q = torch.zeros(1, seq_len, 1, 1, dim)
    K = torch.zeros(1, seq_len, 1, 1, dim)
    Q[..., 0] = 1.0
    K[..., 0] = 1.0
    
    # Apply RoPE to all positions
    Q_rot = rope_1d(Q, seq_dim=1)  # (1, seq_len, 1, 1, dim)
    K_rot = rope_1d(K, seq_dim=1)  # (1, seq_len, 1, 1, dim)
    
    Q_rot = Q_rot[0, :, 0, 0, :]  # (seq_len, dim)
    K_rot = K_rot[0, :, 0, 0, :]  # (seq_len, dim)
    
    # Compute similarity matrix: S[i,j] = ⟨Q_i, K_j⟩
    similarity_1d = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            sim = torch.dot(Q_rot[i], K_rot[j]).item()
            similarity_1d[i, j] = sim
    
    # Now for 2D RoPE
    H, W = 16, 16
    rope_2d = RotaryPositionalEmbedding2D(dim)
    
    # Create 2D grid of queries and keys
    Q_2d = torch.zeros(1, H, W, 1, dim)
    K_2d = torch.zeros(1, H, W, 1, dim)
    Q_2d[..., 0] = 1.0
    K_2d[..., 0] = 1.0
    
    # Apply 2D RoPE
    Q_rot_2d = rope_2d(Q_2d)  # (1, H, W, 1, dim)
    K_rot_2d = rope_2d(K_2d)  # (1, H, W, 1, dim)
    
    Q_rot_2d = Q_rot_2d[0, :, :, 0, :]  # (H, W, dim)
    K_rot_2d = K_rot_2d[0, :, :, 0, :]  # (H, W, dim)
    
    # Flatten to 1D sequences for 2D grid
    Q_flat = Q_rot_2d.view(H*W, dim)  # (H*W, dim)
    K_flat = K_rot_2d.view(H*W, dim)  # (H*W, dim)
    
    # Compute similarity matrix for flattened 2D
    seq_len_2d = H * W
    similarity_2d = torch.zeros(seq_len_2d, seq_len_2d)
    for i in range(seq_len_2d):
        for j in range(seq_len_2d):
            sim = torch.dot(Q_flat[i], K_flat[j]).item()
            similarity_2d[i, j] = sim
    
    # Visualize
    fig = plt.figure(figsize=(16, 8))
    
    # 1. 1D RoPE similarity matrix
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(similarity_1d.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_title('1D RoPE: S[i,j] = ⟨RoPE(Q,i), RoPE(K,j)⟩\nClear DIAGONAL STRIPES')
    plt.colorbar(im1, ax=ax1)
    
    # Draw diagonal lines
    for k in range(-3, 4):
        if k == 0:
            continue
        diag_x = np.arange(max(0, -k), min(seq_len, seq_len - k))
        diag_y = diag_x + k
        if len(diag_x) > 0:
            ax1.plot(diag_x, diag_y, 'w-', alpha=0.5, linewidth=1)
    
    # 2. 2D RoPE similarity matrix (flattened)
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(similarity_2d.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xlabel('Key Position (flattened 2D)')
    ax2.set_ylabel('Query Position (flattened 2D)')
    ax2.set_title('2D RoPE Flattened\nStill shows diagonal structure')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Zoom into 1D RoPE to see pattern clearly
    ax3 = plt.subplot(2, 3, 3)
    zoom_size = 12
    zoom_slice = similarity_1d[:zoom_size, :zoom_size].cpu().numpy()
    im3 = ax3.imshow(zoom_slice, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Annotate with positions
    for i in range(zoom_size):
        for j in range(zoom_size):
            rel_pos = i - j
            ax3.text(j, i, f'{rel_pos:+d}', ha='center', va='center', 
                    color='white' if abs(zoom_slice[i, j]) > 0.5 else 'black',
                    fontsize=8)
    
    ax3.set_xlabel('Key Position j')
    ax3.set_ylabel('Query Position i')
    ax3.set_title(f'Zoomed 1D RoPE (shows rel pos i-j)\nSame color along diagonals')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Show 1D similarity depends only on relative position
    ax4 = plt.subplot(2, 3, 4)
    
    # Extract diagonals
    rel_positions = []
    similarities_by_rel_pos = {}
    
    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos = i - j
            sim = similarity_1d[i, j].item()
            if rel_pos not in similarities_by_rel_pos:
                similarities_by_rel_pos[rel_pos] = []
            similarities_by_rel_pos[rel_pos].append(sim)
    
    # Plot similarity vs relative position
    rel_pos_list = sorted(similarities_by_rel_pos.keys())
    mean_sims = [np.mean(similarities_by_rel_pos[p]) for p in rel_pos_list]
    std_sims = [np.std(similarities_by_rel_pos[p]) for p in rel_pos_list]
    
    ax4.errorbar(rel_pos_list, mean_sims, yerr=std_sims, fmt='o-', 
                capsize=4, linewidth=2)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Self-similarity (i=j)')
    ax4.set_xlabel('Relative Position (i - j)')
    ax4.set_ylabel('Similarity ⟨Q_i, K_j⟩')
    ax4.set_title('1D RoPE: Similarity depends ONLY on i-j\nProof of relative position encoding')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 2D RoPE: Show horizontal and vertical cuts
    ax5 = plt.subplot(2, 3, 5)
    
    # Pick a reference query at center
    center_h, center_w = H//2, W//2
    ref_query_idx = center_h * W + center_w
    
    # Extract row from similarity matrix (fixed query, varying keys)
    similarities_ref_query = similarity_2d[ref_query_idx, :].cpu().numpy()
    
    # Reshape to 2D
    similarities_2d_grid = similarities_ref_query.reshape(H, W)
    
    im5 = ax5.imshow(similarities_2d_grid, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.scatter([center_w], [center_h], color='red', s=100, marker='*', 
                edgecolors='white', linewidths=1.5)
    ax5.set_xlabel('Key Width Position')
    ax5.set_ylabel('Key Height Position')
    ax5.set_title(f'2D RoPE: Similarity from center query (H={center_h}, W={center_w})')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Mathematical property check
    ax6 = plt.subplot(2, 3, 6)
    
    # Check that S[i,j] = S[i+1, j+1] for 1D RoPE
    i, j = 5, 3
    s_ij = similarity_1d[i, j].item()
    s_ip1_jp1 = similarity_1d[i+1, j+1].item()
    s_i2_j2 = similarity_1d[i+2, j+2].item()
    
    x_pos = [0, 1, 2]
    y_vals = [s_ij, s_ip1_jp1, s_i2_j2]
    
    ax6.bar(x_pos, y_vals, color=['blue', 'green', 'orange'])
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'(i={i},j={j})', f'(i={i+1},j={j+1})', f'(i={i+2},j={j+2})'])
    ax6.set_ylabel('Similarity')
    ax6.set_title(f'RoPE Property: S[i,j] = S[i+Δ,j+Δ]\nSame along diagonals (Δ={i-j})')
    
    for idx, (x, y) in enumerate(zip(x_pos, y_vals)):
        ax6.text(x, y/2, f'{y:.3f}', ha='center', va='center', color='white', fontweight='bold')
    
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    # 1D RoPE analysis
    print("\n1. 1D RoPE Properties:")
    print(f"   Sequence length: {seq_len}")
    print(f"   Self-similarity (i=j): {similarity_1d.diag().mean():.4f} ± {similarity_1d.diag().std():.4f}")
    print(f"   Max off-diagonal similarity: {similarity_1d[~torch.eye(seq_len, dtype=bool)].max():.4f}")
    
    # Check translation invariance along diagonals
    print("\n2. Checking relative position encoding:")
    for rel_pos in [-3, -2, -1, 0, 1, 2, 3]:
        diag_vals = torch.diagonal(similarity_1d, offset=rel_pos)
        mean_val = diag_vals.mean().item()
        std_val = diag_vals.std().item()
        print(f"   Relative position {rel_pos:2d}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    print("\n3. Key Insight:")
    print("   • Each diagonal in similarity matrix corresponds to fixed relative position")
    print("   • Values are constant along each diagonal (within numerical precision)")
    print("   • This is the defining property of RoPE: S[i,j] = f(i-j)")
    
    print("\n4. For 2D RoPE:")
    print("   • Similarity depends on relative height AND relative width")
    print("   • S[(h_q,w_q), (h_k,w_k)] = f(h_q-h_k, w_q-w_k)")
    print("   • Creates diagonal stripes in the flattened similarity matrix")
    
    print("=" * 70)


if __name__ == "__main__":
    visualize_correct_rope_similarity()