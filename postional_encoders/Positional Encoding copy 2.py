import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_correct_diagonals():
    """Show CORRECT diagonal pattern for RoPE."""
    
    print("=" * 70)
    print("CORRECT RoPE Diagonal Pattern Analysis")
    print("=" * 70)
    print("\nMathematical expectation:")
    print("  S[i,j] = ⟨RoPE(Q,i), RoPE(K,j)⟩ = f(i-j)")
    print("  Constant along lines: j = i - k  (45° diagonals)")
    print("\nIf we see 135° diagonals (j = -i + k), something is WRONG")
    print("=" * 70)
    
    # Setup
    dim = 64
    seq_len = 20
    
    rope = RotaryPositionalEmbedding1D(dim)
    
    # Create unit vectors
    Q = torch.zeros(1, seq_len, 1, 1, dim)
    K = torch.zeros(1, seq_len, 1, 1, dim)
    Q[..., 0] = 1.0
    K[..., 0] = 1.0
    
    # Apply RoPE
    Q_rot = rope(Q, seq_dim=1)
    K_rot = rope(K, seq_dim=1)
    
    Q_rot = Q_rot[0, :, 0, 0, :]  # (seq_len, dim)
    K_rot = K_rot[0, :, 0, 0, :]  # (seq_len, dim)
    
    # Compute similarity matrix
    similarity = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            sim = torch.dot(Q_rot[i], K_rot[j]).item()
            similarity[i, j] = sim
    
    # Visualize with CORRECT axes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Similarity matrix with proper axes
    ax1 = axes[0, 0]
    im1 = ax1.imshow(similarity.cpu().numpy(), cmap='RdBu_r', 
                    extent=[0, seq_len-1, seq_len-1, 0],  # Note: y-axis reversed for imshow
                    aspect='auto', vmin=-1, vmax=1)
    
    # Draw 45° diagonals (j = i - k)
    for k in range(-seq_len+1, seq_len):
        # j = i - k
        i_vals = np.arange(seq_len)
        j_vals = i_vals - k
        
        # Filter valid indices
        mask = (j_vals >= 0) & (j_vals < seq_len)
        if np.any(mask):
            ax1.plot(j_vals[mask], i_vals[mask], 'w-', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Key Position (j)')
    ax1.set_ylabel('Query Position (i)')
    ax1.set_title('1D RoPE Similarity\nLines: j = i - k (45° diagonals)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Show what WRONG pattern would look like (135° diagonals)
    ax2 = axes[0, 1]
    
    # Create a WRONG pattern for comparison: S[i,j] = f(i+j)
    wrong_similarity = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            # Wrong: depends on i+j instead of i-j
            rel = i + j
            wrong_similarity[i, j] = torch.cos(torch.tensor(rel * 0.3)).item()
    
    im2 = ax2.imshow(wrong_similarity.cpu().numpy(), cmap='RdBu_r',
                    extent=[0, seq_len-1, seq_len-1, 0],
                    aspect='auto', vmin=-1, vmax=1)
    
    # Draw 135° diagonals (j = -i + k)
    for k in range(0, 2*seq_len):
        # j = -i + k
        i_vals = np.arange(seq_len)
        j_vals = -i_vals + k
        
        mask = (j_vals >= 0) & (j_vals < seq_len)
        if np.any(mask):
            ax2.plot(j_vals[mask], i_vals[mask], 'w-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Key Position (j)')
    ax2.set_ylabel('Query Position (i)')
    ax2.set_title('WRONG Pattern (if S[i,j] = f(i+j))\nLines: j = -i + k (135° diagonals)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Check actual diagonals
    ax3 = axes[0, 2]
    
    # Extract values along several diagonals
    diagonals_to_check = [-3, -1, 0, 1, 3]
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for k, color in zip(diagonals_to_check, colors):
        diag_vals = torch.diagonal(similarity, offset=k).cpu().numpy()
        positions = np.arange(len(diag_vals))
        
        ax3.plot(positions, diag_vals, f'{color}-', label=f'k={k}', linewidth=2)
        ax3.scatter(positions, diag_vals, color=color, s=20)
    
    ax3.set_xlabel('Position along diagonal')
    ax3.set_ylabel('Similarity')
    ax3.set_title('Checking constant along diagonals\n(Should be flat lines)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Matrix with annotations to see pattern
    ax4 = axes[1, 0]
    
    zoom_size = 8
    zoom_slice = similarity[:zoom_size, :zoom_size].cpu().numpy()
    
    im4 = ax4.imshow(zoom_slice, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Annotate with i-j values
    for i in range(zoom_size):
        for j in range(zoom_size):
            rel_pos = i - j
            ax4.text(j, i, f'{rel_pos:+d}', ha='center', va='center',
                    color='white' if abs(zoom_slice[i, j]) > 0.5 else 'black',
                    fontsize=10, fontweight='bold')
    
    # Draw 45° line
    i_line = np.arange(zoom_size)
    j_line = i_line  # Main diagonal: i-j=0
    ax4.plot(j_line, i_line, 'y-', linewidth=2, alpha=0.7)
    
    ax4.set_xlabel('j (Key)')
    ax4.set_ylabel('i (Query)')
    ax4.set_title(f'Zoomed with i-j values\nYellow line: i=j (main diagonal)')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Compare i-j vs i+j patterns
    ax5 = axes[1, 1]
    
    # Create sample vectors to understand the rotation
    seq_len_small = 5
    t_small = torch.arange(seq_len_small, dtype=torch.float32)
    
    # Simple 2D rotation example
    def rotate_vector(v, angle):
        return torch.tensor([
            v[0] * math.cos(angle) - v[1] * math.sin(angle),
            v[0] * math.sin(angle) + v[1] * math.cos(angle)
        ])
    
    # Test with simple angles proportional to position
    angles = t_small * 0.5  # θ ∝ position
    vectors = []
    
    for pos in range(seq_len_small):
        v = torch.tensor([1.0, 0.0])  # Start as [1, 0]
        v_rot = rotate_vector(v, angles[pos])
        vectors.append(v_rot)
    
    # Compute similarities
    similarity_small = torch.zeros(seq_len_small, seq_len_small)
    for i in range(seq_len_small):
        for j in range(seq_len_small):
            sim = torch.dot(vectors[i], vectors[j]).item()
            similarity_small[i, j] = sim
    
    im5 = ax5.imshow(similarity_small.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    
    for i in range(seq_len_small):
        for j in range(seq_len_small):
            ax5.text(j, i, f'{similarity_small[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(similarity_small[i,j]) > 0.5 else 'black',
                    fontsize=8)
    
    ax5.set_xlabel('j')
    ax5.set_ylabel('i')
    ax5.set_title(f'Simple 2D rotation example\nθ ∝ position, S[i,j] = cos(θ_i - θ_j)')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Mathematical analysis
    ax6 = axes[1, 2]
    
    # Check: S[i,j] should equal S[i+1,j+1]
    test_pairs = []
    for start_i in [2, 4, 6]:
        for start_j in [1, 3, 5]:
            if start_i < seq_len-2 and start_j < seq_len-2:
                s1 = similarity[start_i, start_j].item()
                s2 = similarity[start_i+1, start_j+1].item()
                s3 = similarity[start_i+2, start_j+2].item()
                diff12 = abs(s1 - s2)
                diff23 = abs(s2 - s3)
                test_pairs.append((start_i, start_j, s1, s2, s3, diff12, diff23))
    
    # Plot the consistency check
    n_tests = len(test_pairs)
    x_pos = np.arange(n_tests)
    widths = [0.2, 0.2, 0.2]
    
    for idx, (i, j, s1, s2, s3, d12, d23) in enumerate(test_pairs):
        ax6.bar(idx - 0.3, s1, width=0.2, color='blue', alpha=0.7)
        ax6.bar(idx, s2, width=0.2, color='green', alpha=0.7)
        ax6.bar(idx + 0.3, s3, width=0.2, color='red', alpha=0.7)
        
        ax6.text(idx - 0.3, s1/2, f'{s1:.3f}', ha='center', va='center', 
                color='white', fontsize=7)
        ax6.text(idx, s2/2, f'{s2:.3f}', ha='center', va='center',
                color='white', fontsize=7)
        ax6.text(idx + 0.3, s3/2, f'{s3:.3f}', ha='center', va='center',
                color='white', fontsize=7)
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'({i},{j})' for i, j, *_ in test_pairs])
    ax6.set_ylabel('Similarity')
    ax6.set_title('Checking S[i,j] = S[i+1,j+1]\nBlue:(i,j), Green:(i+1,j+1), Red:(i+2,j+2)')
    ax6.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Detailed analysis
    print("\n" + "=" * 70)
    print("DIAGONAL PATTERN ANALYSIS")
    print("=" * 70)
    
    # Check if we have 45° or 135° diagonals
    print("\n1. Testing diagonal direction:")
    
    # Method: Check if values are constant along j = i - k lines
    max_k = 3
    for k in range(-max_k, max_k+1):
        diag_vals = torch.diagonal(similarity, offset=k)
        mean_val = diag_vals.mean().item()
        std_val = diag_vals.std().item()
        
        print(f"   Diagonal k={k:2d} (i-j={k}): mean={mean_val:.6f}, std={std_val:.6f}", 
              "✓" if std_val < 0.01 else "✗")
    
    print("\n2. Testing WRONG pattern (i+j):")
    for k in range(0, 2*max_k+1):
        # Extract anti-diagonal: i+j = k
        anti_diag_vals = []
        for i in range(seq_len):
            j = k - i
            if 0 <= j < seq_len:
                anti_diag_vals.append(similarity[i, j].item())
        
        if anti_diag_vals:
            mean_val = np.mean(anti_diag_vals)
            std_val = np.std(anti_diag_vals)
            print(f"   Anti-diagonal i+j={k}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    print("\n3. Conclusion:")
    if torch.diagonal(similarity, offset=1).std() < 0.01:
        print("   ✓ RoPE shows 45° diagonals: S[i,j] depends on i-j")
    else:
        print("   ✗ Something is wrong - check implementation")
    
    print("=" * 70)


if __name__ == "__main__":
    visualize_correct_diagonals()