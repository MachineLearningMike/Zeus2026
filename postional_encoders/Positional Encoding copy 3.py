import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

# Minimal RoPE implementation
def apply_rope_1d(x, positions, dim, base=10000.0):
    """Apply 1D RoPE to x at given positions."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    x_rot = torch.zeros_like(x)
    for idx, pos in enumerate(positions):
        freqs = pos * inv_freq
        cos_vals = torch.cos(freqs).repeat_interleave(2)[:dim]
        sin_vals = torch.sin(freqs).repeat_interleave(2)[:dim]
        
        # Apply rotation to pairs
        x_rot[idx, 0::2] = x[idx, 0::2] * cos_vals[0::2] - x[idx, 1::2] * sin_vals[0::2]
        x_rot[idx, 1::2] = x[idx, 1::2] * cos_vals[0::2] + x[idx, 0::2] * sin_vals[0::2]
    
    return x_rot

def test_rope_diagonals():
    """Test what pattern RoPE actually produces."""
    
    dim = 64
    seq_len = 10
    
    # Create unit vectors at all positions
    positions = torch.arange(seq_len)
    
    # All queries and keys start as same vector: [1, 0, 0, ...]
    Q = torch.zeros(seq_len, dim)
    K = torch.zeros(seq_len, dim)
    Q[:, 0] = 1.0
    K[:, 0] = 1.0
    
    # Apply RoPE
    Q_rot = apply_rope_1d(Q, positions, dim)
    K_rot = apply_rope_1d(K, positions, dim)
    
    # Compute similarity matrix
    similarity = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            similarity[i, j] = torch.dot(Q_rot[i], K_rot[j])
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Show similarity matrix
    ax1 = axes[0]
    im1 = ax1.imshow(similarity.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Key position j')
    ax1.set_ylabel('Query position i')
    ax1.set_title('RoPE Similarity Matrix')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Annotate with i-j values
    ax2 = axes[1]
    im2 = ax2.imshow(similarity.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    
    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos = i - j
            ax2.text(j, i, f'{rel_pos:+d}', ha='center', va='center',
                    color='white' if abs(similarity[i,j]) > 0.5 else 'black',
                    fontsize=8)
    
    # Draw lines for constant i-j
    for k in range(-seq_len+1, seq_len):
        # Points where i - j = k
        points = []
        for i in range(seq_len):
            j = i - k
            if 0 <= j < seq_len:
                points.append((j, i))
        
        if len(points) > 1:
            x_vals, y_vals = zip(*points)
            ax2.plot(x_vals, y_vals, 'w-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Key position j')
    ax2.set_ylabel('Query position i')
    ax2.set_title('Annotated with i-j values\nWhite lines: constant i-j')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Annotate with i+j values
    ax3 = axes[2]
    im3 = ax3.imshow(similarity.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1)
    
    for i in range(seq_len):
        for j in range(seq_len):
            sum_pos = i + j
            ax3.text(j, i, f'{sum_pos}', ha='center', va='center',
                    color='white' if abs(similarity[i,j]) > 0.5 else 'black',
                    fontsize=8)
    
    # Draw lines for constant i+j
    for k in range(0, 2*seq_len):
        # Points where i + j = k
        points = []
        for i in range(seq_len):
            j = k - i
            if 0 <= j < seq_len:
                points.append((j, i))
        
        if len(points) > 1:
            x_vals, y_vals = zip(*points)
            ax3.plot(x_vals, y_vals, 'w-', alpha=0.3, linewidth=1)
    
    ax3.set_xlabel('Key position j')
    ax3.set_ylabel('Query position i')
    ax3.set_title('Annotated with i+j values\nWhite lines: constant i+j')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze which pattern matches
    print("=" * 70)
    print("ANALYZING DIAGONAL PATTERN")
    print("=" * 70)
    
    # Check variance along i-j diagonals
    print("\nChecking i-j diagonals:")
    for k in [-3, -2, -1, 0, 1, 2, 3]:
        diag_vals = torch.diagonal(similarity, offset=k)
        if len(diag_vals) > 0:
            mean_val = diag_vals.mean().item()
            std_val = diag_vals.std().item()
            print(f"  i-j = {k:2d}: mean = {mean_val:.6f}, std = {std_val:.6f}")
    
    # Check variance along i+j anti-diagonals
    print("\nChecking i+j anti-diagonals:")
    for k in [3, 5, 7, 9, 11, 13, 15]:
        anti_diag_vals = []
        for i in range(seq_len):
            j = k - i
            if 0 <= j < seq_len:
                anti_diag_vals.append(similarity[i, j].item())
        
        if anti_diag_vals:
            mean_val = np.mean(anti_diag_vals)
            std_val = np.std(anti_diag_vals)
            print(f"  i+j = {k:2d}: mean = {mean_val:.6f}, std = {std_val:.6f}")
    
    # Let's also check the actual RoPE formula
    print("\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS")
    print("=" * 70)
    
    # For a simple 2D case, let's derive the formula
    dim = 2
    base = 10000.0
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    print(f"\nFor dim={dim}, base={base}:")
    print(f"inv_freq = {inv_freq}")
    
    # Position m: angle = m * inv_freq
    # Position n: angle = n * inv_freq
    m, n = 1, 3
    theta_m = m * inv_freq[0]
    theta_n = n * inv_freq[0]
    
    print(f"\nPosition m={m}: θ_m = {theta_m:.6f}")
    print(f"Position n={n}: θ_n = {theta_n:.6f}")
    
    # Start with vector [1, 0]
    q_m = torch.tensor([math.cos(theta_m), math.sin(theta_m)])
    k_n = torch.tensor([math.cos(theta_n), math.sin(theta_n)])
    
    similarity_mn = torch.dot(q_m, k_n).item()
    print(f"\nq_m = {q_m.tolist()}")
    print(f"k_n = {k_n.tolist()}")
    print(f"⟨q_m, k_n⟩ = {similarity_mn:.6f}")
    
    # Cosine difference formula
    cos_diff = math.cos(theta_m - theta_n)
    print(f"cos(θ_m - θ_n) = {cos_diff:.6f}")
    print(f"Are they equal? {abs(similarity_mn - cos_diff) < 1e-10}")
    
    print(f"\n∴ ⟨RoPE(q,m), RoPE(k,n)⟩ = cos(θ_m - θ_n)")
    print("   = cos((m-n) * inv_freq)")
    print("   Depends ONLY on m-n, not m+n")

if __name__ == "__main__":
    test_rope_diagonals()