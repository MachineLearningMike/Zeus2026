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

def visualize_rope_bottom_left_origin():
    """Visualize RoPE similarity with origin at bottom-left."""
    
    print("=" * 70)
    print("RoPE Similarity with Bottom-Left Origin")
    print("=" * 70)
    print("\nCoordinate system:")
    print("  • Origin at bottom-left")
    print("  • X-axis: Key position j → increasing right")
    print("  • Y-axis: Query position i → increasing UP")
    print("  • Lines of constant i-j have slope = 1 (45°)")
    print("=" * 70)
    
    dim = 64
    seq_len = 15
    
    # Create RoPE module
    rope = RotaryPositionalEmbedding1D(dim)
    
    # Create queries and keys
    Q = torch.zeros(1, seq_len, 1, 1, dim)
    K = torch.zeros(1, seq_len, 1, 1, dim)
    Q[..., 0] = 1.0  # Unit vector in first dimension
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
            similarity[i, j] = torch.dot(Q_rot[i], K_rot[j])
    
    sim_np = similarity.cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Standard imshow (top-left origin) - what most people see
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(sim_np, cmap='RdBu_r', vmin=-1, vmax=1,
                    origin='upper', extent=[0, seq_len, seq_len, 0])
    ax1.set_xlabel('Key position j')
    ax1.set_ylabel('Query position i')
    ax1.set_title('Standard imshow (origin at TOP-LEFT)\n135° lines appear as i-j = constant')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im1, ax=ax1)
    
    # Annotate some points
    for i in [3, 7, 11]:
        for j in [3, 7, 11]:
            ax1.text(j+0.5, i+0.5, f'{i-j:+d}', ha='center', va='center',
                    color='white' if abs(sim_np[i, j]) > 0.5 else 'black',
                    fontsize=8, fontweight='bold')
    
    # 2. Bottom-left origin plot - Cartesian coordinates
    ax2 = plt.subplot(2, 3, 2)
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
    
    # Plot with pcolormesh for bottom-left origin
    im2 = ax2.pcolormesh(X, Y, sim_np, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    
    ax2.set_xlabel('Key position j')
    ax2.set_ylabel('Query position i')
    ax2.set_title('BOTTOM-LEFT origin (Cartesian)\n45° lines = i-j = constant')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im2, ax=ax2)
    
    # Draw 45° lines (i - j = constant)
    for k in [-6, -3, 0, 3, 6]:
        x_vals = np.linspace(max(0, -k), min(seq_len, seq_len - k), 100)
        y_vals = x_vals + k
        ax2.plot(x_vals, y_vals, 'k-', alpha=0.7, linewidth=2,
                label=f'i-j={k}' if abs(k) <= 3 else None)
    
    # Annotate
    for i in [3, 7, 11]:
        for j in [3, 7, 11]:
            ax2.text(j, i, f'{i-j:+d}', ha='center', va='center',
                    color='white' if abs(sim_np[i, j]) > 0.5 else 'black',
                    fontsize=8, fontweight='bold')
    
    # 3. Compare the two coordinate systems side by side
    ax3 = plt.subplot(2, 3, 3)
    
    # Create a transformation diagram
    # Left: Cartesian (bottom-left origin)
    # Right: Matrix/Image (top-left origin)
    
    # Draw coordinate axes
    ax3.axhline(y=0.5, color='k', linestyle='-', linewidth=2)
    ax3.axvline(x=0.5, color='k', linestyle='-', linewidth=2)
    
    # Label the transformation
    ax3.text(0.25, 0.9, 'Cartesian:\n(0,0) at bottom-left\ny increases UP', 
            ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax3.text(0.75, 0.9, 'Matrix/Image:\n(0,0) at top-left\ny increases DOWN', 
            ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Draw a line in Cartesian: y = x (45° line)
    cartesian_line_x = [0.2, 0.4]
    cartesian_line_y = [0.2, 0.4]
    ax3.plot(cartesian_line_x, cartesian_line_y, 'r-', linewidth=3, label='Cartesian: y = x (45°)')
    
    # Transform to matrix coordinates: y_matrix = 1 - y_cartesian
    matrix_line_x = [0.6, 0.8]
    matrix_line_y = [0.8, 0.6]  # 1 - [0.2, 0.4] = [0.8, 0.6]
    ax3.plot(matrix_line_x, matrix_line_y, 'b-', linewidth=3, label='Matrix: appears as 135°')
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.legend(loc='lower center')
    ax3.set_title('Coordinate Transformation:\n45° in Cartesian ↔ 135° in Matrix')
    ax3.axis('off')
    
    # 4. Mathematical verification: check diagonal constancy
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate standard deviation along each diagonal
    k_values = []
    std_values = []
    mean_values = []
    
    for k in range(-seq_len+1, seq_len):
        diag_vals = np.diag(sim_np, k=k)
        if len(diag_vals) > 3:  # Only consider diagonals with enough points
            std_val = np.std(diag_vals)
            mean_val = np.mean(diag_vals)
            k_values.append(k)
            std_values.append(std_val)
            mean_values.append(mean_val)
    
    # Plot standard deviation vs diagonal offset
    bars = ax4.bar(k_values, std_values, alpha=0.7)
    
    # Color bars based on std value
    for bar, std_val in zip(bars, std_values):
        if std_val < 0.01:
            bar.set_color('green')
        elif std_val < 0.05:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    ax4.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Threshold (0.01)')
    ax4.set_xlabel('Diagonal offset k = i - j')
    ax4.set_ylabel('Standard deviation along diagonal')
    ax4.set_title('Checking constancy along diagonals\n(Green = constant, Red = varies)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Plot similarity vs relative position (i-j)
    ax5 = plt.subplot(2, 3, 5)
    
    # Collect all pairs with same i-j
    rel_pos_dict = {}
    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos = i - j
            if rel_pos not in rel_pos_dict:
                rel_pos_dict[rel_pos] = []
            rel_pos_dict[rel_pos].append(sim_np[i, j])
    
    # Plot
    rel_positions = sorted(rel_pos_dict.keys())
    means = [np.mean(rel_pos_dict[p]) for p in rel_positions]
    stds = [np.std(rel_pos_dict[p]) for p in rel_positions]
    
    ax5.errorbar(rel_positions, means, yerr=stds, fmt='o-', capsize=4,
                linewidth=2, markersize=6, label='Mean ± Std')
    
    ax5.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='i=j (self)')
    ax5.set_xlabel('Relative position: i - j')
    ax5.set_ylabel('Similarity')
    ax5.set_title('RoPE: Similarity depends ONLY on i-j')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Simple 2D example to illustrate the math
    ax6 = plt.subplot(2, 3, 6)
    
    # Create a simple 2D rotation example
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Start with vector [1, 0]
    vectors = []
    for angle in angles:
        x = np.cos(angle)
        y = np.sin(angle)
        vectors.append([x, y])
    
    vectors = np.array(vectors)
    
    # Compute similarities
    simple_sim = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            simple_sim[i, j] = np.dot(vectors[i], vectors[j])
    
    # Plot
    X_simple, Y_simple = np.meshgrid(range(8), range(8))
    im6 = ax6.pcolormesh(X_simple, Y_simple, simple_sim, cmap='RdBu_r', 
                        vmin=-1, vmax=1, shading='auto')
    
    # Draw 45° lines
    for k in [-2, 0, 2]:
        x_vals = np.linspace(max(0, -k), min(7, 7 - k), 100)
        y_vals = x_vals + k
        ax6.plot(x_vals, y_vals, 'k-', alpha=0.7, linewidth=2)
    
    ax6.set_xlabel('Position j')
    ax6.set_ylabel('Position i')
    ax6.set_title('Simple 2D rotation example\nS[i,j] = cos(θ_i - θ_j)')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check main diagonal (i=j)
    main_diag = np.diag(sim_np)
    print(f"\n1. Main diagonal (i=j):")
    print(f"   Mean similarity: {main_diag.mean():.6f} (should be ~1.0)")
    print(f"   Std deviation: {main_diag.std():.6f} (should be ~0.0)")
    
    # Check off-diagonals
    print(f"\n2. Off-diagonals (example):")
    for k in [-3, -1, 1, 3]:
        diag = np.diag(sim_np, k=k)
        if len(diag) > 0:
            print(f"   i-j = {k:2d}: mean = {diag.mean():.6f}, std = {diag.std():.6f}")
    
    print(f"\n3. Coordinate system conclusion:")
    print(f"   • In Cartesian (bottom-left origin): 45° lines = constant i-j")
    print(f"   • In Matrix/Image (top-left origin): 135° lines = constant i-j")
    print(f"   • Both show the SAME pattern, just viewed differently")
    
    print("\n4. RoPE property confirmed:")
    if all(np.diag(sim_np, k=1).std() < 0.01 for k in [-2, -1, 0, 1, 2]):
        print("   ✓ Similarity depends only on i-j (relative position)")
    else:
        print("   ✗ Something is wrong with the implementation")
    
    print("=" * 70)

if __name__ == "__main__":
    visualize_rope_bottom_left_origin()