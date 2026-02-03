import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class SymmetricRoPE1D(nn.Module):
    """
    Simple symmetric RoPE where similarity depends on abs(n-m).
    This is achieved by using absolute position values.
    """
    
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, seq_dim=1):
        """
        Apply symmetric RoPE.
        Using absolute positions starting from 0.
        """
        seq_len = x.shape[seq_dim]
        
        # Use absolute positions: 0, 1, 2, ..., seq_len-1
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        freqs = freqs.repeat_interleave(2, dim=-1)
        
        # Expand for broadcasting
        view_shape = [1] * x.ndim
        view_shape[seq_dim] = seq_len
        view_shape[-1] = self.dim
        freqs = freqs.view(*view_shape)
        
        # Apply rotation
        x_rot = torch.empty_like(x)
        cos_f = torch.cos(freqs)
        sin_f = torch.sin(freqs)
        
        x_rot[..., 0::2] = x[..., 0::2] * cos_f[..., 0::2] - x[..., 1::2] * sin_f[..., 0::2]
        x_rot[..., 1::2] = x[..., 1::2] * cos_f[..., 0::2] + x[..., 0::2] * sin_f[..., 0::2]
        
        return x_rot


def visualize_symmetric_rope():
    """Visualize symmetric vs standard RoPE."""
    
    print("=" * 70)
    print("Symmetric RoPE Analysis")
    print("=" * 70)
    print("\nActually, standard RoPE with positions 0,1,2,...")
    print("ALREADY gives S[i,j] that depends on i-j, not abs(i-j)")
    print("\nLet's verify this mathematically...")
    print("=" * 70)
    
    dim = 64
    seq_len = 10
    
    # Create RoPE
    rope = SymmetricRoPE1D(dim)
    
    # Create unit vectors
    Q = torch.zeros(1, seq_len, 1, 1, dim)
    K = torch.zeros(1, seq_len, 1, 1, dim)
    Q[..., 0] = 1.0
    K[..., 0] = 1.0
    
    # Apply RoPE
    Q_rot = rope(Q, seq_dim=1)
    K_rot = rope(K, seq_dim=1)
    
    Q_rot = Q_rot[0, :, 0, 0, :]
    K_rot = K_rot[0, :, 0, 0, :]
    
    # Compute similarity
    similarity = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            similarity[i, j] = torch.dot(Q_rot[i], K_rot[j])
    
    sim_np = similarity.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Show similarity matrix
    ax1 = axes[0, 0]
    X, Y = np.meshgrid(np.arange(seq_len), np.arange(seq_len))
    im1 = ax1.pcolormesh(X, Y, sim_np, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    
    # Annotate with i-j values
    for i in range(seq_len):
        for j in range(seq_len):
            ax1.text(j, i, f'{i-j:+d}', ha='center', va='center',
                    color='white' if abs(sim_np[i, j]) > 0.5 else 'black',
                    fontsize=8)
    
    ax1.set_xlabel('Key position j')
    ax1.set_ylabel('Query position i')
    ax1.set_title(f'RoPE Similarity Matrix\nAnnotations show i-j')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Draw lines for constant i-j (45° lines)
    for k in [-3, 0, 3]:
        x_vals = np.linspace(max(0, -k), min(seq_len-1, seq_len-1-k), 100)
        y_vals = x_vals + k
        ax1.plot(x_vals, y_vals, 'k-', alpha=0.5, linewidth=1)
    
    # 2. Show symmetry check
    ax2 = axes[0, 1]
    
    # Check if S[i,j] = S[j,i]
    symmetry_check = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            symmetry_check[i, j] = abs(sim_np[i, j] - sim_np[j, i])
    
    im2 = ax2.imshow(symmetry_check, cmap='hot', vmin=0, vmax=0.01)
    ax2.set_xlabel('j')
    ax2.set_ylabel('i')
    ax2.set_title('Symmetry check: |S[i,j] - S[j,i]|\n(Should be 0 if symmetric)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Check pairs with same |i-j|
    ax3 = axes[0, 2]
    
    test_cases = []
    # For distance 2: (0,2) vs (2,0), (1,3) vs (3,1), etc.
    distances = [1, 2, 3]
    
    for d in distances:
        pairs_same_dist = []
        for i in range(seq_len - d):
            pairs_same_dist.append(((i, i+d), (i+d, i)))
        
        # Take first few pairs
        for (i1, j1), (i2, j2) in pairs_same_dist[:3]:
            test_cases.append({
                'distance': d,
                'pair1': (i1, j1, sim_np[i1, j1]),
                'pair2': (i2, j2, sim_np[i2, j2]),
                'difference': abs(sim_np[i1, j1] - sim_np[i2, j2])
            })
    
    # Plot the test cases
    x_pos = np.arange(len(test_cases))
    pair1_vals = [tc['pair1'][2] for tc in test_cases]
    pair2_vals = [tc['pair2'][2] for tc in test_cases]
    differences = [tc['difference'] for tc in test_cases]
    
    width = 0.35
    ax3.bar(x_pos - width/2, pair1_vals, width, label='S[i,j] (i<j)', alpha=0.7, color='blue')
    ax3.bar(x_pos + width/2, pair2_vals, width, label='S[j,i] (i>j)', alpha=0.7, color='red')
    
    # Add difference as text
    for idx, diff in enumerate(differences):
        ax3.text(x_pos[idx], max(pair1_vals[idx], pair2_vals[idx]) + 0.05,
                f'diff={diff:.4f}', ha='center', fontsize=8)
    
    ax3.set_xlabel('Test case')
    ax3.set_ylabel('Similarity')
    ax3.set_title('Checking S[i,j] vs S[j,i] for same |i-j|')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'd={tc["distance"]}\n({tc["pair1"][0]},{tc["pair1"][1]}) vs ({tc["pair2"][0]},{tc["pair2"][1]})' 
                        for tc in test_cases], fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mathematical analysis
    ax4 = axes[1, 0]
    
    # Let's derive the formula
    # For a single 2D rotation:
    # v(θ) = [cosθ, sinθ] if starting as [1, 0]
    # ⟨v(θ₁), v(θ₂)⟩ = cosθ₁·cosθ₂ + sinθ₁·sinθ₂ = cos(θ₁ - θ₂)
    
    # For RoPE: θ_i = i * ω, where ω = inv_freq
    # So ⟨v(i), v(j)⟩ = cos(iω - jω) = cos((i-j)ω)
    
    # This is NOT symmetric unless cos((i-j)ω) = cos((j-i)ω)
    # But cos(-x) = cos(x), so actually it IS symmetric!
    
    ax4.text(0.5, 0.7, 'Mathematical Analysis:', ha='center', fontsize=12, fontweight='bold')
    ax4.text(0.5, 0.5, 
            'For single frequency ω:\n'
            'θ_i = i·ω\n'
            '⟨v(i), v(j)⟩ = cos(θ_i - θ_j)\n'
            '             = cos((i-j)·ω)\n'
            'Since cos(-x) = cos(x):\n'
            'cos((i-j)·ω) = cos((j-i)·ω)\n'
            '∴ S[i,j] = S[j,i]',
            ha='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax4.text(0.5, 0.2, 
            'Standard RoPE IS symmetric!\n'
            'S[i,j] depends on (i-j) but\n'
            'cos((i-j)ω) = cos((j-i)ω)',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    # 5. Verify with actual computation
    ax5 = axes[1, 1]
    
    # Test specific pairs
    test_pairs = [
        (2, 5),  # i-j = -3
        (5, 2),  # i-j = 3
        (1, 4),  # i-j = -3
        (4, 1),  # i-j = 3
        (3, 8),  # i-j = -5
        (8, 3),  # i-j = 5
    ]
    
    similarities = []
    for i, j in test_pairs:
        similarities.append(sim_np[i, j])
    
    x_pos = np.arange(len(test_pairs))
    colors = ['blue' if i < j else 'red' for i, j in test_pairs]
    
    bars = ax5.bar(x_pos, similarities, color=colors, alpha=0.7)
    
    # Annotate bars
    for idx, ((i, j), sim) in enumerate(zip(test_pairs, similarities)):
        ax5.text(x_pos[idx], sim/2, f'({i},{j})\n{sim:.4f}', 
                ha='center', va='center', color='white', fontsize=8)
    
    ax5.set_xlabel('Test pair (i,j)')
    ax5.set_ylabel('Similarity S[i,j]')
    ax5.set_title('Pairs with same |i-j| should have same similarity\nBlue: i<j, Red: i>j')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([f'({i},{j})' for i, j in test_pairs])
    ax5.grid(True, alpha=0.3)
    
    # 6. Show cosine property
    ax6 = axes[1, 2]
    
    # Let's compute ω from RoPE
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ω = inv_freq[0].item()  # First frequency
    
    # Generate cosine curve
    x_vals = np.linspace(-seq_len+1, seq_len-1, 1000)
    y_vals = np.cos(ω * x_vals)
    
    ax6.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'cos(ω·x), ω={ω:.6f}')
    
    # Mark specific i-j values
    marked_vals = [-5, -3, -1, 0, 1, 3, 5]
    for x in marked_vals:
        y = np.cos(ω * x)
        ax6.scatter([x], [y], color='red', s=50, zorder=5)
        ax6.text(x, y+0.1, f'({x}, {y:.3f})', ha='center', fontsize=8)
    
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax6.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Relative position: i - j')
    ax6.set_ylabel('Similarity')
    ax6.set_title('RoPE similarity = cos(ω·(i-j))\nSymmetric because cos(-x) = cos(x)')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "=" * 70)
    print("MATHEMATICAL TRUTH REVEALED")
    print("=" * 70)
    
    print("\nStandard RoPE formula:")
    print("  For a single frequency component ω:")
    print("  ⟨RoPE(v, i), RoPE(v, j)⟩ = cos(ω·(i - j))")
    
    print("\nKey observation:")
    print("  cos(ω·(i - j)) = cos(ω·(j - i)) because cos(-x) = cos(x)")
    print("  Therefore: S[i,j] = S[j,i]")
    
    print("\nSo actually:")
    print("  Standard RoPE ALREADY gives symmetric similarity!")
    print("  S[i,j] depends on (i-j) but the cosine makes it symmetric")
    
    print("\nProof:")
    ω = inv_freq[0].item()
    print(f"  First frequency ω = {ω:.6f}")
    
    for (i1, j1), (i2, j2) in [((2,5), (5,2)), ((1,4), (4,1))]:
        s1 = sim_np[i1, j1]
        s2 = sim_np[i2, j2]
        calc = math.cos(ω * (i1 - j1))
        print(f"\n  Test: S[{i1},{j1}] vs S[{j1},{i1}]")
        print(f"    S[{i1},{j1}] = {s1:.6f}")
        print(f"    S[{j1},{i1}] = {s2:.6f}")
        print(f"    Difference = {abs(s1 - s2):.6f}")
        print(f"    cos(ω·({i1}-{j1})) = cos({ω:.6f}·{i1-j1}) = {calc:.6f}")
    
    print("\nConclusion:")
    print("  Standard RoPE with positions 0,1,2,...")
    print("  gives S[i,j] = cos(ω·(i-j)) which IS symmetric")
    print("  because cosine is an even function: cos(-x) = cos(x)")
    
    print("\nWait, but earlier you said...")
    print("  Actually, let me check if there are multiple frequencies...")
    
    # Check multiple frequencies
    print("\nMultiple frequencies:")
    for freq_idx in range(min(3, len(inv_freq))):
        ω_k = inv_freq[freq_idx].item()
        print(f"  Frequency {freq_idx}: ω_{freq_idx} = {ω_k:.6f}")
    
    print("\nFull RoPE similarity = sum of cosines at different frequencies")
    print("  Still symmetric because each cosine term is symmetric!")
    
    print("=" * 70)

if __name__ == "__main__":
    visualize_symmetric_rope()