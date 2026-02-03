import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

class GaussianDistanceEncoding(nn.Module):
    """
    Distance encoding using Gaussian/RBF kernels.
    Similarity = exp(-γ * |i-j|²)
    """
    def __init__(self, dim, num_kernels=16, max_distance=50):
        super().__init__()
        self.dim = dim
        self.num_kernels = num_kernels
        
        # Learnable kernel centers and widths
        self.centers = nn.Parameter(torch.linspace(0, max_distance, num_kernels))
        self.widths = nn.Parameter(torch.ones(num_kernels) * (max_distance / num_kernels))
        
        # Projection to dimension
        self.projection = nn.Linear(num_kernels, dim)
        
    def forward(self, seq_len):
        # Create distance matrix
        positions = torch.arange(seq_len).float()
        distance = torch.abs(positions[:, None] - positions[None, :])  # (seq_len, seq_len)
        
        # Compute Gaussian kernels
        # shape: (seq_len, seq_len, num_kernels)
        kernels = torch.exp(
            -((distance.unsqueeze(-1) - self.centers) ** 2) / 
            (2 * self.widths ** 2)
        )
        
        # Project to embedding dimension
        encoding = self.projection(kernels)  # (seq_len, seq_len, dim)
        
        return encoding

def visualize_distance_encoding():
    # Create the encoding module
    dim = 64
    num_kernels = 16
    seq_len = 50
    
    encoding_module = GaussianDistanceEncoding(dim=dim, num_kernels=num_kernels)
    
    # Get the encoding
    encoding = encoding_module(seq_len)  # (50, 50, 64)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Distance matrix
    ax1 = plt.subplot(2, 3, 1)
    positions = torch.arange(seq_len).float()
    distance_matrix = torch.abs(positions[:, None] - positions[None, :])
    im1 = ax1.imshow(distance_matrix, cmap='viridis')
    ax1.set_title('Distance Matrix |i-j|')
    ax1.set_xlabel('Position j')
    ax1.set_ylabel('Position i')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Gaussian kernels for specific positions
    ax2 = plt.subplot(2, 3, 2)
    # Plot kernels for position 0 and position 25
    kernels = torch.exp(
        -((distance_matrix.unsqueeze(-1) - encoding_module.centers) ** 2) / 
        (2 * encoding_module.widths ** 2)
    )
    
    # Get kernels for position 0 (how it sees all other positions)
    pos0_kernels = kernels[0].detach().numpy()  # (seq_len, num_kernels)
    pos25_kernels = kernels[25].detach().numpy()  # (seq_len, num_kernels)
    
    # Plot a few selected kernels
    for k in range(0, num_kernels, 3):  # Plot every 3rd kernel
        ax2.plot(pos0_kernels[:, k], alpha=0.6, label=f'Kernel {k}' if k == 0 else "")
    
    ax2.set_title(f'Gaussian Kernels (Position 0)')
    ax2.set_xlabel('Position j')
    ax2.set_ylabel('Kernel Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 2D heatmap of kernel activations for position 0
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(pos0_kernels.T, aspect='auto', cmap='hot')
    ax3.set_title('All Kernels for Position 0')
    ax3.set_xlabel('Position j')
    ax3.set_ylabel('Kernel Index')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Encoded similarity matrix (cosine similarity between encodings)
    ax4 = plt.subplot(2, 3, 4)
    
    # Get the full encoding
    full_encoding = encoding.detach().numpy()  # (50, 50, 64)
    
    # Compute cosine similarity between position encodings
    # Flatten the encoding for each position pair
    similarity_matrix = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            # Cosine similarity between encoding[i, j] and encoding[i, j]
            # Actually, let's look at how position i encodes its relationship to j
            # compared to how position k encodes its relationship to l
            # For simplicity, show the norm of the encoding
            similarity_matrix[i, j] = np.linalg.norm(full_encoding[i, j])
    
    im4 = ax4.imshow(similarity_matrix, cmap='plasma')
    ax4.set_title('Encoding Strength Matrix')
    ax4.set_xlabel('Position j')
    ax4.set_ylabel('Position i')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Slice through the encoding dimension
    ax5 = plt.subplot(2, 3, 5)
    # Show how one dimension varies across positions
    dim_slice = 0
    encoding_dim0 = full_encoding[:, :, dim_slice]
    im5 = ax5.imshow(encoding_dim0, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax5.set_title(f'Encoding Dimension {dim_slice}')
    ax5.set_xlabel('Position j')
    ax5.set_ylabel('Position i')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Distance vs Encoding strength
    ax6 = plt.subplot(2, 3, 6)
    
    # Flatten distance and encoding strength
    distances = distance_matrix.flatten().numpy()
    encoding_strengths = similarity_matrix.flatten()
    
    # Sort by distance for cleaner plot
    sort_idx = np.argsort(distances)
    distances_sorted = distances[sort_idx]
    strengths_sorted = encoding_strengths[sort_idx]
    
    # Use moving average for smoother plot
    window_size = 50
    distances_ma = np.convolve(distances_sorted, np.ones(window_size)/window_size, mode='valid')
    strengths_ma = np.convolve(strengths_sorted, np.ones(window_size)/window_size, mode='valid')
    
    ax6.plot(distances_ma, strengths_ma, 'b-', linewidth=2)
    ax6.set_title('Distance vs Encoding Strength')
    ax6.set_xlabel('Distance |i-j|')
    ax6.set_ylabel('Encoding Norm')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Gaussian Distance Encoding Visualization', fontsize=16, y=1.02)
    plt.show()
    
    # Additional analysis
    print("=" * 60)
    print("ANALYSIS OF GAUSSIAN DISTANCE ENCODING")
    print("=" * 60)
    print(f"\n1. Shape of encoding: {encoding.shape}")
    print(f"   - Each position pair (i,j) gets a {dim}-dimensional encoding")
    print(f"   - The encoding captures the distance relationship between i and j")
    
    print(f"\n2. Distance-dependence properties:")
    print(f"   - Symmetric: encoding[i,j] should be similar to encoding[j,i]")
    print(f"   - Distance-based: encoding depends only on |i-j|")
    print(f"   - Banded structure: encoding strength decreases with distance")
    
    print(f"\n3. Kernel parameters (learnable):")
    print(f"   - Centers: {encoding_module.centers.detach().numpy()}")
    print(f"   - Widths: {encoding_module.widths.detach().numpy()}")
    
    # Verify symmetry
    test_i, test_j = 10, 30
    encoding_ij = encoding[test_i, test_j]
    encoding_ji = encoding[test_j, test_i]
    similarity = torch.cosine_similarity(encoding_ij.unsqueeze(0), encoding_ji.unsqueeze(0)).item()
    print(f"\n4. Symmetry test (positions {test_i} and {test_j}):")
    print(f"   Cosine similarity between encoding[{test_i},{test_j}] and encoding[{test_j},{test_i}]: {similarity:.4f}")

# Run the visualization
visualize_distance_encoding()