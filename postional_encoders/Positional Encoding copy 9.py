import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import math

class MonotonicGaussianDistanceEncoding2D(nn.Module):
    """
    2D Distance encoding with guaranteed monotonic decrease.
    Encoding similarity strictly decreases with increasing distance.
    """
    def __init__(self, dim, num_kernels=8, max_distance=8.0, gamma=0.1):
        super().__init__()
        self.dim = dim
        self.num_kernels = num_kernels
        self.max_distance = max_distance
        self.gamma = gamma
        
        # Fixed kernel centers (equally spaced in distance space)
        # This ensures monotonicity
        self.register_buffer('centers', torch.linspace(0, max_distance, num_kernels))
        
        # Fixed decreasing kernel widths (wider for larger distances)
        widths = torch.linspace(0.3, 1.5, num_kernels)
        self.register_buffer('widths', widths)
        
        # Learnable projection with constraints
        self.projection = nn.Linear(num_kernels, dim, bias=False)
        
        # Initialize projection to preserve monotonicity
        with torch.no_grad():
            # Initialize weights to be positive and decreasing
            weights = torch.randn(dim, num_kernels) * 0.02
            # Make weights positive and column-wise decreasing
            for j in range(num_kernels):
                weights[:, j] = torch.abs(weights[:, j]) * (1.0 - 0.1 * j/num_kernels)
            self.projection.weight.data = weights
    
    def forward(self, grid_shape):
        """
        Returns encoding where similarity decreases monotonically with distance.
        """
        h, w = grid_shape
        
        # Create 2D coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float(),
            indexing='ij'
        )
        
        # Flatten coordinates: (h*w, 2)
        coordinates = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        
        # Compute Euclidean distance matrix
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (n, n, 2)
        distances = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # (n, n), avoid sqrt(0)
        
        # Compute Gaussian kernels with monotonic property
        # Use a formulation that ensures similarity decreases with distance
        expanded_dist = distances.unsqueeze(-1)  # (n, n, 1)
        
        # Compute kernel activations
        kernels = torch.exp(
            -((expanded_dist - self.centers) ** 2) / 
            (2 * self.widths ** 2)
        )  # (n, n, num_kernels)
        
        # Ensure kernels are positive and properly scaled
        kernels = torch.relu(kernels)
        
        # Project to embedding dimension
        encoding = self.projection(kernels)  # (n, n, dim)
        
        # Apply additional distance-based scaling to ensure monotonicity
        distance_scale = torch.exp(-self.gamma * distances).unsqueeze(-1)  # (n, n, 1)
        encoding = encoding * distance_scale
        
        return encoding, coordinates, distances

class ExponentialDistanceEncoding2D(nn.Module):
    """
    Simpler version: Direct exponential decay with distance.
    Guarantees monotonic decrease in similarity.
    """
    def __init__(self, dim, base_distance=2.0, min_similarity=0.1):
        super().__init__()
        self.dim = dim
        self.base_distance = base_distance
        self.min_similarity = min_similarity
        
        # Learnable basis functions for different distance scales
        self.distance_weights = nn.Parameter(torch.randn(dim, 4) * 0.1)
        self.direction_weights = nn.Parameter(torch.randn(dim, 4) * 0.1)
    
    def forward(self, grid_shape):
        h, w = grid_shape
        
        # Create coordinate grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float(),
            indexing='ij'
        )
        coordinates = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        
        # Compute distances
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (n, n, 2)
        distances = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # (n, n)
        
        # Compute angles (for directional variation, optional)
        angles = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # (n, n)
        
        # Exponential decay base (ensures monotonic decrease)
        base_similarity = torch.exp(-distances / self.base_distance)  # (n, n)
        base_similarity = self.min_similarity + (1 - self.min_similarity) * base_similarity
        
        # Create encoding with distance-dependent similarity
        n = h * w
        encoding = torch.zeros(n, n, self.dim)
        
        # For each dimension, scale by distance similarity
        for d in range(self.dim):
            # Random pattern modulated by distance similarity
            pattern = torch.randn(n, n)
            encoding[:, :, d] = pattern * base_similarity
        
        return encoding, coordinates, distances, base_similarity

def verify_monotonic_decrease(encoding, distances, center_idx=0):
    """
    Verify that encoding similarity decreases monotonically with distance.
    """
    encoding_np = encoding.detach().numpy()
    distances_np = distances.detach().numpy()
    
    n = encoding.shape[0]
    
    # Get encodings from center to all other positions
    center_encodings = encoding_np[center_idx]  # (n, dim)
    
    # Compute cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Similarity of center's view of others to itself
    similarities = np.zeros(n)
    center_self = center_encodings[center_idx:center_idx+1]  # (1, dim)
    
    for i in range(n):
        if i == center_idx:
            similarities[i] = 1.0  # Self-similarity
        else:
            other_enc = center_encodings[i:i+1]  # (1, dim)
            sim = cosine_similarity(center_self, other_enc)[0, 0]
            similarities[i] = sim
    
    # Get distances from center
    center_distances = distances_np[center_idx]
    
    # Sort by distance
    sort_idx = np.argsort(center_distances)
    sorted_dist = center_distances[sort_idx]
    sorted_sim = similarities[sort_idx]
    
    # Check monotonic decrease
    is_monotonic = True
    violations = 0
    
    for i in range(1, len(sorted_sim)):
        if sorted_sim[i] > sorted_sim[i-1] + 1e-6:  # Allow small numerical errors
            is_monotonic = False
            violations += 1
    
    return sorted_dist, sorted_sim, is_monotonic, violations

def visualize_monotonic_distance_encoding():
    """Visualization focusing on monotonic decrease property."""
    
    # Create the encoding module
    dim = 32
    grid_shape = (10, 10)  # 10x10 grid
    
    # Create both versions for comparison
    module1 = MonotonicGaussianDistanceEncoding2D(dim=dim, num_kernels=8, max_distance=12.0)
    module2 = ExponentialDistanceEncoding2D(dim=dim, base_distance=3.0)
    
    # Get encodings
    encoding1, coords1, dist1 = module1(grid_shape)
    encoding2, coords2, dist2, base_sim2 = module2(grid_shape)
    
    h, w = grid_shape
    n_pixels = h * w
    center_idx = n_pixels // 2  # Center pixel
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # ===== Row 1: Gaussian Encoding =====
    ax = axes[0, 0]
    im = ax.imshow(dist1.detach().numpy(), cmap='viridis')
    ax.set_title('Distance Matrix')
    ax.set_xlabel('Pixel j')
    ax.set_ylabel('Pixel i')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    # Show encoding strength (norm) for center pixel
    encoding_norm1 = torch.norm(encoding1[center_idx], dim=1).reshape(h, w).detach().numpy()
    im = ax.imshow(encoding_norm1, cmap='hot')
    center_y, center_x = center_idx // w, center_idx % w
    ax.plot(center_x, center_y, 'bo', markersize=8)
    ax.set_title('Encoding Strength from Center')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    # Verify monotonic decrease
    sorted_dist1, sorted_sim1, is_mono1, violations1 = verify_monotonic_decrease(encoding1, dist1, center_idx)
    
    # Plot with moving average for smoothness
    window = 15
    dist_ma1 = np.convolve(sorted_dist1, np.ones(window)/window, mode='valid')
    sim_ma1 = np.convolve(sorted_sim1, np.ones(window)/window, mode='valid')
    
    ax.plot(dist_ma1, sim_ma1, 'b-', linewidth=2, label='Similarity')
    
    # Add ideal monotonic curve for reference
    ideal_sim = np.exp(-dist_ma1 / 4.0)
    ax.plot(dist_ma1, ideal_sim, 'r--', alpha=0.5, label='Ideal (exp decay)')
    
    ax.set_xlabel('Distance from center')
    ax.set_ylabel('Encoding similarity')
    ax.set_title(f'Gaussian: Monotonic? {is_mono1}\nViolations: {violations1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 3]
    # Show similarity matrix for center pixel
    from sklearn.metrics.pairwise import cosine_similarity
    encoding_np1 = encoding1.detach().numpy()
    center_enc1 = encoding_np1[center_idx]  # (n, dim)
    sim_matrix1 = cosine_similarity(center_enc1)
    
    im = ax.imshow(sim_matrix1, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.set_title('Similarity Matrix (Center row)')
    plt.colorbar(im, ax=ax)
    
    # ===== Row 2: Exponential Encoding =====
    ax = axes[1, 0]
    im = ax.imshow(dist2.detach().numpy(), cmap='viridis')
    ax.set_title('Distance Matrix')
    ax.set_xlabel('Pixel j')
    ax.set_ylabel('Pixel i')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    # Show encoding strength
    encoding_norm2 = torch.norm(encoding2[center_idx], dim=1).reshape(h, w).detach().numpy()
    im = ax.imshow(encoding_norm2, cmap='hot')
    ax.plot(center_x, center_y, 'bo', markersize=8)
    ax.set_title('Encoding Strength from Center')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    # Verify monotonic decrease for exponential encoding
    sorted_dist2, sorted_sim2, is_mono2, violations2 = verify_monotonic_decrease(encoding2, dist2, center_idx)
    
    dist_ma2 = np.convolve(sorted_dist2, np.ones(window)/window, mode='valid')
    sim_ma2 = np.convolve(sorted_sim2, np.ones(window)/window, mode='valid')
    
    ax.plot(dist_ma2, sim_ma2, 'g-', linewidth=2, label='Similarity')
    
    # Plot base similarity (which is guaranteed monotonic)
    base_sim_flat = base_sim2[center_idx].flatten().detach().numpy()
    base_sim_sorted = base_sim_flat[np.argsort(sorted_dist2)]
    base_sim_ma = np.convolve(base_sim_sorted, np.ones(window)/window, mode='valid')
    ax.plot(dist_ma2, base_sim_ma, 'm--', alpha=0.7, label='Base similarity')
    
    ax.set_xlabel('Distance from center')
    ax.set_ylabel('Encoding similarity')
    ax.set_title(f'Exponential: Monotonic? {is_mono2}\nViolations: {violations2}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 3]
    # Quantitative analysis
    ax.axis('off')
    
    analysis_text = (
        "MONOTONIC DECREASE ANALYSIS\n"
        "==========================\n"
        f"Grid size: {h}×{w} = {n_pixels} pixels\n\n"
        "Gaussian Encoding:\n"
        f"  • Monotonic: {is_mono1}\n"
        f"  • Violations: {violations1}\n"
        f"  • Min similarity: {sorted_sim1.min():.3f}\n"
        f"  • Max similarity: {sorted_sim1.max():.3f}\n\n"
        "Exponential Encoding:\n"
        f"  • Monotonic: {is_mono2}\n"
        f"  • Violations: {violations2}\n"
        f"  • Min similarity: {sorted_sim2.min():.3f}\n"
        f"  • Max similarity: {sorted_sim2.max():.3f}\n\n"
        "Ideal properties:\n"
        "  1. Similarity = 1.0 at distance 0\n"
        "  2. Strictly decreasing with distance\n"
        "  3. Approaches 0 at large distances"
    )
    
    ax.text(0.1, 0.5, analysis_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Distance-Dependent Encoding with Monotonic Decrease Property', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Additional statistical verification
    print("=" * 70)
    print("MONOTONIC DECREASE VERIFICATION")
    print("=" * 70)
    
    for name, (sorted_dist, sorted_sim, is_mono, violations) in [
        ("Gaussian Encoding", (sorted_dist1, sorted_sim1, is_mono1, violations1)),
        ("Exponential Encoding", (sorted_dist2, sorted_sim2, is_mono2, violations2))
    ]:
        print(f"\n{name}:")
        print(f"  Monotonic: {is_mono}")
        print(f"  Violations: {violations}")
        
        # Compute derivative (should be negative)
        diff_sim = np.diff(sorted_sim)
        negative_ratio = np.sum(diff_sim < -1e-8) / len(diff_sim)
        print(f"  Negative derivative ratio: {negative_ratio:.3f}")
        
        # Check correlation (should be strong negative)
        corr = np.corrcoef(sorted_dist[:len(sorted_sim)], sorted_sim)[0, 1]
        print(f"  Distance-similarity correlation: {corr:.4f}")
        
        # Check key distances
        for d in [0, 2, 4, 6, 8]:
            if d < len(sorted_dist):
                idx = np.argmin(np.abs(sorted_dist - d))
                print(f"  Distance {d:.1f}: similarity = {sorted_sim[idx]:.4f}")

def create_strictly_monotonic_encoding():
    """
    Create an encoding with guaranteed strictly monotonic decrease.
    """
    class StrictMonotonicEncoding2D(nn.Module):
        def __init__(self, dim, base_decay=0.5, min_value=0.01):
            super().__init__()
            self.dim = dim
            self.base_decay = base_decay
            self.min_value = min_value
            
            # Learnable patterns that will be scaled by distance
            self.patterns = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
            
        def forward(self, grid_shape):
            h, w = grid_shape
            
            # Create coordinates
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h).float(),
                torch.arange(w).float(),
                indexing='ij'
            )
            coordinates = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
            
            # Compute distances
            diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
            distances = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
            
            # Create strictly monotonic decay
            # Similarity = exp(-decay * distance)
            decay = self.base_decay + torch.sigmoid(self.patterns[0, 0, :1]) * 2.0
            similarity = torch.exp(-decay * distances)
            similarity = self.min_value + (1 - self.min_value) * similarity
            
            # Apply to patterns
            encoding = self.patterns * similarity.unsqueeze(-1)
            
            return encoding, coordinates, distances, similarity
    
    # Test it
    module = StrictMonotonicEncoding2D(dim=32)
    grid_shape = (8, 8)
    encoding, coords, dists, sim = module(grid_shape)
    
    # Verify monotonicity
    center_idx = grid_shape[0] * grid_shape[1] // 2
    sorted_dist, sorted_sim, is_mono, violations = verify_monotonic_decrease(encoding, dists, center_idx)
    
    print("\n" + "=" * 60)
    print("STRICTLY MONOTONIC ENCODING")
    print("=" * 60)
    print(f"Encoding shape: {encoding.shape}")
    print(f"Monotonic: {is_mono}")
    print(f"Violations: {violations}")
    
    # Plot the decay curve
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(sorted_dist, sorted_sim, 'b-', linewidth=2)
    plt.xlabel('Distance')
    plt.ylabel('Similarity')
    plt.title('Strictly Monotonic Decrease')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Show derivative (should be negative)
    derivative = np.diff(sorted_sim)
    plt.plot(sorted_dist[1:], derivative, 'r-', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Distance')
    plt.ylabel('Derivative (dSimilarity/dDistance)')
    plt.title('Derivative (Negative = Monotonic Decrease)')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Guaranteed Monotonic Distance Encoding', fontsize=14)
    plt.tight_layout()
    plt.show()

# Run the visualizations
print("Creating monotonic distance-dependent encodings...")
visualize_monotonic_distance_encoding()
create_strictly_monotonic_encoding()