import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

class GaussianDistanceEncoding2D(nn.Module):
    """
    2D Distance encoding using Gaussian/RBF kernels.
    Encodes Euclidean distance between (x1,y1) and (x2,y2)
    """
    def __init__(self, dim, num_kernels=16, max_distance=10.0):
        super().__init__()
        self.dim = dim
        self.num_kernels = num_kernels
        
        # Learnable kernel centers and widths for 2D distances
        self.centers = nn.Parameter(torch.linspace(0, max_distance, num_kernels))
        self.widths = nn.Parameter(torch.ones(num_kernels) * (max_distance / num_kernels))
        
        # Projection to dimension
        self.projection = nn.Linear(num_kernels, dim)
        
    def forward(self, grid_shape):
        """
        grid_shape: (height, width) or (h, w)
        Returns: (h*w, h*w, dim) encoding for all pairwise distances
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
        
        # Compute Euclidean distance matrix: (h*w, h*w)
        # dist[i,j] = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (h*w, h*w, 2)
        distances = torch.sqrt((diff ** 2).sum(dim=2))  # (h*w, h*w)
        
        # Compute Gaussian kernels: (h*w, h*w, num_kernels)
        kernels = torch.exp(
            -((distances.unsqueeze(-1) - self.centers) ** 2) / 
            (2 * self.widths ** 2)
        )
        
        # Project to embedding dimension
        encoding = self.projection(kernels)  # (h*w, h*w, dim)
        
        return encoding, coordinates, distances
    
    
def visualize_2d_distance_encoding():
    # Create the encoding module
    dim = 32
    num_kernels = 12
    grid_shape = (8, 8)  # 8x8 grid
    
    encoding_module = GaussianDistanceEncoding2D(dim=dim, num_kernels=num_kernels, max_distance=10.0)
    
    # Get the encoding
    encoding, coordinates, distances = encoding_module(grid_shape)
    h, w = grid_shape
    n_pixels = h * w
    
    # Create a larger figure with better layout
    fig = plt.figure(figsize=(22, 14))
    
    # Main grid: 3 rows, 4 columns for main visualizations
    gs_main = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Show the grid coordinates
    ax1 = fig.add_subplot(gs_main[0, 0])
    ax1.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=50, alpha=0.6)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_title('2D Grid Coordinates')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Show distance matrix (flattened)
    ax2 = fig.add_subplot(gs_main[0, 1])
    im2 = ax2.imshow(distances.detach().numpy(), cmap='viridis')
    ax2.set_title('Euclidean Distance Matrix')
    ax2.set_xlabel('Pixel j (flattened)')
    ax2.set_ylabel('Pixel i (flattened)')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Select a specific pixel and show its distances to others
    center_pixel = h * w // 2  # Middle pixel
    center_y, center_x = center_pixel // w, center_pixel % w
    
    ax3 = fig.add_subplot(gs_main[0, 2])
    distances_2d = distances[center_pixel].reshape(h, w).detach().numpy()
    im3 = ax3.imshow(distances_2d, cmap='hot')
    ax3.plot(center_x, center_y, 'bo', markersize=10, label='Center')
    ax3.set_title(f'Distances from Pixel ({center_x},{center_y})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    ax3.legend()
    
    # 4. Show Gaussian kernels for the center pixel
    ax4 = fig.add_subplot(gs_main[0, 3])
    
    # Get kernels for center pixel
    kernels = torch.exp(
        -((distances.unsqueeze(-1) - encoding_module.centers) ** 2) / 
        (2 * encoding_module.widths ** 2)
    )
    
    center_kernels = kernels[center_pixel].detach().numpy()  # (n_pixels, num_kernels)
    
    # Plot kernel centers distribution
    centers = encoding_module.centers.detach().numpy()
    widths = encoding_module.widths.detach().numpy()
    
    x_range = np.linspace(0, centers[-1] * 1.2, 100)
    for i in range(min(num_kernels, 4)):
        y = np.exp(-((x_range - centers[i]) ** 2) / (2 * widths[i] ** 2))
        ax4.plot(x_range, y, label=f'Kernel {i}')
    
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Activation')
    ax4.set_title('Gaussian Kernels (first 4)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Show encoding for a ring of pixels at specific distance
    ax5 = fig.add_subplot(gs_main[1, 0])
    
    # Select pixels at distance ~2.0 from center
    target_distance = 2.0
    tolerance = 0.5
    mask = (torch.abs(distances[center_pixel] - target_distance) < tolerance)
    mask_2d = mask.reshape(h, w).detach().numpy()
    
    # Create visualization
    vis_grid = np.zeros((h, w, 3))
    vis_grid[:, :, 0] = mask_2d  # Red for target ring
    vis_grid[center_y, center_x, 1] = 1  # Green for center
    
    ax5.imshow(vis_grid)
    ax5.set_title(f'Pixels at distance {target_distance}±{tolerance}')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    
    # 6. Compare encoding similarity vs distance
    ax6 = fig.add_subplot(gs_main[1, 1])
    
    # Compute cosine similarity between encodings
    encoding_np = encoding.detach().numpy()
    
    # For center pixel, compare with all others
    center_encoding = encoding_np[center_pixel]  # (n_pixels, dim)
    
    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(center_encoding)  # (n_pixels, n_pixels)
    
    # Plot distance vs similarity
    flat_distances = distances[center_pixel].flatten().detach().numpy()
    flat_similarities = similarities.diagonal()
    
    # Sort by distance
    sort_idx = np.argsort(flat_distances)
    sorted_dist = flat_distances[sort_idx]
    sorted_sim = flat_similarities[sort_idx]
    
    # Moving average
    window = 10
    dist_ma = np.convolve(sorted_dist, np.ones(window)/window, mode='valid')
    sim_ma = np.convolve(sorted_sim, np.ones(window)/window, mode='valid')
    
    ax6.plot(dist_ma, sim_ma, 'b-', linewidth=2)
    ax6.set_xlabel('Distance from center')
    ax6.set_ylabel('Encoding cosine similarity')
    ax6.set_title('Distance vs Encoding Similarity')
    ax6.grid(True, alpha=0.3)
    
    # 7. Radial symmetry demonstration
    ax7 = fig.add_subplot(gs_main[1, 2])
    
    # Get four pixels at same distance but different directions
    same_distance_pixels = []
    directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]  # Right, Up, Left, Down
    
    for dx, dy in directions:
        px = center_x + dx
        py = center_y + dy
        if 0 <= px < w and 0 <= py < h:
            pixel_idx = py * w + px
            same_distance_pixels.append(pixel_idx)
    
    # Compare their encodings to center
    center_to_all = encoding_np[center_pixel]  # How center encodes relationships to others
    
    similarities_to_center = []
    for pixel_idx in same_distance_pixels:
        enc = center_to_all[pixel_idx]
        similarities_to_center.append(enc)
    
    # Show they're similar
    pairwise_similarities = []
    for i, s1 in enumerate(similarities_to_center):
        for j, s2 in enumerate(similarities_to_center):
            if i < j:
                sim = cosine_similarity([s1], [s2])[0, 0]
                pairwise_similarities.append(sim)
    
    avg_similarity = np.mean(pairwise_similarities) if pairwise_similarities else 0
    
    # Visualize the four pixels
    x_coords = coordinates[:, 0].reshape(h, w).numpy()
    y_coords = coordinates[:, 1].reshape(h, w).numpy()
    
    ax7.scatter(x_coords, y_coords, c='lightgray', s=20, alpha=0.3)
    ax7.scatter(center_x, center_y, c='green', s=100, label='Center')
    
    colors = ['red', 'blue', 'orange', 'purple']
    for idx, (color, pixel_idx) in enumerate(zip(colors, same_distance_pixels)):
        px = pixel_idx % w
        py = pixel_idx // w
        ax7.scatter(px, py, c=color, s=80, label=f'Pixel {idx}')
    
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_title(f'Radial Symmetry Test\nAvg similarity: {avg_similarity:.3f}')
    ax7.legend(loc='upper right', fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal')
    
    # 8. Show kernel activations in a separate subplot grid
    ax8 = fig.add_subplot(gs_main[1, 3])
    
    # Show a sample of kernel activations
    kernel_sample = center_kernels.reshape(h, w, num_kernels)
    
    # Take 4 representative kernels
    kernel_indices = [0, 3, 6, 9] if num_kernels > 9 else [0, 1, 2, 3]
    
    for i, k_idx in enumerate(kernel_indices[:4]):
        if k_idx < num_kernels:
            ax = fig.add_subplot(gs_main[2, i])
            im = ax.imshow(kernel_sample[:, :, k_idx], cmap='RdYlBu_r')
            ax.set_title(f'Kernel {k_idx} (center={centers[k_idx]:.1f})')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('2D Gaussian Distance Encoding Visualization', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("=" * 70)
    print("2D DISTANCE-DEPENDENT POSITIONAL ENCODING ANALYSIS")
    print("=" * 70)
    
    print(f"\n1. Grid: {h}×{w} = {n_pixels} pixels")
    print(f"2. Encoding shape: {encoding.shape} (n_pixels × n_pixels × dim)")
    print(f"3. Each encoding[i,j] encodes relationship between pixel i and pixel j")
    
    print(f"\n4. Distance properties:")
    print(f"   - Encoding depends only on Euclidean distance: √[(x₁-x₂)² + (y₁-y₂)²]")
    print(f"   - Rotation invariant: Same distance = similar encoding")
    print(f"   - Translation invariant: Shifting both pixels same amount doesn't change encoding")
    
    print(f"\n5. Radial symmetry example:")
    for i, pixel_idx in enumerate(same_distance_pixels):
        px = pixel_idx % w
        py = pixel_idx // w
        dist = distances[center_pixel, pixel_idx].item()
        print(f"   Pixel ({px},{py}): distance = {dist:.2f}")
    
    print(f"\n6. Kernel parameters:")
    print(f"   - Centers: {centers}")
    print(f"   - Widths: {widths}")
    
    print(f"\n7. Encoding statistics:")
    print(f"   - Mean encoding norm: {np.mean(np.linalg.norm(encoding_np, axis=2)):.4f}")
    print(f"   - Encoding std: {np.std(encoding_np):.4f}")

# Run visualization
visualize_2d_distance_encoding()

