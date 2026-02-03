import torch
from dataclasses import dataclass

@dataclass
class RotationParams:
    cos: torch.Tensor
    sin: torch.Tensor

# Your implementation exactly as you wrote it
class RoPE2D():
    def __init__(self, H, W, dims):
        self.H = H
        self.W = W
        self.dims = dims
        self.D_lat = dims // 2
        self.D_lon = dims - self.D_lat
        
        # Create DIFFERENT lat and lon rotations
        self.rot_lat = self._create_rotation_params(
            H, dims, is_cylindric=False, broadcast_over='H', 
            base_freq=10000.0
        )
        self.rot_lon = self._create_rotation_params(
            W, dims, is_cylindric=True, broadcast_over='W',
            base_harmonic=2.0
        )
    
    def _create_rotation_params(self, N, D, is_cylindric, broadcast_over, 
                               base_freq=10000.0, base_harmonic=1):
        """Create RotationParams with configurable bases."""
        positions = torch.arange(N).float()
        
        # if is_cylindric:
        #     # Start harmonics from base_harmonic
        #     k = torch.arange(base_harmonic, base_harmonic + D//2).float()
        #     theta = positions[:, None] * k[None, :]
        # else:
        #     # Use base_freq
        #     # Standard RoPE: θ = position / (10000^(2i/d_model))
        #     # i = 0, 1, 2, ..., d/2-1
        i = torch.arange(0, D//2).float()  # i = 0, 1, ..., D/2-1
        theta = positions[:, None] / (base_freq ** (2 * i / D))[None, :]  # (N, D/2)
        
        theta = torch.cat([theta, theta], dim=-1)[:, :D]
        
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        if broadcast_over == 'H':
            cos_exp = cos_theta.unsqueeze(0).unsqueeze(2)
            sin_exp = sin_theta.unsqueeze(0).unsqueeze(2)
        else:  # 'W'
            cos_exp = cos_theta.unsqueeze(0).unsqueeze(0)
            sin_exp = sin_theta.unsqueeze(0).unsqueeze(0)
        
        return RotationParams(cos=cos_exp, sin=sin_exp)
    
    def apply_rope(self, x):
        """Same combined rotation formula."""
        cos_combined = self.rot_lat.cos * self.rot_lon.cos - self.rot_lat.sin * self.rot_lon.sin
        sin_combined = self.rot_lat.sin * self.rot_lon.cos + self.rot_lat.cos * self.rot_lon.sin
        
        return x * cos_combined + self._rotate_90(x) * sin_combined
    
    def _rotate_90(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)
    
    def __call__(self, x):
        return self.apply_rope(x)



# Run test if script is executed directly
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np


    def visualize_rope(rope, x=None):
        """
        Visualize RoPE parameters and effects.
        
        Args:
            rope: RoPE2D instance
            x: Optional input tensor to visualize rotation effects
        """
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Extract and plot theta values
        cos_lat_slice = rope.rot_lat.cos[0, :, 0, 0]  # Shape: (H,)
        sin_lat_slice = rope.rot_lat.sin[0, :, 0, 0]  # Shape: (H,)
        theta_lat = torch.atan2(sin_lat_slice, cos_lat_slice).cpu().numpy()
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(theta_lat, 'b-', label='Latitude θ')
        ax1.axvline(x=rope.H//2, color='r', linestyle='--', alpha=0.5, label='Center')
        ax1.set_xlabel('H position')
        ax1.set_ylabel('Theta (radians)')
        ax1.set_title(f'Latitude RoPE (first dim pair)\nθ range: [{theta_lat.min():.3f}, {theta_lat.max():.3f}]')
        ax1.legend()
        ax1.grid(True)
        
        cos_lon_slice = rope.rot_lon.cos[0, 0, :, 0]  # Shape: (W,)
        sin_lon_slice = rope.rot_lon.sin[0, 0, :, 0]  # Shape: (W,)
        theta_lon = torch.atan2(sin_lon_slice, cos_lon_slice).cpu().numpy()
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(theta_lon, 'r-', label='Longitude θ')
        ax2.axvline(x=rope.W//2, color='r', linestyle='--', alpha=0.5, label='Center')
        ax2.set_xlabel('W position')
        ax2.set_ylabel('Theta (radians)')
        ax2.set_title(f'Longitude RoPE (first dim pair)\nθ range: [{theta_lon.min():.3f}, {theta_lon.max():.3f}]')
        ax2.legend()
        ax2.grid(True)
        
        # 2. Plot cos/sin values
        ax3 = plt.subplot(2, 3, 3)
        num_dims_to_plot = min(4, rope.D_lat//2)
        
        for dim in range(num_dims_to_plot):
            cos_vals = rope.rot_lat.cos[0, :, 0, dim*2].cpu().numpy()
            sin_vals = rope.rot_lat.sin[0, :, 0, dim*2].cpu().numpy()
            ax3.plot(cos_vals, '--', label=f'cos(θ) dim{dim*2}')
            ax3.plot(sin_vals, ':', label=f'sin(θ) dim{dim*2}')
        
        ax3.axvline(x=rope.H//2, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('H position')
        ax3.set_ylabel('Value')
        ax3.set_title('Latitude: cos/sin for first few dims')
        ax3.legend(fontsize='small')
        ax3.grid(True)
        
        ax4 = plt.subplot(2, 3, 4)
        num_dims_to_plot = min(4, rope.D_lon//2)
        
        for dim in range(num_dims_to_plot):
            cos_vals = rope.rot_lon.cos[0, 0, :, dim*2].cpu().numpy()
            sin_vals = rope.rot_lon.sin[0, 0, :, dim*2].cpu().numpy()
            ax4.plot(cos_vals, '--', label=f'cos(θ) dim{dim*2}')
            ax4.plot(sin_vals, ':', label=f'sin(θ) dim{dim*2}')
        
        ax4.axvline(x=rope.W//2, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('W position')
        ax4.set_ylabel('Value')
        ax4.set_title('Longitude: cos/sin for first few dims')
        ax4.legend(fontsize='small')
        ax4.grid(True)
        
        # 3. Visualize rotation effect from center perspective
        ax5 = plt.subplot(2, 3, 5)
        
        center_h, center_w = rope.H // 2, rope.W // 2
        
        # Create test vectors at different positions
        test_positions = []
        positions_to_test = [
            (center_h, center_w),  # Center
            (center_h - rope.H//4, center_w),  # North
            (center_h + rope.H//4, center_w),  # South
            (center_h, center_w - rope.W//4),  # West
            (center_h, center_w + rope.W//4),  # East
            (center_h - rope.H//4, center_w - rope.W//4),  # NW
            (center_h - rope.H//4, center_w + rope.W//4),  # NE
            (center_h + rope.H//4, center_w - rope.W//4),  # SW
            (center_h + rope.H//4, center_w + rope.W//4),  # SE
        ]
        
        for h, w in positions_to_test:
            if 0 <= h < rope.H and 0 <= w < rope.W:
                # Create unit vector in first dimension
                vec = torch.zeros(1, rope.H, rope.W, rope.dims)
                vec[0, h, w, 0] = 1.0
                test_positions.append((h, w, vec))
        
        # Apply RoPE
        rotated_vectors = []
        for h, w, vec in test_positions:
            rotated = rope(vec)
            rotated_vectors.append((h, w, rotated[0, h, w]))
        
        # Plot vectors
        for idx, (h, w, rot_vec) in enumerate(rotated_vectors):
            rot_x = rot_vec[0].item()
            rot_y = rot_vec[1].item()
            
            # Color based on position
            color = 'red' if (h == center_h and w == center_w) else f'C{idx%8}'
            label = 'Center' if (h == center_h and w == center_w) else f'({h-center_h},{w-center_w})'
            
            ax5.arrow(0, 0, rot_x, rot_y, 
                    head_width=0.05, head_length=0.05, 
                    fc=color, ec=color, alpha=0.7, label=label)
        
        ax5.set_xlim(-1.2, 1.2)
        ax5.set_ylim(-1.2, 1.2)
        ax5.set_xlabel('Dimension 0')
        ax5.set_ylabel('Dimension 1')
        ax5.set_title('Rotated vectors at different positions\n(All start as (1,0,0,...))')
        ax5.legend(fontsize='small', loc='upper right')
        ax5.grid(True)
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax5.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax5.set_aspect('equal')
        
        # 4. FIXED: Proper similarity heatmap (dot product across ALL dimensions)
        ax6 = plt.subplot(2, 3, 6)
        
        # Create reference vector at CENTER position
        ref_vec = torch.zeros(1, rope.H, rope.W, rope.dims)
        ref_vec[0, center_h, center_w, 0] = 1.0  # Unit vector in first dimension
        ref_rotated = rope(ref_vec)
        ref_vector = ref_rotated[0, center_h, center_w]  # Full rotated vector at center
        
        # Create grid of vectors and compute similarity
        similarity = torch.zeros(rope.H, rope.W)
        
        # Vectorized computation for efficiency
        grid_vecs = torch.zeros(rope.H, rope.W, rope.dims)
        grid_vecs[:, :, 0] = 1.0  # All vectors are (1,0,0,...)
        
        # Reshape for batch processing
        grid_vecs_reshaped = grid_vecs.view(rope.H * rope.W, rope.dims)
        
        # Apply RoPE to all positions at once
        # Need to create proper input shape: (1, H, W, dims)
        grid_input = grid_vecs.unsqueeze(0)
        grid_rotated_all = rope(grid_input)  # (1, H, W, dims)
        grid_rotated = grid_rotated_all[0]  # (H, W, dims)
        
        # Compute dot products
        for h in range(rope.H):
            for w in range(rope.W):
                # Dot product across ALL dimensions
                sim = torch.dot(ref_vector, grid_rotated[h, w]).item()
                similarity[h, w] = sim
        
        # Plot heatmap
        im = ax6.imshow(similarity.cpu().numpy(), cmap='RdBu_r', 
                        extent=[0, rope.W, 0, rope.H], aspect='auto',
                        vmin=-1, vmax=1)
        
        # Mark center
        ax6.scatter([center_w], [center_h], color='red', s=100, marker='*', label='Center')
        
        # Add contour lines to better see pattern
        X, Y = np.meshgrid(np.arange(rope.W), np.arange(rope.H))
        similarity_np = similarity.cpu().numpy()
        contours = ax6.contour(X, Y, similarity_np, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax6.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        ax6.set_xlabel('W position')
        ax6.set_ylabel('H position')
        ax6.set_title(f'Dot product similarity to center\n(Should show radial pattern)')
        plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        plt.show()
        
        # 5. NEW: Analyze the pattern
        print("=" * 60)
        print("RoPE Similarity Pattern Analysis:")
        print(f"Grid: H={rope.H}, W={rope.W}, dims={rope.dims}")
        print(f"Center position: ({center_h}, {center_w})")
        
        # Check if pattern is radial
        print("\nChecking radial symmetry:")
        
        # Check horizontal and vertical symmetry
        horizontal_sym = []
        vertical_sym = []
        
        for offset in range(1, min(center_h, center_w)):
            # Horizontal pairs
            sim_left = similarity[center_h, center_w - offset].item()
            sim_right = similarity[center_h, center_w + offset].item()
            horizontal_sym.append(abs(sim_left - sim_right))
            
            # Vertical pairs
            sim_up = similarity[center_h - offset, center_w].item()
            sim_down = similarity[center_h + offset, center_w].item()
            vertical_sym.append(abs(sim_up - sim_down))
        
        if horizontal_sym:
            avg_horiz_sym = sum(horizontal_sym) / len(horizontal_sym)
            avg_vert_sym = sum(vertical_sym) / len(vertical_sym)
            print(f"  Horizontal symmetry error: {avg_horiz_sym:.6f}")
            print(f"  Vertical symmetry error: {avg_vert_sym:.6f}")
            
            # Check if pattern shows strips
            if avg_horiz_sym > 0.1 and avg_vert_sym < 0.01:
                print("  WARNING: Pattern shows horizontal strips!")
                print("  This means RoPE only varies with H, not with W")
            elif avg_vert_sym > 0.1 and avg_horiz_sym < 0.01:
                print("  WARNING: Pattern shows vertical strips!")
                print("  This means RoPE only varies with W, not with H")
            elif avg_horiz_sym < 0.05 and avg_vert_sym < 0.05:
                print("  GOOD: Pattern shows radial symmetry")
            else:
                print("  Pattern is asymmetric")
        
        # Print similarity extremes
        print(f"\nSimilarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
        print(f"Similarity at center: {similarity[center_h, center_w]:.3f} (should be 1.0)")
        
        # Check corners
        corners = [(0, 0), (0, rope.W-1), (rope.H-1, 0), (rope.H-1, rope.W-1)]
        print("\nSimilarity to corners:")
        for h, w in corners:
            sim = similarity[h, w].item()
            print(f"  Position ({h},{w}): {sim:.3f}")
        
        print("=" * 60)
        
    def confirm_radial_symmetry(rope):
        """Confirm the pattern is truly radial."""
        H, W = rope.H, rope.W
        
        # Create test grid
        grid = torch.zeros(1, H, W, rope.dims)
        grid[0, :, :, 0] = 1.0
        rotated = rope(grid)[0]
        
        center_h, center_w = H//2, W//2
        ref_vec = rotated[center_h, center_w]
        
        # Check multiple directions
        directions = [
            (1, 0),   # Right
            (0, 1),   # Down  
            (1, 1),   # Down-right
            (1, -1),  # Up-right
        ]
        
        print("Checking radial symmetry:")
        for dh, dw in directions:
            similarities = []
            for r in range(1, min(center_h, center_w)):
                h = center_h + dh * r
                w = center_w + dw * r
                if 0 <= h < H and 0 <= w < W:
                    sim = torch.dot(ref_vec, rotated[h, w]).item()
                    similarities.append(sim)
            
            if similarities:
                print(f"  Direction ({dh},{dw}): {similarities[:3]}...")
        
        # All directions should give similar values at same distance!

    # Simple test function
    def test_rope():
        """Quick test of RoPE with visualization."""
        
        H, W, C = 36, 36, 12
        
        rope = RoPE2D(H=H, W=W, dims=C)
        
        # Create test tensor
        x_test = torch.randn(2, H, W, C)
        
        # Apply RoPE
        x_rotated = rope(x_test)
        
        print("Basic RoPE test:")
        print(f"Input shape: {x_test.shape}")
        print(f"Output shape: {x_rotated.shape}")
        print(f"Input norm mean: {x_test.norm(dim=-1).mean().item():.4f}")
        print(f"Output norm mean: {x_rotated.norm(dim=-1).mean().item():.4f}")
        
        # Visualize
        visualize_rope(rope, x_test[0:1])  # Use first batch only
        
        confirm_radial_symmetry(rope)
        
        return rope, x_test, x_rotated
    
    
    test_rope()


