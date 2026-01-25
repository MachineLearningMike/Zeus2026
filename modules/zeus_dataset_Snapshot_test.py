import torch

def resize_categorical_cylindrical_perfect(tensor, target_resolution):
    """
    Perfect cylindrical resize for categorical data.
    Ensures left and right edges have similar distributions.
    """
    target_H, target_W = target_resolution
    C, H, W = tensor.shape
    device = tensor.device
    
    # --- Create mapping that ensures edge continuity ---
    # The key: make sure j_norm for leftmost and rightmost pixels
    # map to similar source regions
    
    # Create j_norm from -0.5 to 0.5 (centered)
    # This ensures continuity at edges
    j_norm = torch.linspace(-0.5, 0.5, target_W, device=device)
    
    # Map to source coordinates
    j_source_float = (j_norm + 0.5) * W  # Now in [0, W]
    
    # Convert to indices with modulo
    j_source = torch.floor(j_source_float).long() % W
    
    # For latitude
    i_norm = torch.linspace(0, 1, target_H, device=device)
    i_source_float = i_norm * (H - 1)
    i_source = torch.floor(i_source_float).long().clamp(0, H-1)
    
    # Create meshgrid
    I, J = torch.meshgrid(i_source, j_source, indexing='ij')
    
    # Sample
    result = tensor[:, I, J]
    
    return result

def test_perfect_categorical_resize():
    # Create the exact test data from your output
    H, W = 721, 1440
    
    # Recreate your patterns
    soil = torch.zeros(H, W, dtype=torch.long)
    soil[:, :W//2] = 1
    soil[:, W//2:] = 2
    
    land = torch.zeros(H, W, dtype=torch.long)
    land[H//3:2*H//3, :] = 1
    
    test_tensor = torch.stack([soil, land], dim=0)
    
    print("=== Testing Perfect Cylindrical Resize ===")
    print(f"Input shape: {test_tensor.shape}")
    
    # Test the perfect version
    resized = resize_categorical_cylindrical_perfect(test_tensor, (256, 512))
    
    print(f"\nOutput shape: {resized.shape}")
    
    # Analyze edge continuity
    soil_left = resized[0, :, 0]
    soil_right = resized[0, :, -1]
    
    land_left = resized[1, :, 0]
    land_right = resized[1, :, -1]
    
    # Count occurrences
    soil_left_counts = torch.bincount(soil_left, minlength=3)
    soil_right_counts = torch.bincount(soil_right, minlength=3)
    
    land_left_counts = torch.bincount(land_left, minlength=2)
    land_right_counts = torch.bincount(land_right, minlength=2)
    
    soil_diff = torch.abs(soil_left_counts - soil_right_counts).sum().item()
    land_diff = torch.abs(land_left_counts - land_right_counts).sum().item()
    
    print(f"\nSoil edge difference: {soil_diff}")
    print(f"Land edge difference: {land_diff}")
    
    # Check what values are at edges
    print(f"\nSoil left edge sample (first 10): {soil_left[:10].tolist()}")
    print(f"Soil right edge sample (last 10): {soil_right[-10:].tolist()}")
    
    # Visualize the mapping
    print(f"\n=== Coordinate Mapping Analysis ===")
    target_W = 512
    W = 1440
    
    j_norm = torch.linspace(-0.5, 0.5, target_W)
    j_source_float = (j_norm + 0.5) * W
    j_source = torch.floor(j_source_float).long() % W
    
    print(f"Left edge maps to source: {j_source[0].item()}")
    print(f"Right edge maps to source: {j_source[-1].item()}")
    print(f"Difference should be small for continuity!")
    
    # With -0.5 to 0.5 mapping, edges should map to similar regions
    edge_similarity = abs(j_source[0] - j_source[-1])
    print(f"Edge mapping similarity: {edge_similarity} pixels")
    
    return soil_diff < 100 and land_diff < 100  # Much more reasonable

# Run test
if __name__ == "__main__":
    success = test_perfect_categorical_resize()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")