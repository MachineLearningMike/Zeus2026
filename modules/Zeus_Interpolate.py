import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape_HFX_1X_MHA_Interpolate(nn.Module):
    anchor = 2
    res_multiplier = [1 / anchor, 1 / anchor]   # clients should use math.ceil(res * res_multiplier)
    dim_multiplier = 1
    
    def __init__(self):
        super().__init__()
        self.original_size = None
        self.target_size = None
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        self.original_size = (H, W)
        self.target_size = (math.ceil(H / self.anchor), math.ceil(W / self.anchor))
        
        # Convert to (B, C, H, W) for interpolate
        x_permuted = x.permute(0, 3, 1, 2)
        x_down = resize_continuous_bilinear_channel_wise_cylindric(x_permuted, self.target_size)
        
        # Convert back to (B, H, W, C)
        return x_down.permute(0, 2, 3, 1)


class Reshape_2FX_1X_MHA_Interpolate(nn.Module):
    anchor = 2
    res_multiplier = [anchor, anchor]   # clients should use math.ceil(res * res_multiplier)
    dim_multiplier = 1

    def __init__(self):
        super().__init__()
        self.original_size = None
        self.target_size = None
        self.down_layer = None
        
    def set_down_layer(self, down_layer):
        """Connect to the corresponding down layer"""
        self.down_layer = down_layer
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Use the down layer's original size as target for perfect reconstruction
        if self.down_layer is not None and self.down_layer.original_size is not None:
            # This ensures level-wise shape matching
            target_res = self.down_layer.original_size
        else:
            # Fallback - but this shouldn't happen in a paired network
            target_res = (math.ceil(H * self.anchor), math.ceil(W * self.anchor))
        
        self.target_size = target_res
        self.original_size = (H, W)
        
        # Convert to (B, C, H, W) for interpolate
        x_permuted = x.permute(0, 3, 1, 2)
        x_up = resize_continuous_bilinear_channel_wise_cylindric(x_permuted, target_res)
        
        # Convert back to (B, H, W, C)
        return x_up.permute(0, 2, 3, 1)


# Your cylindrical resize function
def resize_continuous_bilinear_channel_wise_cylindric(tensor, target_size, mode='bilinear'):
    """
    CORRECT cylindrical resize for cyclic longitude dimension.
    """
    if tensor.dim() == 3:
        x = tensor.unsqueeze(0)
        has_batch = False
    else:
        x = tensor
        has_batch = True
    
    target_h, target_w = target_size
    B, C, H, W = x.shape
    
    # If already at target size, return
    if H == target_h and W == target_w:
        if not has_batch:
            return tensor
        else:
            return x
    
    # Cylindrical padding
    wrap_W = W // 2
    right_half = x[..., -wrap_W:]
    left_half = x[..., :wrap_W]
    x_unwrapped = torch.cat([right_half, x, left_half], dim=-1)
    
    # Resize
    if mode == 'bilinear':
        x_resized = F.interpolate(
            x_unwrapped,
            size=(target_h, target_w + 2*wrap_W),
            mode=mode,
            align_corners=False
        )
    else:
        x_resized = F.interpolate(
            x_unwrapped,
            size=(target_h, target_w + 2*wrap_W),
            mode=mode
        )
    
    # Crop back to target width
    result = x_resized[..., wrap_W:-wrap_W]
    
    # Remove batch dimension if needed
    if not has_batch:
        result = result.squeeze(0)
    
    return result


# Test level-wise shape matching
if __name__ == "__main__":
    print("=" * 60)
    print("Testing LEVEL-WISE shape matching")
    print("=" * 60)
    
    # Create pyramid with connected pairs
    down1 = Reshape_HFX_1X_MHA_Interpolate()  # 45×90 → 23×45
    up1 = Reshape_2FX_1X_MHA_Interpolate()
    up1.set_down_layer(down1)
    
    down2 = Reshape_HFX_1X_MHA_Interpolate()  # 23×45 → 12×23
    up2 = Reshape_2FX_1X_MHA_Interpolate()
    up2.set_down_layer(down2)
    
    down3 = Reshape_HFX_1X_MHA_Interpolate()  # 12×23 → 6×12
    up3 = Reshape_2FX_1X_MHA_Interpolate()
    up3.set_down_layer(down3)
    
    down4 = Reshape_HFX_1X_MHA_Interpolate()  # 6×12 → 3×6
    up4 = Reshape_2FX_1X_MHA_Interpolate()
    up4.set_down_layer(down4)
    
    # Test data
    x = torch.randn(2, 5, 45, 90)
    
    # Downward pass
    x1 = down1(x)      # 45×90 → 23×45
    x2 = down2(x1)     # 23×45 → 12×23
    x3 = down3(x2)     # 12×23 → 6×12
    x4 = down4(x3)     # 6×12 → 3×6
    
    # Upward pass - each up layer should restore the previous level's shape
    x3_up = up4(x4)    # 3×6 → 6×12  (matches x3)
    x2_up = up3(x3_up) # 6×12 → 12×23 (matches x2)
    x1_up = up2(x2_up) # 12×23 → 23×45 (matches x1)
    x_up = up1(x1_up)  # 23×45 → 45×90 (matches x)
    
    print("\nLevel-wise shape verification:")
    print(f"Level 0: {x.shape} (original)")
    print(f"Level 1 down: {x1.shape} → Level 1 up target: {down1.original_size}")
    print(f"Level 2 down: {x2.shape} → Level 2 up target: {down2.original_size}")
    print(f"Level 3 down: {x3.shape} → Level 3 up target: {down3.original_size}")
    print(f"Level 4 down: {x4.shape} → Level 4 up target: {down4.original_size}")
    
    print(f"\nUpward pass results:")
    print(f"up4 (3×6 → 6×12): {x3_up.shape} (should match {x3.shape})")
    print(f"up3 (6×12 → 12×23): {x2_up.shape} (should match {x2.shape})")
    print(f"up2 (12×23 → 23×45): {x1_up.shape} (should match {x1.shape})")
    print(f"up1 (23×45 → 45×90): {x_up.shape} (should match {x.shape})")
    
    # Verify each level matches
    assert x3_up.shape == x3.shape, f"Level 3 mismatch: {x3_up.shape} vs {x3.shape}"
    assert x2_up.shape == x2.shape, f"Level 2 mismatch: {x2_up.shape} vs {x2.shape}"
    assert x1_up.shape == x1.shape, f"Level 1 mismatch: {x1_up.shape} vs {x1.shape}"
    assert x_up.shape == x.shape, f"Level 0 mismatch: {x_up.shape} vs {x.shape}"
    
    print("\n✅ ALL LEVELS MATCH PERFECTLY!")
    
    # Show the level-wise connections
    print("\n" + "=" * 60)
    print("Level-wise architecture:")
    print("=" * 60)
    print("                    [x] 45×90")
    print("                      |")
    print("                  down1 (45→23)")
    print("                      |")
    print("              [x1] 23×45 ←─┐")
    print("                  |         │")
    print("              down2 (23→12) │ up1 (23→45)")
    print("                  |         │")
    print("          [x2] 12×23 ←─┐    │")
    print("              |         │    │")
    print("          down3 (12→6)  │ up2 (12→23)")
    print("              |         │    │")
    print("       [x3] 6×12 ←─┐    │    │")
    print("           |        │    │    │")
    print("       down4 (6→3)  │ up3 (6→12)")
    print("           |        │    │    │")
    print("    [x4] 3×6        │    │    │")
    print("           │        │    │    │")
    print("           └─────up4────┘    │")
    print("                    │         │")
    print("                    └────up3───┘")
    print("                         │      ")
    print("                         └─up2──┘")
    print("                            │    ")
    print("                            └up1─┘")
    print("                              │")
    print("                            [x] 45×90")