import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly

# Custom convolution with horizontal-only cyclicity
class HorizontalCyclicConv2d(nn.Module):
    """Conv2d that is cyclic only in the width dimension"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,  # We'll handle padding manually
            bias=bias
        )
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # Calculate padding needed
        pad_h = self.kernel_size[0] // 2
        pad_w = self.kernel_size[1] // 2
        
        # Pad height with zeros (standard padding)
        if pad_h > 0:
            x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)
        
        # Pad width with circular padding (wrap around)
        if pad_w > 0:
            x = torch.cat([x[:, :, :, -pad_w:], x, x[:, :, :, :pad_w]], dim=3)
        
        # Apply convolution
        x = self.conv(x)
        
        return x

class Reshape_HFX_2X_Interpolate(nn.Module):
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = in_dim * 2
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        self.output_shape = output_shape
        
        self.up = HorizontalCyclicConv2d(dim, 2*dim, kernel_size=3, stride=1, bias=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape

        target_size, _ = self.output_shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.permute(0, 3, 1, 2)
        
        x = self.up(x) 
        
        x = resize_continuous_bilinear_channel_wise_cylindric(x, target_size)
        
        # Convert back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        return x

class Reshape_2FX_HX_Interpolate(nn.Module):

    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        self.output_shape = output_shape
        
        self.down = HorizontalCyclicConv2d(dim, dim//2, kernel_size=3, stride=1, bias=True)
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        target_size, _ = self.output_shape
        
        # Convert to (B, C, H, W) for interpolate
        x = x.permute(0, 3, 1, 2)
        x = resize_continuous_bilinear_channel_wise_cylindric(x, target_size)
        
        x = self.down(x)
        
        # Convert back to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        return x


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
    
    # Create pyramid with connected pairs - pass down_layer directly to up
    down1 = Reshape_HFX_1X_Interpolate()  # 45×90 → 23×45
    up1 = Reshape_2FX_1X_Interpolate(down_layer=down1)  # Connect at creation
    
    down2 = Reshape_HFX_1X_Interpolate()  # 23×45 → 12×23
    up2 = Reshape_2FX_1X_Interpolate(down_layer=down2)
    
    down3 = Reshape_HFX_1X_Interpolate()  # 12×23 → 6×12
    up3 = Reshape_2FX_1X_Interpolate(down_layer=down3)
    
    down4 = Reshape_HFX_1X_Interpolate()  # 6×12 → 3×6
    up4 = Reshape_2FX_1X_Interpolate(down_layer=down4)
    
    # Test data - Note: (batch, height, width, channels)
    x = torch.randn(2, 45, 90, 5)  # (B, H, W, C)
    
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
    
    # Print shapes for verification
    print(f"\nFinal shape verification:")
    print(f"  Original → Down1 → Up1: {x.shape} → {x1.shape} → {x_up.shape}")
    print(f"  Level1 → Down2 → Up2: {x1.shape} → {x2.shape} → {x1_up.shape}")
    print(f"  Level2 → Down3 → Up3: {x2.shape} → {x3.shape} → {x2_up.shape}")
    print(f"  Level3 → Down4 → Up4: {x3.shape} → {x4.shape} → {x3_up.shape}")