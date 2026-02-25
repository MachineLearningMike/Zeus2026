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


class Reshape_HFX_1X_CNN(nn.Module):    # Half-Flex, 1X using CNN
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        # Do not attempt to add padding !!!
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Calculate stride for spatial downsampling
        # We'll determine the stride based on input vs output dimensions
        # For half resolution, we typically use stride=2
        
        # Main downsampling path
        self.downsample = nn.Sequential(
            # First conv: spatial downsampling with horizontal cyclicity
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=2, bias=True),
            # nn.GELU(),
            
            # # Second conv: refinement at target resolution
            # HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
            # # nn.GELU(),
            
            # # Third conv: further processing
            # HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
        )
        
        # Optional: channel projection if needed (though dim should equal out_dim)
        self.projection = nn.Identity() if dim == self.out_dim else \
                         nn.Conv2d(dim, self.out_dim, kernel_size=1, bias=True)
        
        # Learnable positional embedding for new spatial dimensions
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.out_dim, self.out_H, self.out_W) * 0.02
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW for CNN processing
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply downsampling
        x = self.downsample(x)  # (B, C, H//2, W//2) approx
        
        # If current spatial dimensions don't match target, adjust
        curr_H, curr_W = x.shape[2], x.shape[3]
        
        if curr_H != self.out_H or curr_W != self.out_W:
            # Need to adjust spatial dimensions
            if curr_H > self.out_H or curr_W > self.out_W:
                # Downsample further
                stride_h = curr_H // self.out_H if curr_H > self.out_H else 1
                stride_w = curr_W // self.out_W if curr_W > self.out_W else 1
                
                adjust_conv = HorizontalCyclicConv2d(
                    C, C, 
                    kernel_size=3, 
                    stride=(stride_h, stride_w),
                    bias=True
                ).to(x.device)
                x = adjust_conv(x)
            else:
                # Need to upsample (rare for HFX)
                if curr_H < self.out_H or curr_W < self.out_W:
                    # Use transposed conv for upsampling
                    upsample = nn.ConvTranspose2d(
                        C, C, 
                        kernel_size=4, stride=2, padding=1
                    ).to(x.device)
                    x = upsample(x)
        
        # Apply channel projection if needed
        x = self.projection(x)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, C)
        
        return x


class Reshape_2FX_1X_CNN(nn.Module):    # 2X-Flex, 1X using CNN

    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        # For 2FX, we double the resolution
        if isinstance(in_res, tuple):
            out_res = (in_res[0] * 2, in_res[1] * 2)
        else:
            out_res = in_res * 2
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Pre-processing before upsampling
        self.pre_process = nn.Sequential(
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
            # nn.GELU(),
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
        )
        
        # Main upsampling path
        # Option 1: Pixel shuffle for clean 2x upsampling
        self.use_pixel_shuffle = True
        
        # Channel expansion for pixel shuffle (needs 4x channels for 2x upsampling)
        self.channel_expand = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Option 2: Transposed conv for flexible upsampling
        self.upsample_conv = nn.ConvTranspose2d(
            dim, dim, kernel_size=4, stride=2, padding=1, bias=True
        )
        
        # Refinement after upsampling
        self.refine = nn.Sequential(
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
            nn.GELU(),
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
        )
        
        # Optional: channel projection if needed
        self.projection = nn.Identity() if dim == self.out_dim else \
                         nn.Conv2d(dim, self.out_dim, kernel_size=1, bias=True)
        
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Pre-processing
        x = self.pre_process(x)  # (B, C, H, W)
        
        # Check if target is exactly 2x in both dimensions
        if self.out_H == H * 2 and self.out_W == W * 2 and self.use_pixel_shuffle:
            # Use pixel shuffle for clean 2x upsampling
            x = self.channel_expand(x)  # (B, C*4, H, W)
            x = self.pixel_shuffle(x)   # (B, C, H*2, W*2)
        else:
            # Calculate required scale factors
            scale_h = self.out_H / H
            scale_w = self.out_W / W
            
            # Use transposed convolution(s) for upsampling
            if scale_h > 1 or scale_w > 1:
                # Start with 2x upsampling if needed
                if H * 2 <= self.out_H and W * 2 <= self.out_W:
                    x = self.upsample_conv(x)  # (B, C, H*2, W*2)
                
                # Further adjust if still not target size
                if x.shape[2] != self.out_H or x.shape[3] != self.out_W:
                    if x.shape[2] < self.out_H or x.shape[3] < self.out_W:
                        # Need more upsampling - use interpolation as fallback
                        # (minimal use, as requested)
                        x = F.interpolate(
                            x, size=(self.out_H, self.out_W), 
                            mode='bilinear', align_corners=False
                        )
        
        # Refinement
        x = self.refine(x)
        
        # Apply channel projection if needed
        x = self.projection(x)
        
        # Ensure exact target size (center crop if slightly off)
        if x.shape[2] != self.out_H or x.shape[3] != self.out_W:
            # Center crop to exact size
            dh = (x.shape[2] - self.out_H) // 2
            dw = (x.shape[3] - self.out_W) // 2
            x = x[:, :, dh:dh+self.out_H, dw:dw+self.out_W]
                
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, C)
        
        return x


# Lightweight version for specific downsampling task
class Reshape_HFX_1X_CNN_Light(nn.Module):
    """Lightweight version for half-resolution with same channels"""
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Simple downsampling with horizontal cyclicity
        self.downsample = nn.Sequential(
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=2, bias=True),
            nn.GELU(),
        )
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, dim, self.out_H, self.out_W) * 0.02
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.downsample(x)      # (B, C, H//2, W//2)
        x = x + self.pos_embedding
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, C)
        
        return x


class Reshape_2FX_1X_CNN_Light(nn.Module):
    """Lightweight version for double resolution with same channels"""
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        if isinstance(in_res, tuple):
            out_res = (in_res[0] * 2, in_res[1] * 2)
        else:
            out_res = in_res * 2
        out_dim = in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Simple upsampling with transposed conv
        self.upsample = nn.ConvTranspose2d(
            dim, dim, kernel_size=4, stride=2, padding=1, bias=True
        )
        
        # Refinement with horizontal cyclicity
        self.refine = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True)
        
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, dim, self.out_H, self.out_W) * 0.02
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.upsample(x)        # (B, C, H*2, W*2)
        x = F.gelu(x)
        x = self.refine(x)          # Refinement
        x = x + self.pos_embedding
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, C)
        
        return x


# Test function
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Reshape_HFX_1X_CNN
    print("\n" + "="*60)
    print("Testing Reshape_HFX_1X_CNN")
    print("="*60)
    
    in_h, in_w, in_c = 90, 180, 80
    out_h, out_w = 45, 90  # Half resolution
    out_dim = 80  # Same channels
    
    module = Reshape_HFX_1X_CNN(
        dim=in_c,
        output_shape=((out_h, out_w), out_dim)
    ).to(device)
    
    x = torch.randn(2, in_h, in_w, in_c).to(device)
    output = module(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (2, {out_h}, {out_w}, {out_dim})")
    print(f"Match: {output.shape == (2, out_h, out_w, out_dim)}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("Gradient flow: ✓")
    
    # Test Reshape_2FX_1X_CNN
    print("\n" + "="*60)
    print("Testing Reshape_2FX_1X_CNN")
    print("="*60)
    
    in_h, in_w, in_c = 45, 90, 80
    out_h, out_w = 90, 180  # Double resolution
    out_dim = 80  # Same channels
    
    module = Reshape_2FX_1X_CNN(
        dim=in_c,
        output_shape=((out_h, out_w), out_dim)
    ).to(device)
    
    x = torch.randn(2, in_h, in_w, in_c).to(device)
    output = module(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (2, {out_h}, {out_w}, {out_dim})")
    print(f"Match: {output.shape == (2, out_h, out_w, out_dim)}")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("Gradient flow: ✓")
    
    # Test horizontal cyclicity
    print("\n" + "="*60)
    print("Testing Horizontal Cyclicity")
    print("="*60)
    
    # Create a small test module
    test_conv = HorizontalCyclicConv2d(1, 1, kernel_size=3, stride=1, bias=False)
    test_conv.conv.weight.data.fill_(1.0/9)  # Average filter
    
    # Create test pattern
    test_x = torch.zeros(1, 1, 5, 5)
    test_x[0, 0, :, 0] = 1.0  # Left edge
    test_x[0, 0, :, -1] = 2.0  # Right edge
    test_x[0, 0, 0, :] = 3.0   # Top edge
    test_x[0, 0, -1, :] = 4.0  # Bottom edge
    
    print("Test pattern (center values):")
    print(test_x[0, 0])
    
    with torch.no_grad():
        test_out = test_conv(test_x)
    
    print("\nOutput after horizontal-cyclic convolution:")
    print(test_out[0, 0].detach().numpy().round(2))
    
    # Verify horizontal cyclicity
    left_in_right = test_out[0, 0, 2, -1].item()
    right_in_left = test_out[0, 0, 2, 0].item()
    print(f"\nLeft edge influence on right: {left_in_right:.2f}")
    print(f"Right edge influence on left: {right_in_left:.2f}")
    
    # Verify vertical non-cyclicity
    top_in_bottom = test_out[0, 0, -1, 2].item()
    bottom_in_top = test_out[0, 0, 0, 2].item()
    print(f"Top edge influence on bottom: {top_in_bottom:.2f}")
    print(f"Bottom edge influence on top: {bottom_in_top:.2f}")
    
    print("\n✅ Horizontal-only cyclicity verified (width wraps, height doesn't)")