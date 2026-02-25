import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly

class HorizontalCyclicConv2d(nn.Module):
    """Pre-defined convolution with horizontal-only cyclicity"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Calculate effective kernel size with dilation
        effective_kernel = (self.kernel_size[0] + (self.kernel_size[0]-1)*(self.dilation[0]-1),
                           self.kernel_size[1] + (self.kernel_size[1]-1)*(self.dilation[1]-1))
        self.pad = effective_kernel[0] // 2, effective_kernel[1] // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, dilation=dilation, padding=0, bias=True)
        
    def forward(self, x):
        # Pad height with zeros
        if self.pad[0] > 0:
            x = F.pad(x, (0, 0, self.pad[0], self.pad[0]), mode='constant', value=0)
        
        # Pad width with circular padding
        if self.pad[1] > 0:
            x = torch.cat([x[:, :, :, -self.pad[1]:], x, x[:, :, :, :self.pad[1]]], dim=3)
        
        return self.conv(x)


class Reshape_HFX_2X_CNN(nn.Module):    # Half-Flex, 2X using CNN
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        out_res = halve_window_size_optimized_flexibly(in_res, nWin, pWin)
        out_dim = 2 * in_dim
        return (out_res, out_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Feature extraction
        self.features = nn.Sequential(
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1),
            nn.GELU(),
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1),
            nn.GELU(),
        )
        
        # Channel expansion
        self.expand = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True)
        
        # Downsampling - stride=2 halves the dimensions
        self.down = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=2)
        
        # Final projection to exact output dimensions
        # Use adaptive pooling to ensure exact output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.out_H, self.out_W))
        
        # Refinement after pooling
        self.refine = nn.Sequential(
            HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=1),
            nn.GELU(),
            HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=1),
            nn.GELU(),
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Feature extraction
        x = self.features(x)
        
        # Channel expansion
        x = self.expand(x)
        x = F.gelu(x)
        
        # Downsample (halves spatial dimensions)
        x = self.down(x)
        x = F.gelu(x)
        
        # Force exact output size with adaptive pooling
        x = self.adaptive_pool(x)
        
        # Refinement
        x = self.refine(x)
        
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, 2*C)
        
        return x


class Reshape_2FX_HX_CNN(nn.Module):    # 2X-Flex, Half using CNN
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Feature extraction
        self.features = nn.Sequential(
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1),
            nn.GELU(),
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1),
            nn.GELU(),
        )
        
        # Channel reduction
        self.reduce = nn.Conv2d(dim, dim//2, kernel_size=1, bias=True)
        
        # Upsampling - stride=2 doubles the dimensions
        self.up = nn.ConvTranspose2d(dim//2, dim//2, kernel_size=4, stride=2, padding=1, bias=True)
        
        # Force exact output size with interpolation
        # We'll handle this in forward to avoid creating new layers
        
        # Refinement with horizontal cyclicity
        self.refine = nn.Sequential(
            HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1),
            nn.GELU(),
            HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1),
            nn.GELU(),
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Feature extraction
        x = self.features(x)
        
        # Channel reduction
        x = self.reduce(x)
        x = F.gelu(x)
        
        # Upsample (doubles spatial dimensions)
        x = self.up(x)
        x = F.gelu(x)
        
        # Force exact output size with interpolation
        if x.shape[2] != self.out_H or x.shape[3] != self.out_W:
            x = F.interpolate(x, size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
        
        # Refinement with horizontal cyclicity
        x = self.refine(x)
        
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, C//2)
        
        return x


# Test the modules
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Reshape_HFX_2X_CNN")
    print("="*60)
    
    model1 = Reshape_HFX_2X_CNN(
        dim=80,
        output_shape=((45, 90), 160),
        pad_dims=0
    )
    model1.eval()
    
    x1 = torch.randn(2, 90, 180, 80)
    out1 = model1(x1)
    print(f"Input: {x1.shape} -> Output: {out1.shape}")
    print(f"Expected: (2, 45, 90, 160)")
    print(f"Match: {out1.shape == (2, 45, 90, 160)}")
    
    print("\n" + "="*60)
    print("Testing Reshape_2FX_HX_CNN")
    print("="*60)
    
    model2 = Reshape_2FX_HX_CNN(
        dim=80,
        output_shape=((48, 96), 40),
        pad_dims=0
    )
    model2.eval()
    
    x2 = torch.randn(2, 90, 180, 80)
    out2 = model2(x2)
    print(f"Input: {x2.shape} -> Output: {out2.shape}")
    print(f"Expected: (2, 48, 96, 40)")
    print(f"Match: {out2.shape == (2, 48, 96, 40)}")
    
    # Test with different shape for 2FX
    print("\n" + "="*60)
    print("Testing Reshape_2FX_HX_CNN with different input")
    print("="*60)
    
    model3 = Reshape_2FX_HX_CNN(
        dim=64,
        output_shape=((64, 128), 32),
        pad_dims=0
    )
    model3.eval()
    
    x3 = torch.randn(2, 32, 64, 64)
    out3 = model3(x3)
    print(f"Input: {x3.shape} -> Output: {out3.shape}")
    print(f"Expected: (2, 64, 128, 32)")
    print(f"Match: {out3.shape == (2, 64, 128, 32)}")
    
    # Speed test
    print("\n" + "="*60)
    print("Speed Test")
    print("="*60)
    
    import time
    
    # Warmup
    for _ in range(5):
        _ = model1(x1)
        _ = model2(x2)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        out1 = model1(x1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"HFX time (100 runs): {time.time()-start:.3f}s")
    
    start = time.time()
    for _ in range(100):
        out2 = model2(x2)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"2FX time (100 runs): {time.time()-start:.3f}s")