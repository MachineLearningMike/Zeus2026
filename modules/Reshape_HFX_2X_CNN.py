import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Reshape_Half_Flex import halve_window_size_optimized_flexibly

class HorizontalCyclicConv2d(nn.Module):
    """Lightweight convolution with horizontal-only cyclicity"""
    __slots__ = ('conv', 'pad_w')
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.pad_w = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=0, bias=True)
        
    def forward(self, x):
        # Fast path: only pad when needed
        if self.pad_w > 0:
            # Pad height with zeros
            x = F.pad(x, (0, 0, self.pad_w, self.pad_w), mode='constant', value=0)
            # Pad width with circular padding
            x = torch.cat([x[:, :, :, -self.pad_w:], x, x[:, :, :, :self.pad_w]], dim=3)
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
        
        # Use output_shape to configure the layer
        # For HFX, we expect output resolution to be roughly half of input
        # But we'll use adaptive pooling to guarantee exact output_shape
        
        # Feature extraction and channel expansion
        # self.conv = HorizontalCyclicConv2d(dim, 2*dim, kernel_size=3, stride=1)
        
        # Downsampling - stride=2 halves the dimensions
        self.down = HorizontalCyclicConv2d(dim, 2*dim, kernel_size=3, stride=2)
        
        # Adaptive pooling to force exact output_shape
        self.pool = nn.AdaptiveAvgPool2d((self.out_H, self.out_W))
        
        # self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)
        
        # Feature extraction
        # x = self.conv(x)
        
        # Channel expansion and downsampling
        x = self.down(x)
        
        # Force exact output_shape dimensions
        x = self.pool(x)
        
        # Convert back
        return x.permute(0, 2, 3, 1)


class Reshape_2FX_HX_CNN(nn.Module):    # 2X-Flex, Half using CNN
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Feature extraction
        # self.conv1 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1)
                
        # Adaptive pooling to force exact output_shape
        self.pool = nn.AdaptiveAvgPool2d((self.out_H, self.out_W))
        
        # Channel reduction
        self.reduce = nn.Conv2d(dim, dim//2, kernel_size=1, bias=True)
        
        # Refinement
        # self.refine = HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1)
        
        # self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)
        
        # Feature extraction
        # x = self.act(self.conv1(x))
        
        # Force exact output_shape dimensions
        x = self.pool(x)
        
        # Channel reduction
        x = self.reduce(x)
                
        # Refinement with horizontal cyclicity
        # x = self.act(self.refine(x))
        
        # Convert back
        return x.permute(0, 2, 3, 1)


# Even faster specialized version that uses output_shape
class Reshape_2FX_HX_CNN_Fast(nn.Module):
    """Fast version that uses output_shape to configure layers"""
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Single combined operation that directly outputs target dimensions
        # Use adaptive pooling to guarantee exact output_shape
        self.net = nn.Sequential(
            # HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1),
            # nn.GELU(),
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=True),
            # nn.GELU(),
            nn.AdaptiveAvgPool2d((self.out_H, self.out_W)),
            # HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1),
            # nn.GELU(),
        )
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        return x.permute(0, 2, 3, 1)


# Test the modules
if __name__ == "__main__":
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Testing Reshape_HFX_2X_CNN (now using output_shape)")
    print("="*60)
    
    model1 = Reshape_HFX_2X_CNN(
        dim=80,
        output_shape=((45, 90), 160),
        pad_dims=0
    ).to(device)
    model1.eval()
    
    x1 = torch.randn(32, 90, 180, 80).to(device)
    out1 = model1(x1)
    print(f"Input: {x1.shape}")
    print(f"Output: {out1.shape}")
    print(f"Expected: (32, 45, 90, 160)")
    print(f"Match: {out1.shape == (32, 45, 90, 160)}")
    
    print("\n" + "="*60)
    print("Testing Reshape_2FX_HX_CNN")
    print("="*60)
    
    model2 = Reshape_2FX_HX_CNN(
        dim=80,
        output_shape=((48, 96), 40),
        pad_dims=0
    ).to(device)
    model2.eval()
    
    x2 = torch.randn(32, 90, 180, 80).to(device)
    out2 = model2(x2)
    print(f"Input: {x2.shape}")
    print(f"Output: {out2.shape}")
    print(f"Expected: (32, 48, 96, 40)")
    print(f"Match: {out2.shape == (32, 48, 96, 40)}")
    
    print("\n" + "="*60)
    print("Testing Reshape_2FX_HX_CNN_Fast")
    print("="*60)
    
    model3 = Reshape_2FX_HX_CNN_Fast(
        dim=80,
        output_shape=((48, 96), 40),
        pad_dims=0
    ).to(device)
    model3.eval()
    
    out3 = model3(x2)
    print(f"Input: {x2.shape}")
    print(f"Output: {out3.shape}")
    print(f"Expected: (32, 48, 96, 40)")
    print(f"Match: {out3.shape == (32, 48, 96, 40)}")
    
    # Speed test
    print("\n" + "="*60)
    print("Speed Test (1000 runs)")
    print("="*60)
    
    # Warmup
    for _ in range(10):
        _ = model3(x2)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(1000):
        out3 = model3(x2)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"Time for 1000 runs: {time.time()-start:.3f}s")
    print(f"Per run: {(time.time()-start)/1000*1000:.3f}ms")