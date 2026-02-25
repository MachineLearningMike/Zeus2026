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
        
        # Pre-defined layers - all with horizontal cyclicity
        self.conv1 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1)
        self.conv2 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1)
        self.conv3 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, dilation=2)
        self.conv4 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, dilation=3)
        
        # Channel expansion
        self.expand = nn.Conv2d(dim, 2*dim, kernel_size=1, bias=True)
        
        # Downsampling convolutions at different strides
        self.down2 = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=2)
        self.down3 = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=3)
        self.down4 = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=4)
        
        # Refinement
        self.refine1 = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=1)
        self.refine2 = HorizontalCyclicConv2d(2*dim, 2*dim, kernel_size=3, stride=1)
        
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Feature extraction
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        
        # Channel expansion
        x = self.act(self.expand(x))  # (B, 2*C, H, W)
        
        # Calculate required downsampling
        h_ratio = H / self.out_H
        w_ratio = W / self.out_W
        
        # Choose appropriate downsampling based on ratio
        if h_ratio >= 3.5 and w_ratio >= 3.5:
            x = self.act(self.down4(x))
        elif h_ratio >= 2.5 and w_ratio >= 2.5:
            x = self.act(self.down3(x))
        elif h_ratio >= 1.5 and w_ratio >= 1.5:
            x = self.act(self.down2(x))
        
        # Adaptive pooling for exact size (fast and exact)
        if x.shape[2] != self.out_H or x.shape[3] != self.out_W:
            x = F.adaptive_avg_pool2d(x, (self.out_H, self.out_W))
        
        # Refinement
        x = self.act(self.refine1(x))
        x = self.act(self.refine2(x))
        
        # Convert back to NHWC
        x = x.permute(0, 2, 3, 1)  # (B, out_H, out_W, 2*C)
        
        return x


class Reshape_2FX_HX_CNN(nn.Module):    # 2X-Flex, Half using CNN
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        (self.out_H, self.out_W), self.out_dim = output_shape
        self.dim = dim
        
        # Feature extraction (horizontal cyclic)
        self.conv1 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1)
        self.conv2 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1)
        self.conv3 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, dilation=2)
        self.conv4 = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, dilation=3)
        
        # Channel reduction
        self.reduce = nn.Conv2d(dim, dim//2, kernel_size=1, bias=True)
        
        # Upsampling layers
        self.upsample2 = nn.ConvTranspose2d(dim//2, dim//2, kernel_size=4, stride=2, padding=1, bias=True)
        self.upsample3 = nn.ConvTranspose2d(dim//2, dim//2, kernel_size=6, stride=3, padding=1, bias=True)
        self.upsample4 = nn.ConvTranspose2d(dim//2, dim//2, kernel_size=8, stride=4, padding=2, bias=True)
        
        # Refinement (horizontal cyclic)
        self.refine1 = HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1)
        self.refine2 = HorizontalCyclicConv2d(dim//2, dim//2, kernel_size=3, stride=1)
        
        self.act = nn.GELU()
        
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Feature extraction
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        
        # Channel reduction
        x = self.act(self.reduce(x))  # (B, C//2, H, W)
        
        # Calculate required upsampling
        h_ratio = self.out_H / H
        w_ratio = self.out_W / W
        
        # Apply appropriate upsampling
        if h_ratio >= 3.5 and w_ratio >= 3.5:
            x = self.act(self.upsample4(x))
        elif h_ratio >= 2.5 and w_ratio >= 2.5:
            x = self.act(self.upsample3(x))
        elif h_ratio >= 1.5 and w_ratio >= 1.5:
            x = self.act(self.upsample2(x))
        
        # Adaptive upsampling for exact size (fast)
        if x.shape[2] != self.out_H or x.shape[3] != self.out_W:
            x = F.interpolate(x, size=(self.out_H, self.out_W), mode='bilinear', align_corners=False)
        
        # Refinement with horizontal cyclicity
        x = self.act(self.refine1(x))
        x = self.act(self.refine2(x))
        
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
    
    x1 = torch.randn(2, 90, 180, 80)
    import time
    start = time.time()
    out1 = model1(x1)
    print(f"HFX time: {time.time()-start:.3f}s")
    print(f"Input: {x1.shape} -> Output: {out1.shape}")
    
    print("\n" + "="*60)
    print("Testing Reshape_2FX_HX_CNN")
    print("="*60)
    
    model2 = Reshape_2FX_HX_CNN(
        dim=80,
        output_shape=((48, 96), 40),
        pad_dims=0
    )
    
    x2 = torch.randn(2, 90, 180, 80)
    start = time.time()
    out2 = model2(x2)
    print(f"2FX time: {time.time()-start:.3f}s")
    print(f"Input: {x2.shape} -> Output: {out2.shape}")