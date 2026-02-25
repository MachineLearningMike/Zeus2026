import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Zeus_Flash_MHA_Embedded_Query import Seq_Len_Changing_MHA

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

# SwiGLU (unchanged)
class SwiGLU(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension,dimension)
        self.linear_2 = nn.Linear(dimension,dimension)

    def forward(self, x):
        output = self.linear_1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.linear_2(x)
        return swiglu

# ---------------------------
# MLP (unchanged)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DownsampleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            # Use horizontal-cyclic conv for downsampling
            HorizontalCyclicConv2d(in_channels, in_channels, kernel_size=3, stride=2),
            nn.GELU(),
            HorizontalCyclicConv2d(in_channels, in_channels, kernel_size=3, stride=1),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.downsample(x)

class Reshape_IX_Add(nn.Module):
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        return (in_res, target_dim)
    
    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()
        
        (self.out_H, self.out_W), self.out_dim = output_shape
        
        # Mid dimension for pad processing
        mid_dims = pad_dims # // 2
        
        # Deeper pad processing: downsampling + refinement (no GELU)
        # Using custom horizontal-only cyclic convolution
        self.pad_process = nn.Sequential(
            # First conv: downsampling + channel reduction with horizontal-only cyclicity
            HorizontalCyclicConv2d(pad_dims, mid_dims, kernel_size=3, stride=2, bias=True),
        )
        
        # Deeper fusion with larger kernel - horizontal-only cyclicity
        self.fusion = nn.Sequential(
            # First fusion conv with 3x3 kernel to mix spatial info
            HorizontalCyclicConv2d(dim + mid_dims, self.out_dim, kernel_size=3, stride=1, bias=True),
        )

    def forward(self, x, ss=None):
        x_main, x_pad = x
        
        # Process pad: (720,1440,15) -> (360,720,mid_dims)
        x_pad = x_pad.permute(0, 3, 1, 2)  # (B,15,720,1440)
        x_pad = self.pad_process(x_pad)    # (B,mid_dims,360,720)
        x_pad = x_pad.permute(0, 2, 3, 1)  # (B,360,720,mid_dims)
        
        # Concatenate and fuse
        x = torch.cat([x_main, x_pad], dim=-1)  # (B,360,720,dim+mid_dims)
        x = x.permute(0, 3, 1, 2)               # (B,dim+mid_dims,360,720)
        x = self.fusion(x)                       # (B,20,360,720)
        x = x.permute(0, 2, 3, 1)               # (B,360,720,20)
        
        return x


class Reshape_IX_Remove(nn.Module):

    def __init__(self, dim, output_shape=None, pad_dims=0):
        super().__init__()
        self.dim = dim
        self.output_shape = output_shape
        _, out_dim = output_shape

        # Pure CNN-based reduction with horizontal-only cyclicity
        self.reduction_conv = nn.Sequential(
            # Depthwise conv for spatial mixing with horizontal-only cyclicity
            HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True),
            
            # Pointwise conv for channel reduction (1x1 doesn't need cyclicity)
            nn.Conv2d(dim, out_dim, kernel_size=1, bias=True),
            
            # Another spatial conv for refinement with horizontal-only cyclicity
            HorizontalCyclicConv2d(out_dim, out_dim, kernel_size=3, stride=1, bias=True),
        )
            
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # CNN path without normalization
        x_cnn = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x_cnn = self.reduction_conv(x_cnn)  # (B, out_dim, H, W)
        x_cnn = x_cnn.permute(0, 2, 3, 1).contiguous()  # (B, H, W, out_dim)
        
        return x_cnn

# Simplified residual block with horizontal-only cyclicity
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Use horizontal-cyclic convs for spatial layers
        self.conv1 = HorizontalCyclicConv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=True)
        self.conv2 = HorizontalCyclicConv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=True)
        
        # Skip connection if dimensions change (1x1 conv doesn't need cyclicity)
        self.skip = nn.Identity() if in_channels == out_channels else \
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.gelu(x)
        out = self.conv1(out)
        out = F.gelu(out)
        out = self.conv2(out)
        
        out = out + identity
        return out

# Lightweight version with horizontal-only cyclicity
class Reshape_IX_Add_Light(nn.Module):
    """Lightweight version for (720,1440) -> (360,720) downsampling with horizontal-only cyclicity"""
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        return (in_res, target_dim)

    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()

        self.output_shape = output_shape
        self.pad_dims = pad_dims
        (self.out_H, self.out_W), self.out_dim = output_shape
        
        # Channel reduction with horizontal-only cyclicity
        self.pad_reduce = HorizontalCyclicConv2d(pad_dims, pad_dims // 2, kernel_size=3, stride=1, bias=True)
        
        # Simple downsampling (720,1440) -> (360,720) with horizontal-only cyclicity
        self.pad_downsample = nn.Sequential(
            HorizontalCyclicConv2d(pad_dims // 2, pad_dims // 2, kernel_size=3, stride=2, bias=True),
            nn.GELU()
        )
        
        # Simple main tensor processing with horizontal-only cyclicity
        self.main_conv = HorizontalCyclicConv2d(dim, dim, kernel_size=3, stride=1, bias=True)
        
        # Final projection (1x1 conv doesn't need cyclicity)
        self.expansion = nn.Conv2d(dim + pad_dims // 2, self.out_dim, kernel_size=1, bias=True)

    def forward(self, x, ss=None):
        x_main, x_pad = x

        # Process pad
        B, H, W, C = x_pad.shape
        x_pad = x_pad.permute(0, 3, 1, 2).contiguous()
        x_pad = F.gelu(self.pad_reduce(x_pad))
        x_pad = self.pad_downsample(x_pad)  # Downsample to (360,720)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()

        # Process main
        x_main = x_main.permute(0, 3, 1, 2).contiguous()
        x_main = F.gelu(self.main_conv(x_main))
        x_main = x_main.permute(0, 2, 3, 1).contiguous()

        # Combine
        x = torch.cat([x_main, x_pad], dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.expansion(x)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

# Test function to verify horizontal-only cyclicity
def test_horizontal_cyclicity():
    """Test that convolutions are cyclic only in width dimension"""
    
    # Create a simple pattern
    x = torch.zeros(1, 1, 5, 5)
    # Mark leftmost column with 1.0
    x[0, 0, :, 0] = 1.0
    # Mark rightmost column with -1.0
    x[0, 0, :, -1] = -1.0
    # Mark top row with 0.5
    x[0, 0, 0, :] = 0.5
    # Mark bottom row with -0.5
    x[0, 0, -1, :] = -0.5
    
    print("Test pattern:")
    print(x[0, 0])
    
    # Test with standard conv (should have no cyclicity)
    standard_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    standard_conv.weight.data.fill_(1.0)  # Set weights to 1 for simple averaging
    
    # Test with circular conv (both dimensions cyclic)
    circular_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode='circular', bias=False)
    circular_conv.weight.data.fill_(1.0)
    
    # Test with our horizontal-only conv
    horizontal_conv = HorizontalCyclicConv2d(1, 1, kernel_size=3, stride=1, bias=False)
    horizontal_conv.conv.weight.data.fill_(1.0)
    
    with torch.no_grad():
        out_standard = standard_conv(x)
        out_circular = circular_conv(x)
        out_horizontal = horizontal_conv(x)
    
    print("\nStandard conv (no cyclicity):")
    print(out_standard[0, 0].round(decimals=2))
    
    print("\nCircular conv (both dimensions cyclic):")
    print(out_circular[0, 0].round(decimals=2))
    
    print("\nHorizontal-only conv (only width cyclic):")
    print(out_horizontal[0, 0].round(decimals=2))
    
    # Verify horizontal cyclicity: left and right edges should be similar
    left_edge = out_horizontal[0, 0, :, 0].mean()
    right_edge = out_horizontal[0, 0, :, -1].mean()
    print(f"\nLeft edge mean: {left_edge:.2f}, Right edge mean: {right_edge:.2f}")
    
    # Verify vertical non-cyclicity: top and bottom edges should be different
    top_edge = out_horizontal[0, 0, 0, :].mean()
    bottom_edge = out_horizontal[0, 0, -1, :].mean()
    print(f"Top edge mean: {top_edge:.2f}, Bottom edge mean: {bottom_edge:.2f}")

# Simple test for the specific resolution
if __name__ == "__main__":
    # Run cyclicity test
    print("="*60)
    print("Testing horizontal-only cyclicity")
    print("="*60)
    test_horizontal_cyclicity()
    
    print("\n" + "="*60)
    print("Testing with specific resolution")
    print("="*60)
    
    # Test with the specific resolution
    batch_size = 1
    dim = 64
    pad_dims = 128
    out_dim = 96
    
    # Input shapes
    in_H, in_W = 720, 1440
    out_H, out_W = 360, 720
    
    print(f"Testing with input shape: ({in_H}, {in_W}) -> output shape: ({out_H}, {out_W})")
    
    # Create model
    model = Reshape_IX_Add(
        dim=dim,
        output_shape=((out_H, out_W), out_dim),
        pad_dims=pad_dims
    )
    
    # Create dummy inputs
    x_main = torch.randn(batch_size, out_H, out_W, dim)
    x_pad = torch.randn(batch_size, in_H, in_W, pad_dims)
    
    # Forward pass
    output = model((x_main, x_pad))
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {out_H}, {out_W}, {out_dim})")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print("Backward pass successful")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")