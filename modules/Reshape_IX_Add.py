import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.Zeus_Flash_MHA_Embedded_Query import Seq_Len_Changing_MHA

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

# Simplified downsampling module for (720,1440) -> (360,720)
# Option 1: Single conv with stride=2 (simplest, works for all cases)
# class DownsampleCNN(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.downsample = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
#             nn.GELU(),
#         )
    
#     def forward(self, x):
#         return self.downsample(x)  # Always halves the dimensions

# Option 2: If you want two conv layers (both with stride=1 after the first stride=2)
class DownsampleCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),  # (2H,2W) → (H,W)
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),  # Preserves (H,W)
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
        mid_dims = pad_dims // 2
        
        # Deeper pad processing: downsampling + refinement (no GELU)
        self.pad_process = nn.Sequential(
            # First conv: downsampling + channel reduction
            nn.Conv2d(pad_dims, mid_dims, kernel_size=3, stride=2, padding=1, bias=True),
            
            #------------- These extra refinement slows down convergence ---------------
            # # Second conv: refine at target resolution (stride=1)
            # nn.Conv2d(mid_dims, mid_dims, kernel_size=3, stride=1, padding=1, bias=True),
            # # Third conv for even deeper processing
            # nn.Conv2d(mid_dims, mid_dims, kernel_size=3, stride=1, padding=1, bias=True),
        )
        
        # Deeper fusion with larger kernel (no GELU)
        self.fusion = nn.Sequential(
            # First fusion conv with 3x3 kernel to mix spatial info
            nn.Conv2d(dim + mid_dims, self.out_dim, kernel_size=3, padding=1, bias=True),
            # # Final refinement with 3x3 at target channels
            # nn.Conv2d(self.out_dim, self.out_dim, kernel_size=3, padding=1, bias=True),
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

        # Pure CNN-based reduction without any normalization
        self.reduction_conv = nn.Sequential(
            # Depthwise conv for spatial mixing
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            
            # Pointwise conv for channel reduction
            nn.Conv2d(dim, out_dim, kernel_size=1, bias=True),
            
            # Another spatial conv for refinement
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=True),
        )
            
    def forward(self, x, ss=None):
        B, H, W, C = x.shape
        
        # CNN path without normalization
        x_cnn = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x_cnn = self.reduction_conv(x_cnn)  # (B, out_dim, H, W)
        x_cnn = x_cnn.permute(0, 2, 3, 1).contiguous()  # (B, H, W, out_dim)
        
        return x_cnn

# Simplified residual block without normalization
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        
        # Skip connection if dimensions change
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

# Lightweight version for specific downsampling task
class Reshape_IX_Add_Light(nn.Module):
    """Lightweight version for (720,1440) -> (360,720) downsampling"""
    
    @staticmethod
    def get_output_shape(input_shape, nWin, pWin, target_dim=0):
        in_res, in_dim = input_shape
        return (in_res, target_dim)

    def __init__(self, dim, output_shape, pad_dims=0):
        super().__init__()

        self.output_shape = output_shape
        self.pad_dims = pad_dims
        (self.out_H, self.out_W), self.out_dim = output_shape
        
        # Channel reduction
        self.pad_reduce = nn.Conv2d(pad_dims, pad_dims // 2, kernel_size=3, padding=1, bias=True)
        
        # Simple downsampling (720,1440) -> (360,720)
        self.pad_downsample = nn.Sequential(
            nn.Conv2d(pad_dims // 2, pad_dims // 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU()
        )
        
        # Simple main tensor processing
        self.main_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        
        # Final projection
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

# Simple test for the specific resolution
if __name__ == "__main__":
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