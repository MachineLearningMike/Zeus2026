
import torch
import torch.nn as nn
import torch.nn.functional as F

class CylindricalConv2d(nn.Module):
    """
    Convolution on a cylindrical surface:
    - Periodic along width (axis=3)
    - Normal padding along height (axis=2)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0,0), dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.pad_h, self.pad_w = padding
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),  # we handle padding manually
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 1. Pad vertically with zeros (or any other mode you prefer)
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0)

        # 2. Pad horizontally with circular wrap-around
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")

        # 3. Apply convolution
        return self.conv(x)
    
    