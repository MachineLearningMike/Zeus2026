import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.utils import resize_continuous_bilinear_channel_wise_cylindric

class EfficientWeatherInterpolation(torch.nn.Module):
    def __init__(self, target_lat, target_lon, pad_factor=0.25):
        super().__init__()
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.pad_factor = pad_factor
    
    def forward(self, x):
        # x shape: [batch, vars, lat, lon]
        batch, vars, lat, lon = x.shape
        
        # Adaptive padding
        pad = max(int(lon * self.pad_factor), 8)
        pad = min(pad, 64)
        
        # Manual circular padding (works in all PyTorch versions)
        # Split the tensor and concatenate for circular wrapping
        left_pad = x[:, :, :, -pad:]   # Take last 'pad' columns
        right_pad = x[:, :, :, :pad]    # Take first 'pad' columns
        
        # Concatenate: [right_pad, original, left_pad]
        # This creates circular wrapping
        x_padded = torch.cat([left_pad, x, right_pad], dim=3)  # Note: left_pad first for proper wrapping
        
        # Interpolate
        scaled = F.interpolate(
            x_padded,
            size=(self.target_lat, self.target_lon + 2*pad),
            mode='bilinear',
            align_corners=True
        )
        
        # Remove padding
        return scaled[:, :, :, pad:-pad]

# Test
x = torch.randn(4, 5, 45, 90)
interp = EfficientWeatherInterpolation(23, 45, pad_factor=0.25)
output = interp(x)
print(f"Output shape: {output.shape}")  # [4, 5, 23, 45]