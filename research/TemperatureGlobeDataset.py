import torch
from torch.utils.data import Dataset, DataLoader
import math

class TemperatureGlobeDataset(Dataset):
    def __init__(self, B, C, H, W, n, m, length=100):
        """
        Simulated temperature over the globe:
        - B: batch size
        - C: number of channels (e.g., different "variables")
        - H: number of latitude points
        - W: number of longitude points
        - n: number of input frames
        - m: number of target frames
        - length: number of sequences
        """
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.n = n
        self.m = m
        self.length = length

        # Create latitude and longitude grids in radians
        lat = torch.linspace(-math.pi/2, math.pi/2, H)   # -90° to +90°
        lon = torch.linspace(-math.pi, math.pi, W)       # -180° to +180°
        self.lat_grid, self.lon_grid = torch.meshgrid(lat, lon, indexing='ij')
        self.obliqhity = math.pi / 180 * (90-23.44)

    def __len__(self):
        return self.length

class TemperatureGlobeDataset(Dataset):
    def __init__(self, B, C, H, W, n, m, length=100):
        """
        Simulated temperature over the globe:
        - B: batch size
        - C: number of channels (e.g., different "variables")
        - H: number of latitude points
        - W: number of longitude points
        - n: number of input frames
        - m: number of target frames
        - length: number of sequences
        """
        self.B = B
        self.C = C
        self.H = H
        self.W = W
        self.n = n
        self.m = m
        self.length = length

        # Create latitude and longitude grids in radians
        lat = torch.linspace(-math.pi/2, math.pi/2, H)   # -90° to +90°
        lon = torch.linspace(-math.pi, math.pi, W)       # -180° to +180°
        self.lat_grid, self.lon_grid = torch.meshgrid(lat, lon, indexing='ij')
        self.obliquity = math.pi / 180 * 23.44

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seq_len = self.n + self.m
        x_seq = torch.zeros(self.C, seq_len, self.H, self.W)  # (C, seq_len, H, W)

        for t in range(seq_len):
            temp_pattern = (torch.cos(self.lat_grid + self.obliquity) + 1.5) * (torch.cos(self.lon_grid + t * math.pi/12) + 1.5)
            
            for c in range(self.C):
                x_seq[c, t, :, :] = temp_pattern * math.sin(c + 1)

        return x_seq.permute(1, 0, 2, 3)  # (seq_len, C, H, W)
    

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

# Create dataset
dataset = TemperatureGlobeDataset(B=1, C=1, H=72, W=144, n=24, m=24)
x_seq = dataset[0]  # (seq_len, C, H, W)

seq_len, C, H, W = x_seq.shape

# Set up figure
fig, ax = plt.subplots(figsize=(6, 3))
im = ax.imshow(x_seq[0, 0], origin='lower', cmap='coolwarm', extent=[-180, 180, -90, 90])
plt.colorbar(im, ax=ax, label='Temperature (arbitrary units)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Simulated Temperature')

def update(frame_idx):
    im.set_data(x_seq[frame_idx, 0])
    ax.set_title(f"Simulated Temperature - Frame {frame_idx}")
    return [im]

ani = FuncAnimation(fig, update, frames=seq_len, interval=100, blit=True)

plt.show()