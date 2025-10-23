import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Suppose X is your tensor: shape (B, T, C, H, W)
B, T, C, H, W = 2, 4, 3, 32, 32
X = torch.randn(B, T, C, H, W)

# Pick random b, t, c indices
b = torch.randint(0, B, (1,)).item()
t = torch.randint(0, T, (1,)).item()
c = torch.randint(0, C, (1,)).item()
print(f"Selected slice: b={b}, t={t}, c={c}")

# Extract 2D slice
slice_2d = X[b, t, c].cpu().numpy()  # shape (H, W)

# Create coordinate grid
x = np.arange(W)
y = np.arange(H)
Xg, Yg = np.meshgrid(x, y)

# Plot as 3D surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(Xg, Yg, slice_2d, cmap='viridis', edgecolor='none')

ax.set_xlabel('Width')
ax.set_ylabel('Height')
ax.set_zlabel('Value')
ax.set_title(f"3D Surface of X[b={b}, t={t}, c={c}]")

plt.tight_layout()
plt.show()