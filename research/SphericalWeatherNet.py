import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.ndimage import gaussian_filter

S2Activation = F.relu
from s2cnn import S2Convolution
# # from s2cnn import S2Convolution, S2Activation

# ------------------------------
# 1️⃣ Spherical CNN Model
# ------------------------------
class SphericalWeatherNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1, bandwidth=64):
        super().__init__()
        self.conv1 = S2Convolution(in_channels, 32, bandwidth, bandwidth//2)
        self.act1 = S2Activation(nn.ReLU())
        self.conv2 = S2Convolution(32, 64, bandwidth//2, bandwidth//4)
        self.act2 = S2Activation(nn.ReLU())
        self.conv3 = S2Convolution(64, out_channels, bandwidth//4, bandwidth//8)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        return x

# ------------------------------
# 2️⃣ Grid + mask helper
# ------------------------------
def make_era5_to_s2cnn_grid_lat_range(lat, lon, bandwidth=64, lat_min=0, lat_max=90):
    b = bandwidth
    beta = np.linspace(0, np.pi, 2*b, endpoint=False)
    alpha = np.linspace(0, 2*np.pi, 2*b, endpoint=False)
    beta_grid, alpha_grid = np.meshgrid(beta, alpha, indexing="ij")
    lat_sph = 90 - np.degrees(beta_grid)
    lon_sph = np.degrees(alpha_grid)
    lon_norm = (lon_sph / 180.0) - 1.0
    lat_norm = -lat_sph / 90.0
    grid = np.stack([lon_norm, lat_norm], axis=-1)
    grid = torch.from_numpy(grid).unsqueeze(0).float().cuda()
    mask = ((lat_sph >= lat_min) & (lat_sph <= lat_max)).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
    return grid, mask

# ------------------------------
# 3️⃣ ERA5 -> S2CNN conversion
# ------------------------------
def era5_to_s2cnn_batch(X, grid_s2cnn, mask=None):
    is_sequence = (X.dim() == 4)
    if is_sequence:
        T, C, H, W = X.shape
        X_flat = X.reshape(T*C, 1, H, W)
    else:
        C, H, W = X.shape
        X_flat = X.unsqueeze(0)
    X_s2cnn = F.grid_sample(X_flat, grid_s2cnn, align_corners=True)
    if mask is not None:
        X_s2cnn *= mask
    if is_sequence:
        X_s2cnn = X_s2cnn.reshape(T, C, X_s2cnn.shape[-2], X_s2cnn.shape[-1])
    else:
        X_s2cnn = X_s2cnn.squeeze(0)
    return X_s2cnn

# ------------------------------
# 4️⃣ S2CNN -> ERA5 conversion
# ------------------------------
def s2cnn_to_era5_batch(Y_s2cnn, lat, lon):
    is_sequence = (Y_s2cnn.dim() == 5)
    if is_sequence:
        B, T, C, Hs, Ws = Y_s2cnn.shape
        Y_flat = Y_s2cnn.reshape(B*T, C, Hs, Ws)
    else:
        B, C, Hs, Ws = Y_s2cnn.shape
        Y_flat = Y_s2cnn
    H, W = len(lat), len(lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_norm = (lon_grid / 180.0) - 1.0
    lat_norm = -lat_grid / 90.0
    grid = np.stack([lon_norm, lat_norm], axis=-1)
    grid = torch.from_numpy(grid).unsqueeze(0).float().cuda()
    Y_era5 = F.grid_sample(Y_flat, grid, align_corners=True)
    if is_sequence:
        Y_era5 = Y_era5.reshape(B, T, C, H, W)
    return Y_era5

# ------------------------------
# 5️⃣ Dataset (streaming version)
# ------------------------------
class ERA5StreamingDataset(Dataset):
    """
    Dummy ERA5 dataset generator.
    Produces smooth, realistic random data for testing SphericalWeatherNet or similar models.
    """

    def __init__(self, file_list=None, T_in=6, lat_crop=(0, 90), bandwidth=30):
        super().__init__()
        print("[Info] Initializing dummy ERA5StreamingDataset (random synthetic data).")

        # Parameters
        self.T_in = T_in
        self.lat_crop = lat_crop
        self.bandwidth = bandwidth
        self.num_samples = 1000
        self.channels = 4  # Example: [T2M, U10, V10, MSLP]
        self.lat_size = 721
        self.lon_size = 1440

        # Generate base random data
        raw = np.random.randn(self.num_samples, self.channels, self.lat_size, self.lon_size).astype(np.float32)

        # Add spatial smoothness
        for i in range(self.num_samples):
            for c in range(self.channels):
                raw[i, c] = gaussian_filter(raw[i, c], sigma=8.0)

        # Add a latitude-dependent gradient (like temperature decreasing toward poles)
        lats = np.linspace(90, -90, self.lat_size).reshape(1, 1, -1, 1)
        lat_gradient = np.cos(np.deg2rad(lats))
        raw[:, 0] += 0.5 * lat_gradient  # apply to temperature-like channel

        # Crop to selected latitudes
        lat0, lat1 = lat_crop
        lat_idx0 = int((90 - lat1) / 0.25)
        lat_idx1 = int((90 - lat0) / 0.25)
        self.data = raw[:, :, lat_idx1:lat_idx0, :]

        print(f"[Info] Dummy data generated with shape {self.data.shape} (samples, channels, lat, lon)")

    def __len__(self):
        # Example: return total samples minus T_in (to form input-output pairs)
        return self.num_samples - self.T_in

    def __getitem__(self, idx):
        # Return a sequence of T_in timesteps + next-step target
        x = self.data[idx : idx + self.T_in]
        y = self.data[idx + self.T_in]

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y

# ------------------------------
# 6️⃣ Recursive multi-day forecast
# ------------------------------
def recursive_forecast(model, X_init, steps, grid_s2cnn, mask=None, lat=None, lon=None):
    B, T_in, C_in, H, W = X_init.shape
    X_seq = X_init.clone().cuda()
    forecasts = []

    for t in range(steps):
        X_s2cnn = era5_to_s2cnn_batch(X_seq, grid_s2cnn, mask)
        X_flat = X_s2cnn.reshape(B, T_in*C_in, X_s2cnn.shape[-2], X_s2cnn.shape[-1])
        with torch.no_grad():
            y_pred_s2cnn = model(X_flat)
            y_pred_era5 = s2cnn_to_era5_batch(y_pred_s2cnn, lat, lon)
        forecasts.append(y_pred_era5.cpu())

        # Update input sequence
        if T_in > 1:
            X_seq = torch.cat([X_seq[:,1:,:,:,:], y_pred_era5.unsqueeze(1)], dim=1)
        else:
            X_seq = y_pred_era5.unsqueeze(1)

    forecasts = torch.stack(forecasts, dim=1)
    return forecasts

# ------------------------------
# 7️⃣ Visualization
# ------------------------------
def plot_forecasts_global(forecasts, lat, lon, step_indices=None, variable_index=0, cmap='coolwarm'):
    if isinstance(forecasts, torch.Tensor):
        forecasts = forecasts.cpu().numpy()
    B, steps, C_out, H, W = forecasts.shape
    if step_indices is None:
        step_indices = range(steps)
    for b in range(B):
        for t in step_indices:
            data = forecasts[b,t,variable_index]
            plt.figure(figsize=(12,6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            mesh = ax.pcolormesh(lon, lat, data, cmap=cmap, shading='auto')
            plt.title(f"Sample {b}, Forecast step {t}")
            plt.colorbar(mesh, label='Value')
            plt.show()

# ------------------------------
# 8️⃣ Metrics computation
# ------------------------------
def compute_forecast_metrics(forecasts, y_true, step_indices=None, variable_index=0):
    if isinstance(forecasts, torch.Tensor):
        forecasts = forecasts.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    B, steps, C_out, H, W = forecasts.shape
    if step_indices is None:
        step_indices = range(steps)
    metrics = {}
    for t in step_indices:
        pred = forecasts[:, t, variable_index]
        true = y_true[:, t, variable_index]
        error = pred - true
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        mean_error = np.mean(error)
        metrics[t] = {"RMSE":rmse,"MAE":mae,"MeanError":mean_error}
    return metrics

# ------------------------------
# 9️⃣ Example: Training loop
# ------------------------------
# Example: random data for demonstration
H, W = 721, 1440
B, T_in, C_in, C_out = 2, 3, 4, 1
X_train = torch.randn(B,T_in,C_in,H,W).cuda()
Y_train = torch.randn(B,C_out,H,W).cuda()
lat = np.linspace(90,-90,H)
lon = np.linspace(0,359.75,W)
bandwidth=64
grid_s2cnn, mask_north = make_era5_to_s2cnn_grid_lat_range(lat, lon, bandwidth, 0, 90)
grid_s2cnn = grid_s2cnn.cuda()
mask_north = mask_north.cuda()

dataset = ERA5StreamingDataset(file_list=[], T_in=T_in, lat_crop=(0,90), bandwidth=bandwidth)
# Overwrite with random data
dataset.X = X_train
dataset.Y = Y_train
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = SphericalWeatherNet(in_channels=T_in*C_in, out_channels=C_out, bandwidth=bandwidth).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(2):
    model.train()
    for x_batch, y_batch in dataloader:
        B, T, C, Hs, Ws = x_batch.shape
        x_flat = x_batch.reshape(B,T*C,Hs,Ws).cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        y_pred_s2cnn = model(x_flat)
        y_pred_era5 = s2cnn_to_era5_batch(y_pred_s2cnn, lat, lon)
        loss = criterion(y_pred_era5, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ------------------------------
# 10️⃣ Recursive forecast + plot + metrics
# ------------------------------
X_init = X_train.clone()
forecast_steps = 3
forecasts = recursive_forecast(model, X_init, forecast_steps, grid_s2cnn, mask_north, lat, lon)
plot_forecasts_global(forecasts, lat, lon, step_indices=[0,1,2])
metrics = compute_forecast_metrics(forecasts, Y_train.unsqueeze(1).repeat(1,forecast_steps,1,1,1))
print(metrics)
