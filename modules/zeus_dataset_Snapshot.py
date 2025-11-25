
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.config_Mike1 import config

from modules.zeus_dateset_geo_tempo import Geo_Tempo
from zeus_dataset_var_names import mike1_variables, var_name_map

# earth_orbit_angle_tensor_era5 = Geo_Tempo.earth_orbit_angle_tensor_era5(dt_utc, H, W)
# radian_tensor = earth_orbit_angle_tensor_era5 / 24 * 2 * torch.pi
# sin_earth_orbit_angle_tensor = torch.sin(radian_tensor)
# cos_earth_orbit_angle_tensor = torch.cos(radian_tensor)

# #------------ sin/cos true_solar_hours_tensor_era5 --------------
# true_solar_hours_tensor_era5 = Geo_Tempo.true_solar_hours_tensor_era5(dt_utc, H, W)
# radian_solar_tensor = true_solar_hours_tensor_era5 / 24 * 2 * torch.pi
# sin_true_solar_hours_tensor = torch.sin(radian_tensor)
# cos_true_solar_hours_tensor = torch.cos(radian_tensor)

# #------------ (lat, lon) ------------------------------------
# lat_lon_tensor = Geo_Tempo.lat_lon_grid_tensor(H, W)
# cos_lat_sin_lon_tensor_era5 = Geo_Tempo.cos_lat_sin_lon_tensor_era5(H, W)

class Snapshot:
    # temporary. can be added and changed any time
    meteo_dims = len(mike1_variables)
    geo_dims = 6        # 4 constants + 2 embeddings: sin lat, sin/cos lon, orograph, land_sea_mask_emb_placeholder, soil_type_emb_placeholder. lat/lon NOT omitted though they work for position bias.
    tempo_dims = 5      # 4 variables + 1 embedding: sin/cos earth_orbital_angle (2), sin/cos true_solar_hour (2), cascade_emb

    height = 720        # exclude the north pole.
    width = 1440
    
    _geo_tensor = None
    geo_categorical_tensor = None
    geo_nCategories = None
    geo_dense_tensor = None
    geo_sin_lat_sin_cos_lon_tensor = None
    
    def __init__(self, year, month, day, hour, meteo = None, shared_geo = None, tempo = None, device='cpu'):
        assert year is not None and month is not None and day is not None and hour is not None
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        
        self._device = device
        self._fresh = True

        if meteo is not None:
            assert torch.is_tensor(meteo)
            assert meteo.shape == (self.meteo_dims, self.height, self.width)            
        if shared_geo is not None:
            assert torch.is_tensor(shared_geo)
            assert shared_geo.shape == (self.geo_dims, self.height, self.width)            
        if tempo is not None:
            assert torch.is_tensor(tempo)
            assert tempo.shape == (self.tempo_dims, self.height, self.width)

        self._meteo_tensor = meteo
        
        if tempo is not None:
            self._tempo_tensor = tempo
        else:
            tempo_tensor = self.create_tempo_tensor()
            self.tempo = tempo_tensor
        
    def num_get_train_variables():
        return Snapshot.meteo_dims + Snapshot.geo_dims + Snapshot.tempo_dims
        
    #-------------------- device -----------------
    def _get_device(self):
        devices = [
            self._meteo_tensor.device.device if self._meteo_tensor is not None else None,
            Snapshot._geo_tensor.device.device if Snapshot._geo_tensor is not None else None,
            self._tempo_tensor.device.device if self._tempo_tensor is not None else None,
        ]
        true_devices = [d for d in devices if d is not None]
        if len(true_devices) <= 0:
            device = None
        else:
            cpu_devices = [d for d in devices if d == "cpu"]
            if len(cpu_devices) == 0:
                device = "cuda"
            elif len(cpu_devices) == len(true_devices):
                device = "cpu"
            else: device = None # mixed devices
        return device
    def _set_device(self, device):
        tensors = [self._meteo_tensor, self._tempo_tensor]  # _geo_tensor will be always on cuda if not None.
        true_tensors = [arr for arr in tensors if arr is not None]
        [arr.to(device) for arr in true_tensors]
        return
    def to(self, device):
        self._device = device
    device = property(
        fget = _get_device,
        fset = _set_device,
    )

    #------------------- fresh ---------------------
    def _get_fresh(self):
        return self._fresh
    def _set_fresh(self, fresh):
        self._fresh = fresh
    fresh = property(
        fget = _get_fresh,
        fset = _set_fresh,
    )
    
    #------------------ meteo -----------------------
    def _get_meteo_tensor(self):
        return self._meteo_tensor
    def _set_meteo_tensor(self, tensor):
        assert tensor.shape[0] == Snapshot.meteo_dims
        self._meteo_tensor = tensor
    meteo = property(
        fget = _get_meteo_tensor,
        fset = _set_meteo_tensor,
    )
    
    #------------------ geo -----------------------
    def _get_geo_tensor(self):
        return Geo_Tempo._geo_tensor
    def _set_geo_tensor(self, tensor):
        assert tensor.shape[0] == Snapshot.geo_dims
        Geo_Tempo._geo_tensor = tensor
    geo = property(
        fget = _get_geo_tensor,
        fset = _set_geo_tensor,
    )
    
    def create_geo_tensor(categorical_npy_paths, dense_npy_paths):
        def load_npy_files(npy_paths):
            arrays = []
            for path in npy_paths:
                array = np.load(path)
                assert array.shape == (721, 1440)
                has_nan = np.isnan(array).any()
                assert not has_nan
                arrays.append(array)            
            return arrays
        
        final_H, final_W = Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid
        
        # load and normalize category values. Convert to a tensor
        categorical_arrays = load_npy_files(categorical_npy_paths)
        nCategories = []
        processed_arrays = []

        for cat_array in categorical_arrays:
            # Crop safely (if array smaller than final_H/W, pad with zeros)
            H, W = cat_array.shape
            arr = np.zeros((final_H, final_W), dtype=np.int32)
            h = min(H, final_H)
            w = min(W, final_W)
            arr[:h, :w] = cat_array[:h, :w].astype(np.int32)

            # Get unique values and remap to 0..N-1
            unique = np.unique(arr)
            nCategories.append(unique.size)
            for new_idx, u in enumerate(unique):
                arr[arr == u] = new_idx

            # Convert to float32 at the end
            arr = arr.astype(np.float32)
            processed_arrays.append(arr)

        # Now all arrays have shape (final_H, final_W)
        stack = np.stack(processed_arrays, axis=0)
        Snapshot.geo_categorical_tensor = torch.from_numpy(stack)  # shape: (num_arrays, final_H, final_W)
        Snapshot.geo_nCategories = nCategories

        # load and standardize dense arrays. Convert to a tensor
        dense_arrays = load_npy_files(dense_npy_paths)
        processed_arrays = []
        for arr in dense_arrays:
            H, W = arr.shape
            temp = np.zeros((final_H, final_W), dtype=np.float32)
            h = min(H, final_H)
            w = min(W, final_W)
            temp[:h, :w] = arr[:h, :w].astype(np.float32)
            processed_arrays.append(temp)

        # Stack into one array
        stack = np.stack(processed_arrays, axis=0)  # shape: (n, final_H, final_W)

        # Vectorized per-array normalization
        means = stack.mean(axis=(1, 2), keepdims=True)
        stds  = stack.std(axis=(1, 2), keepdims=True)
        stack = (stack - means) / (stds + 1e-8)  # avoid division by zero

        # Convert to PyTorch tensor
        Snapshot.geo_dense_tensor = torch.from_numpy(stack)  # shape: (n, final_H, final_W)
            
        # generate lat_lon tensor
        Snapshot.geo_sin_lat_sin_cos_lon_tensor = Geo_Tempo.sin_lat_sin_cos_lon_tensor_era5() # (3, final_H, final_W)

        assert Snapshot.geo_categorical_tensor.shape[0] + Snapshot.geo_dense_tensor.shape[0] + Snapshot.geo_sin_lat_sin_cos_lon_tensor.shape[0] == Snapshot.geo_dims        
        return

    #------------------ tempo -----------------------
    def _get_tempo_tensor(self):
        return self._tempo_tensor
    def _set_tempo_tensor(self, tensor):
        assert tensor.shape[0] == Snapshot.tempo_dims
        self._tempo_tensor = tensor
    def create_tempo_tensor(self):
        # no need to normalize them.
        final_H, final_W = Snapshot.height, Snapshot.width
        dt_utc = datetime(self.year, self.month, self.day, self.hour, tzinfo=timezone.utc)
        sin_cos_earth_orbit_angle_tensor_era5 = Geo_Tempo.sin_cos_earth_orbit_angle_tensor_era5(dt_utc)[:, :final_H, :final_W]
        sin_cos_true_solar_hours_tensor_era5 = Geo_Tempo.sin_cos_true_solar_hours_tensor_era5(dt_utc)[:, :final_H, :final_W]
        cascade_emb_placeholder = torch.rand(1, self.height, self.width)[:, :final_H, :final_W]
        tempo_tensor = torch.concat([sin_cos_earth_orbit_angle_tensor_era5, sin_cos_true_solar_hours_tensor_era5, cascade_emb_placeholder], dim=0)
        return tempo_tensor
        
    tempo = property(
        fget = _get_tempo_tensor,
        fset = _set_tempo_tensor,
    )
    
    #------------------ input tensor  --------------------
    def _get_input_tensor(self):
        if self._meteo_tensor is None or Geo_Tempo._geo_tensor is None or self._tempo_tensor is None: return None
        tensor = torch.concat([self._meteo_tensor, self._geo_tensor, self._tempo_tensor], dim=0)
        return tensor
    def _set_input_tensor(self, tensor):
        assert tensor.shape[0] == self.meteo_dims + self.geo_dims + self.tempo_dims
        self._meteo_tensor = tensor[:self.meteo_dims]
        # Geo_Tempo._geo_tensor = tensor[self.meteo_dims : self.meteo_dims + self.geo_dims] # Do not change _geo_tensor
        self._tempo_tensor = tensor[self.meteo_dims + self.geo_dims :]
        return
    input_tensor = property(
        fget = _get_input_tensor,
        fset = _set_input_tensor,
    )

    #------------------ output tensor  --------------------
    def _get_output_tensor(self):
        return self._meteo_tensor
    def _set_output_tensor(self, tensor):
        assert tensor.shape[0] == self.meteo_dims
        self._meteo_tensor = tensor
        return
    input_tensor = property(
        fget = _get_output_tensor,
        fset = _set_output_tensor,
    )

    #------------------ miscellaneout --------------------
    def delete(self):
        self._device = 'cpu'
        

categorical_npy_paths = [config['land_sea_mask_npy_path'], config['soil_type_npy_path']]
dense_npy_paths = [config['orograph_npy_path']]

Snapshot.create_geo_tensor(categorical_npy_paths, dense_npy_paths)
# print(Snapshot.geo_categorical_tensor.shape, Snapshot.geo_dense_tensor.shape, Snapshot.geo_sin_lat_sin_cos_lon_tensor.shape)
# print(torch.unique(Snapshot.geo_categorical_tensor[0]), torch.unique(Snapshot.geo_categorical_tensor[1]))
# print(torch.mean(Snapshot.geo_dense_tensor), torch.std(Snapshot.geo_dense_tensor))
# print(torch.mean(Snapshot.geo_sin_lat_sin_cos_lon_tensor[0]), torch.std(Snapshot.geo_sin_lat_sin_cos_lon_tensor[0]))
# print(torch.mean(Snapshot.geo_sin_lat_sin_cos_lon_tensor[1]), torch.std(Snapshot.geo_sin_lat_sin_cos_lon_tensor[1]))
# print(torch.mean(Snapshot.geo_sin_lat_sin_cos_lon_tensor[2]), torch.std(Snapshot.geo_sin_lat_sin_cos_lon_tensor[2]))

if __name__ == "__main__":
    nVars = Snapshot.num_get_train_variables()
    print(f"------------ Total number of train variables: {nVars}")

    now = datetime.now(timezone.utc)
    snapshot_now = Snapshot(now.year, now.month, now.day, now.hour)
    print(snapshot_now._tempo_tensor.shape, snapshot_now.geo_categorical_tensor.shape, snapshot_now.geo_dense_tensor.shape, snapshot_now.geo_sin_lat_sin_cos_lon_tensor.shape, snapshot_now.geo_nCategories)

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    def show_on_map(data):  # np.array

        # # Example: your data tensor (replace with your real tensor)
        # data = np.random.rand(720, 1440)  # shape (lat, lon)
        mean = data.mean()
        # Latitude and longitude vectors at cell centers
        H, W = data.shape
        _, lat, lon = Geo_Tempo.lat_lon_grid_tensor_era5_721()
        lat = lat.numpy()
        lon = lon.numpy()

        # Create a meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Plot
        fig = plt.figure(figsize=(12,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()

        # Plot the data as a colormap
        c = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(), cmap='viridis')

        # Add colorbar
        cb = plt.colorbar(c, ax=ax, orientation='vertical', label='Value')

        # Add country boundaries
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        # Add gridlines for lat/lon
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        plt.title(f"Geographical Distribution. mean = {mean}")
        plt.show()

    #------------- geo, 6 -----------------
    # show_on_map(snapshot_now.geo_categorical_tensor[0].numpy())     # land_sea_mask,    category
    show_on_map(snapshot_now.geo_categorical_tensor[1].numpy())     # soil_type,        category
    # show_on_map(snapshot_now.geo_dense_tensor[0].numpy())           # geopotential_at_surface,  standardized
    # show_on_map(snapshot_now.geo_sin_lat_sin_cos_lon_tensor[0].numpy())     # sin(lat),  as-is
    # show_on_map(snapshot_now.geo_sin_lat_sin_cos_lon_tensor[1].numpy())     # sin(lon),  as-is
    # show_on_map(snapshot_now.geo_sin_lat_sin_cos_lon_tensor[2].numpy())     # cos(lon),  as-is

    #------------- tempo, 5 -----------------
    # show_on_map(snapshot_now.tempo[0].numpy())      # sin(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[1].numpy())      # cos(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[2].numpy())      # sin(true_solar_hour)
    # show_on_map(snapshot_now.tempo[3].numpy())      # cos(true_solar_hour)
    # show_on_map(snapshot_now.tempo[4].numpy())      # cascade_emb_placeholder
