
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import torch.nn.functional as F

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.config_Mike1 import config

from modules.zeus_dateset_geo_tempo import Geo_Tempo
from modules.zeus_dataset_var_names import mike1_variables, var_name_map


class Snapshot:
    export_factor10 = 10
    # temporary. can be added and changed any time
    meteo_dims = len(mike1_variables)
    tempo_dims = 5      # 4 variables + 1 embedding: sin/cos earth_orbital_angle (2), sin/cos true_solar_hour (2), cascade_emb
    geo_dims = 6        # 4 constants + 2 embeddings: sin lat, sin/cos lon, orograph, land_sea_mask_emb_placeholder, soil_type_emb_placeholder. lat/lon NOT omitted though they work for position bias.

    export_height = 720        # exclude the north pole.
    export_width = 1440
    
    _shared_geo_tensor = None   # on device
    geo_nCategories = None
    # geo_categorical_tensor = None
    # geo_dense_tensor = None
    # geo_sin_lat_sin_cos_lon_tensor = None
    
    # scenarios: 
    # Snapshot(year, month, day, hour, meteo): meteo either from dataset or model.forward
    
    def __init__(self, year, month, day, hour, meteo = None, device='cpu'):
        assert year is not None and month is not None and day is not None and hour is not None
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        
        self._device = None
        self.device = device    # operation
        
        self._meteo_tensor, self._tempo_tensor = None, None         # on cpu
        self._meteo_tempo_tensor, self._geo_tensor = None, None     # on device

        #---------------------- temporary ----------------------
        if meteo is None:
            _meteo_tensor = torch.rand(Snapshot.meteo_dims, Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid)
        else:
            # coming from either dataset or model.
            assert torch.is_tensor(meteo)
            assert meteo.shape[0] == Snapshot.meteo_dims
            assert meteo.shape[1] >= Snapshot.export_height
            assert meteo.shape[2] >= Snapshot.export_width
            _meteo_tensor = meteo
            
        _tempo_tensor = self.create_tempo_tensor()
        
        self._meteo_tensor, self._tempo_tensor = Snapshot.crop_tensors_to_common_size([_meteo_tensor, _tempo_tensor])                            

        self._geo_tensor = Snapshot._shared_geo_tensor


    def crop_tensors_to_common_size(tensors):
        # Compute common min height/width
        h = min(t.shape[1] for t in tensors)
        w = min(t.shape[2] for t in tensors)
        
        assert h >= Snapshot.export_height and w >= Snapshot.export_width
        
        cropped = []
        for t in tensors:
            t_h = t[:, -h:] if Geo_Tempo.respect_north else t[:, :h]
            t_hw = t_h[:, :, -w:] if Geo_Tempo.respect_east else t_h[:, :, :w]
            cropped.append(t_hw)
        
        return cropped
    
    def crop_tensor_to_export_size(tensor):
        assert tensor.shape[1] >= Snapshot.export_height and tensor.shape[2] >= Snapshot.export_width
        tensor_cropped = tensor[:, -Snapshot.export_height:] if Geo_Tempo.respect_north else tensor[:, :Snapshot.export_height]
        tensor_cropped = tensor_cropped[:, :, -Snapshot.export_width:] if Geo_Tempo.respect_east else tensor_cropped[:, :, :Snapshot.export_width]
        return tensor_cropped

        
    def num_get_train_variables():
        return Snapshot.meteo_dims + Snapshot.geo_dims + Snapshot.tempo_dims
        
    #-------------------- device -----------------
    def _get_device(self):
        return self._device
    def _set_device(self, device):
        self._device = device
    device = property(
        fget = _get_device,
        fset = _set_device,
    )
    
    #------------------ meteo -----------------------
    def _get_meteo_tensor_internal(self):
        return self._meteo_tensor
    meteo_internal = property(
        fget = _get_meteo_tensor_internal,
    )
    
    #------------------ tempo -----------------------
    def _get_tempo_tensor_internal(self):
        return self._tempo_tensor
    tempo_internal = property(
        fget = _get_tempo_tensor_internal,
    )
    def create_tempo_tensor(self):
        # no need to normalize them.
        dt_utc = datetime(self.year, self.month, self.day, self.hour, tzinfo=timezone.utc)
        sin_cos_earth_orbit_angle_tensor_era5 = Geo_Tempo.sin_cos_earth_orbit_angle_tensor_era5(dt_utc)
        sin_cos_true_solar_hours_tensor_era5 = Geo_Tempo.sin_cos_true_solar_hours_tensor_era5(dt_utc)
        cascade_emb_placeholder = torch.rand(1, Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid)
        tempo_tensor = torch.concat([sin_cos_earth_orbit_angle_tensor_era5, sin_cos_true_solar_hours_tensor_era5, cascade_emb_placeholder], dim=0)
        return tempo_tensor
    
    #------------------ meteo_tempo -----------------------
    def _get_meteo_tempo_tensor_internal(self):
        # These two tensors are only created by the constructor with equal size and don't change later on.
        return torch.concat([self.meteo_internal, self.tempo_internal], dim=0)
    meteo_tempo_internal = property(
        fget = _get_meteo_tempo_tensor_internal,
    )
    
    #------------------ geo -----------------------
    def _get_geo_tensor_internal(self):
        return self._geo_tensor
    geo_internal = property(
        fget = _get_geo_tensor_internal,
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
               
        # load and normalize category values. Convert to a tensor
        categorical_arrays = load_npy_files(categorical_npy_paths)
        nCategories = []
        processed_arrays = []

        for cat_array in categorical_arrays:
            # Crop safely (if array smaller than final_H/W, pad with zeros)
            H, W = cat_array.shape
            arr = np.zeros((Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid), dtype=np.int32)
            h = min(H, Geo_Tempo.H_era5_grid)
            w = min(W, Geo_Tempo.W_era5_grid)
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
        geo_categorical_tensor = torch.from_numpy(stack)  # shape: (num_arrays, final_H, final_W)
        Snapshot.geo_nCategories = nCategories

        # load and standardize dense arrays. Convert to a tensor
        dense_arrays = load_npy_files(dense_npy_paths)
        processed_arrays = []
        for arr in dense_arrays:
            H, W = arr.shape
            temp = np.zeros((Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid), dtype=np.float32)
            h = min(H, Geo_Tempo.H_era5_grid)
            w = min(W, Geo_Tempo.W_era5_grid)
            temp[:h, :w] = arr[:h, :w].astype(np.float32)
            processed_arrays.append(temp)

        # Stack into one array
        stack = np.stack(processed_arrays, axis=0)  # shape: (n, final_H, final_W)

        # Vectorized per-array normalization
        means = stack.mean(axis=(1, 2), keepdims=True)
        stds  = stack.std(axis=(1, 2), keepdims=True)
        stack = (stack - means) / (stds + 1e-8)  # avoid division by zero

        # Convert to PyTorch tensor
        geo_dense_tensor = torch.from_numpy(stack)  # shape: (n, final_H, final_W)
            
        # generate lat_lon tensor
        geo_sin_lat_sin_cos_lon_tensor = Geo_Tempo.sin_lat_sin_cos_lon_tensor_era5() # (3, final_H, final_W)
        assert geo_categorical_tensor.shape[0] + geo_dense_tensor.shape[0] + geo_sin_lat_sin_cos_lon_tensor.shape[0] == Snapshot.geo_dims        
        Snapshot._shared_geo_tensor = torch.concat([geo_categorical_tensor, geo_dense_tensor, geo_sin_lat_sin_cos_lon_tensor], dim=0)
        return
        
    #------------------ export tensors  --------------------
    def get_export_size():
        return (int(Snapshot.export_height/10*Snapshot.export_factor10), int(Snapshot.export_width/10*Snapshot.export_factor10))
    
    def resize_for_export(tensor):
        x = tensor.unsqueeze(0)  # shape (1, C, H, W)
        size=Snapshot.get_export_size()
        x_resized = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x_resized = x_resized.squeeze(0)  # shape (C, H1, W1)
        return x_resized
    
    def _get_input_tensor(self):
        if self.meteo_tempo is None or self.geo_internal is None: return None
        _meteo_tempo, _geo = Snapshot.crop_tensors_to_common_size([self.meteo_tempo_internal, self.geo_internal])                            
        input = torch.concat([_meteo_tempo, _geo], dim=0)
        input = Snapshot.crop_tensor_to_export_size(input)
        
        if Snapshot.export_factor10 != 10:
            input_teinputnsor = Snapshot.resize_for_export(input)

        input = input.to(self.device)
        input.detach()
        input.requires_grad = False
        return input
    
    input = property(
        fget = _get_input_tensor,
    )
    
    def _get_meteo_tempo_tensor(self):
        # These two tensors are only created by the constructor with equal size and don't change later on.
        meteo_tempo = Snapshot.crop_tensor_to_export_size(self.meteo_tempo_internal)
        
        if Snapshot.export_factor10 != 10:
            meteo_tempo = Snapshot.resize_for_export(meteo_tempo)
            
        meteo_tempo = meteo_tempo.to(self.device)
        meteo_tempo.detach()
        meteo_tempo.requires_grad = False
        return meteo_tempo
        
    meteo_tempo = property(
        fget = _get_meteo_tempo_tensor,
    )

    def _get_meteo_tensor(self):
        meteo = Snapshot.crop_tensor_to_export_size(self.meteo_internal)        
        
        if Snapshot.export_factor10 != 10:
            meteo = Snapshot.resize_for_export(meteo)
        
        meteo = meteo.to(self.device)
        meteo.detach()
        meteo.requires_grad = False
        return meteo

    meteo = property(
        fget = _get_meteo_tensor,
    )
    
    def _get_geo_tensor(self):
        geo = Snapshot.crop_tensor_to_export_size(self.geo_internal)        
        
        if Snapshot.export_factor10 != 10:
            geo = Snapshot.resize_for_export(geo)
        
        geo = geo.to(self.device)
        geo = geo.to(self.device)
        geo.detach()
        geo.requires_grad = False
        return geo

    geo = property(
        fget = _get_geo_tensor,
    )

    #------------------ miscellaneout --------------------
    def delete(self):
        self._device = 'cpu'
        

categorical_npy_paths = [config['land_sea_mask_npy_path'], config['soil_type_npy_path']]
dense_npy_paths = [config['orograph_npy_path']]
# Snapshot.create_geo_tensor(categorical_npy_paths, dense_npy_paths)

#------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    nVars = Snapshot.num_get_train_variables()
    print(f"------------ Total number of train variables: {nVars}")

    now = datetime.now(timezone.utc)
    snapshot_now = Snapshot(now.year, now.month, now.day, now.hour)
    # print("meteo_internal, tempo_internal, meteo_tempo_internal, geo_internal, nCat: ", snapshot_now.meteo_internal.shape, snapshot_now.tempo_internal.shape, snapshot_now.meteo_tempo_internal.shape, snapshot_now.geo_internal.shape, snapshot_now.geo_nCategories)
    # print("input, meteo, meteo_tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo.shape, snapshot_now.meteo_tempo.shape, snapshot_now.geo.shape)

    Snapshot.export_factor10 = 2
    print("input_tensor, meteo_tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo_tempo.shape, snapshot_now.geo.shape)    
    Snapshot.export_factor10 = 10
    
    Snapshot.device = "cuda"
    print("input_tensor, meteo_tempo, geo: ", snapshot_now.input.device, snapshot_now.meteo_tempo.device, snapshot_now.geo.device) 

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
    # show_on_map(snapshot_now.geo[0].numpy())     # land_sea_mask,    category
    # show_on_map(snapshot_now.geo[1].numpy())     # soil_type,        category
    # show_on_map(snapshot_now.geo[2].numpy())     # geopotential_at_surface,  standardized
    # show_on_map(snapshot_now.geo[3].numpy())     # sin(lat),  as-is
    # show_on_map(snapshot_now.geo[4].numpy())     # sin(lon),  as-is
    # show_on_map(snapshot_now.geo[5].numpy())     # cos(lon),  as-is

    #------------- tempo, 5 -----------------
    # show_on_map(snapshot_now.tempo[0].numpy())      # sin(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[1].numpy())      # cos(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[2].numpy())      # sin(true_solar_hour)
    # show_on_map(snapshot_now.tempo[3].numpy())      # cos(true_solar_hour)
    # show_on_map(snapshot_now.tempo[4].numpy())      # cascade_emb_placeholder
