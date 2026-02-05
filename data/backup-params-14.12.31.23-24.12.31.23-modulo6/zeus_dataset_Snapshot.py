
import torch
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import torch.nn.functional as F
import math

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.config_Mike1 import config

from modules.zeus_dateset_geo_tempo import Geo_Tempo
from modules.zeus_dataset_var_names import VARIABLE_NAME_LIST_UNCHANGEABLE, mike1_variables_order, num_vars_per_level, num_levels, num_vars_surface

import math

def best_segments_increasing(N, S):
    """
    Return actual segments of numbers 0..N-1
    based on non-decreasing lengths maximizing trailing uniformity.
    Mirrors the logic of best_segments_decreasing.
    """
    # Step 1: compute segment lengths
    def best_segment_lengths_increasing(N, S):
        best_seq = None
        best_trailing = 0

        def dfs(seq, remaining, remaining_count):
            nonlocal best_seq, best_trailing
            if remaining_count == 0:
                if remaining != 0:
                    return
                # trailing uniform length
                last_val = seq[-1]
                count = 0
                for x in reversed(seq):
                    if x == last_val:
                        count += 1
                    else:
                        break
                if count > best_trailing:
                    best_trailing = count
                    best_seq = seq[:]
                return

            # Mirror logic: next value ≥ previous (or floor of even distribution if seq empty)
            start = seq[-1] if seq else max(1, remaining // remaining_count)
            max_val = remaining - (remaining_count - 1)  # leave at least 1 for remaining segments
            for val in range(start, max_val + 1):
                dfs(seq + [val], remaining - val, remaining_count - 1)

        dfs([], N, S)
        return best_seq

    lengths = best_segment_lengths_increasing(N, S)

    # Step 2: build actual segments
    segments = []
    start = 0
    for length in lengths:
        segments.append(list(range(start, start + length)))
        start += length
    return segments

import math

def best_segments_decreasing(N, S):
    """
    Return actual segments of numbers 0..N-1
    based on non-increasing lengths maximizing trailing uniformity.
    """
    # Step 1: compute segment lengths
    def best_segment_lengths_decreasing(N, S):
        best_seq = None
        best_trailing = 0

        def dfs(seq, remaining, remaining_count):
            nonlocal best_seq, best_trailing
            if remaining_count == 0:
                if remaining != 0:
                    return
                # trailing uniform length
                last_val = seq[-1]
                count = 0
                for x in reversed(seq):
                    if x == last_val:
                        count += 1
                    else:
                        break
                if count > best_trailing:
                    best_trailing = count
                    best_seq = seq[:]
                return
            # next value ≤ previous (or ceiling of even distribution if seq empty)
            start = seq[-1] if seq else math.ceil(remaining / remaining_count)
            min_val = 1
            for val in range(start, min_val-1, -1):
                if val * remaining_count < remaining:
                    continue
                dfs(seq + [val], remaining - val, remaining_count - 1)

        dfs([], N, S)
        return best_seq

    lengths = best_segment_lengths_decreasing(N, S)

    # Step 2: build actual segments
    segments = []
    start = 0
    for length in lengths:
        segments.append(list(range(start, start + length)))
        start += length
    return segments

def check_consistency(n, s):
    inc = best_segments_increasing(n, s)
    dec = best_segments_decreasing(n, s)
    def assert_full(A):
        sum = []
        for a in A:
            sum = sum + a
        assert len(A) == s
        assert len(sum) == n
        assert set(sum) == set(range(n))
    assert_full(inc)
    assert_full(dec)
    inc_l = [len(a) for a in inc]
    dec_l = [len(a) for a in dec]
    print(inc_l)
    print(dec_l)
    assert inc_l == dec_l[::-1]
    print(inc)
    print(dec)

#--------------------------------------- Snapshot ----------------------------

class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls):
        return self.f(cls)


class Snapshot:
    assert mike1_variables_order == "[v_l for v ztuvq for l in levels] + surface_vars"
       
    export_factor10 = 10
    
    # temporary. can be added and changed any time
    meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
    tempo_dims = 4      # [sin_orbital_angle', 'cos_orbital_angle', 'sin_solor_hour', 'cos_solar_hour']
    dense_dims = 4  # ['orograph', 'sin_lat', 'sin_lon', 'cos_lon']
    cat_dims = 2    # ['land_sea_mask', 'soil_type']

    original_720_h = 720        # exclude the north pole.
    original_1440_w = 1440
    
    num_air_levels = num_levels
    air_level_dims = num_vars_per_level
   
    delevel_plan = best_segments_increasing(num_air_levels, 7)
    
    dense = None   # on device
    cat = None
    
    def set_original_settings():
        Snapshot.export_factor10 = 10
        # temporary. can be added and changed any time
        Snapshot.meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
        Snapshot.tempo_dims = 4      # 4 variables + 1 embedding: sin/cos earth_orbital_angle (2), sin/cos true_solar_hour (2), cascade_emb
        Snapshot.dense_dims = 4
        Snapshot.cat_dims = 2

        Snapshot.original_720_h = 720        # exclude the north pole.
        Snapshot.original_1440_w = 1440
        
        Snapshot.num_air_levels = num_levels
        Snapshot.air_level_dims = num_vars_per_level
    
        Snapshot.delevel_plan = best_segments_increasing(Snapshot.num_air_levels, 7)
        
        categorical_npy_paths = [config['land_sea_mask_npy_path'], config['soil_type_npy_path']]
        dense_npy_paths = [config['orograph_npy_path']]

        #--------------------------------------------------------------------------------------------------------
        dense = Snapshot.create_dense_era5(dense_npy_paths)
        Snapshot.dense = Snapshot.crop_tensor(dense, size=(Snapshot.original_720_h, Snapshot.original_1440_w))
        cat = Snapshot.create_cat_era5(categorical_npy_paths)
        Snapshot.cat = Snapshot.crop_tensor(cat, size=(Snapshot.original_720_h, Snapshot.original_1440_w))
            
    def __init__(self, year, month, day, hour, meteo_normal, meteo_raw, device='cpu'):
        assert year is not None and month is not None and day is not None and hour is not None
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        
        self._device = None
        self.device = device    # operation
        
        self.meteo_normal, self.tempo, self.dense, self.cat = None, None, None, None

        #---------------------- temporary ----------------------
        exp_h, exp_w = Snapshot.get_export_size()
        full_size = (Snapshot.original_720_h, Snapshot.original_1440_w)

        #----------------- meteo_normal normalized -------------
        # meteo_normal, if not None, comes in two scenarios: 
        # - dataset: (n, 721, 1440)
        # - prediction: (n, 720/10*factor10, 1440/10*factor10) = (n, exp_h, exp_w)
        if meteo_normal is None:
            _meteo_tensor = torch.rand(Snapshot.meteo_dims, exp_h, exp_w) - 0.5     # placeholder for tests.
        else:
            assert torch.is_tensor(meteo_normal)
            assert meteo_normal.shape[0] == Snapshot.meteo_dims
            assert meteo_normal.shape[1] >= exp_h
            assert meteo_normal.shape[2] >= exp_w
            _meteo_tensor = meteo_normal
            
        if _meteo_tensor.shape[1] > Snapshot.original_720_h or _meteo_tensor.shape[2] > Snapshot.original_1440_w:
            _meteo_tensor = Snapshot.crop_tensor(_meteo_tensor, full_size)
            
        self.meteo_normal = _meteo_tensor

        #--------------- meteo_rawized -------------
        # meteo_raw, if not None, comes from dataset: (n, 721, 1440)
        # 
        if meteo_raw is not None:
            meteo_raw = Snapshot.crop_tensor(meteo_raw, full_size)
        self._meteo_raw = meteo_raw

        #--------------- tempo ---------------

        tempo = self.create_tempo_era5()
        self.tempo = Snapshot.crop_tensor(tempo, full_size)

        #---------------- geo ----------------
        
        self.dense = Snapshot.dense
        self.cat = Snapshot.cat

        
    def crop_tensor(tensor, size):
        tensor_cropped = tensor[:, -size[0]:] if Geo_Tempo.respect_north else tensor[:, :size[0]]
        tensor_cropped = tensor_cropped[:, :, -size[1]:] if Geo_Tempo.respect_east else tensor_cropped[:, :, :size[1]]
        return tensor_cropped
        
    def num_get_train_variables():
        return Snapshot.meteo_dims + Snapshot.tempo_dims + Snapshot.dense_dims + Snapshot.cat_dims
        
    #-------------------- device -----------------
    def _get_device(self):
        return self._device
    def _set_device(self, device):
        self._device = device
    def to(self, device):
        self._device = device
    device = property(
        fget = _get_device,
        fset = _set_device,
    )
    
    #------------------ meteo_normal -----------------------
    def _get_meteo(self):
        return self.meteo_normal
    
    meteo_internal = property(
        fget = _get_meteo,
    )
    
    def _get_meteo(self):
        meteo_normal = self.meteo_internal
        
        if not Snapshot.is_export_size(meteo_normal):
            meteo_normal = Snapshot.resize_for_export(meteo_normal)
        
        meteo_normal = meteo_normal.to(self.device) 
        return meteo_normal

    meteo_normal = property(
        fget = _get_meteo,
    )
    
    #------------------ meteo_normal rawized --------------
    def _get_meteo_raw_external(self):
        _meteo_raw = self._meteo_raw
    
        if Snapshot.export_factor10 != 10:
            _meteo_raw = Snapshot.resize_for_export(_meteo_raw)
    
        _meteo_raw = _meteo_raw.to(self.device)    
        return _meteo_raw
    
    meteo_raw = property(
        fget = _get_meteo_raw_external,
    )
    
    #------------------ tempo -----------------------
    def _get_tempo(self):
        return self.tempo
    
    tempo_internal = property(
        fget = _get_tempo,
    )
    
    def _get_tempo_external(self):
        tempo = self.tempo_internal
        
        if not Snapshot.is_export_size(tempo):
            tempo = Snapshot.resize_for_export(tempo)
        
        tempo = tempo.to(self.device)
        return tempo

    tempo = property(
        fget = _get_tempo_external,
    )
    
    #------------------ geo dense -----------------------
    def _get_dense(self):
        return self.dense
    dense_internal = property(
        fget = _get_dense,
    )
    
    def _get_dense_external(self):
        dense = self.dense_internal
        
        if not Snapshot.is_export_size(dense):
            dense = Snapshot.resize_for_export(dense)
        
        dense = dense.to(self.device)
        return dense

    dense = property(
        fget = _get_dense_external,
    )

    @classproperty
    def shared_dense(cls):
        dense = Snapshot.dense
        if not Snapshot.is_export_size(dense):
            dense = Snapshot.resize_for_export(dense)
        return dense
    
    #------------------ geo cat -----------------------
    def _get_cat(self):
        return self.cat
    cat_internal = property(
        fget = _get_cat,
    )
    
    def _get_cat_external(self):
        cat = self.cat_internal
        
        if not Snapshot.is_export_size(cat):
            cat = Snapshot.resize_for_export(cat)
        
        cat = cat.to(self.device)
        return cat

    cat = property(
        fget = _get_cat_external,
    )
    
    @classproperty
    def shared_cat(cls):
        cat = Snapshot.cat
        if not Snapshot.is_export_size(cat):
            cat = Snapshot.resize_for_export(cat)        
        return cat

    #--------------------------------------------------------------------------
    
    def create_tempo_era5(self):
        # Do NOT mormalize this instance-wide. They should be normalized dataset-wide.
        dt_utc = datetime(self.year, self.month, self.day, self.hour, tzinfo=timezone.utc)
        sin_cos_earth_orbit_angle_tensor_era5 = Geo_Tempo.sin_cos_earth_orbit_angle_tensor_era5(dt_utc)
        sin_cos_true_solar_hours_tensor_era5 = Geo_Tempo.sin_cos_true_solar_hours_tensor_era5(dt_utc)
        tempo_tensor = torch.concat([sin_cos_earth_orbit_angle_tensor_era5, sin_cos_true_solar_hours_tensor_era5], dim=0)
        return tempo_tensor
    
    def load_npy_files(npy_paths):
        arrays = []
        for path in npy_paths:
            array = np.load(path)
            assert array.shape == (721, 1440)
            has_nan = np.isnan(array).any()
            assert not has_nan
            arrays.append(array)            
        return arrays
        
    def create_dense_era5(dense_npy_paths):
        #------------------- dense, file -----------------
        # DO normalize them, because this instance are all repeated and dataset-wide normalization is equal to instance-wide normalization.
        dense_arrays = Snapshot.load_npy_files(dense_npy_paths)
        processed_arrays = []
        for arr in dense_arrays:
            assert arr.shape == (Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid)
            processed_arrays.append(arr)

        # Stack into one array
        stack = np.stack(processed_arrays, axis=0)  # shape: (n, final_H, final_W)
        # Vectorized per-array normalization
        means = stack.mean(axis=(1, 2), keepdims=True)
        stds  = stack.std(axis=(1, 2), keepdims=True)
        stack = (stack - means) / (stds + 1e-12)  # avoid division by zero

        # Convert to PyTorch tensor
        dense_tensor = torch.from_numpy(stack)  # shape: (n, final_H, final_W)

        # This code gives the same precision as the above.                
        # processed_arrays = [torch.from_numpy(arr) for arr in processed_arrays]
        # stack = torch.stack(processed_arrays, dim=0)
        # means = torch.mean(stack, dim=(1,2), keepdim=True)
        # stds = torch.std(stack, dim=(1,2), keepdim=True)
        # dense_tensor = (stack - means) / (stds + 1e-12)
        
        #-------------------- dense, lat lon ------------------
        geo_sin_lat_sin_cos_lon_tensor = Geo_Tempo.sin_lat_sin_cos_lon_tensor_era5() # (3, final_H, final_W)
        geo_sin_lat_sin_cos_lon_tensor = (geo_sin_lat_sin_cos_lon_tensor - torch.mean(geo_sin_lat_sin_cos_lon_tensor, dim=(1,2), keepdim=True)) \
        / (torch.std(geo_sin_lat_sin_cos_lon_tensor, dim=(1,2), keepdim=True) + 1e-12)
        
        return torch.concat([dense_tensor, geo_sin_lat_sin_cos_lon_tensor], dim=0)

    
    def create_cat_era5(categorical_npy_paths):
        #----------------- categorical, file --------------
        
        # load and normalize category values. Convert to a tensor
        categorical_arrays = Snapshot.load_npy_files(categorical_npy_paths)
        nCategories = []
        processed_arrays = []

        for cat_array in categorical_arrays:
            # Crop safely (if array smaller than final_H/W, pad with zeros)
            assert cat_array.shape == (Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid)
            arr = cat_array.astype(np.int32)

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
        categorical_tensor = torch.from_numpy(stack)  # shape: (num_arrays, final_H, final_W)

        return categorical_tensor

        
    #------------------ export tensors  --------------------
    def get_export_size():
        return (int(Snapshot.original_720_h/10*Snapshot.export_factor10), int(Snapshot.original_1440_w/10*Snapshot.export_factor10))
    
    def resize_for_export(tensor):
        """
        Resize a tensor (C, H, W) to Snapshot.get_export_size(),
        treating the horizontal dimension as cylindrical (wrap-around).
        """
        x = tensor.unsqueeze(0)  # (1, C, H, W)
        size_h, size_w = Snapshot.get_export_size()  # target size

        C, H, W = x.shape[1], x.shape[2], x.shape[3]

        # 1. Wrap horizontally: pad 1/2 width on both sides
        pad_w = W // 2
        x_wrapped = torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=3)  # (1, C, H, W + W)

        # 2. Interpolate to target size
        x_resized = F.interpolate(x_wrapped, size=(size_h, size_w + 0), mode='bilinear', align_corners=False)

        # 3. Crop central W region
        start = (x_resized.shape[3] - size_w) // 2
        x_resized = x_resized[..., :, start:start + size_w]

        x_resized = x_resized.squeeze(0)  # (C, H, W)
        return x_resized

    def is_export_size(tensor):
        exp_h, exp_w = Snapshot.get_export_size()
        return tensor.shape[-2] == exp_h and tensor.shape[-1] == exp_w

    #----------------------- packinng --------------------
        
    def pack_pair_snapshots(ss1, ss2):
        """
        Packs two snapshots by interleaving meteo_normal, tempo, and geo features per level,
        for vertical levels and surface variables.
        """

        def arrange_by_level(ss):
            _, H, W = ss.meteo_normal.shape
            
            meteo_normal = ss.meteo_normal                        
            meteo_normal = meteo_normal[:Snapshot.air_level_dims*Snapshot.num_air_levels].view(Snapshot.air_level_dims, Snapshot.num_air_levels, H, W)
            
            tempo = ss.tempo # 

            # --- Vertical levels ---            

            delevels = []
            for idx, group in enumerate(Snapshot.delevel_plan):
                start = group[0]
                end = group[-1] + 1
                
                level_slice = torch.concat([meteo_normal[:, start:end, :, :].reshape(-1, H, W), tempo], dim=0)
                delevels.append(level_slice)  # list of length levels, each (different s, H, W)


            # --- Surface variables ---
            meteo_surface = ss.meteo_normal[Snapshot.air_level_dims*Snapshot.num_air_levels:, :, :].view(-1, H, W)  # (m_surf, H, W)
            tempo_surface = tempo                                                                     # (t, H, W)
            surface = torch.concat([meteo_surface, tempo_surface], dim=0)

            return delevels, surface

        # --- Process snapshots ---
        lev1, surf1 = arrange_by_level(ss1)
        lev2, surf2 = arrange_by_level(ss2)
        
        assert ss1.device == ss2.device

        # --- Concatenate corresponding levels ---
        _, H, W = ss1.meteo_normal.shape
        delevels_paired = [torch.cat([l1, l2, ss1.dense], dim=0).to(ss1.device) for l1, l2 in zip(lev1, lev2)]  # list of (m+t+g)*2, H, W
        surface_paired = torch.cat([surf1, surf2, ss1.dense], dim=0).to(ss2.device)                             # (m_surf+t+g)*2, H, W
        assert not torch.isnan(surface_paired).any()

        export_size = Snapshot.get_export_size()
        for lvl, group in zip(delevels_paired, Snapshot.delevel_plan):
            assert lvl.shape == ((Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * 2 + Snapshot.dense_dims, export_size[0], export_size[1])
        assert surface_paired.shape == ((Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * 2 + Snapshot.dense_dims, export_size[0], export_size[1])

        return delevels_paired, surface_paired
    
    #------------------ miscellaneout --------------------

Snapshot.set_original_settings()

#------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #---------------------- Check utility functions ----------
    for n in range(1, 30):
        for s in range(1, n):
            check_consistency(n, s)
    
    #---------------------- Check Snapshot --------------------
    
    Snapshot.set_original_settings()
    ss = Snapshot(2020, 1, 1, 1, meteo_normal=None, meteo_raw=None)
    print("original size: ", ss.meteo_normal.shape, ss.tempo.shape, ss.dense.shape, ss.cat.shape, ss.meteo_normal.dtype, ss.dense.dtype, ss.cat.dtype)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    nVars = Snapshot.num_get_train_variables()
    print(f"------------ Total number of train variables: {nVars}")

    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo_normal=None, meteo_raw=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo_normal=None, meteo_raw=None)
    
    levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    for idx, lvl in enumerate(levels):
        print(idx, lvl.shape)
        break
    print(14, surface.shape)

    now = datetime.now(timezone.utc)
    snapshot_now = Snapshot(now.year, now.month, now.day, now.hour, meteo_normal=None, meteo_raw=None)
    # print("meteo_internal, tempo_internal, meteo_tempo_internal, geo_internal, nCat: ", snapshot_now.meteo_internal.shape, snapshot_now.tempo_internal.shape, snapshot_now.meteo_tempo_internal.shape, snapshot_now.geo_internal.shape, snapshot_now.geo_nCategories)
    # print("input, meteo_normal, meteo_tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo_normal.shape, snapshot_now.meteo_tempo.shape, snapshot_now.geo.shape)

    Snapshot.export_factor10 = 2
    print("meteo_normal, tempo, geo: ", snapshot_now.meteo_normal.shape, snapshot_now.tempo.shape, snapshot_now.dense.shape, snapshot_now.cat.shape)    
    Snapshot.export_factor10 = 10
    
    # Snapshot.device = "cuda"
    Snapshot.device = "cpu"
    print("geo: ", snapshot_now.dense.device) 
    
    print("class property: ", Snapshot.shared_dense.shape)

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    def show_on_map(data, title=""):  # np.array

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

        plt.title(f"data = {title}, mean = {data.mean():.5f}, std = {data.std():.5f}")
        plt.show()

    #------------- cat, 2 -----------------
    # print(Snapshot.delevel_plan)
    # show_on_map(snapshot_now.cat[0].numpy(), title="land_sea_mask")     # land_sea_mask,    category
    # show_on_map(snapshot_now.cat[1].numpy(), title="soil_type")     # soil_type,        category
    
    # #------------- dense, 4 -----------------
    # show_on_map(snapshot_now.dense[0].numpy(), title="geopotential_at_surface")     # geopotential_at_surface,  standardized
    # show_on_map(snapshot_now.dense[1].numpy(), title="sin(lat)")     # sin(lat),  as-is
    # show_on_map(snapshot_now.dense[2].numpy(), title="sin(lon)")     # sin(lon),  as-is
    # show_on_map(snapshot_now.dense[3].numpy(), title="cos(lon)")     # cos(lon),  as-is

    # #------------- tempo, 5 -----------------
    # show_on_map(snapshot_now.tempo[0].numpy(), title="sin(earth_orbital_angle)")      # sin(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[1].numpy(), title="cos(earth_orbital_angle)")      # cos(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[2].numpy(), title="sin(true_solar_hour)")      # sin(true_solar_hour)
    # show_on_map(snapshot_now.tempo[3].numpy(), title="cos(true_solar_hour)")      # cos(true_solar_hour)
    # show_on_map(snapshot_now.tempo[4].numpy(), title="cascade_emb_placeholder")      # cascade_emb_placeholder


    #------------------------------- test packing ----------------------
    Snapshot.set_original_settings()
    
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo_normal=None, meteo_raw=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo_normal=None, meteo_raw=None)
    
    delevels_paired, surface_paired = Snapshot.pack_pair_snapshots(ss1, ss2)

    export_size = Snapshot.get_export_size()    
    for lvl, group in zip(delevels_paired, Snapshot.delevel_plan):
        assert lvl.shape == ((Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * 2 + Snapshot.dense_dims, export_size[0], export_size[1])
    assert surface_paired.shape == ((Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * 2 + Snapshot.dense_dims, export_size[0], export_size[1])

    dense_titles = ['orograph, const.', 'sin lat, const.', 'sin lon., const.', 'cos lon, const.']
    cat_titles = ['land sea mask, const.', 'soil type, const.']
    tempo_titles = ['sin orbit, tempo.', 'cos orbit, tempo.', 'sin solor, tempo.', 'cos solar, tempo.']

    surface_items = []
    for dim in range(Snapshot.dense_dims):
        surface_items.append([(Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * 2 + dim, -1, dense_titles[dim]])
    for dim in range(Snapshot.tempo_dims):
        for ss in range(2):   # 2 snapshots
            surface_items.append([(Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * ss + Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + dim, ss, tempo_titles[dim]])

    level_items = []
    for level_id, group in enumerate(Snapshot.delevel_plan):
        for dim in range(Snapshot.dense_dims):
            level_items.append([level_id, (Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * 2 + dim, -1, dense_titles[dim]])
        for dim in range(Snapshot.tempo_dims):
            for ss in range(2):   # 2 snapshots
                level_items.append([level_id, (Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * ss + Snapshot.air_level_dims * len(group) + dim, ss, tempo_titles[dim]])

    # If packing is correct, this code should show correct geo-distributions.
    for [channel_id, ss, title] in surface_items:
        show_on_map(surface[channel_id].numpy(), title=f"{title} ss = {ss}, surface")
    for [level_id, channel_id, ss, title] in level_items:
        show_on_map(delevels_paired[level_id][channel_id].numpy(), title=f"{title} ss = {ss}, level {level_id}")
