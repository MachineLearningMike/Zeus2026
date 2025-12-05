
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
class Snapshot:
    assert mike1_variables_order == "[v_l for v ztuvq for l in levels] + surface_vars"
       
    export_factor10 = 10
    # temporary. can be added and changed any time
    meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
    tempo_dims = 4      # 4 variables: sin/cos earth_orbital_angle (2), sin/cos true_solar_hour (2)
    geo_dims = 6        # 4 constants + 2 embeddings: sin lat, sin/cos lon, orograph, land_sea_mask_emb_placeholder, soil_type_emb_placeholder. lat/lon NOT omitted though they work for position bias.

    fn_nornalize = None
    fn_unnornalize = None

    export_height_org = 720        # exclude the north pole.
    export_width_org = 1440
    
    num_air_levels = num_levels
    air_level_dims = num_vars_per_level
   
    delevel_plan = best_segments_increasing(num_air_levels, 7)
    
    _shared_geo_tensor = None   # on device
    geo_nCategories = None
    
    def set_original_settings():
        Snapshot.export_factor10 = 10
        # temporary. can be added and changed any time
        Snapshot.meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
        Snapshot.tempo_dims = 4      # 4 variables + 1 embedding: sin/cos earth_orbital_angle (2), sin/cos true_solar_hour (2), cascade_emb
        Snapshot.geo_dims = 6        # 4 constants + 2 embeddings: sin lat, sin/cos lon, orograph, land_sea_mask_emb_placeholder, soil_type_emb_placeholder. lat/lon NOT omitted though they work for position bias.

        Snapshot.fn_nornalize = None
        Snapshot.fn_unnornalize = None

        Snapshot.export_height_org = 720        # exclude the north pole.
        Snapshot.export_width_org = 1440
        
        Snapshot.num_air_levels = num_levels
        Snapshot.air_level_dims = num_vars_per_level
    
        Snapshot.delevel_plan = best_segments_increasing(Snapshot.num_air_levels, 7)
        
        Snapshot._shared_geo_tensor = None   # on device
        Snapshot.geo_nCategories = None
        
        categorical_npy_paths = [config['land_sea_mask_npy_path'], config['soil_type_npy_path']]
        dense_npy_paths = [config['orograph_npy_path']]
        Snapshot.create_geo_tensor(categorical_npy_paths, dense_npy_paths)
    
    def __init__(self, year, month, day, hour, meteo, meteo_unnormal, device='cpu'):
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
        exp_h, exp_w = Snapshot.get_export_size()
        size_org = (Snapshot.export_height_org, Snapshot.export_width_org)
        
        if meteo is None:
            _meteo_tensor = torch.rand(Snapshot.meteo_dims, exp_h, exp_w) - 0.5
        else:
            assert torch.is_tensor(meteo)
            assert meteo.shape[0] == Snapshot.meteo_dims
            assert meteo.shape[1] >= exp_h
            assert meteo.shape[2] >= exp_w
            _meteo_tensor = meteo    
        self._meteo_tensor = Snapshot.crop_tensor(_meteo_tensor, size_org)

        _tempo_tensor = self.create_tempo_tensor()
        self._tempo_tensor = Snapshot.crop_tensor(_tempo_tensor, size_org)

        # self._meteo_tensor, self._tempo_tensor = Snapshot.crop_tensors_to_common_size([_meteo_tensor, _tempo_tensor])

        # if meteo_unnormal is not None:
        #     meteo_unnormal, _ = Snapshot.crop_tensors_to_common_size([meteo_unnormal, _tempo_tensor])
        if meteo_unnormal is not None:
            meteo_unnormal = Snapshot.crop_tensor(meteo_unnormal, size_org)
        self._meteo_unnormal_internal = meteo_unnormal

        self._geo_tensor = Snapshot.crop_tensor(Snapshot._shared_geo_tensor, size_org)

    def crop_tensors_to_common_size(tensors):
        # Compute common min height/width
        h = min(t.shape[1] for t in tensors)
        w = min(t.shape[2] for t in tensors)
        
        exp_h, exp_w = Snapshot.get_export_size()
        assert h >= exp_h and w >= exp_w
        
        cropped = []
        for t in tensors:
            t_h = t[:, -h:] if Geo_Tempo.respect_north else t[:, :h]
            t_hw = t_h[:, :, -w:] if Geo_Tempo.respect_east else t_h[:, :, :w]
            cropped.append(t_hw)
        
        return cropped
    
    def crop_tensor_to_export_size(tensor):
        exp_h, exp_w = Snapshot.get_export_size()
        assert tensor.shape[1] >= exp_h and tensor.shape[2] >= exp_w
        tensor_cropped = tensor[:, -exp_h:] if Geo_Tempo.respect_north else tensor[:, :exp_h]
        tensor_cropped = tensor_cropped[:, :, -exp_w:] if Geo_Tempo.respect_east else tensor_cropped[:, :, :exp_w]
        return tensor_cropped
        
    def crop_tensor(tensor, size):
        tensor_cropped = tensor[:, -size[0]:] if Geo_Tempo.respect_north else tensor[:, :size[0]]
        tensor_cropped = tensor_cropped[:, :, -size[1]:] if Geo_Tempo.respect_east else tensor_cropped[:, :, :size[1]]
        return tensor_cropped
        
    def num_get_train_variables():
        return Snapshot.meteo_dims + Snapshot.geo_dims + Snapshot.tempo_dims
        
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
    
    #------------------ meteo -----------------------
    def _get_meteo_tensor_internal(self):
        return self._meteo_tensor
    def _set_meteo_tensor_internal(self, meteo):
        self._meteo_tensor = meteo
    meteo_internal = property(
        fget = _get_meteo_tensor_internal,
        fset = _set_meteo_tensor_internal
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
        tempo_tensor = torch.concat([sin_cos_earth_orbit_angle_tensor_era5, sin_cos_true_solar_hours_tensor_era5], dim=0)
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
        return (int(Snapshot.export_height_org/10*Snapshot.export_factor10), int(Snapshot.export_width_org/10*Snapshot.export_factor10))
    
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
    
    def _get_input_tensor(self):
        if self.meteo_tempo is None or self.geo_internal is None: return None
        _meteo_tempo, _geo = self.meteo_tempo_internal, self.geo_internal                           
        
        if Snapshot.export_factor10 != 10:
            _meteo_tempo = Snapshot.resize_for_export(_meteo_tempo)
            _geo = Snapshot.resize_for_export(_geo)

        input = torch.concat([_meteo_tempo, _geo], dim=0)

        input = input.to(self.device)
        input.detach()
        # input.requires_grad = False
        return input
    
    input = property(
        fget = _get_input_tensor,
    )
    
    def _get_meteo_tempo_tensor(self):
        # These two tensors are only created by the constructor with equal size and don't change later on.
        meteo_tempo = self.meteo_tempo_internal
        
        if Snapshot.export_factor10 != 10:
            meteo_tempo = Snapshot.resize_for_export(meteo_tempo)
            
        meteo_tempo = meteo_tempo.to(self.device)
        meteo_tempo.detach()
        # meteo_tempo.requires_grad = False
        return meteo_tempo
        
    meteo_tempo = property(
        fget = _get_meteo_tempo_tensor,
    )

    def _get_meteo_tensor(self):
        meteo = self.meteo_internal
        
        if Snapshot.export_factor10 != 10:
            meteo = Snapshot.resize_for_export(meteo)
        
        meteo = meteo.to(self.device)
        meteo.detach()
        # meteo.requires_grad = False   # meteo.detach() is enough. More, requires_grad cannot be changed for non-leaf variables. 
        return meteo

    def _set_meteo_tensor(self, meteo):
        assert torch.is_tensor(meteo)
        self.meteo_internal = meteo    # Risk of size mismatch

    meteo = property(
        fget = _get_meteo_tensor,
        fset = _set_meteo_tensor,
    )
    
    def _get_meteo_unnormal_tensor(self):
        _meteo_unnormal = self._meteo_unnormal_internal
    
        if Snapshot.export_factor10 != 10:
            _meteo_unnormal = Snapshot.resize_for_export(_meteo_unnormal)
    
        _meteo_unnormal = _meteo_unnormal.to(self.device)
        _meteo_unnormal.detach()
    
        return _meteo_unnormal
    
    meteo_unnormal = property(
        fget = _get_meteo_unnormal_tensor,
    )
    
    def _get_tempo_tensor(self):
        tempo = self.tempo_internal
        
        if Snapshot.export_factor10 != 10:
            tempo = Snapshot.resize_for_export(tempo)
        
        tempo = tempo.to(self.device)
        tempo.detach()
        # tempo.requires_grad = False
        return tempo

    tempo = property(
        fget = _get_tempo_tensor,
    )
    
    def _get_geo_tensor(self):
        geo = self.geo_internal
        
        if Snapshot.export_factor10 != 10:
            geo = Snapshot.resize_for_export(geo)
        
        geo = geo.to(self.device)
        geo.detach()
        # geo.requires_grad = False
        return geo

    geo = property(
        fget = _get_geo_tensor,
    )

    #------------------ miscellaneout --------------------
    def delete(self):
        self._device = 'cpu'
        
    def pack_pair_snapshots(ss1, ss2):
        """
        Packs two snapshots by interleaving meteo, tempo, and geo features per level,
        for vertical levels and surface variables.
        """

        def arrange_by_level(ss):
            _, H, W = ss.meteo.shape
            
            meteo = ss.meteo                        
            meteo = meteo[:Snapshot.air_level_dims*Snapshot.num_air_levels].view(Snapshot.air_level_dims, Snapshot.num_air_levels, H, W)
            
            tempo = ss.tempo # 

            # --- Vertical levels ---            

            delevels = []
            for idx, group in enumerate(Snapshot.delevel_plan):
                start = group[0]
                end = group[-1] + 1
                
                level_slice = torch.concat([meteo[:, start:end, :, :].reshape(-1, H, W), tempo], dim=0)
                delevels.append(level_slice)  # list of length levels, each (different s, H, W)


            # --- Surface variables ---
            meteo_surface = ss.meteo[Snapshot.air_level_dims*Snapshot.num_air_levels:, :, :].view(-1, H, W)  # (m_surf, H, W)
            tempo_surface = tempo                                                                     # (t, H, W)
            surface = torch.concat([meteo_surface, tempo_surface], dim=0)

            return delevels, surface

        # --- Process snapshots ---
        lev1, surf1 = arrange_by_level(ss1)
        lev2, surf2 = arrange_by_level(ss2)
        
        assert ss1.device == ss2.device

        # --- Concatenate corresponding levels ---
        _, H, W = ss1.meteo.shape
        geo = ss1.geo #[:, None, :, :].expand(-1, 1, H, W)      # (g, 1, H, W)
        delevels_paired = [torch.cat([l1, l2, geo], dim=0).to(ss1.device) for l1, l2 in zip(lev1, lev2)]  # list of (m+t+g)*2, H, W
        surface_paired = torch.cat([surf1, surf2, geo], dim=0).to(ss2.device)                             # (m_surf+t+g)*2, H, W
        assert not torch.isnan(surface_paired).any()

        export_size = Snapshot.get_export_size()
        for lvl, group in zip(delevels_paired, Snapshot.delevel_plan):
            assert lvl.shape == ((Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * 2 + Snapshot.geo_dims, export_size[0], export_size[1])
        assert surface_paired.shape == ((Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * 2 + Snapshot.geo_dims, export_size[0], export_size[1])

        return delevels_paired, surface_paired

Snapshot.set_original_settings()

#------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #---------------------- Check utility functions ----------
    for n in range(1, 30):
        for s in range(1, n):
            check_consistency(n, s)
    
    #---------------------- Check Snapshot --------------------
    
    Snapshot.set_original_settings()
    ss = Snapshot(2020, 1, 1, 1, meteo=None, meteo_unnormal=None)
    print("original size: ", ss.meteo.shape, ss.tempo.shape, ss.geo.shape, ss.meteo.dtype, ss.geo.dtype)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    nVars = Snapshot.num_get_train_variables()
    print(f"------------ Total number of train variables: {nVars}")

    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
    
    levels, surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    for idx, lvl in enumerate(levels):
        print(idx, lvl.shape)
        break
    print(14, surface.shape)

    now = datetime.now(timezone.utc)
    snapshot_now = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    # print("meteo_internal, tempo_internal, meteo_tempo_internal, geo_internal, nCat: ", snapshot_now.meteo_internal.shape, snapshot_now.tempo_internal.shape, snapshot_now.meteo_tempo_internal.shape, snapshot_now.geo_internal.shape, snapshot_now.geo_nCategories)
    # print("input, meteo, meteo_tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo.shape, snapshot_now.meteo_tempo.shape, snapshot_now.geo.shape)

    Snapshot.export_factor10 = 2
    print("input_tensor, meteo_tempo, meteo, tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo_tempo.shape, snapshot_now.meteo.shape, snapshot_now.tempo.shape, snapshot_now.geo.shape)    
    Snapshot.export_factor10 = 10
    
    # Snapshot.device = "cuda"
    Snapshot.device = "cpu"
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
    print(Snapshot.delevel_plan)
    show_on_map(snapshot_now.geo[0].numpy())     # land_sea_mask,    category
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
