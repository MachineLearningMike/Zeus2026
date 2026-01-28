
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
from modules.zeus_dataset_var_names import VARIABLE_NAME_LIST_UNCHANGEABLE, mike1_variables_order, num_vars_per_level, num_levels, HECTA_PASCAL_UNCHANGEABLE
from modules.utils import check_consistency, resize_continuous_bilinear_channel_wise_cylindric, resize_categorical_nearest_channelwise_cylindric

import math


#--------------------------------------- Snapshot ----------------------------

class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls):
        return self.f(cls)


class Snapshot:
    assert mike1_variables_order == "surface_vars only"
    factor10 = 10
    meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
       
    geo_dense_variables = ["orograph", "sin_lat", "cos_lat", "sin_lon", "cos_lon"]
    geo_cat_variables = ["land_sea_mask", "soil_type"]
    tempo_variables = ["sin_orbit", "cos_orbit", "sin_zenith", "cos_zenith"]

    tempo_dims = len(tempo_variables)    
    geo_dense_dims = len(geo_dense_variables)    
    geo_cat_dims = len(geo_cat_variables)

    original_720_h = 720        # exclude the north pole.
    original_1440_w = 1440
    
    num_air_levels = 0 # num_levels
    air_level_dims = 0 # num_vars_per_level
    
    geo_dense = None
    geo_cat = None
    cos_lat_sin_lon = None
    
    def set_original_settings(factor10=10, increasing=True, packed_levels=7, device='cpu',
        geo_dense_variables = ["orograph", "sin_lat", "cos_lat", "sin_lon", "cos_lon"],
        geo_cat_variables = ["land_sea_mask", "soil_type"],
        tempo_variables = ["sin_orbit", "cos_orbit", "sin_zenith", "cos_zenith"],                            
        ):
        Snapshot.factor10 = factor10
        Snapshot.meteo_dims = len(VARIABLE_NAME_LIST_UNCHANGEABLE)
        
        Snapshot.geo_dense_variables = geo_dense_variables
        Snapshot.geo_cat_variables = geo_cat_variables
        Snapshot.tempo_variables = tempo_variables
        
        Snapshot.tempo_dims = len(tempo_variables)    
        Snapshot.geo_dense_dims = len(geo_dense_variables)    
        Snapshot.geo_cat_dims = len(geo_cat_variables)
       
        Snapshot.original_720_h = 720        # exclude the north pole.
        Snapshot.original_1440_w = 1440
        
        # Snapshot.num_air_levels = num_levels
        Snapshot.air_level_dims = 0 # num_vars_per_level + int(len(HECTA_PASCAL_UNCHANGEABLE )/num_levels)
    
        if increasing:
            Snapshot.delevel_plan = [] # best_segments_increasing(Snapshot.num_air_levels, packed_levels)
        else:
            Snapshot.delevel_plan = [] # best_segments_decreasing(Snapshot.num_air_levels, packed_levels)
        
        dense_npy_paths = [config["orograph_npy_path"]]
        categorical_npy_paths = [config["land_sea_mask_npy_path"], config["soil_type_npy_path"]]

        #--------------------------------------------------------------------------------------------------------
        #-------------------- orograph_npy_path, cos(lat), sin(lon), cos(lon)
        geo_dense = Snapshot.create_geo_dense(dense_npy_paths, geo_dense_variables).to(device)      #---------- to device 
        Snapshot.geo_dense = geo_dense
        size = Snapshot.get_export_size()
        assert Snapshot.geo_dense.shape == (Snapshot.geo_dense_dims, size[0], size[1])
        
        #-------------------- orograph_npy_path, cos(lat), sin(lon), cos(lon)
        geo_cat = Snapshot.create_geo_cat(categorical_npy_paths, geo_cat_variables).to(device)        #---------- to device
        Snapshot.geo_cat = geo_cat
        size = Snapshot.get_export_size()
        assert Snapshot.geo_cat.shape == (Snapshot.geo_cat_dims, size[0], size[1])
        
        cos_lat_sin_lon = Snapshot.create_cos_lat_sin_lon()
        Snapshot.cos_lat_sin_lon = cos_lat_sin_lon
        size = Snapshot.get_export_size()
        assert Snapshot.cos_lat_sin_lon.shape == (2, size[0], size[1])


    def get_season_events(year):
        return Geo_Tempo.get_season_events(year)
            
    def __init__(self, year, month, day, hour, meteo, meteo_unnormal, with_tempo=True):
        assert year is not None and month is not None and day is not None and hour is not None
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
                
        self._meteo, self._tempo, self._geo_dense, self._geo_cat = None, None, None, None

        #---------------------- temporary ----------------------
        exp_h, exp_w = Snapshot.get_export_size()
        full_size = (Snapshot.original_720_h, Snapshot.original_1440_w)

        #----------------- meteo normalized -------------
        # meteo, if not None, comes in two scenarios: 
        # - dataset: (n, 721, 1440)
        # - prediction: (n, 720/10*factor10, 1440/10*factor10) = (n, exp_h, exp_w)
        if meteo is None:
            meteo_tensor = torch.rand(Snapshot.meteo_dims, exp_h, exp_w) - 0.5     # placeholder for tests.
        else:
            meteo_tensor = Snapshot.crop_tensor(meteo, (Snapshot.original_720_h, Snapshot.original_1440_w))
            if Snapshot.factor10 != 10:
                meteo_tensor = resize_continuous_bilinear_channel_wise_cylindric(meteo_tensor, Snapshot.get_export_size())
        self.meteo = meteo_tensor

        #--------------- meteo_unnormalized -------------
        # meteo_unnormal, if not None, comes from dataset: (n, 721, 1440)
        # 
        if meteo_unnormal is not None:
            meteo_unnormal = Snapshot.crop_tensor(meteo_unnormal, full_size)
            if Snapshot.factor10 != 10:
                meteo_unnormal = resize_continuous_bilinear_channel_wise_cylindric(meteo_unnormal, Snapshot.get_export_size())

        self.meteo_unnormal = meteo_unnormal

        #--------------- tempo ---------------

        if with_tempo:
            tempo = self.create_tempo(Snapshot.tempo_variables)        
            self.tempo = tempo

        #---------------- geo ----------------
        
        self._geo_dense = Snapshot.geo_dense
        self._geo_cat = Snapshot.geo_cat
        
        
    def create_cos_lat_sin_lon():
        tensor = Geo_Tempo.cos_lat_sin_lon_tensor_era5()
        tensor = Snapshot.crop_tensor(tensor, (Snapshot.original_720_h, Snapshot.original_1440_w))
        if Snapshot.factor10 != 10:
            tensor = resize_continuous_bilinear_channel_wise_cylindric(tensor, Snapshot.get_export_size())
        return tensor
        

    def crop_tensor(tensor, size):
        tensor_cropped = tensor[:, -size[0]:] if Geo_Tempo.respect_north else tensor[:, :size[0]]
        tensor_cropped = tensor_cropped[:, :, -size[1]:] if Geo_Tempo.respect_east else tensor_cropped[:, :, :size[1]]
        return tensor_cropped
        
    def num_get_train_variables():
        return Snapshot.meteo_dims + Snapshot.tempo_dims + Snapshot.geo_dense_dims + Snapshot.geo_cat_dims

    #--------------------------------------------------------------------------
    
    def create_tempo(self, variables):
        # Do NOT mormalize this instance-wide. They should be normalized dataset-wide.
        dt_utc = datetime(self.year, self.month, self.day, self.hour, tzinfo=timezone.utc)
        sin_cos_earth_orbit_angle_tensor_era5 = Geo_Tempo.sin_cos_earth_orbit_angle_tensor_era5(dt_utc)
        # sin_cos_true_solar_hours_tensor_era5 = Geo_Tempo.sin_cos_true_solar_hours_tensor_era5(dt_utc)
        sin_cos_zenith_tensor_era5 = Geo_Tempo.sin_cos_zeith_angle_tensor_era5(dt_utc)
        
        def select(name):
            tensor = None        
            if name == "sin_orbit":
                tensor = sin_cos_earth_orbit_angle_tensor_era5[0]
            elif name == "cos_orbit":
                tensor = sin_cos_earth_orbit_angle_tensor_era5[1]
            elif name == "sin_zenith":
                tensor = sin_cos_zenith_tensor_era5[0]
            elif name == "cos_zenith":
                tensor = sin_cos_zenith_tensor_era5[1]
            return tensor

        tensor_list = [select(name) for name in variables]
        tensor = torch.stack(tensor_list, dim=0)
        
        tensor = Snapshot.crop_tensor(tensor, (Snapshot.original_720_h, Snapshot.original_1440_w))
        if Snapshot.factor10 != 10:
            tensor = resize_continuous_bilinear_channel_wise_cylindric(tensor, Snapshot.get_export_size())
        
        return tensor
    
    def load_npy_files(npy_paths):
        arrays = []
        for path in npy_paths:
            array = np.load(path)
            assert array.shape == (721, 1440)
            has_nan = np.isnan(array).any()
            assert not has_nan
            arrays.append(array)            
        return arrays
        
    def create_geo_dense(dense_npy_paths, variables):
        #------------------- dense, file -----------------
        # DO normalize them, because this instance are all repeated and dataset-wide normalization is equal to instance-wide normalization.
        dense_arrays = Snapshot.load_npy_files(dense_npy_paths)
        processed_arrays = []
        for arr in dense_arrays:
            assert arr.shape == (Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid)
            processed_arrays.append(arr)

        # Stack into one array
        file_stack = np.stack(processed_arrays, axis=0)  # shape: (n, final_H, final_W)
        file_stack = torch.from_numpy(file_stack)  # shape: (n, final_H, final_W)

        #-------------------- dense, lat lon ------------------
        sin_cos_lat_tensor_era5_tensor = Geo_Tempo.sin_cos_lat_tensor_era5()
        sin_cos_lon_tensor_era5_tensor = Geo_Tempo.sin_cos_lon_tensor_era5()
        
        def select(name):
            tensor = None        
            if name == "orograph":
                idx = 0
                while name not in dense_npy_paths[idx]: idx += 1
                tensor = file_stack[idx]
            elif name == "sin_lat":
                tensor = sin_cos_lat_tensor_era5_tensor[0]
            elif name == "cos_lat":
                tensor = sin_cos_lat_tensor_era5_tensor[1]
            elif name == "sin_lon":
                tensor = sin_cos_lon_tensor_era5_tensor[0]
            elif name == "cos_lon":
                tensor = sin_cos_lon_tensor_era5_tensor[1]
            return tensor

        tensor_list = [select(name) for name in variables]
        tensor = torch.stack(tensor_list, dim=0)
        
        tensor = Snapshot.crop_tensor(tensor, (Snapshot.original_720_h, Snapshot.original_1440_w))
        if Snapshot.factor10 != 10:
            tensor = resize_continuous_bilinear_channel_wise_cylindric(tensor, Snapshot.get_export_size())
        tensor = (tensor - tensor.mean(dim=(1,2), keepdim=True)) / (tensor.std(dim=(1,2), keepdim=True) + 1e-12)
        
        return tensor

    
    def create_geo_cat(categorical_npy_paths, variables):
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

        def select(name):
            tensor = None        
            if name == "land_sea_mask":
                idx = 0
                while name not in categorical_npy_paths[idx]: idx += 1
                tensor = processed_arrays[idx]
            elif name == "soil_type":
                idx = 0
                while name not in categorical_npy_paths[idx]: idx += 1
                tensor = processed_arrays[idx]
            return tensor

        tensor_list = [torch.from_numpy(select(name)) for name in variables]
        tensor = torch.stack(tensor_list, dim=0)
        
        tensor = Snapshot.crop_tensor(tensor, (Snapshot.original_720_h, Snapshot.original_1440_w))
        if Snapshot.factor10 != 10:
            tensor = resize_categorical_nearest_channelwise_cylindric(tensor, Snapshot.get_export_size())                    
        # No normalization for categories !!!!!!!!!!!
        # tensor = (tensor - tensor.mean(dim=(1,2), keepdim=True)) / (tensor.std(dim=(1,2), keepdim=True) + 1e-12)

        return tensor
            
    #------------------ export tensors  --------------------
    def get_export_size():
        return (int(Snapshot.original_720_h/10*Snapshot.factor10), int(Snapshot.original_1440_w/10*Snapshot.factor10))

    def is_export_size(tensor):
        exp_h, exp_w = Snapshot.get_export_size()
        return tensor.shape[-2] == exp_h and tensor.shape[-1] == exp_w

    #----------------------- packinng --------------------
        
    def pack_pair_snapshots(ss1, ss2, device='cpu'):
        """
        Packs two snapshots by interleaving meteo, tempo, and geo features per level,
        for vertical levels and surface variables.
        """

        def arrange_by_level(ss):
            _, H, W = ss.meteo.shape
            
            meteo = ss.meteo                        
            # meteo = meteo[:Snapshot.air_level_dims*Snapshot.num_air_levels].view(Snapshot.air_level_dims, Snapshot.num_air_levels, H, W)
            
            # --- Vertical levels ---            

            # delevels = []
            # for idx, group in enumerate(Snapshot.delevel_plan):
            #     start = group[0]
            #     end = group[-1] + 1
                
            #     level_slice = torch.concat([meteo[:, start:end, :, :].reshape(-1, H, W), ss.tempo], dim=0).to(device)
            #     delevels.append(level_slice)  # list of length levels, each (different s, H, W)

            # --- Surface variables ---
            meteo_surface = ss.meteo # [Snapshot.air_level_dims*Snapshot.num_air_levels:, :, :].view(-1, H, W)  # (m_surf, H, W)
            tempo_surface = ss.tempo                                                                     # (t, H, W)
            surface = torch.concat([meteo_surface, tempo_surface], dim=0)  #.to(device)

            return surface

        # --- Process snapshots ---
        meteo_temop = arrange_by_level(ss1)
        surf1_meteo_temop_dense = torch.cat([meteo_temop.to(device), ss1.geo_dense], dim=0)
        
        surf2_meteo_temop_dense = None
        if ss1 != ss2:
            meteo_temop = arrange_by_level(ss2)
            surf2_meteo_temop_dense = torch.cat([meteo_temop.to(device), ss2.geo_dense], dim=0)
                
        # --- Concatenate corresponding levels ---
        _, H, W = ss1.meteo.shape

        if ss1 == ss2:    
            # delevels_paired = [torch.cat([l1, l2, ss1.geo_dense], dim=0) for l1, l2 in zip(lev1, lev2)]  # list of (m+t+g)*2, H, W
            surface_meteo_tempo_dense = surf1_meteo_temop_dense       # (m_surf+t+g), H, W
        else:
            # delevels_paired = [torch.cat([l1, ss1.geo_dense], dim=0) for l1 in lev1]  # list of (m+t+g)*2, H, W
            surface_meteo_tempo_dense = torch.concat([surf1_meteo_temop_dense, surf2_meteo_temop_dense], dim=0)       # (m_surf+t+g) * 2, H, W

        #---------- asserts ------------        
        # assert not torch.isnan(surface_paired).any()
        # export_size = Snapshot.get_export_size()
        # nSnapshots = 2 if ss1 != ss2 else 1
        # for lvl, group in zip(delevels_paired, Snapshot.delevel_plan):
        #     assert lvl.shape == ((Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * nSnapshots + Snapshot.geo_dense_dims, export_size[0], export_size[1])
        # assert surface_paired.shape == ((Snapshot.meteo_dims - Snapshot.air_level_dims * Snapshot.num_air_levels + Snapshot.tempo_dims) * nSnapshots + Snapshot.geo_dense_dims, export_size[0], export_size[1])

        return surface_meteo_tempo_dense  # requires_grad == False
    
    #------------------ miscellaneout --------------------

Snapshot.set_original_settings(factor10=2)

#------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    #---------------------- Check utility functions ----------
    for n in range(1, 30):
        for s in range(1, n):
            check_consistency(n, s)
    
    #---------------------- Check Snapshot --------------------
    
    Snapshot.set_original_settings()
    ss = Snapshot(2020, 1, 1, 1, meteo=None, meteo_unnormal=None)
    print("original size: ", ss.meteo.shape, ss.tempo.shape, ss.geo_dense.shape, ss.geo_cat.shape, ss.meteo.dtype, ss.geo_dense.dtype, ss.geo_cat.dtype)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    nVars = Snapshot.num_get_train_variables()
    print(f"------------ Total number of train variables: {nVars}")

    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
    
    surface = Snapshot.pack_pair_snapshots(ss1, ss2)
    # for idx, lvl in enumerate(levels):
    #     print(idx, lvl.shape)
    #     break
    print(14, surface.shape)

    now = datetime.now(timezone.utc)
    snapshot_now = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    # print("meteo_internal, tempo_internal, meteo_tempo_internal, geo_internal, nCat: ", snapshot_now.meteo_internal.shape, snapshot_now.tempo_internal.shape, snapshot_now.meteo_tempo_internal.shape, snapshot_now.geo_internal.shape, snapshot_now.geo_nCategories)
    # print("input, meteo, meteo_tempo, geo: ", snapshot_now.input.shape, snapshot_now.meteo.shape, snapshot_now.meteo_tempo.shape, snapshot_now.geo.shape)

    Snapshot.export_factor10 = 2
    print("meteo, tempo, geo: ", snapshot_now.meteo.shape, snapshot_now.tempo.shape, snapshot_now.geo_dense.shape, snapshot_now.geo_cat.shape)    
    Snapshot.export_factor10 = 10
    
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

    import time
    s = time.time()
    zenithh = Geo_Tempo.get_zenithh_angle_faster(dt_utc = datetime.now(timezone.utc))
    e = time.time()
    show_on_map(zenithh.numpy(), title=f"zenithh angle now. time taken: {e-s}")

    #------------- geo_cat, 2 -----------------
    # print(Snapshot.delevel_plan)
    # show_on_map(snapshot_now.geo_cat[0].numpy(), title="land_sea_mask")     # land_sea_mask,    category
    # show_on_map(snapshot_now.geo_cat[1].numpy(), title="soil_type")     # soil_type,        category
    
    # #------------- geo_dense, 4 -----------------
    # show_on_map(snapshot_now.geo_dense[0].numpy(), title="geopotential_at_surface")     # geopotential_at_surface,  standardized
    # show_on_map(snapshot_now.geo_dense[1].numpy(), title="sin(lat)")     # sin(lat),  as-is
    # show_on_map(snapshot_now.geo_dense[2].numpy(), title="sin(lon)")     # sin(lon),  as-is
    # show_on_map(snapshot_now.geo_dense[3].numpy(), title="cos(lon)")     # cos(lon),  as-is

    # #------------- tempo, 5 -----------------
    # show_on_map(snapshot_now.tempo[0].numpy(), title="sin(earth_orbital_angle)")      # sin(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[1].numpy(), title="cos(earth_orbital_angle)")      # cos(earth_orbital_angle)
    # show_on_map(snapshot_now.tempo[2].numpy(), title="sin(true_solar_hour)")      # sin(true_solar_hour)
    # show_on_map(snapshot_now.tempo[3].numpy(), title="cos(true_solar_hour)")      # cos(true_solar_hour)
    # show_on_map(snapshot_now.tempo[4].numpy(), title="cascade_emb_placeholder")      # cascade_emb_placeholder


    #------------------------------- test packing ----------------------
    Snapshot.set_original_settings()
    
    now = datetime.now(timezone.utc)
    ss1 = Snapshot(now.year, now.month, now.day, now.hour, meteo=None, meteo_unnormal=None)
    then = now + timedelta(hours=6)
    ss2 = Snapshot(then.year, then.month, then.day, then.hour, meteo=None, meteo_unnormal=None)
    
    surface_paired = Snapshot.pack_pair_snapshots(ss1, ss2)

    nSnapshots = 2 if ss1 != ss2 else 1
    
    export_size = Snapshot.get_export_size()
    # for lvl, group in zip(delevels_paired, Snapshot.delevel_plan):
    #     assert lvl.shape == ((Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * nSnapshots + Snapshot.geo_dense_dims, export_size[0], export_size[1])
    assert surface_paired.shape == ((Snapshot.meteo_dims + Snapshot.tempo_dims + Snapshot.geo_dense_dims) * nSnapshots, export_size[0], export_size[1])

    geo_dense_variables = Snapshot.geo_dense_variables
    geo_cat_variables = Snapshot.geo_cat_variables
    tempo_variables = Snapshot.tempo_variables

    surface_items = []
    for dim in range(Snapshot.geo_dense_dims):
        surface_items.append([Snapshot.meteo_dims + dim, -1, tempo_variables[dim]])
    for dim in range(Snapshot.tempo_dims):
        for ss in range(nSnapshots):   # 2 snapshots
            surface_items.append([(Snapshot.meteo_dims + Snapshot.tempo_dims) + dim, ss, geo_dense_variables[dim]])

    # level_items = []
    # for level_id, group in enumerate(Snapshot.delevel_plan):
    #     for dim in range(Snapshot.geo_dense_dims):
    #         level_items.append([level_id, (Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * nSnapshots + dim, -1, geo_dense_variables[dim]])
    #     for dim in range(Snapshot.tempo_dims):
    #         for ss in range(2):   # 2 snapshots
    #             level_items.append([level_id, (Snapshot.air_level_dims * len(group) + Snapshot.tempo_dims) * ss + Snapshot.air_level_dims * len(group) + dim, ss, tempo_variables[dim]])

    # If packing is correct, this code should show correct geo-distributions.
    for [channel_id, ss, title] in surface_items:
        show_on_map(surface_paired[channel_id].numpy(), title=f"{title} ss = {ss}, surface")
    # for [level_id, channel_id, ss, title] in level_items:
    #     show_on_map(delevels_paired[level_id][channel_id].numpy(), title=f"{title} ss = {ss}, level {level_id}")
