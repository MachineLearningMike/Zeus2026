import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import threading
import shutil
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import gc
import logging
import math
import platform
import heapq
from multiprocessing import Pool

import json, time
from einops import rearrange
from datetime import datetime, date, timedelta, timezone

import random
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

import os, sys, gc
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.CylindricalResizer import CylindricalResizer
from modules.utils import check_dir
# Make they're from sure modules_zeus
from modules_zeus.constants import ERA5_DATA_VARS
from modules_zeus.sample import Era5Sample
from modules_zeus.coordinates import get_bbox, slice_bbox
from modules_zeus.era5_cds import Era5CDSLoader, get_today
from modules.zeus_dataset_var_names import VARIABLE_NAME_LIST_UNCHANGEABLE, var_name_map, num_vars_per_level, num_levels, HECTA_PASCAL_UNCHANGEABLE

from modules.zeus_dataset_codec_Mike1 import load_and_decode, save
from modules.zeus_dataset_Snapshot import Snapshot
# from modules.zeus_dataset_sharding_dynamic import ERA5ZarrShards

class Zeus_Dataset(Dataset):
    def __init__(self, pm, dataset_path, management_path, download=False):
        super().__init__()
        self.pm = pm
        self.data_limit = pm.data_limit
        
        check_dir(dataset_path)
        check_dir(management_path)

        if not self.pm.dual_drive:
            self.cds_folder = dataset_path / "era5"
            self.cds_folder.mkdir(parents=True, exist_ok=True)
            self.shard_dirs = [self.cds_folder]
        else:
            self.cds_folder_even = dataset_path / "data_even" / "era5"            
            self.cds_folder_even.mkdir(parents=True, exist_ok=True)
            self.cds_folder_odd = dataset_path / "data_odd" / "era5"
            self.cds_folder_odd.mkdir(parents=True, exist_ok=True)
            self.shard_dirs = [self.cds_folder_even, self.cds_folder_odd]
    
        self.param_path = management_path / 'dataset_params.json'
        self.split_hours_path = management_path / 'split_hours.json'
        self.mean_meteo_checkpoint_path = management_path / "mean_meteo_checkpoint.pth"
        self.mean_meteo_onestop_checkpoint_path = management_path / "mean_meteo_onestop_checkpoint.pth"
        self.variance_meteo_checkpoint_path = management_path / "mean_var_meteo_checkpoint.pth"
        self.mean_tempo_checkpoint_path = management_path / "mean_tempo_checkpoint.pth"
        self.variance_tempo_checkpoint_path = management_path / "mean_var_tempo_checkpoint.pth"
        
        self.difficulties = None
        self.zeus_loss_weights_on_target_vars = None

        self.era5_cds_loader = Era5CDSLoader()
        self.train_dates, self.vali_dates, self.test_dates = None, None, None
        self.train_set, self.vali_set, self.test_set = None, None, None
        self.x_set, self.x_set_len, self.data_slots = None, 0, []

        self._fetched_batches = 0
        self.total_batches = 0
        
        self.origin_date = datetime(1970,1,1)
        
        self.resizer = CylindricalResizer()
        
        self.total_ds = None
        if download: self.total_ds = self.get_normalized_dataset()   # For downloading
        
        #-------------------------------------------------------------------------
        self.normal_meteo_buffers = [None] * (self.pm.nRawStepsPerBatch + 2)
        self.unnormal_meteo_buffers = [None] * (self.pm.nRawStepsPerBatch + 2)
        
        self.normal_tempo_buffers = [None] * (self.pm.nRawStepsPerBatch + 2)
        self.unnormal_tempo_buffers = [None] * (self.pm.nRawStepsPerBatch + 2)
        
        self.air_level_pressure_indices = None
        self.air_level_pressure_values = None
        self.taken = None
        
        self.area_weights_cuda = None

        
        # self.shard_dataset = ERA5ZarrShards(
        #     base_dirs=self.shard_dirs,
        #     shard_shapes=self.get_shard_shape(),
        #     max_slices=1000,
        #     chunklen=1,
        #     auto_expand=True
        # )

    #------------------------------------- Download dataset -----------------------
    
    def get_shard_shape(self):
        shape = {}
        for variable in self.pm.meteo_variables:
            shape.update({ variable: (721, 1440) })
        return shape

    def hours_since_epoch(self, year, month, day, hour):
        dt = datetime(year, month, day, hour, tzinfo=timezone.utc)
        epoch = datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc)
        delta = dt - epoch
        return int(delta.total_seconds() // 3600)

    def load_data(self):
        self.mean_meteo, self.std_meteo, self.mean_tempo, self.std_tempo, \
        self.mean_meteo_ready_cpu, self.std_meteo_ready_cpu, self.mean_meteo_ready_cuda, self.std_meteo_ready_cuda, \
        self.mean_tempo_ready_cpu, self.std_tempo_ready_cpu, self.mean_tempo_ready_cuda, self.std_tempo_ready_cuda, \
        self.area_weights_cuda \
        = self.load_dataset_params()
        
        if self.pm.include_difficulty: 
            self.difficulties = self.load_difficulties()
        
        self.zeus_loss_weights_on_target_vars = self.load_era5_data_vars()
        
        self.train_hours, self.vali_hours, self.test_hours = self.split_data_hours_by_date_segents(
            self.pm.train_start_dt, self.pm.vali_start_dt, self.pm.test_start_dt, self.pm.test_end_dt_inclusive,
            renew=True, save=True, seed=self.pm.seed
        )
        return
    
    def get_normalized_dataset(self):
        
        if self.total_ds is None:
        
            dataset = None
            try:
                dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
                # convert latitude and longitude to -90 to 90 and -180 to 180 respectively
                if dataset["longitude"].max() > 180:
                    dataset = dataset.assign_coords(longitude=(dataset["longitude"].values + 180) % 360 - 180)
                if dataset["latitude"].max() > 90:
                    dataset = dataset.assign_coords(latitude=dataset["latitude"].values - 90)

                dataset = dataset.sortby(["latitude", "longitude"])
                dataset = dataset.chunk()
                
                #  -90 <= lat <= +90 [-90, +90], -180 <= lon < 180 [-180, +180)
            except:
                print("Failed to get gcp-era5 dataset")

            self.total_ds = dataset
        else:
            dataset = self.total_ds

        return dataset

    def get_file_path(self, year, month, day, hour, modulo):
        # - self.pm.hourly_offset is not required
        cds_folder = self.cds_folder if not self.pm.dual_drive else self.cds_folder_even if hour % 2 == 0 else self.cds_folder_odd
        return cds_folder / str(year) / str(month).zfill(2) / str(day).zfill(2)
    def get_file_path_npz(self, year, month, day, hour, modulo):
        return self.get_file_path(year, month, day, hour, modulo) / (str(hour).zfill(2) + ".npz")
    def get_file_path_npy(self, year, month, day, hour, modulo):
        return self.get_file_path(year, month, day, hour, modulo) / (str(hour).zfill(2) + ".npy")
    def get_last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]
    
    #--------------------------------------------------------------------------------------

    def next_month(self, dt):
        return dt + relativedelta(month=1)
    def last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]
    def month_string(self, date):
        return str(date.year) + '-' + str(date.month).zfill(2)
    def day_string(self, date):
        return str(date.year) + '-' + str(date.month).zfill(2) + '-' + str(date.day).zfill(2)
    
    #---------------------------------------------------------------------------   
    
    def load_decode_check_for_variable_existency(self, dt, modulo, variables):
        path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
        try:
            dict = load_and_decode(path)
            for var in variables:
                assert var in dict
        except Exception as e:
            print(e)
            dict = None
        return dict
    
    def load_decode_get_array_for_variables(self, dt, modulo, variables):
        path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
        arrays = []
        try:
            dict = load_and_decode(path)
            #----------------------------------- This decides the order of arrays !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #----------------------------------- The order must be consistent across training and inference. !!!!!!!!!!!!!
            #----------------------------------- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for var in variables:
                arr = dict[var]
                arrays.append(arr)
            array = np.stack(arrays, axis=0)
            assert array.shape[0] == self.pm.nVariables
        except Exception as e:
            print(e)
            array = None
        return array
    
    def download_variable(self, idx,  year, month, day, hour, variable, simulated=False):
        """Download a single variable for a given snapshot, interpolating NaNs if needed."""
        
        upper_air = False
        snapshot_time = f"{year}-{month:02d}-{day:02d}T{hour:02d}:00"
        
        if simulated:
            array = np.random.rand(721, 1440)
        else:
            total_ds = self.total_ds
            if total_ds is None:
                print("Getting normal_meteo_buffers ERA5 dataset......", end='')
                total_ds = self.get_normalized_dataset()   # Multiple threads run this line.

            # Check if it's an upper-air variable
            if '_' in variable and variable.split('_')[0] in variable:
                upper_air = True
                var_name, level = variable.split('_')
                era_var = var_name_map[var_name]
                ds = total_ds[era_var].sel(level=int(level), time=snapshot_time)
            else:
                era_var = var_name_map[variable]
                ds = total_ds[era_var].sel(time=snapshot_time)

            array = ds.values.astype(np.float32)

        # If NaNs exist, try interpolation along 'level' for upper-air variables
        if np.isnan(array).any() and upper_air:
            logging.warning(f"NaNs detected in {idx} {variable}, interpolating along pressure levels...")
            # Interpolate from neighboring levels
            # ds_full = total_ds[era_var].sel(time=snapshot_time)
            # Interpolation along 'level' to the desired level
            array = ds.interp(level=int(level)).values.astype(np.float32)

        assert not np.isnan(array).any(), f"Warning: {idx} {variable} still has NaNs at {year}/{month}/{day}/{hour}"
        assert array.shape == (721, 1440), f"array.shape is not (721, 1440) but {array.shape}"
        print(f"Variable downloaded OK: {idx} {variable} at {year}/{month}/{day}/{hour}")

        return variable, array


    def load_meteo_file_downloading_if_not_ready(self, year, month, day, hour, modulo, max_workers_vars, variables, no_return=False, simulated=False):
        """Download all variables for a single snapshot in parallel."""

        dict = None
        
        try:
            dict = self.load_decode_check_for_variable_existency(datetime(year, month, day, hour), modulo, variables)
        except:
            pass
        
        if dict is not None:                       
            print(f"--------------------------------------------------- Meteo file at {year}/{month}/{day}/{hour} ready !")               
        else:       # dict == None
            print(F"--------------------------------------------------- Meteo file at {year}/{month}/{day}/{hour} (re-)downloading ...")

            variance_dict = {}

            # Parallel download of variables
            with ThreadPoolExecutor(max_workers=max_workers_vars) as var_executor:
                var_futures = [var_executor.submit(self.download_variable, idx, year, month, day, hour, var, simulated) for idx, var in enumerate(variables)]

                for idx, future in enumerate(as_completed(var_futures)):
                    var_name, array = future.result()
                    variance_dict[var_name] = array

                success = True
                for idx, var in enumerate(variables):
                    if variance_dict.get(var, None) is None:
                        logging.fatal(f"--------------- variable: {idx} {year}/{month}/{day}/{hour}/{idx}, {variables[idx]} failed to download. Quiting snapshot...")
                        success = False; break
                    else:
                        assert variance_dict[var].shape == (721, 1440)

                if success:
                    snapshot_path = self.get_file_path_npz(year, month, day, hour, modulo)
                    
                    check_dir(snapshot_path.parent)
                    save(snapshot_path, variance_dict)
                    dict = self.load_decode_check_for_variable_existency(datetime(year, month, day, hour), modulo, variables)

                if dict is not None:
                    print(f"==================================== snapshot at {year}/{month}/{day}/{hour} saved         !")
                else:
                    print(f"==================================== snapshot at {year}/{month}/{day}/{hour} failed        !!!")

        if not no_return:
            return dict
        else:
            if dict is not None: del dict
            return None
                    
                    
    def download_meteo_variables(
            self, variables, var_name_map,
            start_dt, end_dt, lat_range=None, lon_range=None,
            modulo=6, max_workers_snapshots=4, max_workers_vars=6,
            memory_limit_mb=1048, simulated=False):  # 1GB limit
        
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        
        # Collect time steps
        time_steps = []                            
        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            time_steps.append((dt.year, dt.month, dt.day, dt.hour))
        
        total = len(time_steps)
        print(f"Processing {total} steps, memory limit: {memory_limit_mb}MB")
        
        with ThreadPoolExecutor(max_workers=max_workers_snapshots) as executor:
            futures = []
            completed = 0
            
            for idx, (year, month, day, hour) in enumerate(time_steps):

                available_memory = psutil.virtual_memory().available / 1024 / 1024
                while available_memory < memory_limit_mb:  # CHANGED: < instead of >
                    print(f"Memory low ({available_memory:.1f}MB available), pausing...")
                    # Wait for some tasks to complete
                    self._wait_for_some_completions(futures, completed)
                    gc.collect()
                    available_memory = psutil.virtual_memory().available / 1024 / 1024

                no_return = True                
                future = executor.submit(
                    self.load_meteo_file_downloading_if_not_ready,
                    year, month, day, hour,
                    modulo, max_workers_vars, variables, no_return, simulated
                )
                futures.append((future, idx))
            
            # Process results
            for future, idx in futures:
                try:
                    result = future.result()
                    completed += 1
                    
                    if completed % 10 == 0:
                        mem = get_memory_usage()
                        print(f"{completed}/{total}, Memory: {mem:.1f}MB")
                        
                except Exception as e:
                    print(f"Task {idx} failed: {e}")

    def _wait_for_some_completions(self, futures, min_complete):
        """Wait for some futures to complete"""
        from concurrent.futures import wait, FIRST_COMPLETED
        if len(futures) > 0:
            wait([f[0] for f in futures], return_when=FIRST_COMPLETED)
            
    def get_geo_data(self, management_path, variables):
        total_ds = self.get_normalized_dataset()
        snapshot_time = "2025-11-10T00:00"

        dict = {}        
        for var in variables:
            ds = total_ds[var]
            if "time" in ds.dims:
                ds = ds.sel(time=snapshot_time)

            array = ds.values
            assert array.shape == (721, 1440)
            array = array.astype(np.float32)
            assert not np.isnan(array).any(), f"{var} has NaNs at {snapshot_time}"

            dict[var] = array
            np.save( management_path / (var + ".npy"), array)

        return dict


    def iter_hourly_paths(self):
        if not self.pm.dual_drive:
            base_folders = [Path(self.cds_folder)]
        else:
            even_root = Path(self.cds_folder_even)
            odd_root  = Path(self.cds_folder_odd)
            base_folders = [even_root, odd_root]

        def extract_hour(name: str):
            # Extract last 1–2 digits before extension
            import re
            m = re.search(r'(\d{1,2})(?=\D*$)', name)
            return int(m.group()) if m else None

        def hourly_iter(base_folder):
            for year_path in sorted(base_folder.iterdir()):
                if not year_path.is_dir() or not year_path.name.isdigit():
                    continue
                year = int(year_path.name)

                for month_path in sorted(year_path.iterdir()):
                    if not month_path.is_dir() or not month_path.name.isdigit():
                        continue
                    month = int(month_path.name)

                    for day_path in sorted(month_path.iterdir()):
                        if not day_path.is_dir() or not day_path.name.isdigit():
                            continue
                        day = int(day_path.name)

                        for file_path in sorted(day_path.iterdir()):
                            if not file_path.is_file():
                                continue

                            hour = extract_hour(file_path.stem)
                            if hour is None or not (0 <= hour <= 23):
                                continue

                            # No dn parity filtering — directories already separate them
                            yield (year, month, day, hour, file_path)

        iterators = [hourly_iter(base) for base in base_folders if base.is_dir()]

        for item in heapq.merge(*iterators,
                                key=lambda x: (x[0], x[1], x[2], x[3])):
            yield item

    def analyze_hourly_paths(self, inclusive_left=None, inclusive_right=None, modulo=6):
        # print("even contents:", list(self.cds_folder_even.iterdir()))
        # print("odd contents:", list(self.cds_folder_odd.iterdir()))
        
        """
        Analyze hourly files with optional datetime boundaries.
        If inclusive_left/right are None, use the dataset's actual min/max.

        Returns:
            {
                "count": int,
                "first_hour": int or None,
                "last_hour": int or None,
                "missing_hours": list[int],
                "existing_hours": list[int],
            }
        """

        # Step 1 — Scan all hours first
        hours = []
        for year, month, day, hour, path in self.iter_hourly_paths():
            assert int(path.stem) in range(24), f"{year}, {month}, {day}, {path}"
            assert path.suffix == ".npz", f"{year}, {month}, {day}, {path}"
            hn = self.hours_since_epoch(year, month, day, hour)
            hours.append(hn)
        # assert sorted(hours) == sorted(list(set(hours))) 
        n_unexpected_hours = len(hours) - len(list(set(hours)))
 
        existing_hours = sorted(hours)
        dataset_min = existing_hours[0]
        dataset_max = existing_hours[-1]

        # Step 2 — Convert boundaries
        if inclusive_left is None:
            left_hn = dataset_min
        else:
            left_hn = self.hours_since_epoch(
                inclusive_left.year, inclusive_left.month,
                inclusive_left.day, inclusive_left.hour
            )

        if inclusive_right is None:
            right_hn = dataset_max
        else:
            right_hn = self.hours_since_epoch(
                inclusive_right.year, inclusive_right.month,
                inclusive_right.day, inclusive_right.hour
            )

        if left_hn > right_hn:
            raise ValueError("inclusive_left must be <= inclusive_right")

        # Step 3 — Keep only hours inside the interval
        existing_hours = [hn for hn in existing_hours if left_hn <= hn <= right_hn]

        first_hour = existing_hours[0]
        last_hour = existing_hours[-1]

        # Step 4 — Identify missing hours in the window
        full_range = sorted(list(set(range(left_hn, right_hn + 1, modulo))))
        missing = sorted(set(full_range) - set(existing_hours))

        return {
            "count": len(existing_hours),
            "n_unexpected_hours" : n_unexpected_hours,
            "first_hour": first_hour,
            "last_hour": last_hour,
            "missing_hours": missing,
            "existing_hours": existing_hours,
        }
    
    def read_checkpoint(self, path, keys):
        checkpoint = torch.load(path, weights_only=False)
        values = []
        for key in keys:
            if checkpoint.get(key, None) is not None:
                values.append(checkpoint[key])
        return tuple(values)
    
    def save_checkpoint(self, path, items):
        try:
            data = torch.load(path, weights_only=False)
        except:
            data = {}
        data.update(items)
        torch.save(data, path)
        return       

    def get_mean_meteo_downloading_if_not_ready(self, inclusive_left=None, inclusive_right=None, checkpoint_interval=10, modulo=6, file_name_suffix=None):
        """
        Compute the mean over all variables across hourly files, with checkpointing.
        """
        cuda_available = torch.cuda.is_available()
        mean_dict = None
        
        original = self.mean_meteo_onestop_checkpoint_path
        checkpoint_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)

        task = "mean_meteo"

        try:
            checkpoint = self.read_checkpoint(
                checkpoint_path,
                ["mean_dict", "step", "inclusive_left", "inclusive_right"]
            )

            mean_dict = checkpoint[0] if len(checkpoint) > 0 else None
            mean = np.array([mean_dict[var] for var in self.pm.meteo_variables]) if mean_dict is not None else None
            
            # step = checkpoint[1] if len(checkpoint) > 1 else 0
            step = int(checkpoint[1]) if len(checkpoint) > 1 else 0
            ck_left_iso = checkpoint[2] if len(checkpoint) > 2 else None
            ck_right_iso = checkpoint[3] if len(checkpoint) > 3 else None

            ck_left = datetime.fromisoformat(ck_left_iso) if ck_left_iso else None
            ck_right = datetime.fromisoformat(ck_right_iso) if ck_right_iso else None

            # Check consistency
            ck_left_match = (ck_left == inclusive_left) or (ck_left is None and inclusive_left is None)
            ck_right_match = (ck_right == inclusive_right) or (ck_right is None and inclusive_right is None)
            if mean is None or not (ck_left_match and ck_right_match):
                print(f"Checkpoint missing or date range mismatch. Starting new {task} fram scratch...")
                mean = np.zeros((len(self.pm.meteo_variables),), dtype=np.float32)
                step = 0
            else:
                print(f"Found step {step}-th record. Resuming {task} from step =. {step+1}..")
                step = step
        except:
            print(f"No checkpoint found for mean meteo. Starting new {task} from scratch...")
            mean = np.zeros((len(self.pm.meteo_variables),), dtype=np.float32)
            step = 0

        skip = -1
                        
        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
            
            skip += 1
            if skip < step:
                continue

            data = self.load_meteo_file_downloading_if_not_ready(
                dt.year, dt.month, dt.day, dt.hour, modulo, 
                max_workers_vars=len(self.pm.meteo_variables), 
                variables=self.pm.meteo_variables,
                )
            if data is None:
                print("Filed to load data. Quiting...") 
                sys.exit(1)
            
            # Make sure the order follows self.pm.meteo_variables
            arr = np.stack([data[key] for key in self.pm.meteo_variables], axis=0)  # (C, H, W)

            if cuda_available:
                tensor_arr = torch.tensor(arr, dtype=torch.float32).cuda()
                _mean = tensor_arr.mean(dim=(1, 2)).cpu().numpy()
            else:
                _mean = arr.mean(axis=(1, 2))

            if np.any(np.isnan(_mean)):
                raise Exception(f"NaN found in hourly mean: {dt}. Quiting...")

            step += 1
            mean = mean * (step - 1)/step + _mean / step
            
            mean_dict = {var: mean[idx] for idx, var in enumerate(self.pm.meteo_variables)}

            if step % checkpoint_interval == 0:
                print(f"Checkpointing mean_meteo... {step}: {year}-{month:02d}-{day:02d} {hour:02d}")
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "mean_dict": mean_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(tmp_path, checkpoint_path)

        if mean_dict is not None:
            # Final save
            tmp_path = str(checkpoint_path) + ".tmp"
            self.save_checkpoint(tmp_path, {
                "mean_dict": mean_dict,
                "step": step,
                "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
            })
            shutil.move(tmp_path, checkpoint_path)

        return mean, step


    def get_mean_meteo(self, inclusive_left=None, inclusive_right=None, checkpoint_interval=10, modulo=6, file_name_suffix=None):
        """
        Compute the mean over all variables across hourly files, with checkpointing.
        """
        cuda_available = torch.cuda.is_available()
        mean_dict = None
        
        original = self.mean_meteo_checkpoint_path
        checkpoint_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)

        task = "mean_meteo"
        
        try:
            checkpoint = self.read_checkpoint(
                checkpoint_path,
                ["mean_dict", "step", "inclusive_left", "inclusive_right"]
            )

            mean_dict = checkpoint[0] if len(checkpoint) > 0 else None
            mean = np.array([mean_dict[var] for var in self.pm.meteo_variables]) if mean_dict is not None else None
            
            # step = checkpoint[1] if len(checkpoint) > 1 else 0
            step = int(checkpoint[1]) if len(checkpoint) > 1 else 0
            ck_left_iso = checkpoint[2] if len(checkpoint) > 2 else None
            ck_right_iso = checkpoint[3] if len(checkpoint) > 3 else None

            ck_left = datetime.fromisoformat(ck_left_iso) if ck_left_iso else None
            ck_right = datetime.fromisoformat(ck_right_iso) if ck_right_iso else None

            # Check consistency
            ck_left_match = (ck_left == inclusive_left) or (ck_left is None and inclusive_left is None)
            ck_right_match = (ck_right == inclusive_right) or (ck_right is None and inclusive_right is None)
            if mean is None or not (ck_left_match and ck_right_match):
                print(f"Checkpoint missing or date range mismatch. Starting new {task} fram scratch...")
                mean = np.zeros((len(self.pm.meteo_variables),), dtype=np.float32)
                step = 0
            else:
                print(f"Found step {step}-th record. Resuming {task} from step =. {step+1}..")
                step = step
        except:
            print(f"No checkpoint found for mean meteo. Starting new {task} from scratch...")
            mean = np.zeros((len(self.pm.meteo_variables),), dtype=np.float32)
            step = 0

        skip = -1

        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
            
            skip += 1
            if skip < step:
                continue

            file_path = self.get_file_path_npz(year, month, day, hour, modulo)
            dict = load_and_decode(file_path)
            
            # Make sure the order follows self.pm.meteo_variables
            arr = np.stack([dict[key] for key in self.pm.meteo_variables], axis=0)  # (C, H, W)

            if cuda_available:
                tensor_arr = torch.tensor(arr, dtype=torch.float32).cuda()
                _mean = tensor_arr.mean(dim=(1, 2)).cpu().numpy()
            else:
                _mean = arr.mean(axis=(1, 2))

            if np.any(np.isnan(_mean)):
                raise Exception(f"NaN found in mean_meteo: {file_path}")

            step += 1
            mean = mean * (step - 1)/step + _mean / step
            
            mean_dict = {var: mean[idx] for idx, var in enumerate(self.pm.meteo_variables)}

            if step % checkpoint_interval == 0:
                print(f"Checkpointing mean_meteo... {step}: {year}-{month:02d}-{day:02d} {hour:02d}")
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "mean_dict": mean_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(checkpoint_path)

        if mean_dict is not None:
            # Final save
            tmp_path = str(checkpoint_path) + ".tmp"
            self.save_checkpoint(tmp_path, {
                "mean_dict": mean_dict,
                "step": step,
                "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
            })
            shutil.move(tmp_path, checkpoint_path)

        return mean, step


    def get_var_meteo(self, mean, inclusive_left=None, inclusive_right=None, checkpoint_interval=10, modulo=6, file_name_suffix=None):
        """
        Compute the standard deviation over all variables across hourly files, with checkpointing.
        Uses incremental variance formula.
        """
        cuda_available = torch.cuda.is_available()
        variance_dict = None
        
        original = self.variance_meteo_checkpoint_path
        checkpoint_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)

        task = "variance_meteo"
        
        try:
            checkpoint = self.read_checkpoint(
                checkpoint_path,
                ["variance_dict", "step", "inclusive_left", "inclusive_right"]
            )

            variance_dict = checkpoint[0] if len(checkpoint) > 0 else None
            variance = np.array([variance_dict[var] for var in self.pm.meteo_variables]) if variance_dict is not None else None

            
            # step = checkpoint[1] if len(checkpoint) > 1 else 0
            step = int(checkpoint[1]) if len(checkpoint) > 1 else 0
            ck_left_iso = checkpoint[2] if len(checkpoint) > 2 else None
            ck_right_iso = checkpoint[3] if len(checkpoint) > 3 else None
            
            ck_left = datetime.fromisoformat(ck_left_iso) if ck_left_iso else None
            ck_right = datetime.fromisoformat(ck_right_iso) if ck_right_iso else None
            
            # Check consistency with provided boundaries
            ck_left_match = (ck_left == inclusive_left) or (ck_left is None and inclusive_left is None)
            ck_right_match = (ck_right == inclusive_right) or (ck_right is None and inclusive_right is None)            
            if variance is None or not (ck_left_match and ck_right_match):
                print(f"Checkpoint missing or date range mismatch. Starting new {task} fram scratch...")
                variance = np.ones((len(self.pm.meteo_variables),), dtype=np.float32)
                step = 0
            else:
                print(f"Found step {step}-th record. Resuming {task} from step =. {step+1}..")
                step = step
        except:
            print(f"No checkpoint found for mean meteo. Starting new {task} from scratch...")
            variance = np.ones((len(self.pm.meteo_variables),), dtype=np.float32)
            step = 0

        skip = -1

        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
            
            skip += 1
            if skip < step:
                continue

            file_path = self.get_file_path_npz(year, month, day, hour, modulo)
            dict = load_and_decode(file_path)

            # Make sure the order follows self.pm.meteo_variables
            arr = np.stack([dict[key] for key in self.pm.meteo_variables], axis=0)  # (C, H, W)

            if cuda_available:
                tensor_arr = torch.tensor(arr, dtype=torch.float32).cuda()
                mean_tensor = torch.tensor(mean, dtype=torch.float32).cuda().view(-1, 1, 1)
                _variance = torch.mean((tensor_arr - mean_tensor) ** 2, dim=(1, 2)).cpu().numpy()
            else:
                _variance = np.mean((arr - mean[:, None, None]) ** 2, axis=(1, 2))

            if np.any(np.isnan(_variance)):
                raise Exception(f"NaN found in var_meteo: {file_path}")

            step += 1
            variance = variance * (step - 1)/step + _variance / step

            variance_dict = {var: variance[idx] for idx, var in enumerate(self.pm.meteo_variables)}

            if step % checkpoint_interval == 0:
                print(f"Checkpointing variance_meteo...{step}: {year}-{month:02d}-{day:02d} {hour:02d}")
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "variance_dict": variance_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(tmp_path, checkpoint_path)

        if variance_dict is not None:
            # Final save
            tmp_path = str(checkpoint_path) + ".tmp"
            self.save_checkpoint(tmp_path, {
                "variance_dict": variance_dict,
                "step": step,
                "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
            })
            shutil.move(tmp_path, checkpoint_path)
        
        return variance, step
    

    def get_mean_tempo(self, inclusive_left=None, inclusive_right=None, checkpoint_interval=10, modulo=6, file_name_suffix=None):
        """
        Compute the mean over all variables across hourly files, with checkpointing.
        """        
        cuda_available = torch.cuda.is_available()
        mean_dict = None
        
        Snapshot.set_original_settings(factor10=self.pm.factor10, 
            geo_dense_variables = self.pm.geo_dense_variables,
            geo_cat_variables = self.pm.geo_cat_variables,
            tempo_variables = self.pm.tempo_variables,            
            )

        original = self.mean_tempo_checkpoint_path
        checkpoint_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)

        task = "mean_tempo"

        try:
            checkpoint = self.read_checkpoint(
                checkpoint_path,
                ["mean_dict", "step", "inclusive_left", "inclusive_right"]
            )

            mean_dict = checkpoint[0] if len(checkpoint) > 0 else None            
            mean = np.array([mean_dict[var] for var in Snapshot.tempo_variables]) if mean_dict is not None else None
            
            # step = checkpoint[1] if len(checkpoint) > 1 else 0
            step = int(checkpoint[1]) if len(checkpoint) > 1 else 0
            ck_left_iso = checkpoint[2] if len(checkpoint) > 2 else None
            ck_right_iso = checkpoint[3] if len(checkpoint) > 3 else None

            ck_left = datetime.fromisoformat(ck_left_iso) if ck_left_iso else None
            ck_right = datetime.fromisoformat(ck_right_iso) if ck_right_iso else None

            # Check consistency
            ck_left_match = (ck_left == inclusive_left) or (ck_left is None and inclusive_left is None)
            ck_right_match = (ck_right == inclusive_right) or (ck_right is None and inclusive_right is None)
            if mean is None or not (ck_left_match and ck_right_match):
                print(f"Checkpoint missing or date range mismatch. Starting new {task} fram scratch...")
                mean = np.zeros((Snapshot.tempo_dims,), dtype=np.float32)
                step = 0
            else:
                print(f"Found step {step}-th record. Resuming {task} from step =. {step+1}..")
                step = step
        except:
            print(f"No checkpoint found for mean meteo. Starting new {task} from scratch...")
            mean = np.zeros((Snapshot.tempo_dims,), dtype=np.float32)
            step = 0

        skip = -1
        
        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
            
            skip += 1
            if skip < step:
                continue

            tensor = Snapshot(year, month, day, hour, meteo=None, meteo_unnormal=None).tempo
            
            if cuda_available:
                tensor = tensor.cuda()
                _mean = tensor.mean(dim=(1, 2)).cpu().numpy()
            else:
                _mean = tensor.numpy().mean(axis=(1, 2))

            if np.any(np.isnan(_mean)):
                raise Exception(f"NaN found in mean_tempo")

            step += 1
            mean = mean * (step - 1)/step + _mean / step
            
            mean_dict = {var: mean[idx] for idx, var in enumerate(Snapshot.tempo_variables)}
            
            if step % checkpoint_interval == 0:
                print(f"Checkpointing mean_tempo... {step}: {year}-{month:02d}-{day:02d} {hour:02d}")
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "mean_dict": mean_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(tmp_path, checkpoint_path)

        if mean_dict is not None:
            # Final save
            if mean_dict:
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "mean_dict": mean_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(tmp_path, checkpoint_path)

        return mean, step


    def get_var_tempo(self, mean, inclusive_left=None, inclusive_right=None, checkpoint_interval=10, modulo=6, file_name_suffix=None):
        """
        Compute the standard deviation over all variables across hourly files, with checkpointing.
        Uses incremental variance formula.
        """
        cuda_available = torch.cuda.is_available()
        variance_dict = None
        
        Snapshot.set_original_settings(factor10=self.pm.factor10, 
            geo_dense_variables = self.pm.geo_dense_variables,
            geo_cat_variables = self.pm.geo_cat_variables,
            tempo_variables = self.pm.tempo_variables,            
            )
        
        original = self.variance_tempo_checkpoint_path
        checkpoint_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)
        
        task = "variance_tempo"

        # Load checkpoint if exists
        try:
            checkpoint = self.read_checkpoint(
                checkpoint_path,
                ["variance_dict", "step", "inclusive_left", "inclusive_right"]
            )

            variance_dict = checkpoint[0] if len(checkpoint) > 0 else None
            variance = np.array([variance_dict[var] for var in Snapshot.tempo_variables]) if variance_dict is not None else None
            
            # step = checkpoint[1] if len(checkpoint) > 1 else 0
            step = int(checkpoint[1]) if len(checkpoint) > 1 else 0
            ck_left_iso = checkpoint[2] if len(checkpoint) > 2 else None
            ck_right_iso = checkpoint[3] if len(checkpoint) > 3 else None
            
            ck_left = datetime.fromisoformat(ck_left_iso) if ck_left_iso else None
            ck_right = datetime.fromisoformat(ck_right_iso) if ck_right_iso else None
            
            # Check consistency with provided boundaries
            ck_left_match = (ck_left == inclusive_left) or (ck_left is None and inclusive_left is None)
            ck_right_match = (ck_right == inclusive_right) or (ck_right is None and inclusive_right is None)
            if variance is None or not (ck_left_match and ck_right_match):
                print(f"Checkpoint missing or date range mismatch. Starting new {task} fram scratch...")
                variance = np.ones((Snapshot.tempo_dims,), dtype=np.float32)
                step = 0
            else:
                print(f"Found step {step}-th record. Resuming {task} from step =. {step+1}..")
                step = step
        except:
            print(f"No checkpoint found for mean meteo. Starting new {task} from scratch...")
            variance = np.ones((Snapshot.tempo_dims,), dtype=np.float32)
            step = 0
            
        skip = -1
        
        dt = inclusive_left - timedelta(hours=modulo)
        right = inclusive_right - timedelta(hours=modulo)
        while dt <= right:
            dt += timedelta(hours=modulo)
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
            
            skip += 1
            if skip < step:
                continue
            
            tensor = Snapshot(year, month, day, hour, meteo=None, meteo_unnormal=None).tempo
            
            if cuda_available:
                tensor = tensor.cuda()
                mean_tensor = torch.tensor(mean, dtype=torch.float32).cuda().view(-1, 1, 1)
                _variance = torch.mean((tensor - mean_tensor) ** 2, dim=(1, 2)).cpu().numpy()
            else:
                _variance = np.mean((tensor.numpy() - mean[:, None, None]) ** 2, axis=(1, 2))

            if np.any(np.isnan(_variance)):
                raise Exception(f"NaN found in var_tempo")

            step += 1
            variance = variance * (step - 1)/step + _variance / step
            
            variance_dict = {var: variance[idx] for idx, var in enumerate(Snapshot.tempo_variables)}

            if step % checkpoint_interval == 0:
                print(f"Checkpointing var_tempo...{step}: {year}-{month:02d}-{day:02d} {hour:02d}")
                tmp_path = str(checkpoint_path) + ".tmp"
                self.save_checkpoint(tmp_path, {
                    "variance_dict": variance_dict,
                    "step": step,
                    "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                    "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
                })
                shutil.move(tmp_path, checkpoint_path)

        if variance_dict is not None:
            # Final save
            tmp_path = str(checkpoint_path) + ".tmp"
            self.save_checkpoint(tmp_path, {
                "variance_dict": variance_dict,
                "step": step,
                "inclusive_left": inclusive_left.isoformat() if inclusive_left else None,
                "inclusive_right": inclusive_right.isoformat() if inclusive_right else None
            })
            shutil.move(tmp_path, checkpoint_path)

        return variance, step


    def analyze_hourly_dataset(self, save=False, inclusive_left=None, inclusive_right=None, modulo=6, file_name_suffix=None):
        """
        Analyze dataset over hourly paths: compute mean and std for all variables.

        Parameters:
            save (bool): If True, save the results to self.param_path (JSON).
            inclusive_left (datetime or None): start datetime to include.
            inclusive_right (datetime or None): end datetime to include.

        Returns:
            mean (np.ndarray): mean values (C,)
            std  (np.ndarray): std values (C,)
        """

        # # Compute mean with checkpointing
        # mean_meteo, m = self.get_mean_meteo(
        #     inclusive_left=inclusive_left,
        #     inclusive_right=inclusive_right,
        #     modulo = modulo
        # )

        mean_meteo, m = self.get_mean_meteo_downloading_if_not_ready(
            inclusive_left=inclusive_left,
            inclusive_right=inclusive_right,
            modulo = modulo, file_name_suffix=file_name_suffix
        )

        # Compute std with checkpointing
        var_meteo, n = self.get_var_meteo(
            mean_meteo,
            inclusive_left=inclusive_left,
            inclusive_right=inclusive_right,
            modulo = modulo, file_name_suffix=file_name_suffix
        )

        assert m == n, f"Step mismatch: mean step ({m}) vs std step ({n})"


        # Compute mean with checkpointing
        mean_tempo, m = self.get_mean_tempo(
            inclusive_left=inclusive_left,
            inclusive_right=inclusive_right,
            modulo = modulo, file_name_suffix=file_name_suffix
        )

        # Compute std with checkpointing
        var_tempo, n = self.get_var_tempo(
            mean_tempo,
            inclusive_left=inclusive_left,
            inclusive_right=inclusive_right,
            modulo = modulo, file_name_suffix=file_name_suffix
        )

        assert m == n, f"Step mismatch: mean step ({m}) vs std step ({n})"

        # Save results if requested
        if save:
            # Map variable names to values
            # Make sure the order follows self.pm.meteo_variables
            mean_meteo_dict = {var: float(val) for var, val in zip(self.pm.meteo_variables, list(mean_meteo))}  # mean_, a numpy array, acts as a list
            variance_meteo_dict  = {var: float(val) for var, val in zip(self.pm.meteo_variables, list(var_meteo))}   # var_, a numpy array, acts as a list

            mean_tempo_dict = {var: float(val) for var, val in zip(Snapshot.tempo_variables, list(mean_tempo))}  # mean_, a numpy array, acts as a list
            variance_tempo_dict  = {var: float(val) for var, val in zip(Snapshot.tempo_variables, list(var_tempo))}   # var_, a numpy array, acts as a list

            original = self.param_path
            save_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)

            # Load old data if it exists
            if save_path.exists() and save_path.is_file():
                with open(save_path, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {}     # corrupted or empty file → reset
            else:
                data = {}

            # Update only the relevant fields
            data["mean_meteo_dict"] = mean_meteo_dict
            data["variance_meteo_dict"] = variance_meteo_dict
            data["mean_tempo_dict"] = mean_tempo_dict
            data["variance_tempo_dict"] = variance_tempo_dict

            # Save updated data
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
                
            print("dataset params saved.")
            
        return mean_tempo, var_tempo, mean_tempo, var_tempo
    
    
    def load_dataset_params(self, file_name_suffix=None):

        original = self.param_path
        params_path = original if file_name_suffix is None else original.with_stem(original.stem + "_" + file_name_suffix)


        with open(params_path, "r") as f:
            data = json.load(f)
            # print('------', data, torch.tensor(data['mean']).shape)

        mean_meteo_dict, variance_meteo_dict, mean_tempo_dict, variance_tempo_dict = \
        data['mean_meteo_dict'], data['variance_meteo_dict'], (data['mean_tempo_dict']), data['variance_tempo_dict']

        #--------------------- meteo -----------------
        
        mean_meteo = []
        for var in self.pm.meteo_variables:
            mean_meteo.append(mean_meteo_dict[var])
        var_meteo = []
        for var in self.pm.meteo_variables:
            var_meteo.append(variance_meteo_dict[var])

        # This is the place where var changes to std.            
        mean_meteo, std_meteo = torch.tensor(mean_meteo), torch.sqrt(torch.tensor(var_meteo))   #------------- sqrt
            
        self.mean_meteo, self.std_meteo = mean_meteo, std_meteo
        
        # This violence may cause inconsistency, I know.
        mean_meteo_ready_cpu = rearrange(mean_meteo, 'c -> c 1 1').to("cpu")
        std_meteo_ready_cpu = rearrange(std_meteo, 'c -> c 1 1').to("cpu")
        
        mean_meteo_ready_cuda, std_meteo_ready_cuda = None, None
        if torch.cuda.is_available():
            mean_meteo_ready_cuda = rearrange(mean_meteo, 'c -> c 1 1').to("cuda")
            std_meteo_ready_cuda = rearrange(std_meteo, 'c -> c 1 1').to("cuda")


        #----------------- tempo -------------------
        
        # Current Snapshot.tempo_variables should be part of the Snapshot.tempo_variables used to create dataset params
        
        Snapshot.set_original_settings(factor10=self.pm.factor10, 
            geo_dense_variables = self.pm.geo_dense_variables,
            geo_cat_variables = self.pm.geo_cat_variables,
            tempo_variables = self.pm.tempo_variables,            
            )
        
        mean_tempo = []
        for var in Snapshot.tempo_variables:
            mean_tempo.append(mean_tempo_dict[var])
        var_tempo = []
        for var in Snapshot.tempo_variables:
            var_tempo.append(variance_tempo_dict[var])

        # This is the place where var changes to std.            
        mean_tempo, std_tempo = torch.tensor(mean_tempo), torch.sqrt(torch.tensor(var_tempo))           #----------- sqrt         
                
        self.mean_tempo, self.std_tempo = mean_tempo, std_tempo
        
        # This violence may cause inconsistency, I know.
        mean_tempo_ready_cpu = rearrange(mean_tempo, 'c -> c 1 1').to("cpu")
        std_tempo_ready_cpu = rearrange(std_tempo, 'c -> c 1 1').to("cpu")
        
        mean_tempo_ready_cuda, std_tempo_ready_cuda = None, None
        if torch.cuda.is_available():
            mean_tempo_ready_cuda = rearrange(mean_tempo, 'c -> c 1 1').to("cuda")
            std_tempo_ready_cuda = rearrange(std_tempo, 'c -> c 1 1').to("cuda")

        area_weights_cuda = Snapshot.cos_lat_sin_lon[None, :, None].to("cuda")   # for (C, H, W)

        pack = (
            mean_meteo, std_meteo, mean_tempo, std_tempo, 
            mean_meteo_ready_cpu, std_meteo_ready_cpu, mean_meteo_ready_cuda, std_meteo_ready_cuda,
            mean_tempo_ready_cpu, std_tempo_ready_cpu, mean_tempo_ready_cuda, std_tempo_ready_cuda,
            area_weights_cuda,
        )
        
        return pack
    
    #----------------- Visualize ---------------------
    
    def play_weather_snapshots(self, inclusive_left, inclusive_right, modulo):
               
        from modules.utils import show_array, initialize_for_show_array, finalize_for_show_array
       
        def get_key():
            """
            Get the most recent key press without buffering.
            Returns None if no key is currently pressed.
            """
            import keyboard
            import time
            
            # Check all keys we care about
            keys_to_check = {
                'up': 'up',
                'down': 'down', 
                'left': 'left',
                'right': 'right',
                '1': '1',
                '2': '2',
                '3': '3',
                '4': '4',
                '5': '5',
                '6': 'dec',
                '7': 'mar',
                '8': 'jun',
                '9': 'sep',
                'esc': 'quit',
            }
            
            # Check if any key is currently pressed
            for key, action in keys_to_check.items():
                if keyboard.is_pressed(key):
                    # Wait for key release to ensure we don't get multiple triggers
                    while keyboard.is_pressed(key):
                        time.sleep(0.01)
                    return action
            
            return None
        
        initialize_for_show_array()
        
        # Main loop -----------------------------------------------------------------------------
        var_id = 0
        dt = inclusive_left
                
        old_path = None; key_stroked = True
        while inclusive_left <= dt and dt <= inclusive_right:
            if key_stroked:
                file_path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
                if file_path != old_path:
                    dict_data = load_and_decode(file_path)
                    old_path = file_path

                variable = self.pm.meteo_variables[var_id]
                arr_y = dict_data[variable]  # (C, H, W)

                # Display the array
                show_array(arr1=arr_y, arr2=None, variable1=variable, variable2="", dt=dt, modulo=modulo)
            
            # Wait for key input
            key = None
            start_time = time.time()
            
            # Poll for key for up to 0.5 seconds
            while key is None and (time.time() - start_time < 0.5):
                key = get_key()
                if key is None:
                    time.sleep(0.1)  # Small sleep to prevent CPU overuse
            
            # Process key
            key_stroked = True
            
            if key == "up": 
                var_id += 1
                var_id = 0 if var_id >= len(self.pm.meteo_variables) else var_id
            elif key == "down": 
                var_id -= 1
                var_id = len(self.pm.meteo_variables) - 1 if var_id < 0 else var_id
            elif key == "left": 
                dt = dt - timedelta(hours=modulo)
            elif key == "right": 
                dt = dt + timedelta(hours=modulo)
            elif key == '1':
                modulo = 1
            elif key == '2':
                modulo = 6
            elif key == '3':
                modulo = 12
            elif key == '4':
                modulo = 24
            elif key == '5':
                modulo = 24 * 30
            elif key == 'dec':
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['dec_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'mar':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['mar_equinox'].replace(hour=dt.hour, minute=0)
            elif key == 'jun':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['jun_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'sep':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['sep_equinox'].replace(hour=dt.hour, minute=0)
            elif key == "quit":
                print("Quitting...")
                break
            elif key is None:
                # No key pressed, continue to next iteration
                key_stroked = False
                continue

        finalize_for_show_array()
        
    #-------------------------- play_physical_prediction -----------------------
    
    def play_physical_prediction(self, inclusive_left, inclusive_right, modulo):
            
        from modules.utils import show_array, initialize_for_show_array, finalize_for_show_array
        from light.t2m_predict_n_hours import FastPhysicsT2MPredictor
        from light.u100m_predict_n_hours import U100mPredictor
    
        def get_key():
            """
            Get the most recent key press without buffering.
            Returns None if no key is currently pressed.
            """
            import keyboard
            import time
            
            # Check all keys we care about
            keys_to_check = {
                'up': 'up',
                'down': 'down', 
                'left': 'left',
                'right': 'right',
                '1': '1',
                '2': '2',
                '3': '3',
                '4': '4',
                '5': '5',
                '6': 'dec',
                '7': 'mar',
                '8': 'jun',
                '9': 'sep',
                'esc': 'quit',
            }
            
            # Check if any key is currently pressed
            for key, action in keys_to_check.items():
                if keyboard.is_pressed(key):
                    # Wait for key release to ensure we don't get multiple triggers
                    while keyboard.is_pressed(key):
                        time.sleep(0.01)
                    return action
            
            return None
        
        initialize_for_show_array()
        
        t2m_predictor = FastPhysicsT2MPredictor()
        u100m_predictor = U100mPredictor()
        
        # Main loop -----------------------------------------------------------------------------
        var_id = 0
        dt = inclusive_left
                
        old_path = None; key_stroked = True
        while inclusive_left <= dt and dt <= inclusive_right:
            if key_stroked:
                file_path = self.get_file_path_npz(dt.year, dt.month, dt.day, dt.hour, modulo)
                if file_path != old_path:
                    dict_data = load_and_decode(file_path)
                    old_path = file_path

                variable = self.pm.meteo_variables[var_id]
                arr_y = dict_data[variable]  # (C, H, W)

                pred = None; e=s=time.time()
                dt_x = dt - timedelta(hours=modulo)
                if inclusive_left <= dt_x and dt_x <= inclusive_right:
                    meteo_x = torch.from_numpy(self.load_decode_get_array_for_variables(dt_x, modulo=modulo, variables=self.pm.meteo_variables))
                    ss_x = Snapshot(dt_x.year, dt_x.month, dt_x.day, dt_x.hour, meteo=meteo_x, meteo_unnormal=None)
                
                    s = time.time()
                    if variable == "t2m":
                        pred = t2m_predictor.predict(ss_x.meteo, ss_x.geo_dense, ss_x.geo_cat, ss_x.tempo, dt_x, delta_hours=modulo)
                    elif variable == "u100m":
                        pred = u100m_predictor.predict(ss_x.meteo, ss_x.geo_dense, ss_x.geo_cat, ss_x.tempo, dt_x, delta_hours=modulo)
                    e = time.time()
                    # pred = ss_x.meteo[0].numpy() works   # array_x[0].numpy works
                
                show_array(arr1=arr_y, arr2=pred, variable1=variable, variable2=f"time: {e-s:.5f}", dt=dt, modulo=modulo)
            
            
            # Wait for key input
            key = None
            start_time = time.time()
            
            # Poll for key for up to 0.5 seconds
            while key is None and (time.time() - start_time < 0.5):
                key = get_key()
                if key is None:
                    time.sleep(0.1)  # Small sleep to prevent CPU overuse
            
            # Process key
            key_stroked = True
            
            if key == "up": 
                var_id += 1
                var_id = 0 if var_id >= len(self.pm.meteo_variables) else var_id
            elif key == "down": 
                var_id -= 1
                var_id = len(self.pm.meteo_variables) - 1 if var_id < 0 else var_id
            elif key == "left": 
                dt = dt - timedelta(hours=modulo)
            elif key == "right": 
                dt = dt + timedelta(hours=modulo)
            elif key == '1':
                modulo = 1
            elif key == '2':
                modulo = 6
            elif key == '3':
                modulo = 12
            elif key == '4':
                modulo = 24
            elif key == '5':
                modulo = 24 * 30
            elif key == 'dec':
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['dec_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'mar':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['mar_equinox'].replace(hour=dt.hour, minute=0)
            elif key == 'jun':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['jun_solstice'].replace(hour=dt.hour, minute=0)
            elif key == 'sep':
                h = dt.hour
                dict = Snapshot.get_season_events(dt.year)
                dt = dict['sep_equinox'].replace(hour=dt.hour, minute=0)
            elif key == "quit":
                print("Quitting...")
                break
            elif key is None:
                # No key pressed, continue to next iteration
                key_stroked = False
                continue

        finalize_for_show_array()
    
        
    #-------------------------------------------------
    
    def show_array_single(self, array, variable):
        from modules.utils import show_array_single        
        show_array_single(arr1=array, arr2=None, variable1=variable, variable2="", dt=datetime(2000,1,1,0), modulo=0)
        
    #----------------- Standardize --------------------

    def normalize_meteo(self, input, output):
        mean = self.mean_meteo_ready_cuda if input.is_cuda else self.mean_meteo_ready_cpu
        std  = self.std_meteo_ready_cuda if input.is_cuda else self.std_meteo_ready_cpu
        if input is not output:
            output.copy_(input)        # copy input into the output tensor
        output.sub_(mean).div_(std)  # normalize in-place
        
    def unnormalize_meteo(self, input, output):
        std  = self.std_meteo_ready_cuda if input.is_cuda else self.std_meteo_ready_cpu
        mean = self.mean_meteo_ready_cuda if input.is_cuda else self.mean_meteo_ready_cpu
        if input is not output:    
            output.copy_(input)   # copy input into output buffer
        output.mul_(std).add_(mean)  # in-place unnormalize
        
    def normalize_tempo(self, input, output):    # data: (... C, H, W)
        mean = self.mean_tempo_ready_cuda if input.is_cuda else self.mean_tempo_ready_cpu
        std  = self.std_tempo_ready_cuda if input.is_cuda else self.std_tempo_ready_cpu
        if input is not output:
            output.copy_(input)        # copy input into the output tensor
        output.sub_(mean).div_(std)  # normalize in-place
            
    def unnormalize_tempo(self, input, output):    # data: (... C, H, W)
        std  = self.std_tempo_ready_cuda if input.is_cuda else self.std_tempo_ready_cpu
        mean = self.mean_tempo_ready_cuda if input.is_cuda else self.mean_tempo_ready_cpu
        if input is not output:    
            output.copy_(input)   # copy input into output buffer
        output.mul_(std).add_(mean)  # in-place unnormalize
    
    def load_era5_data_vars(self):
        # vars = [ERA5_DATA_VARS[var_name_map[k]] for k in self.pm.meteo_variables if var_name_map[k] in ERA5_DATA_VARS.keys()]
        vars = list(ERA5_DATA_VARS.values())  # Make it as secure as the above line, later.
        vars = [v * len(vars) for v in vars]
        # vars = [1.0 for _ in range(len(self.p_intercalated_vars) - len(vars))] + vars
        vars = [1.0 for _ in range(len(self.pm.meteo_variables) - len(vars))] + vars
        zeus_loss_weights_on_target_vars = torch.tensor(vars)
        zeus_loss_weights_on_target_vars = zeus_loss_weights_on_target_vars.to('cuda' if torch.cuda.is_available() else 'cpu')
        return zeus_loss_weights_on_target_vars
   
    #----------------------------------- Difficulty ------------------------------

    def previous_nth_month(self, day: date, n: int):
        """
        Returns (year, month) for the previous n-th month from 'day'.
        n starts at 0.
        - If n == 0 and day is last day of month → return current month
        - Otherwise count backwards from the previous month.
        """
        # Last day in current month
        last_day = calendar.monthrange(day.year, day.month)[1]
        
        # Determine how many months to go back from current month
        if day.day == last_day:
            months_back = n
        else:
            months_back = n + 1
        
        # Calculate target month/year
        target_month = day.month - months_back
        target_year = day.year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        
        return target_year, target_month
    

    def create_difficulties(self, latest_date):
        for variable_idx in range(len(self.pm.meteo_variables)):
            # handle variables one by one, as month_data is of huge size.
            print(self.pm.meteo_variables[variable_idx])
            difficulties = []
            maxDeffVar = 0.0
            for nth in range(12):
                year, month = self.previous_nth_month(latest_date, nth)
                first_day = date(year, month, 1)
                last_day = first_day.replace(day = self.last_day_of_month(year, month))

                month_data = []
                for day in range(1, self.last_day_of_month(year, month) + 1):
                    file_path = self.get_file_path_npz(year, month, day, 'all')

                    if not file_path.exists(): 
                        raise Exception("Data file not found. {}/{}/{}, {}".format(year, month, day))

                    _data = np.load(file_path)[:, variable_idx, :, :]   # (24, 721, 1440)
                    month_data.append(torch.tensor(_data, dtype=self.pm.dtype))

                month_data = torch.cat(month_data, dim=0)               # (n, 721, 1440)
                diffs = torch.diff(month_data, n=1, dim=0)              # (n-1, 721, 1440)
                var_per_loc = diffs.var(dim=0).numpy()                  # (721, 1440)

                month_str = self.month_string(first_day)
                print(f"Min/Mean/Max for {month_str}", var_per_loc.min(), var_per_loc.mean(), var_per_loc.max())

                if maxDeffVar < var_per_loc.max().item(): maxDeffVar = var_per_loc.max().item()
                difficulties.append((var_per_loc, year, month))
            
            for var_per_loc, year, month in difficulties:
                rel_var_per_loc = var_per_loc / maxDeffVar      # (721, 1440)
                current_folder = Path(__file__).resolve().parent
                folder = current_folder / "weights" / self.pm.meteo_variables[variable_idx]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (str(year) + '-' + str(month).zfill(2) + ".npy")
                np.save(path, rel_var_per_loc)
                
    def load_difficulties(self):

        total_days, start_day, end_day, complete, data_dates = self.count_data_days()

        difficulties = {}

        for variable in self.pm.meteo_variables:
            _difficulties = {}

            for nth in range(12):
                year, month = self.previous_nth_month(end_day, nth)
                first_day = date(year, month, 1)
                last_day = first_day.replace(day = self.last_day_of_month(year, month))

                current_folder = Path(__file__).resolve().parent
                (current_folder / "weights" / variable).mkdir(parents=True, exist_ok=True)
                data = np.load(current_folder / "weights" / variable / f"{year}-{str(month).zfill(2)}.npy")
                # print('load', variable, month, data.shape)
                _difficulties.update({month: data})     # data: (721, 1440)
            
            difficulties.update({variable: _difficulties})      # difficulties[variable][month] : data (721, 1440)

        print('load per variable per month', difficulties['2m_temperature'][3].shape)

        return difficulties


    def show_difficulty(self, variable):
        var_per_loc = self.difficulties[variable][7]

        dataset = self.get_normalized_dataset()
        # create artificial grid of latitudes and longitudes for plotting
        x_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.as_tensor(
                        dataset[v].data, dtype=self.pm.dtype
                    )
                    for v in ("latitude", "longitude")
                ],
                indexing="ij",
            ),
            dim=-1,
        )

        # plot the last month's data
        fig = plt.figure(figsize=(4.0, 3.0), dpi=100)

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_axisbelow(True)

        gl = ax.gridlines(draw_labels=True)
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        
        if var_per_loc.shape[0] < 721:  # just for the case where var_per_loc.shape[0] == 720, < 721
            var_per_loc = np.concatenate((var_per_loc, np.ones_like(var_per_loc[:721-var_per_loc.shape[0], :])), axis=0)
        
        scatterplot = ax.scatter(
                x_grid[..., -1], 
                x_grid[..., -2], 
                c=var_per_loc, 
                s= 10,
                marker= "o",
                lw= 0, 
        )

        cbar = fig.colorbar(scatterplot, ax=ax, fraction=0.023, pad=0.05)
        plt.show()


    def get_difficulty_grid(self, varaible, sample: Era5Sample) -> np.ndarray:
        """
        Returns the difficulties for each grid location in the given sample.
        """
        start_month = datetime.fromtimestamp(sample.start_timestamp).month

        # months don't differ that much. If the specified month is not found, use the first one in storage.
        difficulty_matrix = self.difficulties[varaible][7]  # month

        difficulty_grid = slice_bbox(difficulty_matrix, sample.get_bbox())

        return difficulty_grid
    
    #--------------------------------- catalogue data -------------------------------

    def get_T2(self):
        start_timestamp, end_timestamp, num_predict_hours = self.era5_cds_loader.sample_time_range()
        current_hour = np.random.randint(0, 24)
        # Make the time-agnostic. Simulate reality, where time drifts.
        start = start_timestamp - get_today('h') + timedelta(hours=current_hour)
        end = end_timestamp - get_today('h') + timedelta(hours=current_hour)
        choice = end
        # t2 = math.ceil((end.to_pytimedelta().total_seconds()/3600) / 24) + 4
        hours = choice.to_pytimedelta().total_seconds()/3600
        t2 = math.ceil(hours/24) + 5    # starts from 5 days ago

        return t2
    
    def get_T2_new(self):
        start_timestamp, end_timestamp, num_predict_hours = self.era5_cds_loader.sample_time_range()
        current_hour = np.random.randint(0, 24)
        # Make the time-agnostic. Simulate reality, where time drifts.
        start = start_timestamp - get_today('h') + timedelta(hours=current_hour)
        end = end_timestamp - get_today('h') + timedelta(hours=current_hour)
        choice = end
        # t2 = math.ceil((end.to_pytimedelta().total_seconds()/3600) / 24) + 4
        hours = choice.to_pytimedelta().total_seconds()/3600
        t2 = math.ceil(hours / self.pm.FG) + math.ceil(5 * 24 / self.pm.FG)    # starts from 5 days ago

        return t2
    
    def split_data_hours(self, renew=False, save=True, seed=None, inclusive_left=None, inclusive_right=None):
        """
        Split hourly data into train/validation/test sets using full datetime objects,
        optionally restricting to a time range [inclusive_left, inclusive_right].

        Parameters:
            renew (bool): if True, generate new split; otherwise reuse existing.
            save (bool): if True, save the split to JSON.
            seed (int or None): random seed for shuffling.
            inclusive_left (datetime or None): left boundary (inclusive) of hours to include.
            inclusive_right (datetime or None): right boundary (inclusive) of hours to include.

        Returns:
            train_hours, vali_hours, test_hours : list of datetime objects
        """
        old_train_hours = []
        old_vali_hours = []
        old_test_hours = []

        # Load previous split if exists
        try:
            with open(self.split_hours_path, "r") as f:
                data = json.load(f)

            old_train_hours = [datetime.fromisoformat(d) for d in data["train_hours"]]
            old_vali_hours  = [datetime.fromisoformat(d) for d in data["vali_hours"]]
            old_test_hours  = [datetime.fromisoformat(d) for d in data["test_hours"]]
        except:
            print("Loading failed. Creating new data split...")

        # Apply inclusive boundaries to old hours
        def filter_hours(hours):
            filtered = []
            for h in hours:
                if inclusive_left is not None and h < inclusive_left:
                    continue
                if inclusive_right is not None and h > inclusive_right:
                    continue
                filtered.append(h)
            return filtered

        old_train_hours = filter_hours(old_train_hours)
        old_vali_hours  = filter_hours(old_vali_hours)
        old_test_hours  = filter_hours(old_test_hours)

        if not renew:
            return old_train_hours, old_vali_hours, old_test_hours

        # Collect all hourly datetimes using iter_hourly_paths
        all_hours = []
        for year, month, day, hour, _ in self.iter_hourly_paths():
            dt = datetime(year, month, day, hour)
            if inclusive_left is not None and dt < inclusive_left:
                continue
            if inclusive_right is not None and dt > inclusive_right:
                continue
            all_hours.append(dt)

        # Inherit old hours that still exist
        inherited_train_hours = [h for h in old_train_hours if h in all_hours]
        inherited_vali_hours  = [h for h in old_vali_hours  if h in all_hours]
        inherited_test_hours  = [h for h in old_test_hours  if h in all_hours]

        inherited_set = set(inherited_train_hours + inherited_vali_hours + inherited_test_hours)
        new_hours = [h for h in all_hours if h not in inherited_set]

        # Shuffle and split new hours
        if seed is not None:
            random.seed(seed)
        shuffled_hours = new_hours[:]
        random.shuffle(shuffled_hours)

        n = len(shuffled_hours)
        train_size = int(n * self.pm.train_pct / 100)
        vali_size  = int(n * self.pm.vali_pct / 100)

        new_train_hours = shuffled_hours[:train_size]
        new_vali_hours  = shuffled_hours[train_size:train_size + vali_size]
        new_test_hours  = shuffled_hours[train_size + vali_size:]

        # Combine inherited + new
        train_hours = list(set(inherited_train_hours + new_train_hours))[:self.data_limit]
        vali_hours  = list(set(inherited_vali_hours  + new_vali_hours))[:self.data_limit]
        test_hours  = list(set(inherited_test_hours  + new_test_hours))[:self.data_limit]

        random.shuffle(vali_hours)
        random.shuffle(test_hours)

        # Save the split as ISO format datetimes
        if save:
            data = {
                "train_hours": [h.isoformat() for h in train_hours],
                "vali_hours":  [h.isoformat() for h in vali_hours],
                "test_hours":  [h.isoformat() for h in test_hours],
            }
            with open(self.split_hours_path, "w") as f:
                json.dump(data, f, indent=2)

        return train_hours, vali_hours, test_hours


    def split_data_hours_by_date_segents(self, train_start_dt, valid_start_dt, test_start_dt, test_end_dt_inclusive, renew=False, save=True, seed=None):
        """
        Returns:
            train_hours, vali_hours, test_hours : list of datetime objects
        """
        old_train_hours = []
        old_vali_hours = []
        old_test_hours = []

        # Load previous split if exists
        try:
            with open(self.split_hours_path, "r") as f:
                data = json.load(f)

            old_train_hours = [datetime.fromisoformat(d) for d in data["train_hours"]]
            old_vali_hours  = [datetime.fromisoformat(d) for d in data["vali_hours"]]
            old_test_hours  = [datetime.fromisoformat(d) for d in data["test_hours"]]
        except:
            print("Loading failed. Creating new data split...")

        # Apply inclusive boundaries to old hours
        def filter_hours(hours, inclusive_left, exclusive_right):
            filtered = []
            for h in hours:
                if inclusive_left is not None and h < inclusive_left:
                    continue
                if exclusive_right is not None and h >= exclusive_right:
                    continue
                filtered.append(h)
            return filtered

        old_train_hours = filter_hours(old_train_hours, train_start_dt, valid_start_dt)
        old_vali_hours  = filter_hours(old_vali_hours, valid_start_dt, test_start_dt)
        old_test_hours  = filter_hours(old_test_hours, test_start_dt, test_end_dt_inclusive + timedelta(hours=1))

        if not renew:
            return old_train_hours, old_vali_hours, old_test_hours

        # Collect all hourly datetimes using iter_hourly_paths
        
        def get_all_hours(inclusive_left, exclusive_right):
            all_hours = []
            for year, month, day, hour, _ in self.iter_hourly_paths():
                dt = datetime(year, month, day, hour)
                if inclusive_left is not None and dt < inclusive_left:
                    continue
                if exclusive_right is not None and dt >= exclusive_right:
                    continue
                all_hours.append(dt)
            return all_hours

        # Inherit old hours that still exist
        train_hours_all =  get_all_hours(train_start_dt, valid_start_dt)
        vali_hours_all = get_all_hours(valid_start_dt, test_start_dt)
        test_hours_all = get_all_hours(test_start_dt, test_end_dt_inclusive + timedelta(hours=1))

        random.shuffle(train_hours_all)
        random.shuffle(vali_hours_all)
        random.shuffle(test_hours_all)

        # Save the split as ISO format datetimes
        if save:
            data = {
                "train_hours": [h.isoformat() for h in train_hours_all],
                "vali_hours":  [h.isoformat() for h in vali_hours_all],
                "test_hours":  [h.isoformat() for h in test_hours_all],
            }
            with open(self.split_hours_path, "w") as f:
                json.dump(data, f, indent=2)

        return train_hours_all, vali_hours_all, test_hours_all


    def generate_x_hours(self, seed=None):
        if self.train_hours is None or self.vali_hours is None or self.test_hours is None:
            print("Call split_data_hours_by_date_segents first")
            return None, None, None

        # Make copies so the original splits remain unchanged
        train_hours = self.train_hours[:]
        vali_hours  = self.vali_hours[:]
        test_hours  = self.test_hours[:]

        # Shuffle using a local RNG for reproducibility
        rng = random.Random(seed)
        rng.shuffle(train_hours)
        rng.shuffle(vali_hours)
        rng.shuffle(test_hours)

        return train_hours, vali_hours, test_hours
    
    def get_steps_per_epoch(self):
        return math.ceil(len(self.train_hours) / self.pm.batch_size)

    def set_data_slots(self, data_slots):
        self.data_slots = data_slots

    def refresh_x_set(self, x_label, seed=None, renew_if_new_epoch=False):
        renewed = False
        if self.x_set is None or len(self.x_set) <= 0:  # this is a branc-new epoch
            # No need to copy.copy(), as self.x_sets are re-generated above.
            if renew_if_new_epoch is True:
                self.train_hours, self.vali_hours, self.test_hours = self.generate_x_hours(seed=seed)

                if platform.system() == "Windows": 
                    self.x_set = self.train_hours[:] if x_label=='train_set' \
                        else self.vali_hours[:] if x_label=='vali_set' \
                        else self.test_hours[:] if x_label=='test_set' else None
                else:
                    self.x_set = self.train_hours if x_label=='train_set' \
                        else self.vali_hours if x_label=='vali_set' \
                        else self.test_hours if x_label=='test_set' else None            
                
                random.shuffle(self.x_set)
                renewed = True
        
        self._fetched_batches = 0
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)
        
        return renewed
        
    def pack_state(self):
        """Pack the current x_set state for checkpointing."""
        # We assume the take_batch thread has already taken some frrom self.x_set that that's not processed yet.
        if self.taken is not None:
            x_set = self.taken + self.x_set      # return back to self.x_set   
        else:
            x_set = self.x_set
        return {
            "x_set": [dt.isoformat() for dt in x_set],  # full datetime
            "_fetched_batches": self._fetched_batches,
            "total_batches": self.total_batches,
        }

    def unpack_state(self, data):
        """Restore x_set state from checkpoint data."""    
        self.x_set = [datetime.fromisoformat(s) for s in data["x_set"]][:self.data_limit]
        self._fetched_batches = 0
        # Recompute total_batches in case batch_size changed
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)
        
    def no_more_data(self):
        return len(self.x_set) <= 0
    
    def resize_cylindrical(self, x, size_hw):
        """
        Resize cylindrical images along width with circular interpolation.

        Args:
            x: Tensor of shape (..., frames, C, H, W)
            size_hw: tuple (H1, W1)
        Returns:
            Tensor of shape (..., frames, C, H1, W1)
        """
        *leading, Fm, C, H, W = x.shape
        outs = []

        # Flatten leading dims for convenience
        x_flat = x.reshape(-1, Fm, C, H, W)
        n_leading = x_flat.shape[0]

        for f in range(Fm):
            frame = x_flat[:, f]  # (N, C, H, W)

            # circular padding along width
            frame = F.pad(frame, (1, 1, 0, 0), mode="circular")  # (N, C, H, W+2)

            # bicubic interpolation inside each C channel
            frame_resized = F.interpolate(
                frame, size=(size_hw[0], size_hw[1] + 2),
                mode="bicubic", align_corners=False
            )

            # remove circular padding
            frame_resized = frame_resized[:, :, :, 1:-1]  # (N, C, H1, W1)
            outs.append(frame_resized)

        # stack frames back
        output = torch.stack(outs, dim=1)  # (N, Fm, C, H1, W1)

        # reshape back to original leading dims
        output = output.reshape(*leading, Fm, C, size_hw[0], size_hw[1])

        assert output.shape[-5:] == torch.Size([Fm, C, size_hw[0], size_hw[1]])
        return output

    def load_meteo_create_snapshot(self, dt, modulo, with_normalized, idx):
        array = self.load_decode_get_array_for_variables(dt, modulo, self.pm.meteo_variables)
        assert array is not None
        
        if array is not None:
                    
            # if self.unnormal_meteo_buffers[idx] is not None:
            #     cpu_tensor = torch.from_numpy(array).float().pin_memory()
            #     self.unnormal_meteo_buffers[idx].copy_(cpu_tensor, non_blocking=True)
            # else:
            #     self.unnormal_meteo_buffers[idx] = torch.from_numpy(array).float().pin_memory()

            # if with_normalized:
            #     if self.normal_meteo_buffers[idx] is not None:
            #         # In-place normalization, reusing existing buffer
            #         self.normalize_meteo(self.unnormal_meteo_buffers[idx], self.normal_meteo_buffers[idx])
            #     else:
            #         # Allocate once, then reuse for future calls
            #         self.normal_meteo_buffers[idx] = self.unnormal_meteo_buffers[idx].clone().pin_memory()
            #         self.normalize_meteo(self.unnormal_meteo_buffers[idx], self.normal_meteo_buffers[idx])
                    
            # meteo_unnormal = self.unnormal_meteo_buffers[idx]
            # meteo = self.normal_meteo_buffers[idx]
            
            meteo_unnormal = torch.from_numpy(array).float().pin_memory()
            meteo = meteo_unnormal.clone()  # Make meteo the same object as meteo_unnormal.
            self.normalize_meteo(meteo_unnormal, meteo) # Faster if the two arguments are the same objects.
                                                    
            ss = Snapshot(dt.year, dt.month, dt.day, dt.hour, meteo=meteo, meteo_unnormal=meteo_unnormal)

            self.normalize_tempo(ss.tempo, ss.tempo)
                                        
        return ss


    def take_batch(self):        
        if len(self.x_set) <= 0: return None

        taken = self.x_set[:self.pm.batch_size]
        self.taken = taken
        self.x_set = self.x_set[self.pm.batch_size:]
        # assert len(taken) > 0

        batch = []

        months = []
        for b in range(len(taken)):
            dt = taken[b]
            months.append(dt.month)
            
            with_normalized = True
            futures = [
                self._disk_threadPool.submit(
                        self.load_meteo_create_snapshot, dt + timedelta(hours=self.pm.modulo*idx), self.pm.modulo, with_normalized, idx
                    ) for idx in range(self.pm.nRawStepsPerBatch + (2 if self.pm.pair_not_single_snapshot else 1))
                ]

            snapshots = []
            for fu in futures:
                ss = fu.result()
                snapshots.append(ss)
            
            batch.append(snapshots)

        # S, B, _, C, H, W = slots.shape
        # if self.pm.height != H or self.pm.width != W:
        #     slots = self.resizer.resize_cylindrical(slots, (self.pm.height, self.pm.width))
                    
        # self.visualize_frames(batch_x[0])

        if self.pm.include_difficulty:
            # difficulties[variable][month] : data (721, 1440)
            batch_diff = []
            for b in range(B):
                variable_diff = []
                for variable in self.pm.meteo_variables:
                    # self.difficulties[vaiable][months[b]].shape = (720, 1440)
                    diff = self.difficulties[variable][months[b]]   # (721, 1440)
                    variable_diff.append(torch.tensor(diff, dtype=self.pm.dtype))

                variable_diff = torch.stack(variable_diff, dim=0)           # (C, 720, 1440)
                batch_diff.append(variable_diff)
            batch_diff = torch.stack(batch_diff, dim=0)                     # (B, C, 720, 1440)

            # if self.pm.height != H or self.pm.width != W:
            #     batch_diff = self.resizer.resize_cylindrical(batch_diff, (self.pm.height, self.pm.width))
            
        else:
            batch_diff = None
        
        return batch, batch_diff, self.zeus_loss_weights_on_target_vars     # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
    
    def check_accumulation_periods(self):
        """Check accumulation periods for flux variables."""
        if self.total_ds is None:
            self.total_ds = self.get_normalized_dataset()
        
        flux_vars = ['tp', 'ssr', 'ssrd', 'ttr', 'fdir']
        
        for var_name in flux_vars:
            era_var = var_name_map.get(var_name, var_name)
            if era_var in self.total_ds:
                data_var = self.total_ds[era_var]
                print(f"\n{var_name} ({era_var}):")
                print(f"  Shape: {data_var.shape}")
                print(f"  Attributes: {data_var.attrs}")
                
                # Check for accumulation info in attributes
                if 'accumulated' in data_var.attrs:
                    print(f"  Accumulation period: {data_var.attrs['accumulated']}")
                elif 'interval' in data_var.attrs:
                    print(f"  Interval: {data_var.attrs['interval']}")
    
    
    def visualize_frames(self, x_seq):
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
        
    #---------------------------------- Prefetch -------------------------------------
    
    def start_prefetch_thread(self):
        # self.total_batches = None
        # self._fetched_batches = None

        self._lock = threading.Lock()
        self._batch = None
        self._param = None

        self._request_event = threading.Event()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._worker_thread.start()
        
        self._disk_threadPool = ThreadPoolExecutor(max_workers=4)
        
        return

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            # print('prefetch loop... ')
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                param = self._param

            # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
            # batch_x, batch_y, batch_diff, zeus_loss_weights_on_target_vars
            batch = self.take_batch()
            gc.collect()

            with self._lock:
                self._batch = batch
                self._ready_event.set()
            # print(" ...prefetch loop", batch is not None)

    def __prefetch__(self, param):
        with self._lock:
            self._param = param
        self._ready_event.clear()
        self._request_event.set()

    def get_batch(self):    # If __prefetch__ doesn't procedes, it returns all the same batch.
        self._ready_event.wait()
        with self._lock:
            return self._batch

    def __getitem__(self, idx):
        if idx >= self.total_batches:
            raise IndexError("Index exceeds dataset length.")

        batch = self.get_batch()
        if batch is not None:
            self._fetched_batches += 1
        return batch

    def __len__(self):
        return self.total_batches

    def shutdown(self):
        # print('shutdown... ')
        self._stop_event.set()
        self._request_event.set()
        self._worker_thread.join()
        
        self._disk_threadPool.shutdown(wait=True)
        # print(' ...shutdown')
        
def copy_day_foler(day_path, target_path):                
    os.makedirs(target_path, exist_ok=True)
    # Copy all files from day folder
    for filename in os.listdir(day_path):
        src_file = os.path.join(day_path, filename)
        dst_file = os.path.join(target_path, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
    return day_path, target_path

def divide_cds_by_day(src_root, drive_even, drive_odd):
    """
    Divide CDS content between two drives by day number from origin_date.
    Supports multiple year folders.
    
    Parameters:
    - src_root: path to 'CDS' folder
    - drive_even: path to drive for even days
    - drive_odd: path to drive for odd days
    - origin_date: datetime object for day 0 (default 1970-01-01)
    """
    origin_date=datetime(1970,1,1)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for year_folder in sorted(os.listdir(src_root)):
            year_path = os.path.join(src_root, year_folder)
            if not os.path.isdir(year_path) or not year_folder.isdigit():
                continue
            year = int(year_folder)
            
            for month_folder in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month_folder)
                if not os.path.isdir(month_path) or not month_folder.isdigit():
                    continue
                month = int(month_folder)
                
                for day_folder in sorted(os.listdir(month_path)):
                    day_path = os.path.join(month_path, day_folder)
                    if not os.path.isdir(day_path) or not day_folder.isdigit():
                        continue
                    day = int(day_folder)
                    
                    # Compute the day number from origin_date
                    current_date = datetime(year, month, day)
                    day_number = (current_date - origin_date).days
                    
                    # Decide target drive
                    target_drive = drive_even if day_number % 2 == 0 else drive_odd
                    
                    # Create corresponding folder structure in target
                    target_path = os.path.join(target_drive, year_folder, month_folder, day_folder)
                    
                    futures.append(
                        executor.submit(copy_day_foler, day_path, target_path)
                    )
                    
        for future in as_completed(futures):
            try:
                day_path, target_path = future.result()
                print(f"Copied {day_path} → {target_path}")
            except Exception as e:
                print(f"Error processing a day: {e}")
                
def copy_except_cds(src, dst):
    """
    Copy all contents of src to dst, excluding any folder named 'CDS'.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source folder not found: {src}")
    os.makedirs(dst, exist_ok=True)

    for root, dirs, files in os.walk(src):
        # Compute relative path from src
        rel_path = os.path.relpath(root, src)

        # Skip the CDS folder and anything inside it
        if 'CDS' in rel_path.split(os.sep):
            continue

        # Create corresponding folder in destination
        dst_dir = os.path.join(dst, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)

    print(f"All contents (except 'CDS') copied from {src} → {dst}")
    
def list_cds_variables(dataset):
    """Return all variable names for a CDS ERA5 dataset."""
    url = f"https://cds.climate.copernicus.eu/api/v2/resources/{dataset}"
    r = requests.get(url)
    r.raise_for_status()
    info = r.json()
    
    # The "variable" field contains all available parameters
    variables = info["parameters"]["variable"]["options"]
    return variables

from modules.utils import check_dir
import platform 
def setup_logging():
    """
    logging.info("This is an info message")
    logging.warning("Watch out! This is a warning")
    logging.error("An error happened")
    logging.debug("Debug details here (won't show unless level is DEBUG)")
    """
    # Remove all handlers associated with the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if platform.system() == "Windows":
        log_path = Path("d:/downloads/dataset_log.tex")
    else:
        log_path = Path("/home/mike/zeus_manageent/dataset_log.txt")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler (append mode)
    check_dir(log_path.parent)
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    # src = "/mnt/data/zeus_data/CDS"           # source CDS folder
    # drive_even = "/mnt/data_even/CDS"  # target drive for even days
    # drive_odd = "/mnt/data_odd/CDS"   # target drive for odd days
    # divide_cds_by_day(src, drive_even, drive_odd)
    
    # src = "/mnt/data/zeus_data"
    # dst = "/mnt/data_even"
    # copy_except_cds(src, dst)

    from models.config_Mike1 import config
    from modules.params_Mike1 import Params
    from modules.zeus_dataset_var_names import VARIABLE_NAME_LIST_UNCHANGEABLE, var_name_map
    pm = Params(config)

    dataset = Zeus_Dataset(pm, Path(config['dataset_path']), Path(config['management_path']), download=False)
    
    # dataset.check_accumulation_periods()
    
    # variables = ["geopotential_at_surface", "land_sea_mask", "soil_type"]
    # dict = dataset.get_geo_data(Path(config['management_path']), variables)

    # ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
    # # List variables
    # print(ds.data_vars)
    # pass

    # arr = np.load(config["soil_type_npy_path"])
    # dataset.show_array_single(arr, "Soil Type")


    # inclusive_left = datetime(2025, 1, 1, 0)
    # inclusive_right = datetime(2025, 1, 2, 23)

    inclusive_left = datetime(2018, 1, 1, 0, tzinfo=timezone.utc)
    inclusive_right = datetime(2025, 12, 31, 23, tzinfo=timezone.utc)

    # dataset.play_weather_snapshots(inclusive_left, inclusive_right, modulo=6)

    dataset.play_physical_prediction(inclusive_left, inclusive_right, modulo=6)

    # dataset.download_meteo_variables(
    #     variables=VARIABLE_NAME_LIST_UNCHANGEABLE, var_name_map = var_name_map,
    #     # ----- 8 years, 1-hourly, 720 GB (700.8 GB ------
    #     start_dt = inclusive_left,
    #     end_dt = inclusive_right,
    #     modulo=1,
    #     max_workers_snapshots=6, max_workers_vars=6,
    #     simulated=True
    # )

    # mean_meteo, m = dataset.get_mean_meteo_downloading_if_not_ready(
    #     inclusive_left=inclusive_left,
    #     inclusive_right=inclusive_right,
    #     modulo = 1,
    # )

    inclusive_left = datetime(2018, 1, 1, 0)
    inclusive_right = datetime(2025, 12, 31, 23)

    # dict = dataset.analyze_hourly_paths(inclusive_left=inclusive_left, inclusive_right=inclusive_right, modulo=1)
    # print(f"count: {dict['count']}, n_unexpected_hours: {dict['n_unexpected_hours']}, first_hour: {dict['first_hour']}, last_hour: {dict['last_hour']}, n_missing_hours: {len(dict['missing_hours'])}")

    # dataset.analyze_hourly_dataset(save=True, inclusive_left=inclusive_left, inclusive_right=inclusive_right, modulo=6, file_name_suffix="2008_2025_m6")
    # dataset.analyze_hourly_dataset(save=True, inclusive_left=inclusive_left, inclusive_right=inclusive_right, modulo=1, file_name_suffix="2008_2025_m1")

    # mean_meteo, std_meteo, mean_tempo, std_tempo, \
    # mean_meteo_ready_cpu, std_meteo_ready_cpu, mean_meteo_ready_cuda, std_meteo_ready_cuda, \
    # mean_tempo_ready_cpu, std_tempo_ready_cpu, mean_tempo_ready_cuda, std_tempo_ready_cuda, \
    # area_weights_cuda \
    # = dataset.load_dataset_params(file_name_suffix="208_2025_m6")

    # print("-------- mean_meteo: ", mean_meteo.shape, mean_meteo[-3:])
    # print("-------- std_meteo: ", std_meteo.shape, std_meteo[-3:])
    # print("-------- mean_tempo: ", mean_tempo.shape, mean_tempo[-3:])
    # print("-------- std_tempo: ", std_tempo.shape, std_tempo[-3:])

    # train_hours, vali_hours, test_hours = dataset.split_data_hours_by_date_segents(
    #     datetime(2015, 1, 1, 10), datetime(2015, 1, 10, 10), datetime(2015, 1, 20, 10), datetime(2015, 1, 30, 10),
    #     renew=True, save=False, seed=523
    #     )

    # print("Lengths:", len(train_hours), len(vali_hours), len(test_hours))
    # print("train_hours[:3]", train_hours[:3])

    # train_set, vali_set, test_set = dataset.generate_x_sets(seed=523)
    # print("train_set, vali_set, test_set lengths:", len(train_set), len(vali_set), len(test_set))
    # print("train_set[:3]", train_set[:3])

    # dataset.create_difficulties(end_day)
    # dataset.load_difficulties()
    # dataset.show_difficulty("2m_temperature")
  
    # vars = dataset.load_era5_data_vars()
    # print('vars ', vars)

    # dataset.refresh_x_set('train_set', seed=523)

    # batch = dataset.take_batch()
    # print("batch: ", batch)

    # cds_loader = Era5CDSLoader()
    # sample = cds_loader.get_sample()        # get random sample
    # difficult_grid = dataset.get_difficulty_grid("100m_u_component_of_wind", sample)
    # print(difficult_grid)

    # dataset.integrate_variable_data()

    # dataset.decompress_day_all_data()

