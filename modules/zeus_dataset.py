import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import calendar
import json
import gc
import math
import platform
import heapq
from datetime import datetime, date, timedelta
import random
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
from pathlib import Path
from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.params import Params
from modules.utils import check_dir
# Make they're from sure modules_zeus
from modules_zeus.constants import ERA5_DATA_VARS
from modules_zeus.sample import Era5Sample
from modules_zeus.coordinates import get_bbox, slice_bbox
from modules_zeus.era5_cds import Era5CDSLoader, get_today

from modules.zeus_dataset_codec import encode_variables, save_variables, load_decoded_variables

class Zeus_Dataset(Dataset):
    def __init__(self, pm, dataset_path, management_path):
        super().__init__()
        self.pm = pm
        
        check_dir(dataset_path)
        check_dir(management_path)

        if not self.pm.dual_drive:
            self.cds_folder = dataset_path / "CDS"
            self.cds_folder.mkdir(parents=True, exist_ok=True)

            self.param_path = management_path / 'dataset_params.json'
            self.split_hours_path = management_path / 'split_hours.json'
            self.analysis_path = management_path / "analysis.pth"
        else:
            self.cds_folder_even = dataset_path / "data_even" / "CDS"            
            self.cds_folder_even.mkdir(parents=True, exist_ok=True)
            self.cds_folder_odd = dataset_path / "data_odd" / "CDS"
            self.cds_folder_odd.mkdir(parents=True, exist_ok=True)
    
            self.param_path = management_path /'dataset_params.json'
            self.split_hours_path = management_path / 'split_hours.json'
            self.analysis_path = management_path / "analysis.pth"
        
        self.difficulties = None
        self.era5_data_vars = None

        self.loader = Era5CDSLoader()
        self.train_dates, self.vali_dates, self.test_dates = None, None, None
        self.train_set, self.vali_set, self.test_set = None, None, None
        self.x_set, self.x_set_len = None, 0

        self._fetched_batches = 0
        self.total_batches = 0
        
        self.origin_date = datetime(1970,1,1)

    #------------------------------------- Download dataset -----------------------

    def get_day_number(self, year, month, day):
        return (datetime(year, month, day) - datetime(1970,1,1)).days

    def load_data(self):
        self.load_dataset_params()
        if self.pm.include_difficulty: self.load_difficulties()
        self.load_era5_data_vars()
        self.split_data_hours(renew=True, seed=self.pm.seed)
        return
    
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
        
        self._disk_threadPool = ThreadPoolExecutor(max_workers=2)
        
        return

    def get_normalized_dataset(self):
        dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
        # dataset = dataset[self.pm.variables]
        # start_string = self.day_string(self.start_day)
        # end_string = self.day_string(self.end_day)
        # dataset = dataset.sel(time=slice(start_string, end_string))

        # convert latitude and longitude to -90 to 90 and -180 to 180 respectively
        if dataset["longitude"].max() > 180:
            dataset = dataset.assign_coords(longitude=(dataset["longitude"].values + 180) % 360 - 180)
        if dataset["latitude"].max() > 90:
            dataset = dataset.assign_coords(latitude=dataset["latitude"].values - 90)

        dataset = dataset.sortby(["latitude", "longitude"])
        dataset = dataset.chunk()

        return dataset

    def get_file_path(self, year, month, day, variable):
        cds_folder = self.cds_folder if not self.pm.dual_drive else self.cds_folder_even if self.get_day_number(year, month, day) % 2 == 0 else self.cds_folder_odd        
        return cds_folder / str(year) / str(month).zfill(2) / str(day).zfill(2) / (variable + ".npz")
    def get_file_path_npy(self, year, month, day, variable):
        cds_folder = self.cds_folder if not self.pm.dual_drive else self.cds_folder_even if self.get_day_number(year, month, day) % 2 == 0 else self.cds_folder_odd        
        return cds_folder / str(year) / str(month).zfill(2) / str(day).zfill(2) / (variable + ".npy")
    def get_last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]

    def lownload_day_data_old(self, total_ds, year, month, day):
        files_to_remove = []
        file_path_npy = self.get_file_path_npy(year, month, day, 'all')
        file_path_npz = self.get_file_path(year, month, day, 'all')
        
        # Convert existing npz files to npy files.
        try:
            data = np.load(file_path_npz)   # ok
            data = load[load.files[0]]
            np.save(file_path_npy, data)
            if file_path_npz.exists(): files_to_remove.append(file_path_npz)
        except:
            if file_path_npz.exists(): files_to_remove.append(file_path_npz)

            data_list = []
            for variable in self.pm.variables:
                print("{}/{}/{} {}...".format(year, str(month).zfill(2), str(day).zfill(2), variable))

                try:
                    file_path_npz = self.get_file_path(year, month, day, variable)
                    load = np.load(file_path_npz)
                    array = load[load.files[0]]
                    files_to_remove.append(file_path_npz)
                except:
                    ds = total_ds[[variable]]
                    dayStr = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
                    ds = ds.sel(time=slice(dayStr, dayStr))               # day data
                    array = ds[variable].values     # Actual downloading happens.
                    array = np.clip(array, -65504, 65504).astype(np.float16)
                    has_nan = np.isnan(array).any()
                    assert not has_nan                        
                data_list.append(array)

            data = np.stack(data_list, axis=1)      # (24, C, 721, 1440)
            assert data.shape[1] == len(self.pm.variables)

            file_path_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_path_npy, data)

        return files_to_remove
    
    
    def download_day_data(self, total_ds, year, month, day):
        files_to_remove = []
        file_path_npy = self.get_file_path_npy(year, month, day, 'all')
        file_path_npz = self.get_file_path(year, month, day, 'all')
        
        exists = False
        
        try:
            data = None
            data = np.load(file_path_npy) # npY
            # We have no way but trust the data integrity, as there is no vairable name in data.
            
            if file_path_npy.exists(): files_to_remove.append(file_path_npy)
        except:
            try:
                load = load_decoded_variables(file_path_npz)    # npZ
                idx = 0; values = []
                for key, value in load.items():
                    assert key == self.pm.variables[idx]; idx += 1
                    values.append(value)
                data = np.stack(values, axis=1)  # expected: (24, C, 721, 1440), read 
                if data.shape == (24, len(self.pm.variables), 721, 1440): exists = True
            except: pass    # data == None
        
        if not exists:
        
            var_dict = {}
            for idx, variable in enumerate(self.pm.variables):
                print("{}/{}/{} {}...".format(year, str(month).zfill(2), str(day).zfill(2), variable))

                try:
                    array = data[:, idx, :, :]
                    assert array.shape == (24, 721, 1440)
                except:
                    ds = total_ds[[variable]]
                    dayStr = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
                    ds = ds.sel(time=slice(dayStr, dayStr))               # day data
                    array = ds[variable].values     # Actual downloading happens.
                    assert array.shape == (24, 721, 1440)
                    has_nan = np.isnan(array).any()
                    assert not has_nan

                var_dict.update({variable: array})

            encoded, meta = encode_variables(var_dict)
            file_path_npz.parent.mkdir(parents=True, exist_ok=True)
            save_variables(file_path_npz, encoded, meta)        # npz only. npy doesn't support meta data and takes more space. Speed is no problem for us.

        return files_to_remove

    def download_dataset_old(self, start_year, end_date):
        total_ds = self.get_normalized_dataset()

        print(list(total_ds.data_vars))
        print(total_ds["latitude"].min(), total_ds["latitude"].max(), total_ds["longitude"].min(), total_ds["longitude"].max())
        
        year_list = list(range(end_date.year, start_year-1, -1))
        month_list = list(range(12, 0, -1))

        for year in year_list:
            for month in month_list:
                for day in range(self.get_last_day_of_month(year, month), 0, -1):
                    if date(year, month, day) > end_date: continue
                    print(year, month, day)
                    files_to_remove = self.download_day_data(total_ds, year, month, day)
                    for f in files_to_remove: os.remove(f)
        return
    
    def process_day(self, total_ds, year, month, day):
        """Helper to download and clean up one day."""
        files_to_remove = self.download_day_data(total_ds, year, month, day)
        for f in files_to_remove:
            os.remove(f)
        return (year, month, day)  # optional return for logging    
    
    def download_dataset(self, start_year, end_date, max_workers=4):
        """Run self.download_day_data in parallel for multiple days."""
        total_ds = self.get_normalized_dataset()

        print(list(total_ds.data_vars))
        print(total_ds["latitude"].min(), total_ds["latitude"].max(), total_ds["longitude"].min(), total_ds["longitude"].max())
        
        year_list = list(range(end_date.year, start_year-1, -1))
        month_list = list(range(12, 0, -1))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for year in year_list:
                for month in month_list:
                    for day in range(self.get_last_day_of_month(year, month), 0, -1):
                        if date(year, month, day) > end_date:
                            continue

                        # Submit each day's work to thread pool
                        futures.append(
                            executor.submit(self.process_day, total_ds, year, month, day)
                        )

            # Optionally: wait for all to finish and handle results
            for future in as_completed(futures):
                try:
                    y, m, d = future.result()
                    print(f"Finished {y}-{m:02d}-{d:02d}")
                except Exception as e:
                    print(f"Error processing a day: {e}")    
        
    def next_month(self, dt):
        return dt + relativedelta(month=1)
    def last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]
    def month_string(self, date):
        return str(date.year) + '-' + str(date.month).zfill(2)
    def day_string(self, date):
        return str(date.year) + '-' + str(date.month).zfill(2) + '-' + str(date.day).zfill(2)

    #---------------------------- Analyse dataset ----------------------------

    def sort_files(self, pathlib_dir, stem_order):
        # Your predefined order
        desired_stem_order = self.pm.variables

        # Convert to a dict for fast lookup
        stem_rank = {stem: i for i, stem in enumerate(desired_stem_order)}

        # Get and sort file paths
        files = list(pathlib_dir.glob("*.npz"))
        sorted_files = sorted(files, key=lambda f: stem_rank.get(f.stem, float('inf')))
        return sorted_files

    def iter_day_folders(self):
        """
        Yield (year, month, day, Path) for all day folders under the correct cds folder(s),
        ordered by date, lazily and memory-efficiently.
        """
        if not self.pm.dual_drive:
            base_folders = [Path(self.cds_folder)]
        else:
            base_folders = [Path(self.cds_folder_even), Path(self.cds_folder_odd)]

        def day_iter(base_folder):
            """Yield (year, month, day, Path) for one base folder, sorted by date."""
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

                        dn = self.get_day_number(year, month, day)

                        if self.pm.dual_drive:
                            if base_folder == Path(self.cds_folder_even) and dn % 2 != 0:
                                continue
                            if base_folder == Path(self.cds_folder_odd) and dn % 2 == 0:
                                continue

                        yield (year, month, day, day_path)

        # Create iterators for all base folders
        iterators = [day_iter(base) for base in base_folders if base.is_dir()]

        # Merge all iterators in order by (year, month, day)
        # key=lambda x: (year, month, day)
        for item in heapq.merge(*iterators, key=lambda x: (x[0], x[1], x[2])):
            yield item

    def count_data_days(self, inclusive_left=None, inclusive_right=None):
        start, end = None, None
        missing = False
        total_days = 0
        
        for year, month, day, day_dir in self.iter_day_folders():   # assume the return is chronologically increasingly ordered.
            current = date(year, month, day)
            if inclusive_left is not None and current < inclusive_left: continue
            if inclusive_right is not None and current > inclusive_right: continue
                        
            files = {f.stem for f in day_dir.glob("*.npz")}
            missing_vars = [var for var in ['all'] if var not in files]
            if len(missing_vars) > 0:
                print('missing...', day_dir)
                missing = True

            total_days += 1
            if start is None:
                start = day_dir
            end = day_dir
            end_day = date(year=int(end.parts[-3]), month=int(end.parts[-2]), day=int(end.parts[-1]))

        complete = not missing
        start_day = date(year=int(start.parts[-3]), month=int(start.parts[-2]), day=int(start.parts[-1]))
        end_day = date(year=int(end.parts[-3]), month=int(end.parts[-2]), day=int(end.parts[-1]))
        
        data_dates = []
        for delta in range(total_days):
            data_dates.append(start_day + timedelta(days=delta))

        assert data_dates[0] == start_day
        assert data_dates[-1] == end_day
        # assert data_dates[-1] == end

        return total_days, start_day, end_day, complete, data_dates
    
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
        
    def get_mean_of_dataset(self, inclusive_left=None, inclusive_right=None):
        cuda_available = torch.cuda.is_available()

        try:
            mean, step = self.read_checkpoint(self.analysis_path, ["mean", "mean_step"])
            print(f"mean: {mean}, step: {step}")
        except:
            print("read_checkpoint failed. Starting a new analysis...")
            mean = np.zeros((len(self.pm.variables),), dtype=np.float32) # (C,)
            step = 0
        
        skip = -1
        for year, month, day, day_dir in self.iter_day_folders():
            current = date(year, month, day)
            if inclusive_left is not None and current < inclusive_left: continue
            if inclusive_right is not None and current > inclusive_right: continue

            skip += 1   # the 0-based id of data item
            if skip < step: continue

            file_path = day_dir / 'all.npz'
            
            values = []
            load = load_decoded_variables(file_path)    # npZ
            idx = 0; values = []
            for key, value in load.items():
                assert key == self.pm.variables[idx]; idx += 1
                values.append(value)
            arr = np.stack(values, axis=1)  # expected: (24, C, 721, 1440), read 
            arr = arr[:, :, :, :]        # (24, C, 721, 1440)    # Do not exclude the north pole

            if cuda_available:
                _mean = torch.tensor(arr, dtype=torch.float32).cuda().mean(dim=(0,2,3)).cpu().numpy()   # (C,)
            else:
                _mean = np.mean(arr, axis=(0, 2, 3))   # (C,)

            has_nan = np.any(np.isnan(_mean))
            if has_nan:
                raise Exception("mean nan found: ", str(day_dir))
            formatter = {'float_kind': lambda x: f"{x:.4g}"}
            print("mean", str(day_dir), np.array2string(_mean, formatter=formatter))
            step += 1
            mean = mean * (step-1)/step + _mean / step   # (C,) # mean of mean = total mean 

            if step % 200 == 0:
                self.save_checkpoint(self.analysis_path, {"mean": mean, "mean_step": step})
                # print(f"saved. step: {step}")

            # break
        self.save_checkpoint(self.analysis_path, {"mean": mean, "mean_step": step})
        return mean, step  # numpy, scalar
    
    def get_std_of_dataset(self, mean, inclusive_left=None, inclusive_right=None): 
        cuda_available = torch.cuda.is_available()
        if cuda_available: mean = torch.tensor(mean).cuda()

        try:
            var, step = self.read_checkpoint(self.analysis_path, ["var", "var_step"])
            print(f"var: {var}, step: {step}")
        except:
            print("read_checkpoint failed. Starting a new analysis...")
            var = np.zeros((len(self.pm.variables),), dtype=np.float32) # (C,)
            step = 0
        
        skip = -1
        for year, month, day, day_dir in self.iter_day_folders():
            current = date(year, month, day)
            if inclusive_left is not None and current < inclusive_left: continue
            if inclusive_right is not None and current > inclusive_right: continue
                                                   
            skip += 1   # the 0-based id of data item
            if skip < step: continue
                                                                        
            file_path = day_dir / 'all.npz'

            values = []
            load = load_decoded_variables(file_path)    # npZ
            idx = 0; values = []
            for key, value in load.items():
                assert key == self.pm.variables[idx]; idx += 1
                values.append(value)
            arr = np.stack(values, axis=1)  # expected: (24, C, 721, 1440), read 
            arr = arr[:, :, :, :]        # (24, C, 721, 1440)    # Do not exclude the north pole

            arr = arr.transpose(0, 2, 3, 1) # (24, 721, 1440, C)                    

            if cuda_available:
                tensor_arr = torch.tensor(arr, dtype=torch.float32).cuda() # (24, 721, 1440, C)
                _var = torch.mean(torch.pow(tensor_arr - mean, 2), dim=(0,1,2)).cpu().numpy() # (C,)                    
            else:
                _var = np.mean(np.power(arr - mean, 2), axis=(0,1,2)) # (C,)

            has_nan = np.any(np.isnan(_var))
            if has_nan:
                raise Exception("Std nan found: ", str(day_dir))
            formatter = {'float_kind': lambda x: f"{x:.4g}"}
            print("var", str(day_dir), np.array2string(_var, formatter=formatter))
            step += 1
            var = var * (step-1)/step + _var / step   # (C,) # mean of mean = total mean

            if step % 200 == 0:
                self.save_checkpoint(self.analysis_path, {"var": var, "var_step": step})
                # print(f"saved. step: {step}")

            # break                
        self.save_checkpoint(self.analysis_path, {"var": var, "var_step": step})
        return np.sqrt(var), step   # numpy, scalar

    def analyze_dataset(self, save=False, inclusive_left=None, inclusive_right=None):

        mean, m = self.get_mean_of_dataset(inclusive_left=inclusive_left, inclusive_right=inclusive_right)        # (C,)
        std, n = self.get_std_of_dataset(mean, inclusive_left=inclusive_left, inclusive_right=inclusive_right)        # (C,)
        assert m == n

        if save:
            data = {
                "mean": mean.tolist(),
                "std":  std.tolist(),
            }
            if self.param_path.exists() and self.param_path.is_file():
                self.param_path.unlink()  # deletes the file
            with open(self.param_path, "w") as f:
                json.dump(data, f)

        return mean, std  # (C,), (C,)
            
    def load_dataset_params(self):
        with open(self.param_path, "r") as f:
            data = json.load(f)
            # print('------', data, torch.tensor(data['mean']).shape)
        mean, std = torch.tensor(data['mean']), torch.tensor(data['std'])
        
        std = std # * (self.pm.factor10 / 10)   # Mike!!! This multiplication make sdt deviate from 1.
        
        self.mean, self.std = mean, std
        
        # This violence may cause inconsistency, I know.
        self.mean_ready_cpu = rearrange(self.mean, 'c -> c 1 1').to("cpu")
        self.std_ready_cpu = rearrange(self.std, 'c -> c 1 1').to("cpu")
        if torch.cuda.is_available():
            self.mean_ready_cuda = rearrange(self.mean, 'c -> c 1 1').to("cuda")
            self.std_ready_cuda = rearrange(self.std, 'c -> c 1 1').to("cuda")
        
        return mean, std    # (C,), (C,)

    def normalize(self, data):    # data: (... C, H, W)
        # device = data.device
        # return ( data - rearrange(self.mean, 'c -> c 1 1').to(device) ) / rearrange(self.std, 'c -> c 1 1').to(device)
        if data.is_cuda:
            return ( data - self.mean_ready_cuda ) / self.std_ready_cuda
        else:
            return ( data - self.mean_ready_cpu ) / self.std_ready_cpu
    
    def unnormalize(self, data):    # data: (... C, H, W)
        # Mike!!! I found this error Oct 23 only. 
        # Over 2 months of test training was based on this wrong formula, forcing neual networks to distort accordingly.
        # return ( data - rearrange(self.mean, 'c -> c 1 1').to(device) ) / rearrange(self.std, 'c -> c 1 1').to(device)

        # device = data.device
        # return data * rearrange(self.std, 'c -> c 1 1').to(device) + rearrange(self.mean, 'c -> c 1 1').to(device)
        if data.is_cuda:
            return data * self.std_ready_cuda + self.mean_ready_cuda
        else:
            return data * self.std_ready_cpu + self.mean_ready_cpu
    
    def load_era5_data_vars(self):
        vars = [ERA5_DATA_VARS[k] for k in self.pm.variables if k in ERA5_DATA_VARS.keys()]
        self.era5_data_vars = torch.tensor(vars, dtype=self.pm.dtype)
        return self.era5_data_vars
   
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
        for variable_idx in range(len(self.pm.variables)):
            # handle variables one by one, as month_data is of huge size.
            print(self.pm.variables[variable_idx])
            difficulties = []
            maxDeffVar = 0.0
            for nth in range(12):
                year, month = self.previous_nth_month(latest_date, nth)
                first_day = date(year, month, 1)
                last_day = first_day.replace(day = self.last_day_of_month(year, month))

                month_data = []
                for day in range(1, self.last_day_of_month(year, month) + 1):
                    file_path = self.get_file_path(year, month, day, 'all')

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
                folder = current_folder / "weights" / self.pm.variables[variable_idx]
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / (str(year) + '-' + str(month).zfill(2) + ".npy")
                np.save(path, rel_var_per_loc)
                
    def load_difficulties(self):

        total_days, start_day, end_day, complete, data_dates = self.count_data_days()

        difficulties = {}

        for variable in self.pm.variables:
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

        self.difficulties = difficulties
        print('load per variable per month', self.difficulties['2m_temperature'][3].shape)


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
        start_timestamp, end_timestamp, num_predict_hours = self.loader.sample_time_range()
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
        start_timestamp, end_timestamp, num_predict_hours = self.loader.sample_time_range()
        current_hour = np.random.randint(0, 24)
        # Make the time-agnostic. Simulate reality, where time drifts.
        start = start_timestamp - get_today('h') + timedelta(hours=current_hour)
        end = end_timestamp - get_today('h') + timedelta(hours=current_hour)
        choice = end
        # t2 = math.ceil((end.to_pytimedelta().total_seconds()/3600) / 24) + 4
        hours = choice.to_pytimedelta().total_seconds()/3600
        t2 = math.ceil(hours / self.pm.FG) + math.ceil(5 * 24 / self.pm.FG)    # starts from 5 days ago

        return t2
    
    def split_data_hours(self, renew=False, save=True, seed=None):
        old_train_hours = []
        old_vali_hours = []
        old_test_hours = []

        try:
            # print("Loading data dates...")
            with open(self.split_hours_path, "r") as f:
                data = json.load(f)

            old_train_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["train_hours"]]
            old_vali_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["vali_hours"]]
            old_test_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["test_hours"]]
        except:
            print("Loading failed. Creating data hours...")
            
        if renew == False:
            self.train_hours = old_train_hours
            self.vali_hours = old_vali_hours
            self.test_hours = old_test_hours
        else:
            total_days, start_day, end_day, complete, data_dates = self.count_data_days()   # assume: data_dates is ordered.
            train_pct = self.pm.train_pct; vali_pct = self.pm.vali_pct

            T1 = self.pm.T1
            data_dates = data_dates[:-1]    # Remove the last day for safety.
            
            # Remove dates that are not found in data_dates
            inherited_train_hours = [(d, h) for (d, h) in old_train_hours if d in data_dates]
            inherited_vali_hours = [(d, h) for (d, h) in old_vali_hours if d in data_dates]
            inherited_test_hours = [(d, h) for (d, h) in old_test_hours if d in data_dates]
            
            inherited_dates = [d for (d, _) in inherited_train_hours] + [ d for (d, _) in inherited_vali_hours] + [d for (d, _) in inherited_test_hours]
            inherited_dates = list(set(inherited_dates))
            new_data_dates = [d for d in data_dates if d not in inherited_dates]
            
            data_hours = []
            for day in new_data_dates:
                # data_hours += [(day, hour) for hour in range(24)]
                data_hours += [(day, hour) for hour in range(1)]    # day, 0 hour

            if seed is not None: random.seed(seed)

            shuffled_hours = data_hours[:]
            random.shuffle(shuffled_hours)
            n = len(shuffled_hours)
            train_size = int(n * train_pct / 100)
            vali_size = int(n * vali_pct / 100)
            new_train_hours = shuffled_hours[:train_size]
            new_vali_hours = shuffled_hours[train_size : train_size + vali_size]
            new_test_hours = shuffled_hours[train_size + vali_size :]
            
            self.train_hours = list(set(inherited_train_hours + new_train_hours))
            self.vali_hours = list(set(inherited_vali_hours + new_vali_hours))
            self.test_hours = list(set(inherited_test_hours + new_test_hours))
            random.shuffle(self.train_hours)
            random.shuffle(self.vali_hours)
            random.shuffle(self.test_hours)
            
            if save:
                data = {
                    "train_hours":  [(d.isoformat(), h) for (d, h) in self.train_hours],
                    "vali_hours":   [(d.isoformat(), h) for (d, h) in self.vali_hours],
                    "test_hours":   [(d.isoformat(), h) for (d, h) in self.test_hours],
                }
                with open(self.split_hours_path, "w") as f:
                    json.dump(data, f, indent=2)
        
        return self.train_hours, self.vali_hours, self.test_hours

    def generate_x_hours(self, seed=None):
        if seed is not None: random.seed(seed)

        if self.train_hours is None or self.vali_hours is None or self.test_hours is None:
            print("Call split_data_hours first")
            return None, None, None

        random.shuffle(self.train_hours)
        random.shuffle(self.vali_hours)
        random.shuffle(self.test_hours)
        
        return self.train_hours, self.vali_hours, self.test_hours
    
    def get_steps_per_epoch(self):
        return math.ceil(len(self.train_hours) / self.pm.batch_size)

    def refresh_x_set(self, x_label, seed=None):
        self.generate_x_hours(seed=seed)

        # Mike!!! remove [:2]
        # No need to copy.copy(), as self.x_sets are re-generated above.
        if platform.system() == "Windows": 
            self.x_set = self.train_hours[:] if x_label=='train_set' \
                else self.vali_hours[:] if x_label=='vali_set' \
                else self.test_hours[:] if x_label=='test_set' else None
        else:
            self.x_set = self.train_hours if x_label=='train_set' \
                else self.vali_hours if x_label=='vali_set' \
                else self.test_hours if x_label=='test_set' else None            
        
        random.shuffle(self.x_set)

        self._fetched_batches = 0
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)
        
    def pack_state(self):
           return {
            "x_set": [(d.strftime("%Y-%m-%d"), t2) for d, t2 in self.x_set],
            "_fetched_batches": self._fetched_batches,
            "total_batches": self.total_batches,
        }

    def unpack_state(self, data):
        self.x_set = [
            (datetime.strptime(item[0], "%Y-%m-%d").date(), item[1])
            for item in data["x_set"]
        ]
        self._fetched_batches = 0
        # self.total_batches = data["total_batches"] - data["_fetched_batches"]
        self.total_batches = math.ceil(len(self.x_set) / self.pm.batch_size)    # This line allows to change batch_size mid apoch

    def no_more_data(self):
        return len(self.x_set) <= 0

    def load_day_vdata(self, day):
        day_data = []
        for variable in self.pm.variables:
            file_path = self.get_file_path(day.year, day.month, day.day, variable)
            if file_path.exists():
                load = np.load(file_path)
                variable_data = torch.tensor(load[load.files[0]], dtype=self.pm.dtype)[:, :720, :]    # (24, H, W)
                day_data.append(variable_data)
        if len(day_data) == len(self.pm.variables):
            day_data = torch.stack(day_data, dim=1)     # (24, C, H, W)
        else:
            day_data = None
        return day_data
    
    def load_day_all_data_outdated(self, day):
        all_path = self.get_file_path(day.year, day.month, day.day, 'all')
        if all_path.exists():
            try:
                load = np.load(all_path)
                data = torch.tensor(load[load.files[0]], dtype=self.pm.dtype)[:, :, :720, :]    # (24, C, H, W)
            except:
                total_ds = self.get_normalized_dataset()
                self.lownload_day_vdata(total_ds, day.year, day.month, day.day)
                data = self.load_day_vdata(day)
            return data
        else:
            return None
        
    def load_day_all_data(self, day):
        all_path = self.get_file_path(day.year, day.month, day.day, 'all')
        try:
            load = load_decoded_variables(all_path)
            idx = 0; values = []
            for key, value in load.items():
                assert key == self.pm.variables[idx]; idx += 1
                values.append(value)
            data = np.stack(values, axis=1)  # expected: (24, C, 721, 1440), read 
            assert data.shape == (24, len(self.pm.variables), 721, 1440)            
            return data
        except:
            raise Exception("load_day_all_data error for {}".format(day))
        
    def load_hours_all_data(self, day, start_hour, hours):
        try:
            assert isinstance(start_hour, int)
            assert isinstance(hours, int)
            assert 0 <= start_hour and start_hour < 24 and 0 < hours and hours <= 48

            day_data = self.load_day_all_data(day)
            data = day_data[start_hour : start_hour + hours]
            if start_hour + hours <= 24:
                # print(day, start_hour)
                return data
            else:
                day2 = day + timedelta(days=1)
                day_data = self.load_day_all_data(day2)
                data2 = day_data[: start_hour + hours - 24]
                data = np.concatenate([data, data2], axis=0)
                # print(day, start_hour, day2, start_hour + hours - 24)
                return data
        except:
            raise Exception("No data found for {}, start_hour {}, hours {}".format(day, start_hour, hours))

    def remove_day_vdata(self, day):
        day_data = []
        all_path = self.get_file_path(day.year, day.month, day.day, 'all')
        if all_path.exists():
            for variable in self.pm.variables:
                v_path = self.get_file_path(day.year, day.month, day.day, variable)
                if v_path.exists(): v_path.unlink(missing_ok=True)
        return
    
    def remove_variable_data(self):
        total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
        day = start_day
        while day <= end_day:
            self.remove_day_vdata(day)
            day = day + timedelta(days=1)

    def remove_all(self, folder):
        for filename in os.listdir(folder):
            file_path = folder / filename
            if file_path.is_file():
                file_path.unlink(missing_ok=True)

    def decompress_day_all_data(self):
        total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
        day = start_day
        while day <= end_day:
            print('int', day)
            file_path_npy = self.get_file_path(day.year, day.month, day.day, 'all')
            try:
                data = np.load(file_path_npy)
                np.save(file_path_npy, data[:, :720, :])
            except:
                data = self.load_day_all_data_outdated(day)
            if data is not None:
                folder_path = file_path_npy.parent
                self.remove_all(folder_path)
                np.save(file_path_npy, data)
            day = day + timedelta(days=1)

    def integrate_variable_data(self):
        total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
        day = start_day
        while day <= end_day:
            print('int', day)
            file_path = self.get_file_path(day.year, day.month, day.day, 'all')
            if not file_path.eixsts():
                array = self.load_day_vdata(day).numpy()
                if array is not None:
                    np.savez_compressed(file_path, array)
                    self.remove_day_vdata(day)
            else:
                self.remove_day_vdata(day)
            day = day + timedelta(days=1)
        return
    
    def resize_cylindrical(self, x, size_hw):
        """
        x: (B, frames, C, H, W)
        size_hw: (H1, W1)
        """
        B, Fm, C, H, W = x.shape
        outs = []

        for f in range(Fm):
            frame = x[:, f]  # (B, C, H, W)

            # Mike!!! DO we need circular interpolation?

            # circular padding along W (last dimension)
            # pad format: (pad_left, pad_right, pad_top, pad_bottom)
            frame = F.pad(frame, (1, 1, 0, 0), mode="circular")  # (B, C, H, W+2)

            # interpolate. Pls make sure it interpolate inside each C plane.
            frame_resized = F.interpolate(
                frame, size=(size_hw[0], size_hw[1] + 2),
                mode="bicubic", align_corners=False    # mode="bilinear"
            )

            # remove the extra columns to restore circular wrap
            frame_resized = frame_resized[:, :, :, 1:-1]  # (B, C, H1, W1)

            outs.append(frame_resized)

        # stack frames back
        output = torch.stack(outs, dim=1)
        # print(B, Fm, C, H, W, size_hw[0], size_hw[1],output.shape)
        assert output.shape == torch.Size([B, Fm, C, size_hw[0], size_hw[1]])
        
        return output  # (B, frames, C, H1, W1)

    def load_from_drive(self, day, with_normalized=True):
        unnormalied = torch.tensor(self.load_day_all_data(day))
        normalized = None
        if with_normalized:
            normalized = self.normalize(unnormalied)
        return unnormalied, normalized

    def take_batch(self):
        
        # batch_x = torch.randn(self.pm.batch_size, self.pm.T1, self.pm.nVariables, 72 * self.pm.factor10, 144 * self.pm.factor10)
        # batch_y = torch.randn(self.pm.batch_size, self.pm.T1, self.pm.nVariables, 72 * self.pm.factor10, 144 * self.pm.factor10)
        
        if len(self.x_set) <= 0: return None

        taken = self.x_set[:self.pm.batch_size]
        self.x_set = self.x_set[self.pm.batch_size:]
        assert len(taken) > 0

        # B, C, T1, FG, H, W = len(taken), len(self.pm.variables), self.pm.T1, self.pm.FG, self.pm.height, self.pm.width

        # batch_x: (B, T1*FG, C, H, W), batch_y: (B, T2*FG, C, H, W). batch_y[i] immediately follows batch_x[i]
        batch_x, batch_y = [], []  #torch.empty((B, 0, C, H, W), dtype=self.pm.dtype), torch.empty((B, 0, C, H, W), dtype=self.pm.dtype)

        months = []
        for b in range(len(taken)):
            day, start_hour = taken[b]
            months.append(day.month)
            # hours = self.pm.T1 * self.pm.FG
            nHours = self.pm.nFrames
            mHours = self.pm.mFrames
            
            if nHours != 24 or mHours != 24:
                data = self.load_hours_all_data(day, start_hour, nHours + mHours)                    # (2 * T1FG, C, H, W)
                
                d = torch.tensor(data[:nHours])
                d = self.normalize(d)
                batch_x.append(d)

                d = torch.tensor(data[nHours:])
                d = self.normalize(d)
                batch_y.append(d)
                                
            else:
                with_normalized = False
                future_x = self._disk_threadPool.submit(self.load_from_drive, day, with_normalized)
                future_y = self._disk_threadPool.submit(self.load_from_drive, day + timedelta(days=1), with_normalized)                
                unnormalized_x, _ = future_x.result()
                unnormalized_y, _ = future_y.result()
                batch_x.append(unnormalized_x)
                batch_y.append(unnormalized_y)
                
        batch_x = torch.stack(batch_x, dim=0)     # (B, T1FG, C, 721, 1440)
        batch_y = torch.stack(batch_y, dim=0)     # (B, T2FG, C, 721, 1440)

        # Required.
        batch_x = batch_x[:, :, :, :self.pm.height, :]  # 721 -> 720 for transformer and factor10 == 10
        batch_y = batch_y[:, :, :, :self.pm.height, :]  # 721 -> 720 for transformer and factor10 == 10
        
        B, _, C, H, W = batch_x.shape
        if self.pm.height != H or self.pm.width != W:
            batch_x = self.resize_cylindrical(batch_x, (self.pm.height, self.pm.width))
            batch_y = self.resize_cylindrical(batch_y, (self.pm.height, self.pm.width))
                    
        # self.visualize_frames(batch_x[0])

        if self.pm.include_difficulty:
            # difficulties[variable][month] : data (721, 1440)
            batch_diff = []
            for b in range(B):
                variable_diff = []
                for variable in self.pm.variables:
                    # self.difficulties[vaiable][months[b]].shape = (720, 1440)
                    diff = self.difficulties[variable][months[b]]   # (721, 1440)
                    variable_diff.append(torch.tensor(diff, dtype=self.pm.dtype))

                variable_diff = torch.stack(variable_diff, dim=0)           # (C, 720, 1440)
                batch_diff.append(variable_diff)
            batch_diff = torch.stack(batch_diff, dim=0)                     # (B, C, 720, 1440)

            if self.pm.height != H or self.pm.width != W:
                batch_diff = self.resize_cylindrical(batch_diff, (self.pm.height, self.pm.width))
            
        else:
            batch_diff = None
        # print('batch_diff', batch_diff.shape)

        # print("batch_x, y shape: ", batch_x.shape, batch_y.shape)
        # print('-------- type ', batch_x.dtype, batch_y.dtype, batch_diff.dtype, self.era5_data_vars.dtype)
        if self.pm.exclude_poles > 0:
            batch_x = batch_x[:, :, :, self.pm.exclude_poles:-self.pm.exclude_poles, :]
            batch_y = batch_y[:, :, :, self.pm.exclude_poles:-self.pm.exclude_poles, :]
            if batch_diff is not None:
                batch_diff = batch_diff[:, self.pm.exclude_poles:-self.pm.exclude_poles, :]
        
        return batch_x, batch_y, batch_diff, self.era5_data_vars     # (B, T1FG, C, H, W), (B, T2FG, C, H, W), (B, 1, C, H, W), (C,)
    
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
            # batch_x, batch_y, batch_diff, era5_data_vars
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

        # print('get_batch... ')
        batch = self.get_batch()
        # print(' ...get_batch: ', batch is not None)
        if batch is not None: self._fetched_batches += 1
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

    print(f"✅ All contents (except 'CDS') copied from {src} → {dst}")

#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    # src = "/mnt/data/zeus_data/CDS"           # source CDS folder
    # drive_even = "/mnt/data_even/CDS"  # target drive for even days
    # drive_odd = "/mnt/data_odd/CDS"   # target drive for odd days
    # divide_cds_by_day(src, drive_even, drive_odd)
    
    # src = "/mnt/data/zeus_data"
    # dst = "/mnt/data_even"
    # copy_except_cds(src, dst)

    from models.config import config
    pm = Params(config)

    dataset = Zeus_Dataset(pm, Path(config['dataset_path']), Path(config['management_path']))

    end_date = date(2018, 10, 10)
    dataset.download_dataset(2010, end_date, max_workers=10)   # downloads day data in reverse order.
    
    total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
    print(f"total_days {total_days}, start_day {start_day}, end_day {end_day}, complete {complete}")
    
    # mean, std = dataset.analyze_dataset(save=False)
    # print(mean, std)

    # for _ in range(100):
    #     print(dataset.get_T2_new())

    # mean, std = dataset.analyze_dataset()
    # print(mean, std)

    # mean, std = dataset.load_dataset_params()
    # print("mean: {}, std: {}".format(mean, std))

    train_hours, vali_hours, test_hours = dataset.split_data_hours(renew=False, save=False, seed=523)
    print("Lengths:", len(train_hours), len(vali_hours), len(test_hours))
    print("train_hours[:3]", train_hours[:3])

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

