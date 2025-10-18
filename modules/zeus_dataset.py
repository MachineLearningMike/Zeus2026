import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import threading
import calendar
import json
import gc
import math
import platform
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
from modules_zeus.constants import ERA5_DATA_VARS
from modules_zeus.sample import Era5Sample
from modules_zeus.coordinates import get_bbox, slice_bbox
from modules_zeus.era5_cds import Era5CDSLoader, get_today


class Zeus_Dataset(Dataset):
    def __init__(self, config, root_path):
        super().__init__()
        self.pm = Params(config)
        self.root_path = root_path

        self.cds_folder = root_path / "CDS"
        self.cds_folder.mkdir(parents=True, exist_ok=True)
        
        self.param_path = root_path / 'dataset_params.json'
        self.split_hours_path = root_path / 'split_hours.json'
        
        self.difficulties = None
        self.era5_data_vars = None

        self.loader = Era5CDSLoader()
        self.train_dates, self.vali_dates, self.test_dates = None, None, None
        self.train_set, self.vali_set, self.test_set = None, None, None
        self.x_set, self.x_set_len = None, 0

        self.load_data()
        
        self._fetched_batches = 0
        self.total_batches = 0

    #------------------------------------- Download dataset -----------------------

    def load_data(self):
        self.load_dataset_params()
        if self.pm.include_difficulty: self.load_difficulties()
        self.load_era5_data_vars()
        self.split_data_hours(seed=self.pm.seed)
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
        return self.cds_folder / str(year) / str(month).zfill(2) / str(day).zfill(2) / (variable + ".npz")
    def get_file_path_npy(self, year, month, day, variable):
        return self.cds_folder / str(year) / str(month).zfill(2) / str(day).zfill(2) / (variable + ".npy")
    def get_last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]

    def lownload_day_data(self, total_ds, year, month, day):
        files_to_remove = []
        file_path_npy = self.get_file_path_npy(year, month, day, 'all')
        file_path_npz = self.get_file_path(year, month, day, 'all')
        try:
            data = np.load(file_path_npy)   # ok
            if file_path_npz.exists(): files_to_remove.append(file_path_npz)
        except:
            try:
                load = np.load(file_path_npz)
                data = load[load.files[0]]
                np.save(file_path_npy, data)
                files_to_remove.append(file_path_npz)
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
                        array = array.astype(np.float16)    # (24, 721, 1440)           #-------- float16
                    data_list.append(array)

                data = np.stack(data_list, axis=1)      # (24, C, 721, 1440)
                assert data.shape[1] == len(self.pm.variables)

                file_path_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(file_path_npy, data)

        return files_to_remove

    def download_dataset(self, start_year, end_date):
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
                    files_to_remove = self.lownload_day_data(total_ds, year, month, day)
                    for f in files_to_remove: os.remove(f)
        return
    
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


    def count_data_days(self):
        start, end = None, None
        missing = False
        total_days = 0
        for year_dir in sorted(self.cds_folder.iterdir())[ - self.pm.data_last_years : ]:
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue

                    files = {f.stem for f in day_dir.glob("*.npy")}
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
    

    def analyze_dataset(self):
        mean = np.zeros((len(self.pm.variables),), dtype=np.float32)
        n = 0
        for year_dir in sorted(self.cds_folder.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                                        
                    file_path = day_dir / 'all.npy'
                    arr = np.load(str(file_path))   # (24, C, 721, 1440)
                    arr = arr[:, :, :720, :]        # (24, C, 720, 1440)    # Exclude the north pole at this time.
                    # _mean = np.mean(arr, axis=(0, 2, 3))    # (C,)
                    _mean = torch.tensor(arr, dtype=torch.float32).cuda().mean(dim=(0,2,3)).cpu().numpy()
                    has_nan = np.any(np.isnan(_mean))
                    if has_nan:
                        raise Exception("NAN found: ", str(day_dir))
                    print("mean", str(day_dir), _mean)
                    n += 1
                    mean = mean * (n-1)/n + _mean / n   # (C,)
            #         break
            #     break
            # break
        self.mean = mean        # (C,)

        #-------------------------------- std -------------------
        C = len(self.pm.variables)
        # mean = np.zeros((C,), dtype=np.float64)
        # M2   = np.zeros((C,), dtype=np.float64)
        # n = 0

        self_mean = torch.tensor(self.mean).cuda()
        var = np.zeros((len(self.pm.variables),), dtype=np.float32)
        n = 0

        for year_dir in sorted(self.cds_folder.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue

                    file_path = day_dir / 'all.npy'

                    # arr = np.load(file_path)[:, :, :720, :]   # (24, C, 720, 1440)
                    # x = arr.transpose(0, 2, 3, 1).reshape(-1, C)  # flatten all samples
                    # for batch in np.array_split(x, 10):  # optional: process in chunks
                    #     n_batch = batch.shape[0]
                    #     batch_mean = batch.mean(axis=0)
                    #     batch_var  = ((batch - batch_mean)**2).sum(axis=0)
                    #     # Update running stats (Welford)
                    #     delta = batch_mean - mean
                    #     total_n = n + n_batch
                    #     mean += delta * n_batch / total_n
                    #     M2   += batch_var + delta**2 * n * n_batch / total_n
                    #     n = total_n
                    # print("var", str(day_dir), M2)

                    arr = np.load(str(file_path))   # (24, C, 721, 1440)
                    arr = arr[:, :, :720, :]        # (24, C, 720, 1440)    # Exclude the north pole, at this time.

                    arr = arr.transpose(0, 2, 3, 1) # (24, 720, 1440, C)
                    # _var = np.mean(np.power(arr - self.mean, 2), axis=(0,1,2))    # (C,)
                    
                    ten = torch.tensor(arr, dtype=torch.float32).cuda()
                    _var = torch.mean(torch.pow(ten - self_mean, 2), dim=(0,1,2)).cpu().numpy()

                    has_nan = np.any(np.isnan(_var))
                    if has_nan:
                        raise Exception("NAN found: ", str(day_dir))
                    print("var", str(day_dir), _var)
                    n += 1
                    var = var * (n-1)/n + _var / n   # (C,)

            #         break
            #     break
            # break
            
        # var = M2 / n
        std = np.sqrt(var)
        self.std = std

        data = {
            "mean": self.mean.tolist(),
            "std":  self.std.tolist(),
        }
        if self.param_path.exists() and self.param_path.is_file():
            self.param_path.unlink()  # deletes the file
        with open(self.param_path, "w") as f:
            json.dump(data, f)

        return self.mean, self.std  # (C,), (C,)

    def load_dataset_params(self):
        with open(self.param_path, "r") as f:
            data = json.load(f)
            # print('------', data, torch.tensor(data['mean']).shape)
        mean, std = torch.tensor(data['mean']), torch.tensor(data['std'])
        
        std = std * self.pm.factor10 / 10   #=============================== Mike!!!
        
        self.mean, self.std = mean, std
        return mean, std    # (C,), (C,)

    def normalize(self, data):    # data: (... C, H, W)
        ## Shall we compute it on cuda? This data is on cpu. cuda is working now. Choose not to interfere cuda with new momory allocation.
        device = data.device
        return ( data - rearrange(self.mean, 'c -> c 1 1').to(device) ) / rearrange(self.std, 'c -> c 1 1').to(device)
    
    def unnormalize(self, data):    # data: (... C, H, W)
        # This data is already on cuda. So computing uses the GPU(s).
        device = data.device
        return data * rearrange(self.std, 'c -> c 1 1').to(device) + rearrange(self.mean, 'c -> c 1 1').to(device)
    
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
                    file_path = self.get_file_path_npy(year, month, day, 'all')

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
    
    def split_data_hours(self, renew=False, seed=None):
        total_days, start_day, end_day, complete, data_dates = self.count_data_days()

        train_pct = self.pm.train_pct; vali_pct = self.pm.vali_pct

        try:
            assert renew == False
            # print("Loading data dates...")
            with open(self.split_hours_path, "r") as f:
                data = json.load(f)

            self.train_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["train_hours"]]
            self.vali_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["vali_hours"]]
            self.test_hours = [(datetime.fromisoformat(d).date(), h) for (d, h) in data["test_hours"]]
        except:
            print("Loading failed. Creating data hours...")

            T1 = self.pm.T1
            maxT2 = self.pm.maxT2                      #------------- This is an implicit dependency
            data_dates = data_dates[:-1]    # Remove the last day for safety.
            
            data_hours = []
            for day in data_dates:
                # data_hours += [(day, hour) for hour in range(24)]
                data_hours += [(day, hour) for hour in range(1)]    # day, 0 hour

            if seed is not None: random.seed(seed)

            shuffled_hours = data_hours[:]
            random.shuffle(shuffled_hours)
            n = len(shuffled_hours)
            train_size = int(n * train_pct / 100)
            vali_size = int(n * vali_pct / 100)
            self.train_hours = shuffled_hours[:train_size]
            self.vali_hours = shuffled_hours[train_size : train_size + vali_size]
            self.test_hours = shuffled_hours[train_size + vali_size :]

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
    
    def load_day_all_data(self, day):
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
        
    def load_day_all_data_npy(self, day):
        all_path = self.get_file_path_npy(day.year, day.month, day.day, 'all')
        try:
            load = np.load(all_path)
            data = torch.tensor(load, dtype=self.pm.dtype)[:, :, :720, :]    # (24, C, H, W)
            return data
        except:
            raise Exception("No data found for {}".format(day))
        
    def load_hours_all_data_npy(self, day, start_hour, hours):
        try:
            assert isinstance(start_hour, int)
            assert isinstance(hours, int)
            assert 0 <= start_hour and start_hour < 24 and 0 < hours and hours <= 48

            day_data = self.load_day_all_data_npy(day)
            data = day_data[start_hour : start_hour + hours]
            if start_hour + hours <= 24:
                # print(day, start_hour)
                return data
            else:
                day2 = day + timedelta(days=1)
                day_data = self.load_day_all_data_npy(day2)
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
            file_path_npy = self.get_file_path_npy(day.year, day.month, day.day, 'all')
            try:
                data = np.load(file_path_npy)
                np.save(file_path_npy, data[:, :720, :])
            except:
                data = self.load_day_all_data(day)
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

            # circular padding along W (last dimension)
            # pad format: (pad_left, pad_right, pad_top, pad_bottom)
            frame = F.pad(frame, (1, 1, 0, 0), mode="circular")  # (B, C, H, W+2)

            # interpolate
            frame_resized = F.interpolate(
                frame, size=(size_hw[0], size_hw[1] + 2),
                mode="bicubic", align_corners=False    # mode="bilinear"
            )

            # remove the extra columns to restore circular wrap
            frame_resized = frame_resized[:, :, :, 1:-1]  # (B, C, H1, W1)

            outs.append(frame_resized)

        # stack frames back
        return torch.stack(outs, dim=1)  # (B, frames, C, H1, W1)

    def take_batch(self):
        if len(self.x_set) <= 0: return None

        taken = self.x_set[:self.pm.batch_size]
        self.x_set = self.x_set[self.pm.batch_size:]
        assert len(taken) > 0

        B, C, T1, FG, H, W = len(taken), len(self.pm.variables), self.pm.T1, self.pm.FG, self.pm.image_height, self.pm.image_width

        # batch_x: (B, T1*FG, C, H, W), batch_y: (B, T2*FG, C, H, W). batch_y[i] immediately follows batch_x[i]
        batch_x, batch_y = [], []  #torch.empty((B, 0, C, H, W), dtype=self.pm.dtype), torch.empty((B, 0, C, H, W), dtype=self.pm.dtype)

        months = []
        for b in range(B):
            day, start_hour = taken[b]
            months.append(day.month)

            # hours = self.pm.T1 * self.pm.FG
            nHours = self.pm.nFrames
            mHours = self.pm.mFrames
            data = self.load_hours_all_data_npy(day, start_hour, nHours + mHours)                    # (2 * T1FG, C, H, W)
            batch_x.append(torch.tensor(data[:nHours]))
            batch_y.append(torch.tensor(data[nHours:]))

        batch_x = torch.stack(batch_x, dim=0)     # (B, T1FG, C, H, W)
        batch_y = torch.stack(batch_y, dim=0)     # (B, T2FG, C, H, W)        
        
        B, _, C, H, W = batch_x.shape
        
        H1 = self.pm.image_height
        W1 = self.pm.image_width
        if H1 != H or W1 != W:
            batch_x = self.resize_cylindrical(batch_x, (H1, W1))
            batch_y = self.resize_cylindrical(batch_y, (H1, W1))
            
        batch_x = self.normalize(batch_x)
        batch_y = self.normalize(batch_y)
            
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
        else:
            batch_diff = None
        # print('batch_diff', batch_diff.shape)

        # print("batch_x, y shape: ", batch_x.shape, batch_y.shape)
        # print('-------- type ', batch_x.dtype, batch_y.dtype, batch_diff.dtype, self.era5_data_vars.dtype)
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
        # print(' ...shutdown')


#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    from models.config import config, root_path

    dataset = Zeus_Dataset(config, root_path)

    # end_date = date(2025, 10, 5)
    # dataset.download_dataset(2015, end_date)   # downloads day data in reverse order.

    total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
    print(start_day, end_day)
    print(total_days, complete)

    # for _ in range(100):
    #     print(dataset.get_T2_new())

    # mean, std = dataset.analyze_dataset()
    # print(mean, std)

    # mean, std = dataset.load_dataset_params()
    # print("mean: {}, std: {}".format(mean, std))

    # train_dates, vali_dates, test_dates = dataset.split_data_dates(renew=True, seed=523)
    # print("Lengths of dates:", len(train_dates), len(vali_dates), len(test_dates))
    # print("train_dates[:3]", train_dates[:3])

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

