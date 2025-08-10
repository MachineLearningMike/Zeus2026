import numpy as np
import xarray as xr
import torch
import threading
import calendar
import json
import math
from datetime import datetime, date, timedelta
import random
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.params import Params
from modules_zeus.sample import Era5Sample
from modules_zeus.coordinates import get_bbox, slice_bbox
from modules_zeus.era5_cds import Era5CDSLoader, get_today
from modules_zeus._sample_time_range import sample_time_range

class Zeus_Dataset(Dataset):
    def __init__(self, config, root_dir, param_path, split_dates_path, variables):
        super().__init__()
        self.pm = Params(config)
        self.root_dir = root_dir
        self.param_path = param_path
        self.split_dates_path = split_dates_path
        self.variables = variables
        self.difficulties = None

        self.loader = Era5CDSLoader()
        self.train_dates, self.vali_dates, self.test_dates = None, None, None
        self.train_set, self.vali_set, self.test_set = None, None, None

        self.total_batches = 10
        self._fetched_batches = 0

        self._lock = threading.Lock()
        self._batch = None
        self._param = None

        self._request_event = threading.Event()
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()

        self._worker_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._worker_thread.start()

    #------------------------------------- Download dataset -----------------------

    def get_normalized_dataset(self):
        dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
        # dataset = dataset[self.variables]
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
        return os.path.join(self.root_dir, str(year), str(month).zfill(2), str(day).zfill(2), variable + ".npz")
    def get_last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]

    def download_dataset(self, start_year):
        total_ds = self.get_normalized_dataset()

        print(list(total_ds.data_vars))
        print(total_ds["latitude"].min(), total_ds["latitude"].max(), total_ds["longitude"].min(), total_ds["longitude"].max())
        
        # 2. Select variables and time range
        year_list = list(range(date.today().year, start_year-1, -1))
        month_list = list(range(12, 0, -1))

        for year in year_list:
            for month in month_list:
                for day in range(self.get_last_day_of_month(year, month), 0, -1):

                    if date(year, month, day) > date.today() - timedelta(days=6): continue
                    
                    for variable in self.variables:
                        print("{}/{}/{} {}...".format(year, str(month).zfill(2), str(day).zfill(2), variable), end='')

                        file_path = self.get_file_path(year, month, day, variable)
                        if os.path.exists(file_path): continue

                        ds = total_ds[[variable]]
                        monthDayStr = str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)
                        ds = ds.sel(time=slice(monthDayStr, monthDayStr))

                        # 3. Extract numpy arrays
                        array = ds[variable].values
                        array = array.astype(np.float16)
                        print(" {}".format(array.shape))

                        # 4. Save to disk as raw binary numpy files
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        np.savez_compressed(file_path, array)
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
        desired_stem_order = self.variables

        # Convert to a dict for fast lookup
        stem_rank = {stem: i for i, stem in enumerate(desired_stem_order)}

        # Get and sort file paths
        files = list(pathlib_dir.glob("*.npz"))
        sorted_files = sorted(files, key=lambda f: stem_rank.get(f.stem, float('inf')))
        return sorted_files


    def count_data_days(self):
        start, end = None, None
        missing = []
        total_days = 0
        for year_dir in sorted(self.root_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue

                    files = {f.stem for f in day_dir.glob("*.npz")}
                    missing_vars = [var for var in self.variables if var not in files]
                    if missing_vars:  missing.append((year_dir, month_dir, day_dir, missing_vars))

                    total_days += 1
                    if start is None:
                        start = day_dir
                    end = day_dir
                    end_day = date(year=int(end.parts[-3]), month=int(end.parts[-2]), day=int(end.parts[-1]))

        complete = missing == []
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
        mean = []
        for year_dir in self.root_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue

                    sorted_files = self.sort_files(day_dir, self.variables)
                    # arr = np.load(sorted_files[0])[np.load(sorted_files[0]).files[0]]   # (24, 721, 1440)
                    file_means = [np.mean(np.load(f)[np.load(f).files[0]], axis=(0,1,2)) for f in sorted_files] # A list of C scalars.
                    mean.append(file_means)
                    has_nan = np.any(np.isnan(file_means))
                    if has_nan:
                        raise Exception("NAN found: ", str(year_dir).split("\\")[-1], str(month_dir).split("\\")[-1], str(day_dir).split("\\")[-1])
                    print("mean", str(year_dir).split("\\")[-1], str(month_dir).split("\\")[-1], str(day_dir).split("\\")[-1], file_means)
            #         break
            #     break
            # break
        mean = np.array(mean)   # (days, C)
        mean = np.mean(mean, axis=0)    # (C,)
        self.mean = mean        # (C,)

        var = []
        for year_dir in self.root_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.iterdir():
                    if not day_dir.is_dir():
                        continue
                    
                    sorted_files = self.sort_files(day_dir, self.variables)
                    # print('sorted_files', sorted_files)
                    file_data = [np.load(f)[np.load(f).files[0]] for f in sorted_files]
                    file_data = np.array(file_data)
                    file_data = file_data.transpose(1,2,3,0)   # (24, 721, 1440, C)
                    file_data = np.mean(np.power(file_data - self.mean, 2), axis=(0,1,2))    # (C,)
                    var.append(file_data)
                    print("var", str(year_dir).split("\\")[-1], str(month_dir).split("\\")[-1], str(day_dir).split("\\")[-1], file_data)
            #         break
            #     break
            # break
        
        var = np.array(var)     # (days, C)
        self.std = np.power( np.mean(var, axis=0), 0.5)

        data = {
            "mean": self.mean.tolist(),
            "std":  self.std.tolist(),
        }
        with open(self.param_path, "w") as f:
            json.dump(data, f)

        return self.mean, self.std


    def load_dataset_params(self):
        with open(self.param_path, "r") as f:
            data = json.load(f)        
        mean, std = np.array(data['mean']), np.array(data['std'])
        if mean is not None and std is not None:
            self.mean, self.std = mean, std

        return self.mean, self.std
   
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
    

    def create_difficulties(self, latest_date, WRITE_FILES = False):
        for variable in self.variables:
            print(variable)
            difficulties = []

            maxDeffVar = 0.0
            for nth in range(12):
                year, month = self.previous_nth_month(latest_date, nth)
                first_day = date(year, month, 1)
                last_day = first_day.replace(day = self.last_day_of_month(year, month))

                data = []
                for day in range(1, self.last_day_of_month(year, month) + 1):
                    file_path = self.get_file_path(year, month, day, variable)
                    if not os.path.exists(file_path): 
                        raise Exception("Data file not found. {}/{}/{}, {}".format(year, month, day, variable))
                    
                    load = np.load(file_path)
                    _data = load[load.files[0]]     # (24, 721, 1440)
                    data.append(torch.tensor(_data))

                data = torch.concat(data, dim=0)               # (n, 721, 1440)
                diffs = torch.diff(data, n=1, dim=0)    # (n-1, 721, 1440)
                var_per_loc = diffs.var(dim=0).numpy()  # (721, 1440)

                month_str = self.month_string(first_day)
                print(f"Min/Mean/Max for {month_str}", var_per_loc.min(), var_per_loc.mean(), var_per_loc.max())

                if maxDeffVar < var_per_loc.max().item(): maxDeffVar = var_per_loc.max().item()
                
                difficulties.append((var_per_loc, year, month))
            
            for var_per_loc, year, month in difficulties:
                rel_var_per_loc = var_per_loc / maxDeffVar
                current_folder = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(os.path.join(current_folder,"weights", variable), exist_ok=True)
                np.save(os.path.join(current_folder, "weights", variable, str(year) + '-' + str(month).zfill(2) + ".npy"), rel_var_per_loc)


    def load_difficulties(self):

        total_days, start_day, end_day, complete, data_dates = self.count_data_days()

        difficulties = {}

        for variable in self.variables:
            _difficulties = {}

            for nth in range(12):
                year, month = self.previous_nth_month(end_day, nth)
                first_day = date(year, month, 1)
                last_day = first_day.replace(day = self.last_day_of_month(year, month))

                current_folder = os.path.dirname(os.path.abspath(__file__))
                os.makedirs(os.path.join(current_folder,"weights", variable), exist_ok=True)
                data = np.load(os.path.join(current_folder, "weights", variable, str(year) + '-' + str(month).zfill(2) + ".npy"))
                # print('load', variable, month, data.shape)
                _difficulties.update({month: data})
            
            difficulties.update({variable: _difficulties})

        self.difficulties = difficulties


    def show_difficulty(self, variable):
        var_per_loc = self.difficulties[variable][7]

        dataset = self.get_normalized_dataset()
        # create artificial grid of latitudes and longitudes for plotting
        x_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.as_tensor(
                        dataset[v].data, dtype=torch.float
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
    
    def split_data_dates(self, train_pct=0.5, vali_pct=0.3, seed=None):
        total_days, start_day, end_day, complete, data_dates = self.count_data_days()

        try:
            print("Loading data dates...")
            with open(self.split_dates_path, "r") as f:
                data = json.load(f)

            self.train_dates = [datetime.fromisoformat(d).date() for d in data["train_dates"]]
            self.vali_dates = [datetime.fromisoformat(d).date() for d in data["vali_dates"]]
            self.test_dates = [datetime.fromisoformat(d).date() for d in data["test_dates"]]
        except:
            print("Loading failed. Creating data dates...")

            T1 = self.pm.T1
            maxT2 = 14                          #------------- This is an implicit dependency
            data_dates = data_dates[: -(T1 + maxT2)]    # latest days cannot be a start date.

            if seed is not None: random.seed(seed)

            shuffled_dates = data_dates[:]
            random.shuffle(shuffled_dates)
            n = len(shuffled_dates)
            train_size = int(n * train_pct)
            vali_size = int(n * vali_pct)
            self.train_dates = shuffled_dates[:train_size]
            self.vali_dates = shuffled_dates[train_size : train_size + vali_size]
            self.test_dates = shuffled_dates[train_size + vali_size :]

            data = {
                "train_dates":  [d.isoformat() for d in self.train_dates],
                "vali_dates":   [d.isoformat() for d in self.vali_dates],
                "test_dates":   [d.isoformat() for d in self.test_dates],
            }
            with open(self.split_dates_path, "w") as f:
                json.dump(data, f, indent=2)
        
        return self.train_dates, self.vali_dates, self.test_dates


    def generate_x_sets(self, seed=None):
        if seed is not None: random.seed(seed)

        if self.train_dates is None or self.vali_dates is None or self.test_dates is None:
            print("Call split_data_dates first")
            return None, None, None

        N = 3
        train_dates = self.train_dates * N
        random.shuffle(train_dates)
        print(len(self.train_dates), len(train_dates))
        train_set = [(day, self.get_T2()) for day in train_dates]

        vali_set = [(day, self.get_T2()) for day in self.vali_dates]
        test_set = [(day, self.get_T2()) for day in self.test_dates]

        self.train_set, self.vali_set, self.test_set = train_set, vali_set, test_set

        return train_set, vali_set, test_set

    def take_batch(self, x_label, batch_size):
        x_set = self.train_set if x_label=='train_set' \
            else self.vali_set if x_label=='vali_set' \
            else self.test_set if x_label=='test_set' else None
        if x_set is None or len(x_set) <= 0: return None

        _, T2 = x_set[0]
        print("T2: ", T2)

        taken = []
        i = len(x_set) - 1
        while i >= 0 and len(taken) < batch_size:
            if x_set[i][1] == T2:                   # x_set[0][1] == T2, at least.
                taken.append(x_set.pop(i))          # This way, self.train_set, self.vali_set, and self.test_set diminish.
            i -= 1

        def get_day_data(day):
            day_data = []
            for c in range(C):
                variable = self.variables[c]
                file_path = self.get_file_path(day.year, day.month, day.day, variable)
                load = np.load(file_path)
                variable_data = torch.tensor(load[load.files[0]])[:, :-1, :]    # (24, H, W)
                day_data.append(variable_data)
            day_data = torch.stack(day_data, dim=1)     # (24, C, H, W)
            print("day_data: ", day_data.shape)
            return day_data

        B, C, T1, FG, H, W = len(taken), len(self.variables), self.pm.T1, self.pm.FG, self.pm.image_height, self.pm.image_width

        # batch_x: (B, T1*FG, C, H, W), batch_y: (B, T2*FG, C, H, W). batch_y[i] immediately follows batch_x[i]
        batch_x, batch_y = [], []  #torch.empty((B, 0, C, H, W)), torch.empty((B, 0, C, H, W))

        for b in range(B):
            start_date = taken[b][0]

            print("x...")
            days_data_x = []
            for t1 in range(T1):
                day = start_date + timedelta(days=t1)
                day_data = get_day_data(day)                    # (24, C, H, W)
                days_data_x.append(day_data)
            days_data_x = torch.concat(days_data_x, dim=0)      # (T1*24, C, H, W)
            batch_x.append(days_data_x)

            print("y...")
            days_data_y = []
            for t2 in range(T2):
                day = start_date + timedelta(days=T1+t2) 
                day_data = get_day_data(day)                    # (24, C, H, W)
                days_data_y.append(day_data)
            days_data_y = torch.concat(days_data_y, dim=0)      # (T2*24, C, H, W)
            batch_y.append(days_data_y)

        batch_x = torch.stack(batch_x, dim=0)
        batch_y = torch.stack(batch_y, dim=0)

        print("batch_x, y shape: ", batch_x.shape, batch_y.shape)
        
        return batch_x, batch_y

    #---------------------------------- Prefetch -------------------------------------

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            self._request_event.wait()
            self._request_event.clear()
            if self._stop_event.is_set():
                break

            with self._lock:
                param = self._param

            # Simulate data loading
            # time.sleep(random.uniform(0.000002, 0.000005))
            batch_x = torch.rand((3,4))
            batch_y = torch.rand(3,4)
            batch = (batch_x, batch_y)


            with self._lock:
                self._batch = batch
                self._ready_event.set()

    def __prefetch__(self, param):
        with self._lock:
            self._param = param
        self._ready_event.clear()
        self._request_event.set()

    def get_batch(self):
        self._ready_event.wait()
        with self._lock:
            return self._batch

    def __getitem__(self, idx):
        if idx >= self.total_batches:
            raise IndexError("Index exceeds dataset length.")

        batch = self.get_batch()

        self._fetched_batches += 1
        if self._fetched_batches >= self.total_batches:
            self.shutdown()

        return batch

    def __len__(self):
        return self.total_batches

    def shutdown(self):
        self._stop_event.set()
        self._request_event.set()
        self._worker_thread.join()


#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":

    root_dir = Path("D:\Download\CDS")
    param_path = Path("D:\Download\dataset_params.json")
    split_dates_path = Path("D:\Download\split_dates.json")
    variables = ["2m_temperature", 'total_precipitation', "100m_u_component_of_wind", '100m_v_component_of_wind']   # order matters.

    config = {
        # image h w c
        'height': 72,
        'width': 144,
        'channels': 4,
        # video length in and out
        'T1': 5,    # 5 later.
        'MaxT2': 5,
        'FG': 24,   # frames in a frame group
        # patch size
        'patch_size': 8,
        # transformer
        'heads': 8,
        'dim_head': 32,
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'patch_dim_to_dim' : 1,
        'dim_to_mlp_dim': 2,
        # layers
        'attLayers' : 1,
        'zeusLayers': 6,
    }

    dataset = Zeus_Dataset(config, root_dir, param_path, split_dates_path, variables)

    # dataset.download_dataset(2015)   # downloads day data in reverse order.

    total_days, start_day, end_day, complete, data_dates = dataset.count_data_days()
    print(start_day, end_day)
    print(total_days, complete)

    # complete, mean, std = dataset.analyze_dataset()
    # print(complete, mean, std)

    mean, std = dataset.load_dataset_params()
    print("mean: {}, std: {}".format(mean, std))

    train_dates, vali_dates, test_dates = dataset.split_data_dates(train_pct=0.5, vali_pct=0.3, seed=523)
    print("Lengths of dates:", len(train_dates), len(vali_dates), len(test_dates))
    print("train_dates[:3]", train_dates[:3])

    train_set, vali_set, test_set = dataset.generate_x_sets(seed=523)
    print("train_set, vali_set, test_set lengths:", len(train_set), len(vali_set), len(test_set))
    print("train_set[:3]", train_set[:3])
    
    batch = dataset.take_batch('vali_set', batch_size=2)
    # print("batch: ", batch)
    
    # dataset.create_difficulties(end_day, WRITE_FILES=True)

    dataset.load_difficulties()

    # dataset.show_difficulty("100m_u_component_of_wind")

    # cds_loader = Era5CDSLoader()
    # sample = cds_loader.get_sample()        # get random sample
    # difficult_grid = dataset.get_difficulty_grid("100m_u_component_of_wind", sample)
    # print(difficult_grid)