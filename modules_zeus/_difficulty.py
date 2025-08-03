
import xarray as xr
import torch
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from dateutil.relativedelta import relativedelta

import os
import re
import pathlib
import matplotlib.pyplot as plt
import calendar
from datetime import datetime

from sample import Era5Sample
from coordinates import get_bbox, slice_bbox

MONTHS = 1  # 12

class difficulty():

    def __init__(
        self,
        start_year,
        start_month,
        variables,
        data_folder: str = "weights/",  # part starting from the folder this script itself is located in
    ):
        self.start_year = start_year
        self.start_month = start_month
        self.variables = variables
        self.data_folder = data_folder

        self.dataset = self.get_normalized_dataset()

    def next_month(self, dt):
        return dt + relativedelta(month=1)
    def last_day_of_month(self, year, month):
        return calendar.monthrange(year, month)[1]
    def month_string(self, year, month):
        return str(year) + '-' + str(month).zfill(2)
    def day_string(self, year, month, day):
        return str(year) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2)

    def get_normalized_dataset(self):
        dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
        dataset = dataset[self.variables]
        dataset = dataset.sel(time=slice("2023-10-31", "2024-10-31"))

        # convert latitude and longitude to -90 to 90 and -180 to 180 respectively
        if dataset["longitude"].max() > 180:
            dataset = dataset.assign_coords(longitude=(dataset["longitude"].values + 180) % 360 - 180)
        if dataset["latitude"].max() > 90:
            dataset = dataset.assign_coords(latitude=dataset["latitude"].values - 90)

        dataset = dataset.sortby(["latitude", "longitude"])
        dataset = dataset.chunk()

        return dataset

    def create_month_list(self, start, end):
        months = []
        current = start
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            current += relativedelta(months=1)
        return months

    def create_difficulty_map(self, WRITE_FILES = False):
        absolute_max = 0
        stored_difficulties = []

        first_day = datetime(year=self.start_year, month=self.start_month, day=1)
        last_day = self.last_day_of_month(first_day.year, first_day.month)

        variable = self.variables[0]

        for _ in range(MONTHS):
            first_day_str = self.day_string(first_day.year, first_day.month, first_day.day)
            last_day_str = self.day_string(first_day.year, first_day.month, last_day)

            month_data = self.dataset.sel(time=slice(first_day_str, last_day_str))

            # fetch that data -- takes 1-2 minutes per call
            month_data = month_data.compute()

            # calculate the 'variance of time-derivative' for each location
            t2m_data = torch.as_tensor(month_data[variable].data, dtype=torch.float)
            print("shape: ", t2m_data.shape)
            t2m_diffs = torch.diff(t2m_data, n=1, dim=0)
            var_per_loc = t2m_diffs.var(dim=0)

            month_str = self.month_string(first_day.year, first_day.month)
            print(f"Min/Mean/Max for {month_str}", var_per_loc.min(), var_per_loc.mean(), var_per_loc.max())

            if var_per_loc.max() > absolute_max:
                absolute_max = var_per_loc.max()
            
            stored_difficulties.append((month_str, var_per_loc.numpy()))

            first_day = self.next_month(first_day)
            last_day = self.last_day_of_month(first_day.year, first_day.month)

            
        if WRITE_FILES:
            current_folder = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(current_folder,"weights"), exist_ok=True)
            for month_str, var_per_loc in stored_difficulties:
                var_per_loc = var_per_loc / absolute_max
                np.save(os.path.join(current_folder, "weights", "difficulty_" + month_str + "_grid_025.npy"), var_per_loc.numpy())


    def show_difficulty_map(self):
        data_folder = pathlib.Path(os.path.abspath(__file__)).parent / self.data_folder
        self.weight_files = data_folder.glob("*.npy")

        for weight_file in self.weight_files:
            weight_matrix = np.load(weight_file)
            break

        var_per_loc = weight_matrix

        # create artificial grid of latitudes and longitudes for plotting
        x_grid = torch.stack(
            torch.meshgrid(
                *[
                    torch.as_tensor(
                        self.dataset[v].data, dtype=torch.float
                    )
                    for v in ("latitude", "longitude")
                ],
                indexing="ij",
            ),
            dim=-1,
        )

        # plot the last month's data
        fig = plt.figure(figsize=(8.0, 6.0), dpi=100)

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


    def load_difficulty(self):
        
        data_folder = pathlib.Path(os.path.abspath(__file__)).parent / self.data_folder
        self.weight_files = data_folder.glob("*.npy")

        self.difficulty_matrices = {}

        for weight_file in self.weight_files:
            weight_matrix = np.load(weight_file)
            month = re.search(r"difficulty_(\d+)", weight_file.stem)
            if not month:
                # bt.logging.error(
                #     f"Could not parse weight file {weight_file.stem}! This means another month will be used for month {month}."
                # )     # Mike
                print(
                    f"Could not parse weight file {weight_file.stem}! This means another month will be used for month {month}."
                )
            month = month.group(1)
            self.difficulty_matrices[int(month)] = weight_matrix

        if not self.difficulty_matrices:
            raise AssertionError(
                f"No difficulty matrices found in {data_folder}, scoring cannot be calculated!"
            ) 

    def get_difficulty_grid(self, sample: Era5Sample) -> np.ndarray:
        """
        Returns the difficulties for each grid location in the given sample.
        """
        start_month = datetime.fromtimestamp(sample.start_timestamp).month

        # months don't differ that much. If the specified month is not found, use the first one in storage.
        difficulty_matrix = self.difficulty_matrices.get(
            start_month,
            self.difficulty_matrices[list(self.difficulty_matrices.keys())[0]],
        )

        difficulty_grid = slice_bbox(difficulty_matrix, sample.get_bbox())

        return difficulty_grid



if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    import era5_cds

    variables = ['2m_temperature'] # , 'total_precipitation', '100m_u_component_of_wind', '100m_v_component_of_wind']
    difficulty = difficulty(2024, 7, variables)

    difficulty.create_difficulty_map(WRITE_FILES = True)
    difficulty.load_difficulty()
    difficulty.show_difficulty_map()

    cds_loader = era5_cds.Era5CDSLoader()

    sample = cds_loader.get_sample()        # get random sample
    difficult_grid = difficulty.get_difficulty_grid(sample)
    print(difficult_grid)

    sample = cds_loader.get_sample()        # get random sample
    difficult_grid = difficulty.get_difficulty_grid(sample)
    print(difficult_grid)