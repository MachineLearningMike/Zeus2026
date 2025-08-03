
import xarray as xr
import torch
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from dateutil.relativedelta import relativedelta

import os
import pathlib
import matplotlib.pyplot as plt


class difficulty_map():

    def __init__(
        self,
        data_folder: str = "weights/",  # part starting from the folder this script itself is located in
    ):
        self.dataset = self.get_normalized_dataset()

        data_folder = pathlib.Path(os.path.abspath(__file__)).parent / data_folder
        self.weight_files = data_folder.glob("*.npy")

    def get_normalized_dataset(self):
        dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", chunks=None)
        dataset = dataset[["2m_temperature"]]
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

        start = datetime(2024, 6, 1)    # inclusive
        end = datetime(2025, 7, 1)      # inclusive

        MONTHS = self.create_month_list(start, end)

        absolute_max = 0
        stored_difficulties = []
        for i, start in enumerate(MONTHS[:-1]):
            end = MONTHS[i+1]
            month_data = self.dataset.sel(time=slice(start, end))

            # fetch that data -- takes 1-2 minutes per call
            month_data = month_data.compute()

            # calculate the 'variance of time-derivative' for each location
            t2m_data = torch.as_tensor(month_data["2m_temperature"].data, dtype=torch.float)
            t2m_diffs = torch.diff(t2m_data, n=1, dim=0)
            var_per_loc = t2m_diffs.var(dim=0)

            month_str = start.split('-')[-1]
            print(f"Min/Mean/Max for {month_str}", var_per_loc.min(), var_per_loc.mean(), var_per_loc.max())

            if var_per_loc.max() > absolute_max:
                absolute_max = var_per_loc.max()
            
            stored_difficulties.append((month_str, var_per_loc.numpy()))

            
        if WRITE_FILES:
            for month_str, var_per_loc in stored_difficulties:
                var_per_loc = var_per_loc / absolute_max
                np.save(f"../weights/difficulty_{month_str}_grid_025.npy", var_per_loc.numpy())


    def show_difficulty_map(self):
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


if __name__ == "__main__":

    difficulty = difficulty_map()

    difficulty.show_difficulty_map()