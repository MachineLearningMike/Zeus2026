#!/usr/bin/env python3
"""
preprocess_hres_fc0.py

Preprocess ECMWF HRES fc0 data into ML-ready 0.25° tensor for mike1_variables.

Steps:

1. Load fc0 NetCDF
2. Select variables
3. Interpolate upper-air variables to ERA5 pressure levels
4. Downsample horizontally to 0.25° grid
5. Assemble ML input tensor
6. Optional normalization
   """

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import xesmf as xe  # for horizontal remapping

def preprocess_fc0(nc_file):
# ------------------------------
# 1. Load fc0 dataset
# ------------------------------
ds = xr.open_dataset(nc_file)

```
# ------------------------------
# 2. Define variables used in ML
# ------------------------------
mike1_variables = [
    # Upper-air
    "z_50","t_50","u_50","v_50","q_50",
    "z_100","t_100","u_100","v_100","q_100",
    "z_150","t_150","u_150","v_150","q_150",
    "z_200","t_200","u_200","v_200","q_200",
    "z_250","t_250","u_250","v_250","q_250",
    "z_300","t_300","u_300","v_300","q_300",
    "z_400","t_400","u_400","v_400","q_400",
    "z_500","t_500","u_500","v_500","q_500",
    "z_600","t_600","u_600","v_600","q_600",
    "z_700","t_700","u_700","v_700","q_700",
    "z_850","t_850","u_850","v_850","q_850",
    "z_925","t_925","u_925","v_925","q_925",
    "z_1000","t_1000","u_1000","v_1000","q_1000",
    # Surface
    "u10m","v10m","msl","lcc","mcc","hcc","tcc",
    "ssr","ssrd","fdir","ttr",
    "t2m","tp","u100m","v100m","d2m","sp"
]

# ------------------------------
# 3. Vertical interpolation for upper-air variables
# ------------------------------
target_levels = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])  # hPa
upper_vars = ["z","t","u","v","q"]
upper_interpolated = {}

for var in upper_vars:
    data = ds[var].values  # shape (level, lat, lon)
    fc0_levels = ds.level.values  # hybrid levels
    interp_func = interp1d(np.log(fc0_levels), data, axis=0, bounds_error=False, fill_value="extrapolate")
    for lev in target_levels:
        key = f"{var}_{lev}"
        upper_interpolated[key] = interp_func(np.log(lev))

# ------------------------------
# 4. Surface variables
# ------------------------------
surface_vars = ["u10","v10","msl","lcc","mcc","hcc","tcc",
                "ssr","ssrd","fdir","ttr",
                "t2m","tp","u100","v100","d2m","sp"]

surface_data = {var: ds[var].values for var in surface_vars}

# ------------------------------
# 5. Horizontal remapping to 0.25° grid
# ------------------------------
lat_new = np.arange(-90,90.01,0.25)
lon_new = np.arange(0,360,0.25)
grid_out = xr.Dataset({"lat": (["lat"], lat_new),
                       "lon": (["lon"], lon_new)})
regridder = xe.Regridder(ds, grid_out, "bilinear", reuse_weights=True)

for key in upper_interpolated:
    upper_interpolated[key] = regridder(upper_interpolated[key])

for key in surface_data:
    surface_data[key] = regridder(surface_data[key])

# ------------------------------
# 6. Assemble ML input tensor
# ------------------------------
tensor = np.stack([upper_interpolated.get(v, surface_data.get(v)) for v in mike1_variables], axis=0)
# shape: (num_vars, lat, lon)

# ------------------------------
# 7. Optional normalization
# ------------------------------
# mean_array = np.load("means.npy")
# std_array = np.load("stds.npy")
# tensor = (tensor - mean_array[:,None,None]) / std_array[:,None,None]

return tensor
```

if **name** == "**main**":
tensor = preprocess_fc0("hres_fc0.nc")
print("ML-ready tensor shape:", tensor.shape)
# np.save("ml_input_tensor.npy", tensor)
