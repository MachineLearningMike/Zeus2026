import xarray as xr
import requests
import subprocess
import os

# Example: Download IMERG Early Run 30‑minute file via HTTP
url = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/2025/12/05/3B-HHR-E.MS.MRG.3IMERG.20251205-S000000-E002959.0300.V07B.30min.tif'
local_path = '/tmp/imerg_20251205_0000.tif'
subprocess.run(['wget', '-O', local_path, url])

ds = xr.open_rasterio(local_path)
precip = ds.sel(band=1)  # precipitation accumulation or rate

# Example: Load OISST from NOAA via OpenDAP / NetCDF
oisst_url = 'https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.current.nc'
ds_sst = xr.open_dataset(oisst_url)
sst = ds_sst['sst']
print(sst.sel(time='2025-12-01'))
