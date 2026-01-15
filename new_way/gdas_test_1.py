import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import requests
import numpy as np
import os
import json

def test_gfs_access_simple():
    """Simple test to see if we can access GFS data."""
    print("Testing GFS Access...")
    print("=" * 50)
    
    # Try a known working URL
    test_urls = [
        "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs20240110/gfs_0p25_00z",
        "https://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs20240110/gfs_0p25_1hr_00z",
        "https://nomads.ncep.noaa.gov/dods/gfs_0p50/gfs20240110/gfs_0p50_00z",
    ]
    
    for url in test_urls:
        print(f"\nTrying: {url}")
        try:
            response = requests.head(url, timeout=10)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                print("✓ URL is accessible!")
                
                # Try to get basic info
                try:
                    ds = xr.open_dataset(url)
                    print(f"Dataset opened successfully!")
                    print(f"Dimensions: {dict(ds.dims)}")
                    
                    # List variables that match our needs
                    print("\nRelevant variables found:")
                    for var in ds.variables:
                        var_lower = var.lower()
                        if any(keyword in var_lower for keyword in ['tmp', 'pres', 'ugrd', 'vgrd', 'apcp', 'dpt']):
                            print(f"  - {var}")
                    
                    ds.close()
                    return url
                    
                except Exception as e:
                    print(f"Error opening dataset: {e}")
        except Exception as e:
            print(f"Connection error: {e}")
    
    return None

def download_small_region_gfs(url, lat_range, lon_range, region_name):
    """Download data for a small region."""
    print(f"\nDownloading data for {region_name}...")
    print(f"Latitude: {lat_range[0]}° to {lat_range[1]}°")
    print(f"Longitude: {lon_range[0]}° to {lon_range[1]}°")
    
    try:
        # Open the dataset
        ds = xr.open_dataset(url, decode_times=True)
        
        # Convert longitude to 0-360 if needed
        lon_min, lon_max = lon_range
        if lon_min < 0:
            lon_min_360 = lon_min + 360
            lon_max_360 = lon_max + 360
        else:
            lon_min_360, lon_max_360 = lon_min, lon_max
        
        # Select region
        # GFS has descending latitude (90 to -90)
        lat_min, lat_max = sorted(lat_range)
        
        # Try to select our variables
        data_vars = {}
        
        # Temperature at 2m
        if 'tmp2m' in ds:
            temp_data = ds['tmp2m'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            data_vars['temperature_2m'] = temp_data
        
        # Surface pressure
        if 'pressfc' in ds:
            pres_data = ds['pressfc'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            data_vars['surface_pressure'] = pres_data
        
        # Wind at 10m (closest to 100m available)
        if 'ugrd10m' in ds and 'vgrd10m' in ds:
            u_data = ds['ugrd10m'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            v_data = ds['vgrd10m'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            data_vars['wind_u_10m'] = u_data
            data_vars['wind_v_10m'] = v_data
        
        # Precipitation
        if 'apcpsfc' in ds:
            precip_data = ds['apcpsfc'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            data_vars['precipitation'] = precip_data
        
        # Dew point
        if 'dpt2m' in ds:
            dew_data = ds['dpt2m'].sel(
                lat=slice(lat_max, lat_min),
                lon=slice(lon_min_360, lon_max_360)
            )
            data_vars['dew_point'] = dew_data
        
        if not data_vars:
            print("No target variables found in dataset")
            ds.close()
            return None
        
        # Create new dataset
        result = xr.Dataset(data_vars)
        
        # Get first time step
        if 'time' in result.dims and len(result.time) > 0:
            result = result.isel(time=0)
        
        # Add metadata
        result.attrs.update({
            'region': region_name,
            'download_time': datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            'source_url': url,
            'lat_range': f"{lat_range[0]} to {lat_range[1]}",
            'lon_range': f"{lon_range[0]} to {lon_range[1]}",
            'grid_size': f"{result.sizes.get('lat', 0)}×{result.sizes.get('lon', 0)}"
        })
        
        ds.close()
        
        print(f"✓ Downloaded {len(data_vars)} variables")
        print(f"  Grid size: {result.attrs['grid_size']}")
        
        return result
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def create_simple_test_data():
    """Create simple test data without netcdf issues."""
    print("\nCreating simple test data...")
    
    # Create a simple grid
    lats = np.linspace(40.0, 42.0, 5)  # 5 points across 2 degrees
    lons = np.linspace(-75.0, -73.0, 5)  # 5 points across 2 degrees
    
    # Create synthetic data
    data = {
        'temperature_2m': 290 + np.random.randn(5, 5) * 2,  # ~17°C with noise
        'surface_pressure': 101325 + np.random.randn(5, 5) * 100,  # ~1013 hPa
        'wind_u_10m': np.random.randn(5, 5) * 3,  # U component
        'wind_v_10m': np.random.randn(5, 5) * 3,  # V component
        'precipitation': np.random.rand(5, 5) * 0.1,  # 0-0.1 kg/m²
        'dew_point': 285 + np.random.randn(5, 5) * 2,  # ~12°C
    }
    
    # Create pandas DataFrames for each variable
    dfs = {}
    for var_name, var_data in data.items():
        df = pd.DataFrame(var_data, index=lats, columns=lons)
        dfs[var_name] = df
    
    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for var_name, df in dfs.items():
        filename = f"test_{var_name}_{timestamp}.csv"
        df.to_csv(filename)
        print(f"  Saved {var_name} to {filename}")
    
    # Also save metadata
    metadata = {
        'region': 'Test_Region',
        'creation_time': datetime.utcnow().isoformat(),
        'lat_range': [40.0, 42.0],
        'lon_range': [-75.0, -73.0],
        'grid_resolution': '0.5 degrees',
        'variables': list(data.keys())
    }
    
    with open(f"test_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created test data files")
    
    return dfs

def main():
    """Main function."""
    print("GFS SMALL REGION DATA DOWNLOADER")
    print("=" * 60)
    
    # Test if we can access GFS
    gfs_url = test_gfs_access_simple()
    
    if gfs_url:
        print(f"\n{'='*60}")
        print("GFS ACCESS SUCCESSFUL - Downloading real data")
        print("=" * 60)
        
        # Define small 2-degree regions to try
        regions = [
            {
                'name': 'New_York',
                'lat_range': (40.5, 43.5),      # 3 degrees
                'lon_range': (-74.5, -73.5),    # 3 degrees
            },
            {
                'name': 'San_Francisco',
                'lat_range': (37.5, 40.5),      # 3 degrees
                'lon_range': (-123.5, -122.5),  # 3 degrees
            },
            {
                'name': 'London',
                'lat_range': (51.0, 54.0),      # 3 degrees
                'lon_range': (-1.0, 2.0),       # 3 degrees
            }
        ]
        
        # Try each region
        for region in regions:
            print(f"\nTrying region: {region['name']}")
            data = download_small_region_gfs(
                url=gfs_url,
                lat_range=region['lat_range'],
                lon_range=region['lon_range'],
                region_name=region['name']
            )
            
            if data is not None:
                # Display data info
                print(f"\nData information:")
                print(f"  Variables: {list(data.data_vars.keys())}")
                
                if 'temperature_2m' in data:
                    temp = data['temperature_2m']
                    if temp.size > 0:
                        temp_values = temp.values.flatten()
                        temp_values = temp_values[~np.isnan(temp_values)]
                        if len(temp_values) > 0:
                            temp_c = temp_values - 273.15
                            print(f"  Temperature: {temp_c.min():.1f} to {temp_c.max():.1f} °C")
                
                # Try to save (but skip if netcdf has issues)
                try:
                    filename = f"gfs_{region['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
                    data.to_netcdf(filename)
                    print(f"  Saved to: {filename}")
                except:
                    print(f"  Could not save as NetCDF, saving as CSV instead...")
                    # Save each variable as CSV
                    for var_name in data.data_vars:
                        var_data = data[var_name]
                        if var_data.ndim == 2:
                            df = pd.DataFrame(var_data.values, 
                                            index=var_data.lat.values,
                                            columns=var_data.lon.values)
                            csv_name = f"gfs_{region['name']}_{var_name}.csv"
                            df.to_csv(csv_name)
                            print(f"    Saved {var_name} to {csv_name}")
                
                break  # Stop after first successful region
        else:
            print("\nCould not download data for any region")
            print("Creating test data instead...")
            create_simple_test_data()
    
    else:
        print(f"\n{'='*60}")
        print("CANNOT ACCESS GFS - Creating test data")
        print("=" * 60)
        create_simple_test_data()
    
    print(f"\n{'='*60}")
    print("PROGRAM COMPLETE")
    print("=" * 60)

# Alternative: Minimal working script
def minimal_working_example():
    """Minimal working example that should always work."""
    print("Minimal GFS Download Example")
    print("=" * 40)
    
    # Use a URL that should work
    url = "https://www.ncei.noaa.gov/thredds/dodsC/model-gfs-g4-anl-files/2024/202401/20240101/gfsanl_4_20240101_0000_000.grb2"
    
    print(f"Trying NCEI server: {url}")
    
    try:
        # Try to open
        ds = xr.open_dataset(url)
        print("✓ Successfully opened dataset")
        
        # Get available variables
        print(f"\nAvailable variables (first 10):")
        for i, var in enumerate(list(ds.variables)[:10]):
            print(f"  {var}")
        
        # Try to get temperature
        if 'Temperature_surface' in ds:
            temp = ds['Temperature_surface']
            print(f"\nSurface temperature:")
            print(f"  Shape: {temp.shape}")
            print(f"  Units: {temp.attrs.get('units', 'unknown')}")
            
            # Select small region
            subset = temp.isel(time=0, lat=slice(100, 105), lon=slice(100, 105))
            print(f"  Small subset values:")
            print(subset.values)
        
        ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying ERDDAP as alternative...")
        
        # ERDDAP example
        erddap_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NCEP_Global_Best.csv?air[(2024-01-01T00:00:00Z)][(40):(42)][(-75):(-73)]"
        print(f"\nERDDAP query URL (copy to browser):")
        print(erddap_url)
        
        print("\nYou can download CSV data from the above URL")
        print("Then load it with pandas: pd.read_csv('filename.csv')")

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment to run minimal example
    # print("\n" + "="*60)
    # print("MINIMAL EXAMPLE")
    # print("="*60)
    # minimal_working_example()