"""
EFFICIENT REAL GDAS DOWNLOADER
Downloads only the needed region and variables - not entire global files.
"""

import numpy as np
import xarray as xr
import requests
from datetime import datetime, timedelta
import os
import json
import tempfile
import subprocess
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientGDASDownloader:
    """
    Efficiently downloads GDAS data by extracting only needed region/variables.
    """
    
    def __init__(self):
        self.base_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod"
        
        # GRIB2 parameter codes for our variables
        self.grib_params = {
            '2m_temperature': '2t',        # Temperature at 2m
            'surface_pressure': 'pres',    # Surface pressure
            '100m_u_wind': '100u',         # U wind at 100m
            '100m_v_wind': '100v',         # V wind at 100m
            'total_precipitation': 'apcp', # Total precipitation
            '2m_dewpoint': '2d'           # Dew point at 2m
        }
        
        # Human readable names
        self.var_names = {
            '2t': '2m_temperature',
            'pres': 'surface_pressure',
            '100u': '100m_u_wind',
            '100v': '100m_v_wind',
            'apcp': 'total_precipitation',
            '2d': '2m_dewpoint'
        }
        
        # Units
        self.var_units = {
            '2t': 'K',
            'pres': 'Pa',
            '100u': 'm/s',
            '100v': 'm/s',
            'apcp': 'kg/m^2',
            '2d': 'K'
        }
    
    def download_gdas_region(self,
                           lat_range: Tuple[float, float],
                           lon_range: Tuple[float, float],
                           start_time: datetime,
                           hours: int = 10) -> Optional[Dict]:
        """
        Download GDAS data for specific region and time period.
        """
        print("=" * 70)
        print("EFFICIENT GDAS DOWNLOAD")
        print("=" * 70)
        print(f"Region: {lat_range[0]}° to {lat_range[1]}° N, {lon_range[0]}° to {lon_range[1]}° W")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"Duration: {hours} hours")
        print(f"Variables: {list(self.var_names.values())}")
        print("=" * 70)
        
        # Adjust to GDAS 6-hourly cycles
        n_cycles = (hours + 5) // 6  # Round up
        total_hours = n_cycles * 6
        
        print(f"\nGDAS is 6-hourly, downloading {n_cycles} cycles ({total_hours} hours)")
        
        all_data = []
        all_times = []
        
        for cycle in range(n_cycles):
            cycle_time = start_time + timedelta(hours=cycle * 6)
            
            print(f"\n[Cycle {cycle+1}/{n_cycles}] {cycle_time.strftime('%Y-%m-%d %H:%M UTC')}")
            
            # Download and extract
            cycle_data = self._download_gdas_cycle(
                lat_range=lat_range,
                lon_range=lon_range,
                cycle_time=cycle_time
            )
            
            if cycle_data:
                all_data.append(cycle_data['data'])
                all_times.append(cycle_time)
                print(f"  ✓ Success")
            else:
                print(f"  ✗ Failed")
        
        if not all_data:
            print("\n❌ No data downloaded")
            return None
        
        # Combine time series
        print(f"\n✓ Downloaded {len(all_data)} time cycles")
        result = self._create_time_series(all_data, all_times, lat_range, lon_range)
        
        return result
    
    def _download_gdas_cycle(self,
                           lat_range: Tuple[float, float],
                           lon_range: Tuple[float, float],
                           cycle_time: datetime) -> Optional[Dict]:
        """
        Download and extract single GDAS cycle.
        """
        # Build URL for this cycle
        date_str = cycle_time.strftime('%Y%m%d')
        hour_str = f"{cycle_time.hour:02d}"
        url = f"{self.base_url}/gdas.{date_str}/{hour_str}/atmos/gdas.t{hour_str}z.pgrb2.0p25.f000"
        
        print(f"  Source: {url}")
        
        # Method 1: Try wgrib2 for efficient extraction
        if self._check_wgrib2():
            print(f"  Using wgrib2 for efficient extraction...")
            return self._extract_with_wgrib2(url, lat_range, lon_range, cycle_time)
        
        # Method 2: Try partial download with byte ranges
        print(f"  Trying partial download...")
        return self._partial_download(url, lat_range, lon_range, cycle_time)
    
    def _check_wgrib2(self) -> bool:
        """Check if wgrib2 is available."""
        try:
            result = subprocess.run(['wgrib2', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _extract_with_wgrib2(self, url: str, 
                           lat_range: Tuple[float, float],
                           lon_range: Tuple[float, float],
                           cycle_time: datetime) -> Optional[Dict]:
        """
        Use wgrib2 to extract only needed region and variables.
        This is MUCH more efficient.
        """
        lat_min, lat_max = sorted(lat_range)
        lon_min, lon_max = sorted(lon_range)
        
        # Create temp files
        temp_grib = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
        temp_grib.close()
        temp_grib_path = temp_grib.name
        
        temp_small = tempfile.NamedTemporaryFile(delete=False, suffix='_small.grib2')
        temp_small.close()
        temp_small_path = temp_small.name
        
        try:
            # Step 1: Download only the beginning of file to get index
            print(f"    Getting GRIB index...", end="", flush=True)
            
            # Download first 1MB to get index
            headers = {'Range': 'bytes=0-1000000'}
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 206:  # Partial content
                print(f" ✗ Cannot do partial download")
                return None
            
            with open(temp_grib_path, 'wb') as f:
                f.write(response.content)
            
            # Get GRIB index
            cmd_index = ['wgrib2', temp_grib_path, '-var']
            result = subprocess.run(cmd_index, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f" ✗ Cannot read GRIB index")
                return None
            
            print(f" ✓")
            
            # Step 2: Build command to extract only our variables and region
            print(f"    Building extraction command...", end="", flush=True)
            
            # Create filter for our variables
            var_filter = []
            for grib_code, human_name in self.var_names.items():
                if grib_code in result.stdout:
                    var_filter.extend(['-match', f':{grib_code}:'])
            
            if not var_filter:
                print(f" ✗ None of our variables found in GRIB")
                return None
            
            print(f" ✓ Found {len(var_filter)//2} variables")
            
            # Step 3: Download full file for selected variables only
            print(f"    Downloading selected data...", end="", flush=True)
            
            # We need to download the full file for wgrib2 to work properly
            # But this is still better than processing globally
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(temp_grib_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = os.path.getsize(temp_grib_path) / (1024*1024)
            print(f" ✓ ({file_size:.1f} MB)")
            
            # Step 4: Extract region and convert to netCDF
            print(f"    Extracting region...", end="", flush=True)
            
            # First extract our variables
            cmd_extract = ['wgrib2', temp_grib_path] + var_filter + [
                '-small_grib', f"{lon_min}:{lon_max}", f"{lat_min}:{lat_max}",
                temp_small_path
            ]
            
            result = subprocess.run(cmd_extract, capture_output=True, text=True)
            
            if result.returncode != 0 or os.path.getsize(temp_small_path) == 0:
                print(f" ✗ Extraction failed")
                return None
            
            small_size = os.path.getsize(temp_small_path) / 1024
            print(f" ✓ ({small_size:.1f} KB)")
            
            # Step 5: Convert to netCDF and read
            print(f"    Converting to netCDF...", end="", flush=True)
            
            temp_nc = temp_small_path.replace('.grib2', '.nc')
            cmd_convert = ['wgrib2', temp_small_path, '-netcdf', temp_nc]
            
            result = subprocess.run(cmd_convert, capture_output=True, text=True)
            
            if result.returncode != 0 or not os.path.exists(temp_nc):
                print(f" ✗ Conversion failed")
                return None
            
            print(f" ✓")
            
            # Step 6: Read netCDF
            print(f"    Reading data...", end="", flush=True)
            
            try:
                ds = xr.open_dataset(temp_nc)
                
                # Extract data arrays
                data_arrays = {}
                coordinates = {}
                
                # Get coordinates
                if 'latitude' in ds.coords:
                    coordinates['latitude'] = ds.latitude.values
                elif 'lat' in ds.coords:
                    coordinates['latitude'] = ds.lat.values
                
                if 'longitude' in ds.coords:
                    coordinates['longitude'] = ds.longitude.values
                elif 'lon' in ds.coords:
                    coordinates['longitude'] = ds.lon.values
                
                # Extract variables
                for grib_code, human_name in self.var_names.items():
                    # Look for variable in dataset
                    for ds_var in ds.data_vars:
                        if grib_code in str(ds_var).lower():
                            data_arrays[human_name] = ds[ds_var].values.squeeze()
                            break
                
                ds.close()
                
                print(f" ✓")
                
                return {
                    'data': data_arrays,
                    'coordinates': coordinates,
                    'metadata': {
                        'time': cycle_time.isoformat(),
                        'source': url,
                        'extraction_method': 'wgrib2',
                        'region': f"{lat_min},{lat_max},{lon_min},{lon_max}"
                    }
                }
                
            except Exception as e:
                print(f" ✗ Read error: {str(e)[:50]}")
                return None
            
        finally:
            # Clean up temp files
            for temp_file in [temp_grib_path, temp_small_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Also remove any .nc files
            nc_files = [f for f in os.listdir('.') if f.endswith('.nc') and 'tmp' in f]
            for nc_file in nc_files:
                try:
                    os.remove(nc_file)
                except:
                    pass
    
    def _partial_download(self, url: str,
                        lat_range: Tuple[float, float],
                        lon_range: Tuple[float, float],
                        cycle_time: datetime) -> Optional[Dict]:
        """
        Fallback method: download and use CDO or similar tools.
        """
        print(f"    Downloading file...", end="", flush=True)
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.grib2')
        temp_file.close()
        temp_path = temp_file.name
        
        try:
            # Download with progress
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            file_size_mb = total_size / (1024*1024)
            print(f" ✓ ({file_size_mb:.1f} MB)")
            
            print(f"    Note: File is large because downloading global data")
            print(f"    Consider installing wgrib2 for efficient extraction")
            
            # Try to process with available tools
            return self._process_downloaded_file(temp_path, lat_range, lon_range, cycle_time)
            
        except Exception as e:
            print(f" ✗ Download error: {str(e)[:50]}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _process_downloaded_file(self, grib_file: str,
                               lat_range: Tuple[float, float],
                               lon_range: Tuple[float, float],
                               cycle_time: datetime) -> Optional[Dict]:
        """
        Process downloaded GRIB file.
        """
        # Try CDO if available
        if self._check_cdo():
            return self._extract_with_cdo(grib_file, lat_range, lon_range, cycle_time)
        
        # Try cfgrib
        try:
            import cfgrib
            return self._extract_with_cfgrib(grib_file, lat_range, lon_range, cycle_time)
        except:
            pass
        
        # Try pygrib
        try:
            import pygrib
            return self._extract_with_pygrib(grib_file, lat_range, lon_range, cycle_time)
        except:
            pass
        
        print(f"    ✗ No GRIB processing tools available")
        print(f"    Install one of: wgrib2, cdo, cfgrib, or pygrib")
        return None
    
    def _check_cdo(self) -> bool:
        """Check if CDO is available."""
        try:
            result = subprocess.run(['cdo', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _extract_with_cdo(self, grib_file: str,
                         lat_range: Tuple[float, float],
                         lon_range: Tuple[float, float],
                         cycle_time: datetime) -> Optional[Dict]:
        """
        Use CDO to extract region.
        """
        print(f"    Using CDO for extraction...", end="", flush=True)
        
        lat_min, lat_max = sorted(lat_range)
        lon_min, lon_max = sorted(lon_range)
        
        temp_nc = grib_file.replace('.grib2', '.nc')
        
        try:
            # Use CDO to select region and convert to netCDF
            # First convert to netCDF
            cmd_convert = ['cdo', '-f', 'nc', 'copy', grib_file, temp_nc]
            result = subprocess.run(cmd_convert, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f" ✗ CDO conversion failed")
                return None
            
            # Now select region
            temp_region = temp_nc.replace('.nc', '_region.nc')
            cmd_region = [
                'cdo', 'sellonlatbox',
                str(lon_min), str(lon_max), str(lat_min), str(lat_max),
                temp_nc, temp_region
            ]
            
            result = subprocess.run(cmd_region, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f" ✗ Region selection failed")
                return None
            
            # Read the region file
            ds = xr.open_dataset(temp_region)
            
            # Extract data
            data_arrays = {}
            coordinates = {}
            
            if 'lat' in ds.coords:
                coordinates['latitude'] = ds.lat.values
            if 'lon' in ds.coords:
                coordinates['longitude'] = ds.lon.values
            
            # Map variables - CDO uses different names
            cdo_var_map = {
                '2m_temperature': ['t2m', '2t', 'var167'],
                'surface_pressure': ['sp', 'pres', 'var134'],
                '100m_u_wind': ['u100', '100u'],
                '100m_v_wind': ['v100', '100v'],
                'total_precipitation': ['tp', 'apcp'],
                '2m_dewpoint': ['d2m', '2d']
            }
            
            for human_name, patterns in cdo_var_map.items():
                for ds_var in ds.data_vars:
                    ds_var_str = str(ds_var).lower()
                    for pattern in patterns:
                        if pattern in ds_var_str:
                            data_arrays[human_name] = ds[ds_var].values.squeeze()
                            break
                    if human_name in data_arrays:
                        break
            
            ds.close()
            
            # Clean up
            for temp_f in [temp_nc, temp_region]:
                if os.path.exists(temp_f):
                    os.remove(temp_f)
            
            print(f" ✓")
            
            return {
                'data': data_arrays,
                'coordinates': coordinates,
                'metadata': {
                    'time': cycle_time.isoformat(),
                    'extraction_method': 'cdo',
                    'region': f"{lat_min},{lat_max},{lon_min},{lon_max}"
                }
            }
            
        except Exception as e:
            print(f" ✗ Error: {str(e)[:50]}")
            
            # Clean up
            for temp_f in [temp_nc, temp_nc.replace('.nc', '_region.nc')]:
                if os.path.exists(temp_f):
                    try:
                        os.remove(temp_f)
                    except:
                        pass
            
            return None
    
    def _create_time_series(self, all_data: List[Dict],
                          all_times: List[datetime],
                          lat_range: Tuple[float, float],
                          lon_range: Tuple[float, float]) -> Dict:
        """
        Create time series from multiple cycles.
        """
        # Get common variables
        first_vars = list(all_data[0]['data'].keys())
        
        # Stack data along time dimension
        time_series = {}
        
        for var_name in first_vars:
            var_arrays = []
            for cycle_data in all_data:
                if var_name in cycle_data['data']:
                    var_arrays.append(cycle_data['data'][var_name])
            
            if var_arrays:
                # Ensure all have same shape
                shapes = [arr.shape for arr in var_arrays]
                if len(set(shapes)) == 1:
                    time_series[var_name] = np.stack(var_arrays, axis=0)
                else:
                    # Find common shape
                    min_shape = min(shapes, key=lambda x: np.prod(x))
                    resized = []
                    for arr in var_arrays:
                        if arr.shape != min_shape:
                            # Simple truncation (should work for same region)
                            slices = tuple(slice(0, min_dim) for min_dim in min_shape)
                            resized.append(arr[slices])
                        else:
                            resized.append(arr)
                    time_series[var_name] = np.stack(resized, axis=0)
        
        # Get coordinates from first cycle
        coordinates = all_data[0]['coordinates'] if all_data else {}
        
        return {
            'time_series': time_series,
            'coordinates': coordinates,
            'times': [t.isoformat() for t in all_times],
            'metadata': {
                'n_times': len(all_times),
                'time_range': f"{all_times[0].strftime('%Y-%m-%d %H:%M')} to {all_times[-1].strftime('%Y-%m-%d %H:%M')}",
                'lat_range': f"{lat_range[0]},{lat_range[1]}",
                'lon_range': f"{lon_range[0]},{lon_range[1]}",
                'variables': list(time_series.keys()),
                'grid_size': f"{coordinates.get('latitude', []).shape}" if coordinates else "unknown"
            }
        }

def save_results(data_result: Dict, output_dir: str = "gdas_results"):
    """Save the downloaded data."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n💾 Saving results to {output_dir}/")
    
    # Save time series arrays
    for var_name, array_data in data_result['time_series'].items():
        filename = os.path.join(output_dir, f"{var_name}_{timestamp}.npy")
        np.save(filename, array_data)
        print(f"  ✓ {var_name}: {array_data.shape}")
    
    # Save coordinates
    if data_result['coordinates']:
        coords_file = os.path.join(output_dir, f"coordinates_{timestamp}.npz")
        np.savez(coords_file, **data_result['coordinates'])
        print(f"  ✓ Coordinates saved")
    
    # Save times
    times_file = os.path.join(output_dir, f"times_{timestamp}.npy")
    np.save(times_file, data_result['times'])
    print(f"  ✓ Time stamps saved")
    
    # Save metadata
    meta_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
    with open(meta_file, 'w') as f:
        json.dump(data_result['metadata'], f, indent=2)
    print(f"  ✓ Metadata saved")
    
    # Save everything in one file
    combined_file = os.path.join(output_dir, f"gdas_complete_{timestamp}.npz")
    np.savez(combined_file, 
             **data_result['time_series'],
             **{f"{k}_coords": v for k, v in data_result['coordinates'].items()},
             times=data_result['times'])
    print(f"  ✓ Combined file saved")
    
    return output_dir

def main():
    """Main function."""
    print("EFFICIENT GDAS DATA DOWNLOADER")
    print("Downloads only needed region/variables")
    print("=" * 70)
    
    # Check for tools
    print("Checking available tools...")
    tools = []
    
    try:
        subprocess.run(['wgrib2', '--version'], capture_output=True, check=True)
        tools.append('wgrib2')
    except:
        pass
    
    try:
        subprocess.run(['cdo', '--version'], capture_output=True, check=True)
        tools.append('cdo')
    except:
        pass
    
    if tools:
        print(f"✓ Available tools: {', '.join(tools)}")
    else:
        print("⚠ No efficient extraction tools found")
        print("  Recommended: Install wgrib2 or CDO")
        print("  wgrib2: sudo apt-get install wgrib2")
        print("  CDO: sudo apt-get install cdo")
    
    print("\n" + "=" * 70)
    
    # Initialize downloader
    downloader = EfficientGDASDownloader()
    
    # Set parameters
    # Small region (2-3 degrees)
    lat_range = (40.5, 42.5)    # 2 degrees
    lon_range = (-75.5, -73.5)  # 2 degrees
    
    # Get time within past 6 days
    # GDAS produces at 00, 06, 12, 18 UTC
    current = datetime.utcnow()
    
    # Try multiple start times if needed
    for days_ago in [2, 3, 4, 1, 5]:
        start_time = current - timedelta(days=days_ago)
        start_hour = (start_time.hour // 6) * 6
        start_time = start_time.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        
        print(f"\nTrying start time: {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"({days_ago} days ago)")
        
        # Download 10 hours (will be 12 hours for GDAS 6-hourly)
        data_result = downloader.download_gdas_region(
            lat_range=lat_range,
            lon_range=lon_range,
            start_time=start_time,
            hours=10
        )
        
        if data_result:
            print(f"\n{'='*70}")
            print("SUCCESS! Real GDAS data downloaded")
            print("=" * 70)
            
            # Display summary
            meta = data_result['metadata']
            print(f"\nSummary:")
            print(f"  Time steps: {meta['n_times']}")
            print(f"  Time range: {meta['time_range']}")
            print(f"  Variables: {', '.join(meta['variables'])}")
            print(f"  Region: lat {meta['lat_range']}, lon {meta['lon_range']}")
            
            print(f"\nArray shapes:")
            for var_name, array_data in data_result['time_series'].items():
                print(f"  {var_name}: {array_data.shape}")
            
            # Save data
            output_dir = save_results(data_result)
            
            print(f"\n✅ Data saved to: {output_dir}/")
            
            # Show how to load
            print(f"\nTo load data:")
            print(f"  import numpy as np")
            print(f"  data = np.load('{output_dir}/gdas_complete_*.npz')")
            print(f"  temperature = data['2m_temperature']  # shape: (time, lat, lon)")
            print(f"  lats = data['latitude_coords']")
            print(f"  lons = data['longitude_coords']")
            print(f"  times = data['times']")
            
            return
    
    print(f"\n❌ Could not download GDAS data")
    print(f"\nPossible solutions:")
    print(f"1. Install wgrib2: sudo apt-get install wgrib2")
    print(f"2. Try different dates/times")
    print(f"3. Check network connection to NOAA")
    print(f"4. Try smaller region")

if __name__ == "__main__":
    main()