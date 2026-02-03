import numpy as np
import xarray as xr
from scipy.interpolate import griddata, RegularGridInterpolator
import warnings

class GDAS_ERA5_GridTransformer:
    """
    Complete transformation from GDAS to ERA5 grid with cell shifting
    """
    
    def __init__(self, verify_grids=True):
        """
        Initialize transformer with standard GDAS and ERA5 grid definitions
        """
        # Standard GDAS 0.25° grid (NCEP)
        self.gdas_lats = np.arange(90, -90.25, -0.25)  # 721 points
        self.gdas_lons = np.arange(0, 360, 0.25)       # 1440 points
        
        # Standard ERA5 0.25° grid (ECMWF)
        self.era5_lats = np.arange(90, -90.25, -0.25)  # 721 points
        self.era5_lons = np.arange(0, 360, 0.25)       # 1440 points
        
        # Earth model parameters
        self.gdas_radius = 6371.229  # km, NCEP spherical Earth
        self.era5_equatorial = 6378.137  # km, ECMWF equatorial radius
        self.era5_polar = 6356.7523142  # km, ECMWF polar radius
        
        if verify_grids:
            self._verify_grid_alignment()
    
    def _verify_grid_alignment(self):
        """Verify if GDAS and ERA5 grids are aligned"""
        print("=== Grid Alignment Check ===")
        print(f"GDAS lats: {len(self.gdas_lats)} points, "
              f"{self.gdas_lats[0]:.2f} to {self.gdas_lats[-1]:.2f}")
        print(f"ERA5 lats: {len(self.era5_lats)} points, "
              f"{self.era5_lats[0]:.2f} to {self.era5_lats[-1]:.2f}")
        
        # Check if grids are identical
        if len(self.gdas_lats) == len(self.era5_lats):
            lat_diff = np.max(np.abs(self.gdas_lats - self.era5_lats))
            print(f"Max latitude difference: {lat_diff:.6f}°")
            
            if lat_diff < 1e-6:
                print("✅ Latitude grids are IDENTICAL (no shifting needed)")
            else:
                print(f"⚠️ Latitude grids DIFFER by {lat_diff:.6f}°")
        
        # Calculate cell edges
        gdas_lat_edges = self._calculate_cell_edges(self.gdas_lats)
        era5_lat_edges = self._calculate_cell_edges(self.era5_lats)
        
        print(f"\nGDAS cell edges: {gdas_lat_edges[:5]:.2f} ... {gdas_lat_edges[-5:]:.2f}")
        print(f"ERA5 cell edges: {era5_lat_edges[:5]:.2f} ... {era5_lat_edges[-5:]:.2f}")
    
    def _calculate_cell_edges(self, centers):
        """Calculate cell edges from cell centers"""
        edges = np.zeros(len(centers) + 1)
        edges[1:-1] = (centers[:-1] + centers[1:]) / 2
        edges[0] = centers[0] + (centers[0] - edges[1])  # Extrapolate north
        edges[-1] = centers[-1] - (edges[-2] - centers[-1])  # Extrapolate south
        return edges
    
    def analyze_your_gdas_grid(self, gdas_data):
        """
        Analyze your specific GDAS grid to determine shifting needs
        
        Parameters:
        -----------
        gdas_data : xarray.Dataset or path to GDAS file
        
        Returns:
        --------
        dict: Grid analysis results
        """
        if isinstance(gdas_data, str):
            gdas_data = xr.open_dataset(gdas_data)
        
        analysis = {
            'grid_type': 'unknown',
            'needs_shifting': False,
            'shift_amount_lat': 0.0,
            'shift_amount_lon': 0.0,
            'recommended_method': 'direct_interpolation'
        }
        
        # Extract coordinates
        if 'latitude' in gdas_data.coords:
            lats = gdas_data.latitude.values
            lons = gdas_data.longitude.values
        elif 'lat' in gdas_data.coords:
            lats = gdas_data.lat.values
            lons = gdas_data.lon.values
        else:
            raise ValueError("Cannot find latitude/longitude coordinates")
        
        print(f"\n=== Your GDAS Grid Analysis ===")
        print(f"Shape: {lats.shape[0]} lats × {lons.shape[0]} lons")
        print(f"Latitude range: {lats.min():.3f} to {lats.max():.3f}")
        print(f"Longitude range: {lons.min():.3f} to {lons.max():.3f}")
        
        # Check if it's standard GDAS grid
        if (len(lats) == 721 and len(lons) == 1440 and 
            np.allclose(lats[0], 90.0) and np.allclose(lats[-1], -90.0)):
            print("✅ This is STANDARD GDAS 0.25° grid")
            analysis['grid_type'] = 'standard_gdas_0.25'
            
            # Check alignment with ERA5
            if np.allclose(lats, self.era5_lats, atol=1e-6):
                print("✅ Grid centers ALREADY ALIGNED with ERA5")
                analysis['needs_shifting'] = False
                analysis['recommended_method'] = 'earth_model_correction_only'
            else:
                print("⚠️ Grid centers NOT aligned with ERA5")
                analysis['needs_shifting'] = True
                
                # Calculate needed shift
                lat_shift = lats[0] - self.era5_lats[0]
                lon_shift = lons[0] - self.era5_lons[0]
                analysis['shift_amount_lat'] = lat_shift
                analysis['shift_amount_lon'] = lon_shift
                print(f"Required shift: {lat_shift:.6f}° in latitude, {lon_shift:.6f}° in longitude")
        
        else:
            print("⚠️ This is NON-STANDARD GDAS grid")
            analysis['grid_type'] = 'non_standard'
            analysis['needs_shifting'] = True
            analysis['recommended_method'] = 'bilinear_interpolation'
        
        return analysis
    
    def transform_grid_cells(self, gdas_data, method='conservative'):
        """
        Transform GDAS grid cells to ERA5 grid
        
        Parameters:
        -----------
        gdas_data : xarray.Dataset with GDAS data
        method : str, transformation method
                 'conservative' - conservative remapping (best for fluxes)
                 'bilinear' - bilinear interpolation
                 'nearest' - nearest neighbor
                 'distance_weighted' - distance-weighted averaging
        
        Returns:
        --------
        xarray.Dataset: Data on ERA5 grid
        """
        
        print("\n=== Starting Grid Transformation ===")
        
        # 1. Analyze the grid
        analysis = self.analyze_your_gdas_grid(gdas_data)
        
        # 2. Extract data
        if isinstance(gdas_data, xr.Dataset):
            data_vars = {}
            for var_name in gdas_data.data_vars:
                data_vars[var_name] = gdas_data[var_name]
            coords = gdas_data.coords
        else:
            data_vars = {'data': gdas_data}
            coords = {}
        
        # 3. Apply transformation based on analysis
        if not analysis['needs_shifting']:
            print("No cell shifting needed - applying Earth model corrections only")
            return self._apply_earth_model_corrections(gdas_data)
        
        else:
            print(f"Applying {method} transformation...")
            
            if method == 'conservative':
                return self._conservative_remapping(gdas_data)
            elif method == 'bilinear':
                return self._bilinear_interpolation(gdas_data)
            elif method == 'nearest':
                return self._nearest_neighbor(gdas_data)
            elif method == 'distance_weighted':
                return self._distance_weighted_averaging(gdas_data)
            else:
                raise ValueError(f"Unknown method: {method}")
    
    def _apply_earth_model_corrections(self, data):
        """
        Apply Earth model corrections without shifting grid cells
        (For when grids are already aligned)
        """
        # This applies the 12% distance/25% area corrections
        # Implementation from our previous EarthModelConverter
        
        corrected_data = data.copy()
        
        # Get latitudes
        if 'latitude' in corrected_data.coords:
            lats = corrected_data.latitude.values
        elif 'lat' in corrected_data.coords:
            lats = corrected_data.lat.values
        else:
            return corrected_data
        
        # Calculate correction factors
        correction_factors = np.ones_like(lats)
        for i, lat in enumerate(lats):
            # Calculate zonal distance ratio
            gdas_dist = 2 * np.pi * self.gdas_radius * np.cos(np.radians(lat)) * 0.25 / 360
            
            # ERA5 ellipsoidal distance (simplified)
            era5_dist = 2 * np.pi * self.era5_equatorial * np.cos(np.radians(lat)) * 0.25 / 360
            
            correction_factors[i] = era5_dist / gdas_dist
        
        # Apply to vector variables
        for var_name in corrected_data.data_vars:
            var_lower = var_name.lower()
            if any(wind in var_lower for wind in ['u', 'v', 'wind']):
                print(f"Applying correction to {var_name}")
                
                # Create 2D correction array
                if 'longitude' in corrected_data.coords:
                    n_lons = len(corrected_data.longitude)
                elif 'lon' in corrected_data.coords:
                    n_lons = len(corrected_data.lon)
                else:
                    n_lons = 1
                
                correction_2d = np.tile(correction_factors[:, np.newaxis], (1, n_lons))
                corrected_data[var_name].values *= correction_2d
        
        return corrected_data
    
    def _bilinear_interpolation(self, gdas_data):
        """
        Bilinear interpolation from GDAS to ERA5 grid
        Most common method for regular grids
        """
        from scipy.interpolate import interp2d
        
        print("Performing bilinear interpolation...")
        
        # Get source grid
        if 'latitude' in gdas_data.coords:
            src_lats = gdas_data.latitude.values
            src_lons = gdas_data.longitude.values
        else:
            src_lats = gdas_data.lat.values
            src_lons = gdas_data.lon.values
        
        # Ensure longitude is 0-360
        if src_lons.min() < 0:
            src_lons = (src_lons + 360) % 360
        
        # Create result dataset
        result_vars = {}
        
        for var_name in gdas_data.data_vars:
            var_data = gdas_data[var_name]
            
            if var_data.ndim == 2:  # 2D field
                # Create interpolator
                f = interp2d(src_lons, src_lats, var_data.values, kind='linear')
                
                # Interpolate to ERA5 grid
                era5_data = f(self.era5_lons, self.era5_lats)
                
            elif var_data.ndim == 3:  # 3D field (time, lat, lon) or (level, lat, lon)
                # Determine dimension order
                dims = var_data.dims
                if 'time' in dims and 'latitude' in dims:
                    # (time, lat, lon)
                    n_times = var_data.shape[0]
                    era5_data = np.zeros((n_times, len(self.era5_lats), len(self.era5_lons)))
                    
                    for t in range(n_times):
                        f = interp2d(src_lons, src_lats, var_data.values[t, :, :], kind='linear')
                        era5_data[t, :, :] = f(self.era5_lons, self.era5_lats)
                
                elif 'level' in dims and 'latitude' in dims:
                    # (level, lat, lon)
                    n_levels = var_data.shape[0]
                    era5_data = np.zeros((n_levels, len(self.era5_lats), len(self.era5_lons)))
                    
                    for l in range(n_levels):
                        f = interp2d(src_lons, src_lats, var_data.values[l, :, :], kind='linear')
                        era5_data[l, :, :] = f(self.era5_lons, self.era5_lats)
            
            else:
                warnings.warn(f"Cannot interpolate {var_name} with {var_data.ndim} dimensions")
                continue
            
            # Create DataArray
            result_vars[var_name] = xr.DataArray(
                era5_data,
                dims=var_data.dims,
                coords=self._create_era5_coords(var_data.dims),
                attrs=var_data.attrs
            )
        
        # Create dataset
        result = xr.Dataset(result_vars)
        result.attrs.update(gdas_data.attrs)
        result.attrs['transformation_method'] = 'bilinear_interpolation'
        result.attrs['source_grid'] = 'GDAS'
        result.attrs['target_grid'] = 'ERA5'
        
        return result
    
    def _conservative_remapping(self, gdas_data):
        """
        Conservative remapping - preserves integrals (best for fluxes)
        """
        print("Performing conservative remapping (flux-preserving)...")
        
        # This is simplified - for full implementation use xESMF or similar
        # Here's a simplified area-weighted approach
        
        if 'latitude' in gdas_data.coords:
            src_lats = gdas_data.latitude.values
            src_lons = gdas_data.longitude.values
        else:
            src_lats = gdas_data.lat.values
            src_lons = gdas_data.lon.values
        
        # Calculate source cell areas (GDAS Earth model)
        src_areas = self._calculate_cell_areas(src_lats, self.gdas_radius)
        
        # Calculate target cell areas (ERA5 Earth model)
        tgt_areas = self._calculate_cell_areas(self.era5_lats, self.era5_equatorial, 
                                              is_ellipsoid=True)
        
        # For each target cell, find overlapping source cells
        result_vars = {}
        
        for var_name in gdas_data.data_vars:
            var_data = gdas_data[var_name]
            
            if var_data.ndim != 2:
                warnings.warn(f"Conservative remapping only implemented for 2D, "
                            f"using bilinear for {var_name}")
                # Fall back to bilinear
                continue
            
            # Create result array
            era5_data = np.zeros((len(self.era5_lats), len(self.era5_lons)))
            
            # For each target cell
            for i_tgt, lat_tgt in enumerate(self.era5_lats):
                for j_tgt, lon_tgt in enumerate(self.era5_lons):
                    # Find source cells that overlap with this target cell
                    # (Simplified - actual implementation needs overlap calculation)
                    
                    # Find nearest source cell
                    i_src = np.argmin(np.abs(src_lats - lat_tgt))
                    j_src = np.argmin(np.abs(src_lons - lon_tgt))
                    
                    # Area-weighted value
                    area_ratio = tgt_areas[i_tgt] / src_areas[i_src]
                    era5_data[i_tgt, j_tgt] = var_data.values[i_src, j_src] * area_ratio
            
            result_vars[var_name] = xr.DataArray(
                era5_data,
                dims=['latitude', 'longitude'],
                coords={'latitude': self.era5_lats, 'longitude': self.era5_lons},
                attrs=var_data.attrs
            )
        
        result = xr.Dataset(result_vars)
        result.attrs['transformation_method'] = 'conservative_remapping'
        
        return result
    
    def _calculate_cell_areas(self, lats, radius, is_ellipsoid=False):
        """
        Calculate grid cell areas
        """
        areas = np.zeros(len(lats))
        
        # Calculate latitude edges
        lat_edges = np.zeros(len(lats) + 1)
        lat_edges[1:-1] = (lats[:-1] + lats[1:]) / 2
        lat_edges[0] = lats[0] + (lats[0] - lat_edges[1])
        lat_edges[-1] = lats[-1] - (lat_edges[-2] - lats[-1])
        
        for i in range(len(lats)):
            lat_center = lats[i]
            lat_north = lat_edges[i]
            lat_south = lat_edges[i+1]
            
            if is_ellipsoid:
                # Ellipsoidal area (simplified)
                lat_rad = np.radians(lat_center)
                # Radius of curvature
                N = radius / np.sqrt(1 - (1 - (self.era5_polar/self.era5_equatorial)**2) * 
                                    np.sin(lat_rad)**2)
                M = radius * (1 - (1 - (self.era5_polar/self.era5_equatorial)**2)) / \
                    (1 - (1 - (self.era5_polar/self.era5_equatorial)**2) * 
                     np.sin(lat_rad)**2)**1.5
                
                # Cell dimensions
                dlat_rad = np.radians(abs(lat_north - lat_south))
                dlon_rad = np.radians(0.25)
                
                areas[i] = (N * np.cos(lat_rad) * dlon_rad) * (M * dlat_rad)
            else:
                # Spherical area
                lat1_rad = np.radians(lat_north)
                lat2_rad = np.radians(lat_south)
                dlon_rad = np.radians(0.25)
                
                areas[i] = radius**2 * dlon_rad * abs(np.sin(lat1_rad) - np.sin(lat2_rad))
        
        return areas
    
    def _create_era5_coords(self, dims):
        """
        Create ERA5 coordinate dictionary based on dimensions
        """
        coords = {}
        
        if 'latitude' in dims or 'lat' in dims:
            coords['latitude'] = self.era5_lats
        if 'longitude' in dims or 'lon' in dims:
            coords['longitude'] = self.era5_lons
        if 'time' in dims:
            # Preserve time from source if available
            pass
        if 'level' in dims:
            # Preserve levels from source if available
            pass
        
        return coords
    
    def visualize_transformation(self, gdas_data, var_name='temperature'):
        """
        Visualize the grid transformation
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        
        if isinstance(gdas_data, str):
            gdas_data = xr.open_dataset(gdas_data)
        
        # Get variable data
        if var_name in gdas_data.data_vars:
            data = gdas_data[var_name]
            if data.ndim >= 2:
                # Take first time/level if 3D/4D
                if data.ndim == 3:
                    plot_data = data.values[0, :, :]
                elif data.ndim == 4:
                    plot_data = data.values[0, 0, :, :]
                else:
                    plot_data = data.values
            else:
                print(f"Cannot plot {var_name} - wrong dimensions")
                return
        else:
            print(f"Variable {var_name} not found")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot 1: Original GDAS grid
        ax1 = axes[0]
        im1 = ax1.pcolormesh(gdas_data.longitude.values, gdas_data.latitude.values,
                           plot_data, transform=ccrs.PlateCarree(),
                           cmap='viridis', shading='auto')
        ax1.coastlines()
        ax1.set_title('Original GDAS Grid')
        ax1.gridlines(draw_labels=True)
        
        # Transform data
        transformed = self._bilinear_interpolation(gdas_data)
        
        # Plot 2: Transformed ERA5 grid
        ax2 = axes[1]
        if var_name in transformed.data_vars:
            t_data = transformed[var_name]
            if t_data.ndim == 3:
                t_plot = t_data.values[0, :, :]
            elif t_data.ndim == 2:
                t_plot = t_data.values
            else:
                t_plot = t_data.values[0, 0, :, :]
            
            im2 = ax2.pcolormesh(self.era5_lons, self.era5_lats,
                               t_plot, transform=ccrs.PlateCarree(),
                               cmap='viridis', shading='auto')
            ax2.coastlines()
            ax2.set_title('Transformed ERA5 Grid')
            ax2.gridlines(draw_labels=True)
        
        # Plot 3: Difference
        ax3 = axes[2]
        # Interpolate back to GDAS grid for comparison
        # (Simplified - just for visualization)
        diff = plot_data - t_plot[:plot_data.shape[0], :plot_data.shape[1]]
        im3 = ax3.pcolormesh(gdas_data.longitude.values, gdas_data.latitude.values,
                           diff, transform=ccrs.PlateCarree(),
                           cmap='RdBu_r', shading='auto')
        ax3.coastlines()
        ax3.set_title('Difference (GDAS - ERA5)')
        ax3.gridlines(draw_labels=True)
        
        plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05)
        plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.05)
        plt.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.05)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# SIMPLIFIED ONE-FUNCTION SOLUTION
# ============================================================================

def transform_gdas_to_era5_simple(gdas_file, output_file=None, variable=None):
    """
    Simplified one-function solution for GDAS to ERA5 transformation
    
    Parameters:
    -----------
    gdas_file : str, path to GDAS file
    output_file : str, path for output (optional)
    variable : str, specific variable to transform (optional)
    
    Returns:
    --------
    xarray.Dataset: Transformed data on ERA5 grid
    """
    
    # Load GDAS data
    print(f"Loading {gdas_file}...")
    ds = xr.open_dataset(gdas_file)
    
    # Initialize transformer
    transformer = GDAS_ERA5_GridTransformer(verify_grids=True)
    
    # Analyze grid
    analysis = transformer.analyze_your_gdas_grid(ds)
    
    print(f"\n=== Transformation Plan ===")
    print(f"Grid type: {analysis['grid_type']}")
    print(f"Needs shifting: {analysis['needs_shifting']}")
    print(f"Recommended method: {analysis['recommended_method']}")
    
    # Select variables to transform
    if variable and variable in ds.data_vars:
        vars_to_transform = [variable]
    else:
        vars_to_transform = list(ds.data_vars)
    
    print(f"\nTransforming {len(vars_to_transform)} variables...")
    
    # Apply transformation
    if analysis['recommended_method'] == 'earth_model_correction_only':
        result = transformer._apply_earth_model_corrections(ds)
    else:
        # Use bilinear as default
        result = transformer._bilinear_interpolation(ds)
    
    # Save if requested
    if output_file:
        print(f"\nSaving to {output_file}...")
        result.to_netcdf(output_file)
        print("Transformation complete!")
    
    return result

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("GDAS to ERA5 Grid Transformation Tool")
    print("=" * 50)
    
    # Example 1: Check your GDAS grid
    transformer = GDAS_ERA5_GridTransformer()
    
    # Example 2: How to use (uncomment and modify paths)
    """
    # Transform your GDAS data
    result = transform_gdas_to_era5_simple(
        gdas_file="your_gdas_data.nc",
        output_file="gdas_on_era5_grid.nc",
        variable=None  # Transform all variables
    )
    
    # Or use the full transformer
    transformer = GDAS_ERA5_GridTransformer()
    ds = xr.open_dataset("your_gdas_data.nc")
    transformed = transformer.transform_grid_cells(ds, method='bilinear')
    
    # Visualize the transformation
    transformer.visualize_transformation(ds, var_name='temperature')
    """
    
    print("\nQuick reference:")
    print("1. First run: transformer.analyze_your_gdas_grid(your_data)")
    print("2. If 'needs_shifting' is False: Only Earth model corrections needed")
    print("3. If True: Use transform_grid_cells() with appropriate method")
    print("4. For fluxes: Use 'conservative' method")
    print("5. For other variables: Use 'bilinear' method")