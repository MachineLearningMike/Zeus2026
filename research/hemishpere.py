import numpy as np
from scipy.interpolate import RegularGridInterpolator

def latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=300, R=6371):
    """
    Convert lat-lon gridded data to Fibonacci points on the sphere.
    
    Parameters:
    - lat_grid: 1D array of latitudes (degrees)
    - lon_grid: 1D array of longitudes (degrees)
    - data_grid: 2D array of shape (len(lat_grid), len(lon_grid))
    - d_target: approximate spacing between Fibonacci points (km)
    - R: radius of the sphere (Earth radius in km by default)
    
    Returns:
    - lat_fib: 1D array of Fibonacci point latitudes (degrees)
    - lon_fib: 1D array of Fibonacci point longitudes (degrees)
    - data_fib: 1D array of interpolated data at Fibonacci points
    - x, y, z: Cartesian coordinates of Fibonacci points (optional, for 3D plotting)
    """
    
    # --- Number of Fibonacci points ---
    A = 4 * np.pi * R**2
    N = int(A / d_target**2)
    
    # --- Fibonacci spherical coordinates ---
    indices = np.arange(0, N, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/N)           # polar angle
    theta = np.pi * (1 + 5**0.5) * indices    # azimuthal angle
    
    lat_fib = 90 - np.degrees(phi)             # latitude in degrees
    lon_fib = np.degrees(theta) % 360          # longitude in degrees
    
    # --- Interpolation ---
    interp_func = RegularGridInterpolator(
        (lat_grid, lon_grid), data_grid,
        method='linear', bounds_error=False, fill_value=None
    )
    
    points_fib = np.column_stack((lat_fib, lon_fib))
    data_fib = interp_func(points_fib)
    
    # --- Cartesian coordinates (optional) ---
    phi_rad = np.radians(90 - lat_fib)
    theta_rad = np.radians(lon_fib)
    x = R * np.sin(phi_rad) * np.cos(theta_rad)
    y = R * np.sin(phi_rad) * np.sin(theta_rad)
    z = R * np.cos(phi_rad)
    
    return lat_fib, lon_fib, data_fib, x, y, z


#-------------------------------------- test ----------------------------------------
if __name__ == "__main__":
# Example lat-lon grid
    import time

    lat_grid = np.linspace(-90, 90.25, 721)
    lon_grid = np.linspace(0, 360, 180)
    data_grid = np.random.rand(len(lat_grid), len(lon_grid))  # example data, e.g., temperature

    start = time.time()
    for _ in range(24*2):
        lat_fib, lon_fib, data_fib, x, y, z = latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=30, R=6371)
    end = time.time()

    print(lat_fib.shape, lon_fib.shape, data_fib.shape, end-start)
