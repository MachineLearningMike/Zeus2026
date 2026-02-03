import numpy as np
import matplotlib.pyplot as plt

def visualize_grids():
    """Compare GDAS and ERA5 grid geometries"""
    
    # GDAS: Regular spacing
    gdas_lats = np.linspace(90, -90, 721)
    gdas_lons = np.linspace(0, 360, 1440, endpoint=False)
    
    # ERA5 Gaussian latitudes (approximation)
    # Actual Gaussian latitudes from N320 grid
    n_lats = 640
    # Simplified representation
    era5_lats = np.sort(np.sin(np.linspace(-1, 1, n_lats)))
    era5_lats = np.arcsin(era5_lats) * 180/np.pi  # Approximate Gaussian latitudes
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Latitude spacing
    axes[0, 0].plot(gdas_lats[:-1], np.diff(gdas_lats), 'b-', label='GDAS', alpha=0.7)
    axes[0, 0].plot(era5_lats[:-1], np.diff(era5_lats), 'r-', label='ERA5 Gaussian', alpha=0.7)
    axes[0, 0].set_xlabel('Latitude (°)')
    axes[0, 0].set_ylabel('Latitude Spacing (°)')
    axes[0, 0].set_title('Latitude Spacing Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Grid cell area at equator
    lat_range = np.linspace(-90, 90, 181)
    
    # GDAS area (spherical)
    R_gdas = 6371.229  # km
    gdas_area = (2*np.pi*R_gdas/360 * 0.25) * (2*np.pi*R_gdas*np.cos(np.radians(lat_range))/360 * 0.25)
    
    # ERA5 area (ellipsoidal)
    a_era5 = 6378.137  # km
    b_era5 = 6356.752  # km
    era5_area = []
    for lat in lat_range:
        lat_rad = np.radians(lat)
        # Radius of curvature in prime vertical
        N = a_era5 / np.sqrt(1 - (1 - (b_era5/a_era5)**2) * np.sin(lat_rad)**2)
        # Radius of curvature in meridian
        M = a_era5 * (1 - (1 - (b_era5/a_era5)**2)) / (1 - (1 - (b_era5/a_era5)**2) * np.sin(lat_rad)**2)**1.5
        area = (N * np.cos(lat_rad) * np.radians(0.25)) * (M * np.radians(0.25))
        era5_area.append(area)
    era5_area = np.array(era5_area)
    
    axes[0, 1].plot(lat_range, gdas_area/1e6, 'b-', label='GDAS', alpha=0.7)
    axes[0, 1].plot(lat_range, era5_area/1e6, 'r-', label='ERA5', alpha=0.7)
    axes[0, 1].set_xlabel('Latitude (°)')
    axes[0, 1].set_ylabel('Grid Cell Area (km²)')
    axes[0, 1].set_title('Grid Cell Area at 0.25° Resolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Zonal distance per 0.25°
    gdas_zonal = 2*np.pi*R_gdas*np.cos(np.radians(lat_range))/360 * 0.25
    era5_zonal = []
    for lat in lat_range:
        lat_rad = np.radians(lat)
        N = a_era5 / np.sqrt(1 - (1 - (b_era5/a_era5)**2) * np.sin(lat_rad)**2)
        zonal = N * np.cos(lat_rad) * np.radians(0.25)
        era5_zonal.append(zonal)
    era5_zonal = np.array(era5_zonal)
    
    axes[1, 0].plot(lat_range, gdas_zonal, 'b-', label='GDAS', alpha=0.7)
    axes[1, 0].plot(lat_range, era5_zonal, 'r-', label='ERA5', alpha=0.7)
    axes[1, 0].set_xlabel('Latitude (°)')
    axes[1, 0].set_ylabel('Zonal Distance per 0.25° (km)')
    axes[1, 0].set_title('East-West Grid Spacing')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Percentage difference
    diff_percent = 100 * (era5_zonal - gdas_zonal) / gdas_zonal
    axes[1, 1].plot(lat_range, diff_percent, 'g-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Latitude (°)')
    axes[1, 1].set_ylabel('Difference (%)')
    axes[1, 1].set_title('ERA5 vs GDAS: Zonal Distance Difference')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].fill_between(lat_range, 0, diff_percent, alpha=0.3, color='green')
    
    plt.tight_layout()
    plt.show()

visualize_grids()  # Uncomment to run