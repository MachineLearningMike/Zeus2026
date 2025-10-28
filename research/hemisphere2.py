import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

R = 6371  # sphere radius

# --- Generate Fibonacci points ---
def latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=300):
    A = 4 * np.pi * R**2
    N = int(A / d_target**2)
    indices = np.arange(0, N, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/N)
    theta = np.pi * (1 + 5**0.5) * indices
    lat_fib = 90 - np.degrees(phi)
    lon_fib = np.degrees(theta) % 360
    interp_func = RegularGridInterpolator(
        (lat_grid, lon_grid), data_grid,
        method='linear', bounds_error=False, fill_value=None
    )
    points_fib = np.column_stack((lat_fib, lon_fib))
    data_fib = interp_func(points_fib)
    phi_rad = np.radians(90 - lat_fib)
    theta_rad = np.radians(lon_fib)
    x = R * np.sin(phi_rad) * np.cos(theta_rad)
    y = R * np.sin(phi_rad) * np.sin(theta_rad)
    z = R * np.cos(phi_rad)
    return lat_fib, lon_fib, data_fib, x, y, z

# --- Lat-lon grid for data ---
lat_grid = np.linspace(-90, 90, 181)
lon_grid = np.linspace(0, 360, 361)
data_grid = 20 + 10*np.outer(np.cos(np.radians(lat_grid)), np.ones_like(lon_grid))

lat_fib, lon_fib, data_fib, x, y, z = latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=300)

# --- Circle definitions (radius = 1/15 of globe = 24°) ---
r_deg = 360 / 30
circles = [
    (0, 0, r_deg),
    (45, 90, r_deg),
    (-30, -60, r_deg)
]

# --- Base sphere for surface coloring ---
phi_surf, theta_surf = np.mgrid[0:np.pi:200j, 0:2*np.pi:400j]
lat_surf = 90 - np.degrees(phi_surf)
lon_surf = np.degrees(theta_surf)

# Start with navy
surface_colors = np.full(lat_surf.shape, 'navy', dtype=object)

# --- Paint circles green ---
for lat_c, lon_c, r in circles:
    lat1 = np.radians(lat_surf)
    lon1 = np.radians(lon_surf)
    lat2 = np.radians(lat_c)
    lon2 = np.radians(lon_c)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    angular_distance = 2 * np.arcsin(np.sqrt(a))
    mask = angular_distance <= np.radians(r)
    surface_colors[mask] = 'green'

# --- Surface coordinates ---
x_surf = R*np.sin(phi_surf)*np.cos(theta_surf)
y_surf = R*np.sin(phi_surf)*np.sin(theta_surf)
z_surf = R*np.cos(phi_surf)

# --- Surface trace ---
surface = go.Surface(
    x=x_surf, y=y_surf, z=z_surf,
    surfacecolor=np.zeros_like(x_surf),  # color values not needed, using `colorscale` by object array
    colorscale=[[0, 'navy'], [1, 'green']],
    cmin=0, cmax=1,
    showscale=False,
    opacity=1.0
)

# --- Points trace: green points inside circles, else colored by data ---
mask_points_green = np.zeros_like(lat_fib, dtype=bool)
for lat_c, lon_c, r in circles:
    lat1 = np.radians(lat_fib)
    lon1 = np.radians(lon_fib)
    lat2 = np.radians(lat_c)
    lon2 = np.radians(lon_c)
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    angular_distance = 2 * np.arcsin(np.sqrt(a))
    mask_points_green |= angular_distance <= np.radians(r)

colors_points = np.array(data_fib, dtype=object)
colors_points[mask_points_green] = 'green'

points_trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=3, color=colors_points),
    name='Fibonacci Points'
)

fig = go.Figure(data=[surface, points_trace])
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.2))
    ),
    title='Globe with Green Circles and Points'
)

import plotly.offline as pyo
pyo.plot(fig, filename='fibonacci_green_circles_surface.html')
