import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

# --- Your function ---
def latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=300, R=6371):
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

# --- Example lat-lon grid ---
lat_grid = np.linspace(-90, 90, 181)
lon_grid = np.linspace(0, 360, 361)
data_grid = 20 + 10*np.outer(np.cos(np.radians(lat_grid)), np.ones_like(lon_grid))  # e.g., temperature

# --- Convert to Fibonacci points ---
lat_fib, lon_fib, data_fib, x, y, z = latlon_to_fibonacci(lat_grid, lon_grid, data_grid, d_target=300)

# --- Plotting ---
surface_phi, surface_theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:200j]
x_surf = 6371 * np.sin(surface_phi) * np.cos(surface_theta)
y_surf = 6371 * np.sin(surface_phi) * np.sin(surface_theta)
z_surf = 6371 * np.cos(surface_phi)

# Sphere surface
surface = go.Surface(
    x=x_surf, y=y_surf, z=z_surf,
    colorscale='Earth', cmin=-6371, cmax=6371, opacity=0.7, showscale=False
)

# Fibonacci points colored by data
points_trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=3, color=data_fib, colorscale='Jet', colorbar=dict(title='Value')),
    name='Fibonacci Points'
)

fig = go.Figure(data=[surface, points_trace])
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.2)),
    ),
    title='Weather Data on Fibonacci Sphere',
)

import plotly.offline as pyo
pyo.plot(fig, filename='fibonacci_weather.html')
