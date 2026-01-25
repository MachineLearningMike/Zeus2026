"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = numpy.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (721, 1440) in lat and lon,
    together with earth orbital angle, zenith angle, and soil type, where all are of shape (721, 1440) in lat and lon.
The time t is given in UTC.
Note my soil types are 
soil_type_descriptions = {
0:	'Ocean/Lake',
1:	'Coarse',
2:	'Medium',
3:	'Medium Fine',
4:	'Fine',
5:	'Very Fine',
6:	'Organic',
7:	'Tropical Organic',
}
Please write code that predicts t2m at time t + 6 hours, by fastest physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import numpy as np
import torch

def fast_physics_t2m_6hr(meteo_t, earth_orbital_angle, zenith_t, soil_type,
                          lat_grid, lon_grid, time_t_utc, delta_hours=6):
    """
    FAST physics-based 6-hour t2m prediction.
    
    Inputs:
        meteo_t: numpy array shape (6, 721, 1440) containing:
            [t2m, tp, u100, v100, d2m, sp] in that order
        earth_orbital_angle: (721, 1440) orbital angle (degrees)
        zenith_t: (721, 1440) zenith angle at time t (degrees)
        soil_type: (721, 1440) integer code 0-7
        lat_grid, lon_grid: (721, 1440) coordinates (°)
        time_t_utc: datetime of time t
        delta_hours: prediction horizon (default 6)
        
    Returns:
        t2m_t6: predicted temperature at t+6 hours (°C), shape (721, 1440)
    """
    # --- Extract meteo variables ---
    T0 = meteo_t[0].copy()        # t2m (°C)
    precip = meteo_t[1].copy()    # tp (m water)
    U = meteo_t[2].copy()         # u100 (m/s)
    V = meteo_t[3].copy()         # v100 (m/s)
    Td = meteo_t[4].copy()        # d2m (°C)
    Psfc = meteo_t[5].copy()      # sp (Pa)
    
    z0 = zenith_t.copy()
    soil = soil_type.copy()
    lat = lat_grid.copy()
    lon = lon_grid.copy()
    
    H, W = T0.shape
    
    # --- Precompute reusable arrays (FAST) ---
    phi_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    z0_rad = np.radians(z0)
    
    # Constants (precomputed)
    SOLAR_CONSTANT = 1361.0
    SIGMA = 5.67e-8
    HOURS_TO_RAD = np.radians(15)  # 15 degrees per hour
    
    # --- 1. SOIL PROPERTIES (vectorized) ---
    # Create property grids using broadcasting
    soil_codes = np.arange(8)
    # Heat capacity (J/m²K)
    C_vals = np.array([4.18e6, 1.2e6, 1.6e6, 2.0e6, 2.4e6, 2.8e6, 3.0e6, 3.2e6])
    # Albedo
    alpha_vals = np.array([0.06, 0.25, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10])
    # Response factor
    resp_vals = np.array([0.6, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5])
    
    C_grid = C_vals[soil]
    alpha_grid = alpha_vals[soil]
    resp_grid = resp_vals[soil]
    
    # --- 2. HUMIDITY & CLOUDS (FAST approximations) ---
    # Fast RH: 100% when T=Td, ~50% when T-Td=10°C
    T_diff = np.maximum(T0 - Td, 0.1)
    RH = np.exp(-0.07 * T_diff)  # Fast exponential decay
    cloud_cover = np.clip(1.5 * RH**3 - 0.5, 0.0, 1.0)
    
    # --- 3. SOLAR RADIATION (optimized) ---
    # Solar declination (single value for all grid points)
    delta_deg = 23.45 * np.sin(np.radians(earth_orbital_angle[0, 0] - 270))
    delta_rad = np.radians(delta_deg)
    
    # Precompute trig terms
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)
    sin_delta = np.sin(delta_rad)
    cos_delta = np.cos(delta_rad)
    
    # Current hour angle from zenith (vectorized)
    # cos(z) = sinφ sinδ + cosφ cosδ cos(h)
    cos_h0 = (np.cos(z0_rad) - sin_phi * sin_delta) / np.maximum(cos_phi * cos_delta, 1e-6)
    cos_h0 = np.clip(cos_h0, -1.0, 1.0)
    h0_rad = np.arccos(cos_h0)
    
    # Determine morning/afternoon (crude but fast)
    local_hour = (12 + lon/15.0) % 24
    is_afternoon = local_hour > 12
    h0_rad = np.where(is_afternoon, -h0_rad, h0_rad)
    
    # Hour angle 6 hours later
    h6_rad = h0_rad + HOURS_TO_RAD * delta_hours
    
    # Zenith at t+6
    cos_z6 = sin_phi * sin_delta + cos_phi * cos_delta * np.cos(h6_rad)
    cos_z6 = np.clip(cos_z6, -1.0, 1.0)
    
    # Solar insolation with clouds
    I0_clear = SOLAR_CONSTANT * 0.75 * np.maximum(0, np.cos(z0_rad))
    I6_clear = SOLAR_CONSTANT * 0.75 * np.maximum(0, cos_z6)
    cloud_reduction = 1.0 - 0.7 * cloud_cover  # Clouds block 70% max
    I_avg = 0.5 * (I0_clear + I6_clear) * cloud_reduction
    
    # --- 4. LOCAL HEATING (vectorized energy balance) ---
    T0_K = T0 + 273.15
    
    # Absorbed solar
    absorbed = (1.0 - alpha_grid) * I_avg
    
    # Outgoing longwave (clouds trap at night)
    is_night = I_avg < 20
    cloud_trapping = 1.0 + 0.4 * cloud_cover * is_night
    emitted = SIGMA * 0.9 * (T0_K ** 4) / cloud_trapping
    
    # Sensible heat (simplified)
    sensible = 10.0 * resp_grid * (T0_K - (T0_K - 1.5))  # ~1.5K gradient
    
    # Net flux
    Q_net = absorbed - emitted - sensible
    Q_net = np.clip(Q_net, -150, 300)
    
    # Local temperature change
    delta_t_sec = delta_hours * 3600
    delta_T_local = (Q_net * delta_t_sec) / C_grid
    delta_T_local = delta_T_local * resp_grid
    
    # --- 5. PRECIPITATION COOLING (fast) ---
    precip_mm = precip * 1000.0
    # Rain cooling: -2°C per log(mm+1)
    rain_cooling = -2.0 * np.log1p(precip_mm)
    # No rain cooling if T < -5°C (snow)
    snow_mask = T0 < -5.0
    rain_cooling[snow_mask] = -0.5 * np.log1p(precip_mm[snow_mask])
    
    delta_T_precip = rain_cooling * resp_grid
    
    # --- 6. PRESSURE EFFECTS (fast) ---
    # Low pressure -> cooling, High pressure -> warming
    P_dev = (Psfc - 101325.0) / 1000.0  # kPa from standard
    delta_T_pressure = -0.08 * P_dev  # Fast linear approx
    
    # --- 7. WIND ADVECTION (optimized with roll) ---
    # Use numpy.roll for fast finite differences
    dx = 27830 * np.cos(phi_rad)  # 0.25° at equator in meters (approx)
    dy = 27830 * np.ones_like(phi_rad)  # 0.25° in meters
    
    # Temperature gradients using roll (fast!)
    dT_dx = (np.roll(T0, -1, axis=1) - np.roll(T0, 1, axis=1)) / (2 * dx)
    dT_dy = (np.roll(T0, -1, axis=0) - np.roll(T0, 1, axis=0)) / (2 * dy)
    
    # Fix edges (simple reflection)
    dT_dx[:, 0] = dT_dx[:, 1]
    dT_dx[:, -1] = dT_dx[:, -2]
    dT_dy[0, :] = dT_dy[1, :]
    dT_dy[-1, :] = dT_dy[-2, :]
    
    # Advection (vectorized)
    advection = -(U * dT_dx + V * dT_dy)
    delta_T_adv = advection * delta_t_sec
    
    # --- 8. JET-STREAM ENHANCEMENT (fast) ---
    # sin²φ cos²φ peaks at 30°, 60° latitude
    jet_strength = 4.0 * (sin_phi**2) * (cos_phi**2)
    
    # Enhanced advection in jet regions
    jet_factor = 1.0 + 0.5 * jet_strength
    delta_T_adv = delta_T_adv * jet_factor
    
    # Poleward heat transport
    delta_T_jet = 0.12 * jet_strength * np.sign(lat)
    
    # --- 9. COMBINE ALL EFFECTS (fast) ---
    delta_T_total = (delta_T_local + delta_T_adv + delta_T_jet + 
                     delta_T_precip + delta_T_pressure)
    
    # Rate limiting based on soil type
    max_change = 10.0 * resp_grid
    delta_T_total = np.clip(delta_T_total, -max_change, max_change)
    
    T6_pred = T0 + delta_T_total
    
    # --- 10. PHYSICAL CONSTRAINTS (fast) ---
    # Absolute bounds
    T6_pred = np.clip(T6_pred, -60, 55)
    
    # Ocean stability (very slow changes)
    ocean_mask = soil == 0
    T6_pred[ocean_mask] = 0.7 * T6_pred[ocean_mask] + 0.3 * T0[ocean_mask]
    
    # Desert extremes (fast response)
    desert_mask = (soil == 1) & (T_diff > 8.0)  # Dry deserts
    T6_pred[desert_mask] = np.clip(T6_pred[desert_mask], 
                                    T0[desert_mask] - 15, 
                                    T0[desert_mask] + 15)
    
    # Organic soils stability
    organic_mask = soil >= 6
    T6_pred[organic_mask] = np.clip(T6_pred[organic_mask],
                                     T0[organic_mask] - 4,
                                     T0[organic_mask] + 4)
    
    # --- 11. DIURNAL CORRECTIONS (fast) ---
    solar_hour = (np.degrees(h0_rad) / 15.0 + 12.0) % 24
    
    # Morning boost (6-10 AM solar time, clear skies)
    morning = (solar_hour >= 6) & (solar_hour < 10) & (cloud_cover < 0.5)
    T6_pred[morning] += 0.5 * resp_grid[morning]
    
    # Afternoon decline (2-6 PM solar time)
    afternoon = (solar_hour >= 14) & (solar_hour < 18)
    T6_pred[afternoon] -= 0.3 * resp_grid[afternoon]
    
    # Polar night cooling
    polar_night = (I_avg < 5) & (np.abs(lat) > 60)
    T6_pred[polar_night] -= 1.5 * resp_grid[polar_night]
    
    # --- 12. FAST SMOOTHING (3x3 convolution) ---
    # Simple 3x3 averaging (fast with slicing)
    T6_smooth = np.zeros_like(T6_pred)
    T6_smooth[1:-1, 1:-1] = (
        T6_pred[0:-2, 0:-2] + T6_pred[0:-2, 1:-1] + T6_pred[0:-2, 2:] +
        T6_pred[1:-1, 0:-2] + 2 * T6_pred[1:-1, 1:-1] + T6_pred[1:-1, 2:] +
        T6_pred[2:, 0:-2] + T6_pred[2:, 1:-1] + T6_pred[2:, 2:]
    ) / 10.0
    
    # Keep edges
    T6_smooth[0, :] = T6_pred[0, :]
    T6_smooth[-1, :] = T6_pred[-1, :]
    T6_smooth[:, 0] = T6_pred[:, 0]
    T6_smooth[:, -1] = T6_pred[:, -1]
    
    return T6_smooth