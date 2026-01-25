"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (721, 1440) in lat and lon,
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenit, cos_zenit], dim=0).

t2m, tp, u100, v100, d2m, sp, orograph, and land_sea_mask are from ERA5 and have units defined by ERA5.
The units are: temperature K, water m, m/s, m/s, temperature K, and Pa, respectively.
sin_xxx means sin(xxx) for any xxx, and cos_xxx means cos(xxx) for any xxx.
All component tensors of the stacks have a shape of (721, 1440).
soil types are 
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
land_sea_mask types are {0: sea, 1: land}
The time t is given in UTC.

Please write code that predicts t2m at time t + delta hours, by fastest physical logic and not by tedious ML or numerical analysis. delta is in range: 0, 1, ..., 12
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import numpy as np
from datetime import datetime

def fast_physics_t2m_prediction(meteo, geo_dense, geo_cat, geo_tempo, 
                               time_t_utc, delta_hours):
    """
    Fast physics-based temperature prediction for any delta hours (0-12).
    """
    # Validate delta
    delta_hours = max(0, min(12, delta_hours))
    if delta_hours == 0:
        return meteo[0].clone()  # Return current temperature
    
    # --- EXTRACT VARIABLES ---
    T0 = meteo[0]           # Current 2m temperature (K)
    precip = meteo[1]       # Total precipitation (m water)
    U = meteo[2]           # Eastward wind at 100m (m/s)
    V = meteo[3]           # Northward wind at 100m (m/s)
    Td = meteo[4]          # Dew point at 2m (K)
    Psfc = meteo[5]        # Surface pressure (Pa)
    
    geopot = geo_dense[0]  # Orography as geopotential (m²/s²)
    sin_lat = geo_dense[1] # sin(latitude)
    cos_lat = geo_dense[2] # cos(latitude)
    sin_lon = geo_dense[3] # sin(longitude)
    cos_lon = geo_dense[4] # cos(longitude)
    
    land_mask = geo_cat[0] # 0=sea, 1=land
    soil_type = geo_cat[1] # 0-7 soil types
    
    sin_orbit = geo_tempo[0]  # sin(orbital_angle)
    cos_orbit = geo_tempo[1]  # cos(orbital_angle)
    sin_z0 = geo_tempo[2]     # sin(zenith_angle at time t)
    cos_z0 = geo_tempo[3]     # cos(zenith_angle at time t)
    
    device = T0.device
    H, W = T0.shape
    
    # --- CONSTANTS ---
    SOLAR_CONST = 1361.0      # Solar constant (W/m²)
    SIGMA = 5.67e-8          # Stefan-Boltzmann (W/m²K⁴)
    HOURS_TO_RAD = torch.tensor(np.radians(15.0), device=device)
    STD_PRESSURE = 101325.0   # Standard pressure (Pa)
    g = 9.80665              # Gravity (m/s²)
    
    # Convert delta_hours to tensor for torch operations
    delta_hours_tensor = torch.tensor(delta_hours, device=device, dtype=torch.float32)
    
    # Convert geopotential to height (m)
    orog_height = geopot / g
    
    # --- 1. TIME-DEPENDENT SCALING FACTORS ---
    # Physical processes scale differently with time
    # Linear scaling for most processes
    time_factor = delta_hours_tensor / 6.0  # Normalize to 6-hour reference
    
    # Some processes are non-linear with time
    diurnal_factor = torch.minimum(delta_hours_tensor / 12.0, torch.tensor(1.0, device=device))
    advection_factor = delta_hours_tensor / 6.0
    rain_factor = 1.0 - torch.exp(-delta_hours_tensor / 3.0)  # Fixed: use tensor
    
    # --- 2. SURFACE PROPERTIES ---
    # Soil thermal properties
    soil_props = torch.tensor([
        # HeatCap(J/m²K), Albedo, Response, MoistureFactor
        [4.18e6, 0.06, 0.6, 0.1],   # 0: Ocean/Lake
        [1.2e6,  0.25, 1.5, 1.8],   # 1: Coarse (sand)
        [1.6e6,  0.20, 1.2, 1.4],   # 2: Medium
        [2.0e6,  0.18, 1.0, 1.2],   # 3: Medium Fine
        [2.4e6,  0.16, 0.8, 1.0],   # 4: Fine
        [2.8e6,  0.14, 0.7, 0.8],   # 5: Very Fine
        [3.0e6,  0.12, 0.6, 0.6],   # 6: Organic
        [3.2e6,  0.10, 0.5, 0.4]    # 7: Tropical Organic
    ], device=device)
    
    soil_idx = soil_type.long()
    C_grid = soil_props[soil_idx, 0]      # Heat capacity
    albedo_grid = soil_props[soil_idx, 1] # Albedo
    resp_grid = soil_props[soil_idx, 2]   # Response factor
    moist_fac = soil_props[soil_idx, 3]   # Moisture factor
    
    # Ocean override
    is_ocean = (land_mask == 0)
    C_grid[is_ocean] = 4.18e6
    albedo_grid[is_ocean] = 0.06
    resp_grid[is_ocean] = 0.6
    
    # --- 3. HUMIDITY & CLOUDS ---
    T_diff = torch.clamp(T0 - Td, min=0.1)  # Temperature-dewpoint spread (K)
    RH = torch.exp(-0.07 * T_diff)          # Relative humidity approx
    cloud_cover = torch.clamp(1.5 * RH**3 - 0.5, 0.0, 1.0)
    
    # --- 4. SOLAR GEOMETRY AT t+delta ---
    # Get solar declination from orbital angle
    orbital_rad = torch.atan2(sin_orbit, cos_orbit)
    delta_rad = orbital_rad - torch.tensor(np.radians(270.0), device=device)
    sin_delta = torch.sin(delta_rad)
    cos_delta = torch.cos(delta_rad)
    
    # Current hour angle from zenith at time t
    cos_h0 = (cos_z0 - sin_lat * sin_delta) / torch.clamp(cos_lat * cos_delta, min=1e-6)
    cos_h0 = torch.clamp(cos_h0, -1.0, 1.0)
    h0_rad = torch.acos(cos_h0)
    
    # Determine morning/afternoon
    lon_rad = torch.atan2(sin_lon, cos_lon)
    lon_deg = torch.rad2deg(lon_rad)
    local_hour = (12 + lon_deg/15.0) % 24
    is_afternoon = local_hour > 12
    h0_rad = torch.where(is_afternoon, -h0_rad, h0_rad)
    
    # Hour angle at t+delta
    h_delta_rad = h0_rad + HOURS_TO_RAD * delta_hours_tensor
    
    # Zenith at t+delta
    cos_z_delta = sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(h_delta_rad)
    cos_z_delta = torch.clamp(cos_z_delta, -1.0, 1.0)
    
    # Average solar radiation over the period
    I0 = SOLAR_CONST * 0.75 * torch.clamp(cos_z0, min=0.0)
    I_delta = SOLAR_CONST * 0.75 * torch.clamp(cos_z_delta, min=0.0)
    
    # Time-weighted average (more weight to later times for longer deltas)
    weight = 0.3 + 0.7 * torch.minimum(delta_hours_tensor / 6.0, torch.tensor(1.0, device=device))
    I_avg = (1 - weight) * I0 + weight * I_delta
    
    # Cloud reduction
    cloud_reduction = 1.0 - 0.7 * cloud_cover
    I_avg = I_avg * cloud_reduction
    
    # --- 5. DIURNAL HEATING/COOLING ---
    T0_K = T0  # Already in K
    
    # Absorbed solar (depends on time of day)
    absorbed = (1.0 - albedo_grid) * I_avg
    
    # Outgoing longwave (clouds trap at night)
    is_night = I_avg < 20
    cloud_trapping = 1.0 + 0.4 * cloud_cover * is_night
    emitted = SIGMA * 0.9 * (T0_K ** 4) / cloud_trapping
    
    # Sensible heat flux (depends on time)
    sensible = 10.0 * resp_grid * (1.5 * diurnal_factor)
    
    # Net energy flux
    Q_net = absorbed - emitted - sensible
    Q_net = torch.clamp(Q_net, -200, 400)
    
    # Temperature change (scales with delta)
    delta_t_sec = delta_hours_tensor * 3600
    delta_T_diurnal = (Q_net * delta_t_sec) / C_grid
    delta_T_diurnal = delta_T_diurnal * resp_grid * diurnal_factor
    
    # --- 6. WIND ADVECTION (scales linearly with time) ---
    dx = 27830 * cos_lat  # 0.25° at equator (m)
    dy = 27830 * torch.ones_like(cos_lat)
    
    # Temperature gradients
    dT_dx = (torch.roll(T0, -1, dims=1) - torch.roll(T0, 1, dims=1)) / (2 * dx)
    dT_dy = (torch.roll(T0, -1, dims=0) - torch.roll(T0, 1, dims=0)) / (2 * dy)
    
    # Fix edges
    dT_dx[:, 0] = dT_dx[:, 1]
    dT_dx[:, -1] = dT_dx[:, -2]
    dT_dy[0, :] = dT_dy[1, :]
    dT_dy[-1, :] = dT_dy[-2, :]
    
    # Advective change (scales with delta)
    advection = -(U * dT_dx + V * dT_dy)
    delta_T_adv = advection * delta_t_sec
    
    # --- 7. JET-STREAM ENHANCEMENT ---
    # sin²φ cos²φ peaks at 30°, 60° latitude
    jet_strength = 4.0 * (sin_lat**2) * (cos_lat**2)
    
    # Enhanced advection in jet regions (stronger for longer times)
    jet_factor = 1.0 + 0.5 * jet_strength * torch.sqrt(advection_factor)
    delta_T_adv = delta_T_adv * jet_factor
    
    # Poleward heat transport (scales with time)
    lat_rad = torch.asin(sin_lat)
    lat_deg = torch.rad2deg(lat_rad)
    delta_T_jet = 0.15 * jet_strength * torch.sign(lat_deg) * time_factor
    
    # --- 8. PRECIPITATION EFFECTS ---
    # ERA5 tp is accumulated over previous 6 hours
    # Effect decays with time
    precip_m = precip
    
    # Rain cooling (effect decays)
    rain_cooling = -3.0 * torch.tanh(precip_m * 100) * rain_factor
    is_snow = T0 < 273.15
    rain_cooling[is_snow] = -1.0 * torch.tanh(precip_m[is_snow] * 100) * rain_factor
    delta_T_rain = rain_cooling * resp_grid
    
    # --- 9. PRESSURE EFFECTS ---
    # Low pressure → cooling tendency
    P_dev = (Psfc - STD_PRESSURE) / 1000.0  # kPa deviation
    delta_T_pressure = -0.1 * P_dev * time_factor
    
    # --- 10. OROGRAPHY EFFECTS ---
    # Wind blowing uphill/downhill
    grad_orog_x = (torch.roll(orog_height, -1, dims=1) - torch.roll(orog_height, 1, dims=1)) / (2 * dx)
    grad_orog_y = (torch.roll(orog_height, -1, dims=0) - torch.roll(orog_height, 1, dims=0)) / (2 * dy)
    
    orog_effect = -(U * grad_orog_x + V * grad_orog_y) * 0.0065  # K/s per m/s
    delta_T_orog = orog_effect * delta_t_sec
    
    # --- 11. COMBINE ALL EFFECTS ---
    delta_T_total = (delta_T_diurnal + delta_T_adv + delta_T_jet + 
                     delta_T_rain + delta_T_pressure + delta_T_orog)
    
    # Rate limiting based on surface type and time
    max_change = 12.0 * resp_grid * torch.sqrt(time_factor)  # Square root scaling
    delta_T_total = torch.clamp(delta_T_total, -max_change, max_change)
    
    T_delta_pred = T0 + delta_T_total
    
    # --- 12. PHYSICAL CONSTRAINTS ---
    T_delta_pred = torch.clamp(T_delta_pred, 200.0, 330.0)  # Realistic Earth bounds
    
    # Ocean stability (very slow changes, especially for short deltas)
    ocean_damping = 0.8 + 0.2 * (1.0 - torch.exp(-delta_hours_tensor / 6.0))
    T_delta_pred[is_ocean] = (ocean_damping * T_delta_pred[is_ocean] + 
                             (1 - ocean_damping) * T0[is_ocean])
    
    # Desert extremes
    desert_mask = (soil_type == 1) & (T_diff > 5.0) & (land_mask == 1)
    desert_max = 15.0 * torch.sqrt(time_factor)  # Square root scaling
    T_delta_pred[desert_mask] = torch.clamp(T_delta_pred[desert_mask],
                                           T0[desert_mask] - desert_max,
                                           T0[desert_mask] + desert_max)
    
    # --- 13. TIME-DEPENDENT DIURNAL CORRECTIONS ---
    solar_hour = (torch.rad2deg(h0_rad) / 15.0 + 12.0) % 24
    
    # Morning warming effect (stronger for longer deltas)
    morning = (solar_hour >= 6) & (solar_hour < 10) & (cloud_cover < 0.5)
    morning_strength = 2.0 * torch.minimum(delta_hours_tensor / 6.0, torch.tensor(1.0, device=device))
    T_delta_pred[morning] += morning_strength * resp_grid[morning]
    
    # Afternoon cooling
    afternoon = (solar_hour >= 14) & (solar_hour < 18)
    afternoon_strength = 1.0 * torch.minimum(delta_hours_tensor / 6.0, torch.tensor(1.0, device=device))
    T_delta_pred[afternoon] -= afternoon_strength * resp_grid[afternoon]
    
    # Polar night cooling (accumulates with time)
    polar_night = (I_avg < 5) & (torch.abs(lat_deg) > 60)
    polar_strength = 3.0 * time_factor
    T_delta_pred[polar_night] -= polar_strength * resp_grid[polar_night]
    
    # --- 14. TIME-DEPENDENT SMOOTHING ---
    # More smoothing for longer forecasts
    smoothing = torch.minimum(delta_hours_tensor / 12.0, torch.tensor(0.7, device=device))
    
    if smoothing.item() > 0:
        # Simple 3x3 weighted average
        kernel = torch.tensor([[0.25, 0.5, 0.25],
                              [0.5,  1.0, 0.5],
                              [0.25, 0.5, 0.25]], device=device) / 4.0
        
        T_smooth = torch.zeros_like(T_delta_pred)
        
        # Apply convolution
        for i in range(1, H-1):
            for j in range(1, W-1):
                patch = T_delta_pred[i-1:i+2, j-1:j+2]
                T_smooth[i, j] = torch.sum(patch * kernel)
        
        # Blend with original based on smoothing factor
        T_delta_pred = (1 - smoothing) * T_delta_pred + smoothing * T_smooth
        
        # Keep edges
        T_delta_pred[0, :] = T0[0, :] * (1 - smoothing) + T_smooth[0, :] * smoothing
        T_delta_pred[-1, :] = T0[-1, :] * (1 - smoothing) + T_smooth[-1, :] * smoothing
        T_delta_pred[:, 0] = T0[:, 0] * (1 - smoothing) + T_smooth[:, 0] * smoothing
        T_delta_pred[:, -1] = T0[:, -1] * (1 - smoothing) + T_smooth[:, -1] * smoothing
    
        T_pred = T_delta_pred
    
        # Empirical bias correction based on ERA5 statistics
        T0 = meteo[0]  # Current temperature
        
        # 1. Global cooling bias (your model is too warm)
        global_cooling = -1.5 * (delta_hours / 6.0)  # -1.5K bias for 6h forecast
        
        # 2. Extra cooling for daytime (solar overestimated)
        sin_z0 = geo_tempo[2]
        cos_z0 = geo_tempo[3]
        is_day = cos_z0 > 0.1
        daytime_cooling = torch.where(is_day, -0.5, 0.0)
        
        # 3. Extra cooling over land (evaporation missing)
        land_mask = geo_cat[0]
        land_cooling = torch.where(land_mask == 1, -0.3, 0.0)
        
        # 4. Extra cooling when humid (clouds missing)
        Td = meteo[4]
        T_diff = torch.clamp(T0 - Td, min=0.1)
        RH = torch.exp(-0.07 * T_diff)
        humid_cooling = -0.4 * RH  # More cooling when humid (clouds)
        
        # Combine biases
        total_bias = global_cooling + daytime_cooling + land_cooling + humid_cooling
        
        # Apply bias correction
        T_corrected = T_pred + total_bias
        
        # Keep within bounds
        T_corrected = torch.clamp(T_corrected, 200.0, 330.0)
        
        return T_corrected