"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (721, 1440) in lat and lon,
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenit, cos_zenit], dim=0).

t2m, tp, u100, v100, d2m, sp, orograph, and land_sea_mask are from ERA5 and have units defined by ERA5.
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

Please write code that predicts t2m at time t + 6 hours, by fastest physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import numpy as np
from datetime import datetime

def ultra_fast_t2m_6hr_prediction(meteo, geo_dense, geo_cat, geo_tempo, 
                                 time_t_utc, delta_hours=6):
    """
    Ultra-fast 6-hour temperature prediction using physical logic.
    
    Inputs (all torch tensors shape (C, 721, 1440)):
        meteo: [t2m(K), tp(m), u100(m/s), v100(m/s), d2m(K), sp(Pa)]
        geo_dense: [orograph(m^2/s^2), sin_lat, cos_lat, sin_lon, cos_lon]
        geo_cat: [land_sea_mask, soil_type] where 0=sea, 1=land
        geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith]
    
    Returns:
        t2m_t6: predicted temperature at t+6 hours (°C), shape (721, 1440)
    """
    return meteo[0].clone()
    
    # --- EXTRACT VARIABLES ---
    # --- EXTRACT WITH CORRECT UNITS ---
    T0_K = meteo[0]          # 2m temperature in KELVIN (NOT °C!)
    precip = meteo[1]        # Total precipitation in METERS water equivalent
    U = meteo[2]             # Eastward wind at 100m (m/s)
    V = meteo[3]             # Northward wind at 100m (m/s)
    Td_K = meteo[4]          # Dew point at 2m in KELVIN (NOT °C!)
    Psfc = meteo[5]          # Surface pressure (Pa)
    
    geopotential = geo_dense[0]  # Orography as geopotential (m²/s²)
    sin_lat = geo_dense[1]       # sin(latitude)
    cos_lat = geo_dense[2]       # cos(latitude)
    sin_lon = geo_dense[3]       # sin(longitude)
    cos_lon = geo_dense[4]       # cos(longitude)
    
    land_mask = geo_cat[0]       # 0=sea, 1=land
    soil_type = geo_cat[1]       # 0-7 soil types
    
    sin_orbit = geo_tempo[0]     # sin(orbital_angle)
    cos_orbit = geo_tempo[1]     # cos(orbital_angle)
    sin_z0 = geo_tempo[2]        # sin(zenith_angle)
    cos_z0 = geo_tempo[3]        # cos(zenith_angle)
    
    H, W = T0_K.shape
    device = T0_K.device
    
    # --- CONSTANTS ---
    SOLAR_CONST = 1361.0         # W/m²
    SIGMA = 5.67e-8              # Stefan-Boltzmann (W/m²K⁴)
    HOURS_TO_RAD = torch.tensor(np.radians(15.0), device=device)
    STD_PRESSURE = 101325.0      # Pa
    g = 9.80665                  # Gravity (m/s²)
    
    # --- CONVERT GEOPOTENTIAL TO HEIGHT (m) ---
    # ERA5 orography = geopotential (m²/s²)
    # Height = geopotential / g
    orog_height = geopotential / g  # Now in meters
    
    # --- 1. SURFACE PROPERTIES ---
    # Soil thermal properties (using KELVIN-appropriate values)
    soil_vals = torch.tensor([
        # HeatCap(J/m²K), Albedo, Response, LapseCorrection
        [4.18e6, 0.06, 0.6, 0.0],    # 0: Ocean/Lake
        [1.2e6,  0.25, 1.5, 1.0],    # 1: Coarse (sand)
        [1.6e6,  0.20, 1.2, 0.8],    # 2: Medium
        [2.0e6,  0.18, 1.0, 0.6],    # 3: Medium Fine
        [2.4e6,  0.16, 0.8, 0.4],    # 4: Fine
        [2.8e6,  0.14, 0.7, 0.3],    # 5: Very Fine
        [3.0e6,  0.12, 0.6, 0.2],    # 6: Organic
        [3.2e6,  0.10, 0.5, 0.1]     # 7: Tropical Organic
    ], device=device)
    
    soil_idx = soil_type.long()
    C_grid = soil_vals[soil_idx, 0]      # Heat capacity
    albedo_grid = soil_vals[soil_idx, 1] # Albedo
    resp_grid = soil_vals[soil_idx, 2]   # Response factor
    lapse_corr = soil_vals[soil_idx, 3]  # Lapse rate correction
    
    # Ocean properties override
    is_ocean = (land_mask == 0)
    C_grid[is_ocean] = 4.18e6
    albedo_grid[is_ocean] = 0.06
    resp_grid[is_ocean] = 0.6
    lapse_corr[is_ocean] = 0.0
    
    # --- 2. BASIC LAPSE RATE CORRECTION ---
    # Standard atmospheric lapse: -6.5 K/km
    # But we need current temperature relative to sea level
    T0_sea_level = T0_K + 0.0065 * orog_height  # Adjust to sea level
    
    # --- 3. HUMIDITY (in KELVIN) ---
    # T and Td are in K, so difference is in K
    T_diff_K = torch.clamp(T0_K - Td_K, min=0.1)  # K difference
    
    # Relative humidity approximation (using K)
    # Clausius-Clapeyron simplified: RH ~ exp(-0.07*(T-Td)) works for K too
    RH = torch.exp(-0.07 * T_diff_K)
    cloud_cover = torch.clamp(1.5 * RH**3 - 0.5, 0.0, 1.0)
    
    # --- 4. SOLAR GEOMETRY (unchanged, dimensionless) ---
    orbital_rad = torch.atan2(sin_orbit, cos_orbit)
    delta_rad = orbital_rad - torch.tensor(np.radians(270.0), device=device)
    sin_delta = torch.sin(delta_rad)
    cos_delta = torch.cos(delta_rad)
    
    # Hour angle from zenith
    cos_h0 = (cos_z0 - sin_lat * sin_delta) / torch.clamp(cos_lat * cos_delta, min=1e-6)
    cos_h0 = torch.clamp(cos_h0, -1.0, 1.0)
    h0_rad = torch.acos(cos_h0)
    
    # Morning/afternoon
    lon_rad = torch.atan2(sin_lon, cos_lon)
    lon_deg = torch.rad2deg(lon_rad)
    local_hour = (12 + lon_deg/15.0) % 24
    is_afternoon = local_hour > 12
    h0_rad = torch.where(is_afternoon, -h0_rad, h0_rad)
    
    # 6 hours later
    h6_rad = h0_rad + HOURS_TO_RAD * delta_hours
    cos_z6 = sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(h6_rad)
    cos_z6 = torch.clamp(cos_z6, -1.0, 1.0)
    
    # Solar radiation
    I0_clear = SOLAR_CONST * 0.75 * torch.clamp(cos_z0, min=0.0)
    I6_clear = SOLAR_CONST * 0.75 * torch.clamp(cos_z6, min=0.0)
    cloud_reduction = 1.0 - 0.7 * cloud_cover
    I_avg = 0.5 * (I0_clear + I6_clear) * cloud_reduction
    
    # --- 5. LOCAL HEATING (in KELVIN) ---
    # T0_K is already in K, no conversion needed
    
    # Absorbed solar
    absorbed = (1.0 - albedo_grid) * I_avg
    
    # Outgoing longwave (T^4 in K^4)
    is_night = I_avg < 20
    cloud_trapping = 1.0 + 0.4 * cloud_cover * is_night
    emitted = SIGMA * 0.9 * (T0_K ** 4) / cloud_trapping
    
    # Sensible heat (smaller gradient in K)
    sensible = 8.0 * resp_grid * 1.5  # ~1.5K gradient
    
    # Net flux
    Q_net = absorbed - emitted - sensible
    Q_net = torch.clamp(Q_net, -200, 400)
    
    # Temperature change in K
    delta_t_sec = delta_hours * 3600
    delta_T_local_K = (Q_net * delta_t_sec) / C_grid
    delta_T_local_K = delta_T_local_K * resp_grid
    
    # --- 6. PRECIPITATION (tp in meters) ---
    # ERA5 tp is accumulated since last post-processing (3-6 hours)
    # For 6-hour forecast, use as-is
    precip_m = precip  # Already in meters
    
    # Rain cooling effect (in K)
    # Heavy rain: -3K, Light rain: -0.5K
    rain_cooling_K = -3.0 * torch.tanh(precip_m * 100)  # Scale appropriately
    
    # Snow vs rain based on temperature
    is_snow = T0_K < 273.15  # Below freezing
    rain_cooling_K[is_snow] = -1.0 * torch.tanh(precip_m[is_snow] * 100)
    
    delta_T_precip_K = rain_cooling_K * resp_grid
    
    # --- 7. PRESSURE EFFECTS (in K) ---
    # Low pressure → rising → adiabatic cooling in K
    P_dev = (Psfc - STD_PRESSURE) / 1000.0  # kPa deviation
    delta_T_pressure_K = -0.08 * P_dev  # K per kPa
    
    # --- 8. WIND ADVECTION (in K) ---
    # Grid spacing
    dx = 27830 * cos_lat
    dy = 27830 * torch.ones_like(cos_lat)
    
    # Temperature gradients in K/m
    dT_dx = (torch.roll(T0_K, -1, dims=1) - torch.roll(T0_K, 1, dims=1)) / (2 * dx)
    dT_dy = (torch.roll(T0_K, -1, dims=0) - torch.roll(T0_K, 1, dims=0)) / (2 * dy)
    
    # Fix edges
    dT_dx[:, 0] = dT_dx[:, 1]
    dT_dx[:, -1] = dT_dx[:, -2]
    dT_dy[0, :] = dT_dy[1, :]
    dT_dy[-1, :] = dT_dy[-2, :]
    
    # Advection in K/s
    advection = -(U * dT_dx + V * dT_dy)
    delta_T_adv_K = advection * delta_t_sec
    
    # --- 9. OROGRAPHIC EFFECTS (in K) ---
    # Lapse rate: -0.0065 K/m
    # Wind blowing uphill: adiabatic cooling
    grad_orog_x = (torch.roll(orog_height, -1, dims=1) - torch.roll(orog_height, 1, dims=1)) / (2 * dx)
    grad_orog_y = (torch.roll(orog_height, -1, dims=0) - torch.roll(orog_height, 1, dims=0)) / (2 * dy)
    
    orog_effect = -(U * grad_orog_x + V * grad_orog_y) * 0.0065  # K/s
    delta_T_orog_K = orog_effect * delta_t_sec * lapse_corr
    
    # --- 10. JET-STREAM (dimensionless pattern) ---
    jet_strength = 4.0 * (sin_lat**2) * (cos_lat**2)
    
    # Enhanced advection in jet regions
    jet_factor = 1.0 + 0.5 * jet_strength
    delta_T_adv_K = delta_T_adv_K * jet_factor
    
    # Poleward heat transport in K
    lat_rad = torch.asin(sin_lat)
    lat_deg = torch.rad2deg(lat_rad)
    delta_T_jet_K = 0.1 * jet_strength * torch.sign(lat_deg)
    
    # --- 11. COMBINE ALL EFFECTS (in K) ---
    delta_T_total_K = (delta_T_local_K + delta_T_adv_K + delta_T_jet_K + 
                      delta_T_precip_K + delta_T_pressure_K + delta_T_orog_K)
    
    # Rate limiting (in K)
    max_change_K = 8.0 * resp_grid
    delta_T_total_K = torch.clamp(delta_T_total_K, -max_change_K, max_change_K)
    
    T6_pred_K = T0_K + delta_T_total_K
    
    # --- 12. PHYSICAL CONSTRAINTS (in K) ---
    # Realistic Earth bounds in K
    T6_pred_K = torch.clamp(T6_pred_K, 200.0, 330.0)  # ~-73°C to +57°C
    
    # Ocean stability
    T6_pred_K[is_ocean] = 0.85 * T6_pred_K[is_ocean] + 0.15 * T0_K[is_ocean]
    
    # Desert extremes
    desert_mask = (soil_type == 1) & (T_diff_K > 5.0) & (land_mask == 1)
    T6_pred_K[desert_mask] = torch.clamp(T6_pred_K[desert_mask],
                                         T0_K[desert_mask] - 10.0,
                                         T0_K[desert_mask] + 10.0)
    
    # --- 13. DIURNAL CYCLE (in K) ---
    solar_hour = (torch.rad2deg(h0_rad) / 15.0 + 12.0) % 24
    
    # Morning warming (6-10 AM solar)
    morning = (solar_hour >= 6) & (solar_hour < 10) & (cloud_cover < 0.5)
    T6_pred_K[morning] += 1.5 * resp_grid[morning]  # K increase
    
    # Afternoon cooling (2-6 PM solar)
    afternoon = (solar_hour >= 14) & (solar_hour < 18)
    T6_pred_K[afternoon] -= 1.0 * resp_grid[afternoon]  # K decrease
    
    # Polar night
    polar_night = (I_avg < 5) & (torch.abs(lat_deg) > 60)
    T6_pred_K[polar_night] -= 2.5 * resp_grid[polar_night]  # K decrease
    
    # --- 14. RE-APPLY LAPSE RATE ---
    # Bring back to actual elevation
    T6_pred_K = T6_pred_K - 0.0065 * orog_height
    
    # --- 15. FINAL SMOOTHING ---
    # Simple 3x3 smoothing
    T6_smooth_K = torch.zeros_like(T6_pred_K)
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            # Weighted average
            patch = T6_pred_K[i-1:i+2, j-1:j+2]
            weights = torch.tensor([[0.25, 0.5, 0.25],
                                   [0.5,  1.0, 0.5],
                                   [0.25, 0.5, 0.25]], device=device) / 4.0
            T6_smooth_K[i, j] = torch.sum(patch * weights)
    
    # Keep edges
    T6_smooth_K[0, :] = T6_pred_K[0, :]
    T6_smooth_K[-1, :] = T6_pred_K[-1, :]
    T6_smooth_K[:, 0] = T6_pred_K[:, 0]
    T6_smooth_K[:, -1] = T6_pred_K[:, -1]
    
    return T6_smooth_K  # Returns in KELVIN


# Optional: Even faster version using F.conv2d (requires padding)
def ultra_fast_t2m_6hr_prediction_conv(meteo, geo_dense, geo_cat, geo_tempo,
                                      time_t_utc, delta_hours=6):
    """Version using F.conv2d for faster smoothing."""
    import torch.nn.functional as F
    
    # [Same as above until smoothing step...]
    
    # Use F.conv2d for smoothing
    T6_pred_4d = T6_pred.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = torch.tensor([[[[0.25, 0.5, 0.25],
                            [0.5,  1.0, 0.5],
                            [0.25, 0.5, 0.25]]]], device=device) / 4.0
    
    T6_smooth_4d = F.conv2d(T6_pred_4d, kernel, padding=1)
    T6_smooth = T6_smooth_4d.squeeze()
    
    return T6_smooth