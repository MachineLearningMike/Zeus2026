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
    Oceans are much less sensitive to sunshine than land.
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
    HOURS_TO_RAD = torch.deg2rad(torch.tensor(15.0, device=device))
    STD_PRESSURE = 101325.0   # Standard pressure (Pa)
    g = 9.80665              # Gravity (m/s²)
    
    # Convert delta_hours to tensor
    delta_hours_tensor = torch.tensor(delta_hours, device=device, dtype=torch.float32)
    
    # Convert geopotential to height (m)
    orog_height = geopot / g
    
    # --- 1. TIME SCALING ---
    time_factor = delta_hours_tensor / 6.0
    time_factor = torch.clamp(time_factor, 0.0, 1.0)
    
    is_ocean = (land_mask == 0)
    is_land = (land_mask == 1)
    
    # --- 2. SURFACE PROPERTIES - OCEANS MUCH MORE STABLE ---
    # Oceans have much higher heat capacity and lower response
    C_land = 1.8e6      # Land heat capacity
    C_ocean = 5.0e6     # Ocean: HIGHER heat capacity (more stable)
    
    albedo_land = 0.18  # Land albedo
    albedo_ocean = 0.08 # Ocean albedo (slightly higher than before)
    
    response_land = 1.8    # Land: fast response
    response_ocean = 0.15  # Ocean: VERY SLOW response (reduced from 0.4)
    
    C_grid = torch.where(is_land, C_land, C_ocean)
    albedo_grid = torch.where(is_land, albedo_land, albedo_ocean)
    resp_grid = torch.where(is_land, response_land, response_ocean)
    
    # --- 3. HUMIDITY & CLOUDS ---
    T_diff = torch.clamp(T0 - Td, min=0.1)
    RH = torch.exp(-0.05 * T_diff)
    cloud_cover = torch.clamp(1.0 * RH**2, 0.0, 1.0)
    
    # --- 4. SOLAR GEOMETRY ---
    orbital_rad = torch.atan2(sin_orbit, cos_orbit)
    
    # Solar declination
    declination_rad = torch.deg2rad(torch.tensor(23.44, device=device)) * torch.sin(orbital_rad)
    sin_delta = torch.sin(declination_rad)
    cos_delta = torch.cos(declination_rad)
    
    lat_rad = torch.asin(sin_lat)
    lon_rad = torch.atan2(sin_lon, cos_lon)
    
    # CURRENT local solar time
    UTC_hour = time_t_utc.hour + time_t_utc.minute/60.0
    local_solar_current = (UTC_hour + torch.rad2deg(lon_rad)/15.0) % 24
    
    # FUTURE local solar time
    local_solar_future = (local_solar_current + delta_hours) % 24
    
    # Solar hour angle at FUTURE time
    hour_angle_future = torch.deg2rad(15.0 * (local_solar_future - 12.0))
    
    # Solar zenith at FUTURE time
    cos_z_future = sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(hour_angle_future)
    cos_z_future = torch.clamp(cos_z_future, 0.0, 1.0)
    
    # Solar radiation at FUTURE time
    I_future = SOLAR_CONST * 0.75 * cos_z_future
    cloud_reduction = 1.0 - 0.6 * cloud_cover
    I_future = I_future * cloud_reduction
    
    # Day/night at FUTURE time
    is_day_future = I_future > 20.0
    is_night_future = ~is_day_future
    
    # --- 5. TIME PERIOD MASKS ---
    # Pre-dawn hours (3 AM - 6 AM): coldest time
    pre_dawn = (local_solar_future >= 3) & (local_solar_future < 6)
    
    # Early night (6 PM - 9 PM): rapid cooling (land only)
    early_night = (local_solar_future >= 18) & (local_solar_future < 21)
    
    # Late night (9 PM - 3 AM): gradual cooling
    late_night = (local_solar_future >= 21) | (local_solar_future < 3)
    
    # Morning (6 AM - 9 AM): warming begins
    morning = (local_solar_future >= 6) & (local_solar_future < 9)
    
    # Midday (9 AM - 3 PM): strong warming (land only)
    midday = (local_solar_future >= 9) & (local_solar_future < 15)
    
    # Afternoon (3 PM - 6 PM): transition
    afternoon = (local_solar_future >= 15) & (local_solar_future < 18)
    
    # --- 6. DIURNAL CYCLE - OCEANS MUCH FLATTER ---
    diurnal_adjustment = torch.zeros_like(T0)
    
    # --- 6a. NIGHTTIME ---
    # Pre-dawn (3-6 AM): Land cools, ocean very stable
    if torch.any(pre_dawn):
        land_pre_dawn = pre_dawn & is_land
        ocean_pre_dawn = pre_dawn & is_ocean
        
        # Land: slight cooling
        pre_dawn_factor = (local_solar_future - 3) / 3.0
        land_cooling = -0.4 * pre_dawn_factor * (1.0 - cloud_cover)
        diurnal_adjustment[land_pre_dawn] = land_cooling[land_pre_dawn]
        
        # Ocean: VERY minimal change (±0.05K)
        ocean_cooling = -0.05 * pre_dawn_factor * (1.0 - cloud_cover)
        diurnal_adjustment[ocean_pre_dawn] = ocean_cooling[ocean_pre_dawn]
    
    # Early night (6-9 PM): Land cools rapidly, ocean barely changes
    if torch.any(early_night):
        land_early_night = early_night & is_land
        ocean_early_night = early_night & is_ocean
        
        hours_after = local_solar_future - 18
        cooling_factor = hours_after / 3.0
        
        # Land: -2K by 9 PM
        land_cooling = -2.0 * cooling_factor * (1.0 - cloud_cover)
        diurnal_adjustment[land_early_night] = land_cooling[land_early_night]
        
        # Ocean: barely changes (-0.3K max)
        ocean_cooling = -0.3 * cooling_factor * (1.0 - cloud_cover)
        diurnal_adjustment[ocean_early_night] = ocean_cooling[ocean_early_night]
    
    # Late night (9 PM - 3 AM): Land continues cooling, ocean stable
    if torch.any(late_night):
        land_late_night = late_night & is_land
        ocean_late_night = late_night & is_ocean
        
        night_factor = torch.where(
            local_solar_future >= 21,
            (local_solar_future - 21) / 3.0,
            (local_solar_future + 3) / 6.0
        )
        
        # Land: -1.2K additional cooling
        land_cooling = -1.2 * night_factor * (1.0 - cloud_cover)
        diurnal_adjustment[land_late_night] = land_cooling[land_late_night]
        
        # Ocean: barely changes (-0.2K max)
        ocean_cooling = -0.2 * night_factor * (1.0 - cloud_cover)
        diurnal_adjustment[ocean_late_night] = ocean_cooling[ocean_late_night]
    
    # --- 6b. DAYTIME - BIG DIFFERENCE BETWEEN LAND AND OCEAN ---
    # Morning (6-9 AM): Land warms quickly, ocean slowly
    if torch.any(morning):
        land_morning = morning & is_land
        ocean_morning = morning & is_ocean
        
        hours_after_sunrise = local_solar_future - 6
        warming_factor = hours_after_sunrise / 3.0
        
        # Land: +3.5K by 9 AM
        land_warming = 3.5 * warming_factor
        diurnal_adjustment[land_morning] = land_warming[land_morning]
        
        # Ocean: +0.6K by 9 AM (very slow)
        ocean_warming = 0.6 * warming_factor
        diurnal_adjustment[ocean_morning] = ocean_warming[ocean_morning]
    
    # Midday (9 AM - 3 PM): Land warms strongly, ocean slightly
    if torch.any(midday):
        land_midday = midday & is_land
        ocean_midday = midday & is_ocean
        
        hours_from_noon = torch.abs(local_solar_future - 12.0)
        warming_factor = 1.0 - (hours_from_noon / 3.0) ** 2
        
        # Land: +5K peak at noon
        land_warming = 5.0 * warming_factor
        diurnal_adjustment[land_midday] = land_warming[land_midday]
        
        # Ocean: +1K peak at noon (much smaller)
        ocean_warming = 1.0 * warming_factor
        diurnal_adjustment[ocean_midday] = ocean_warming[ocean_midday]
    
    # Afternoon (3-6 PM): Land transitions to cooling, ocean stable
    if torch.any(afternoon):
        land_afternoon = afternoon & is_land
        ocean_afternoon = afternoon & is_ocean
        
        hours_after_3pm = local_solar_future - 15
        transition_factor = 1.0 - hours_after_3pm / 3.0
        
        # Land: +1.5K at 3 PM, cooling to 0 by 6 PM
        land_effect = 1.5 * transition_factor
        diurnal_adjustment[land_afternoon] = land_effect[land_afternoon]
        
        # Ocean: +0.3K at 3 PM, stable
        ocean_effect = 0.3 * transition_factor
        diurnal_adjustment[ocean_afternoon] = ocean_effect[ocean_afternoon]
    
    # Scale diurnal adjustment by time factor
    diurnal_adjustment = diurnal_adjustment * time_factor
    
    # --- 7. SOLAR RADIATION EFFECT - OCEANS MUCH LESS SENSITIVE ---
    # Solar heating - oceans respond much less
    solar_heating_land = 6.0 * cos_z_future * time_factor  # Land: strong
    solar_heating_ocean = 1.0 * cos_z_future * time_factor  # Ocean: weak
    solar_heating = torch.where(is_land, solar_heating_land, solar_heating_ocean)
    
    # NO solar heating at night
    solar_heating = torch.where(is_day_future, solar_heating, 0.0)
    
    # Radiative cooling - oceans cool less
    radiative_cooling_land = -1.8 * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
    radiative_cooling_ocean = -0.4 * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
    radiative_cooling = torch.where(is_land, radiative_cooling_land, radiative_cooling_ocean)
    
    # Only at night
    radiative_cooling = torch.where(is_night_future, radiative_cooling, 0.0)
    
    # Combine
    delta_T_solar = solar_heating + radiative_cooling
    
    # --- 8. ENERGY BALANCE - OCEANS HAVE SMALLER NET ENERGY ---
    # Daytime net heating
    Q_day_land = 50.0 * cos_z_future - 25.0  # Land: large positive
    Q_day_ocean = 15.0 * cos_z_future - 10.0  # Ocean: small positive
    Q_day = torch.where(is_land, Q_day_land, Q_day_ocean)
    
    # Nighttime net cooling
    Q_night_land = -30.0 * (1.0 - cos_z_future)  # Land: strong cooling
    Q_night_ocean = -8.0 * (1.0 - cos_z_future)  # Ocean: mild cooling
    Q_night = torch.where(is_land, Q_night_land, Q_night_ocean)
    
    # Combine
    Q_simple = torch.where(is_day_future, Q_day, Q_night)
    Q_simple = torch.clamp(Q_simple, -25.0, 55.0)  # Smaller range for ocean
    
    delta_t_sec = delta_hours_tensor * 3600
    delta_T_energy = (Q_simple * delta_t_sec) / C_grid
    delta_T_energy = delta_T_energy * resp_grid * time_factor
    
    # --- 9. ADVECTION - OCEANS MAY HAVE SLIGHTLY MORE ADVECTION ---
    dx = 27830 * cos_lat
    dy = 27830 * torch.ones_like(cos_lat)
    
    dT_dx = (torch.roll(T0, -1, dims=1) - torch.roll(T0, 1, dims=1)) / (2 * dx)
    dT_dy = (torch.roll(T0, -1, dims=0) - torch.roll(T0, 1, dims=0)) / (2 * dy)
    
    dT_dx[:, 0] = dT_dx[:, 1]
    dT_dx[:, -1] = dT_dx[:, -2]
    dT_dy[0, :] = dT_dy[1, :]
    dT_dy[-1, :] = dT_dy[-2, :]
    
    advection = -(U * dT_dx + V * dT_dy)
    # Oceans can have slightly more advection (ocean currents)
    advection_factor = torch.where(is_land, 0.08, 0.12)
    delta_T_adv = advection * delta_t_sec * time_factor * advection_factor
    
    # --- 10. COMBINE ALL EFFECTS ---
    delta_T_total = diurnal_adjustment + delta_T_solar + delta_T_energy + delta_T_adv
    
    # --- 11. STRICT RATE LIMITS - OCEANS VERY STABLE ---
    # Land: larger changes allowed
    max_warming_land = 10.0 * time_factor
    max_cooling_land = -8.0 * time_factor
    
    # Ocean: very small changes allowed
    max_warming_ocean = 3.0 * time_factor  # Reduced from 5.0
    max_cooling_ocean = -2.0 * time_factor  # Reduced from -3.0
    
    max_warming = torch.where(is_land, max_warming_land, max_warming_ocean)
    max_cooling = torch.where(is_land, max_cooling_land, max_cooling_ocean)
    
    # Special pre-dawn limits
    if torch.any(pre_dawn):
        # Pre-dawn: even stricter for oceans
        max_pre_dawn_warming = torch.where(is_land[pre_dawn], 0.1, 0.05) * time_factor
        pre_dawn_max_cooling = max_cooling[pre_dawn]
        
        delta_T_total[pre_dawn] = torch.clamp(
            delta_T_total[pre_dawn], 
            pre_dawn_max_cooling, 
            max_pre_dawn_warming
        )
    
    # Apply general rate limits
    delta_T_total = torch.clamp(delta_T_total, max_cooling, max_warming)
    
    T_pred = T0 + delta_T_total
    
    # --- 12. OCEAN STABILIZATION - STRONGER ---
    # Oceans are very stable - damp changes heavily
    ocean_damping = 0.95  # Increased from 0.92
    T_pred[is_ocean] = (ocean_damping * T_pred[is_ocean] + 
                       (1 - ocean_damping) * T0[is_ocean])
    
    # For land, also apply some damping but less
    land_damping = 0.85
    T_pred[is_land] = (land_damping * T_pred[is_land] + 
                      (1 - land_damping) * T0[is_land])
    
    # --- 13. SPATIAL SMOOTHING ---
    smoothing = 0.25  # Increased smoothing for more realistic fields
    
    kernel = torch.ones(3, 3, device=device) / 9.0
    
    T_smooth = torch.zeros_like(T_pred)
    for i in range(1, H-1):
        for j in range(1, W-1):
            patch = T_pred[i-1:i+2, j-1:j+2]
            T_smooth[i, j] = torch.sum(patch * kernel)
    
    T_smooth[0, :] = T_pred[0, :]
    T_smooth[-1, :] = T_pred[-1, :]
    T_smooth[:, 0] = T_pred[:, 0]
    T_smooth[:, -1] = T_pred[:, -1]
    
    T_pred = (1 - smoothing) * T_pred + smoothing * T_smooth
    
    # --- 14. TEMPERATURE RANGE ---
    T_pred = torch.clamp(T_pred, 240.0, 310.0)
    
    # Don't go below dew point
    min_T = Td - 3.0
    T_pred = torch.maximum(T_pred, min_T)
    
    # --- 15. FINAL ADJUSTMENTS ---
    global_adjust = -0.1 * time_factor
    # Land gets slight warm bias, ocean gets slight cool bias
    land_adjust = torch.where(is_land, 0.1 * time_factor, -0.05 * time_factor)
    humid_adjust = -0.05 * RH
    
    total_adjust = global_adjust + land_adjust + humid_adjust
    T_final = T_pred + total_adjust
    
    # Final bounds
    T_final = torch.clamp(T_final, 240.0, 310.0)
    
    # --- 16. DEBUG: Check land vs ocean sensitivity ---
    if delta_hours == 6:
        # Compare land and ocean daytime warming
        land_day = is_land & is_day_future
        ocean_day = is_ocean & is_day_future
        
        if torch.any(land_day) and torch.any(ocean_day):
            avg_land_warming = torch.mean(T_final[land_day] - T0[land_day]).item()
            avg_ocean_warming = torch.mean(T_final[ocean_day] - T0[ocean_day]).item()
            avg_land_cos_z = torch.mean(cos_z_future[land_day]).item()
            avg_ocean_cos_z = torch.mean(cos_z_future[ocean_day]).item()
            
            print(f"6h forecast - Daytime sensitivity:")
            print(f"  Land: ΔT={avg_land_warming:.2f}K (cos_z={avg_land_cos_z:.2f})")
            print(f"  Ocean: ΔT={avg_ocean_warming:.2f}K (cos_z={avg_ocean_cos_z:.2f})")
            if avg_land_cos_z > 0.1 and avg_ocean_cos_z > 0.1:
                sensitivity_ratio = avg_land_warming / max(avg_ocean_warming, 0.1)
                print(f"  Land is {sensitivity_ratio:.1f}x more sensitive to sunshine")
    
    return T_final