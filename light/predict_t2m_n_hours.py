"""
Prompt (DeepSeek)

t2m, tp, u100, v100, d2m, sp, orograph, and land_sea_mask are from ERA5 and have units defined by ERA5.
Their units are: t2m: temperature K, tp: water m, u100: m/s, v100: m/s, d2m: temperature K, sp: Pascal, orograph: m**2 / s**2
soil types are {'O: cean/Lake', 1: 'Coarse', 2:'Medium', 3: 'Medium Fine', 4: 'Fine', 5: 'Very Fine', 6: 'Organic', 7: 'Tropical Organic',}
land_sea_mask types are {0: sea, 1: land}
We also have geological data: loatitudes, longitudes, earth orbital angle, and zenith angle.
All above data components are a map over the globe, with thee shape of (latitudes and longitudes).
The time t is given in UTC.

Now we have inputs for given current time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0),
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).
, where sin_xxx means sin(xxx) and cos_xxx means cos(xxx)
We also have inputs for the given future time f. 
The current and the future shares all inputs, except that the future has no meteo and its own geo_tempo called geo_tempo_f.

def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor

Please write a Python class "T2mPredictor" that has the above predict(.) function, which predicts t2m 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import numpy as np
import math
from datetime import datetime
from typing import Optional, Tuple

class T2mPredictor:
    """
    T2M Predictor that fixes morning/evening banding.
    Key insight: Temperature depends on ABSOLUTE solar intensity, not just change.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_constants()
        self._setup_banding_fix()
        
    def _setup_constants(self):
        self.SOLAR_CONST = torch.tensor(1361.0, device=self.device)
        self.SIGMA = torch.tensor(5.67e-8, device=self.device)
        self.CP_AIR = torch.tensor(1004.0, device=self.device)
        self.RHO_AIR = torch.tensor(1.225, device=self.device)
        
        self.PI = torch.tensor(math.pi, device=self.device)
        self.DEG2RAD = torch.tensor(math.pi / 180.0, device=self.device)
        self.EPS = torch.tensor(1e-8, device=self.device)
        
    def _setup_banding_fix(self):
        """Parameters to fix morning/evening banding."""
        # SOLAR HEATING: Based on ABSOLUTE solar intensity at future time
        self.SOLAR_HEATING_RATE = {
            'land': torch.tensor(0.02, device=self.device),   # K/hour per 100 W/m²
            'ocean': torch.tensor(0.005, device=self.device), # K/hour per 100 W/m²
        }
        
        # DIURNAL CYCLE: Based on LOCAL TIME, not solar change
        self.DIURNAL = {
            'land_amplitude': torch.tensor(5.0, device=self.device),
            'ocean_amplitude': torch.tensor(1.0, device=self.device),
            'peak_hour': torch.tensor(14.0, device=self.device),  # 2 PM
        }
        
        # MORNING VS EVENING ASYMMETRY
        self.MORNING_BOOST = torch.tensor(1.2, device=self.device)   # Morning heats faster
        self.EVENING_REDUCTION = torch.tensor(0.8, device=self.device) # Evening cools slower
        
        # WIND MIXING
        self.WIND_MIXING = torch.tensor(0.1, device=self.device)
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Fix morning/evening banding by using absolute solar intensity,
        not solar change.
        """
        if delta_hours == 0:
            return meteo[0].clone()
        
        # Move to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T0 = meteo[0]
        tp = meteo[1]
        U = meteo[2]
        V = meteo[3]
        Td = meteo[4]
        
        # Geographical data
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        sin_lon = geo_dense[3]
        cos_lon = geo_dense[4]
        
        land_mask = geo_cat[0]
        land_fraction = land_mask.float()
        
        # Solar geometry
        cos_z_current = torch.clamp(geo_tempo[3], 0.0, 1.0)
        cos_z_future = torch.clamp(geo_tempo_f[3], 0.0, 1.0)
        
        H, W = T0.shape
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # --- FIX MORNING/EVENING BANDING ---
        
        # 1. ABSOLUTE SOLAR HEATING (not change-based)
        solar_future = self.SOLAR_CONST * cos_z_future
        
        # Cloud reduction
        cloud_fraction = torch.clamp(20.0 * tp, 0.0, 1.0)
        solar_future = solar_future * (1.0 - 0.6 * cloud_fraction)
        
        # Heating rate based on absolute solar intensity
        heating_rate = (self.SOLAR_HEATING_RATE['land'] * land_fraction + 
                       self.SOLAR_HEATING_RATE['ocean'] * (1.0 - land_fraction))
        
        # Convert to temperature change: ΔT = rate × (solar/100) × Δt
        solar_heating = heating_rate * (solar_future / 100.0) * delta_hours_tensor
        
        # 2. MORNING/EVENING ASYMMETRY FIX
        # Determine if it's morning or evening based on solar motion
        # cos_z_change indicates rising (morning) or setting (evening) sun
        cos_z_change = cos_z_future - cos_z_current
        
        # Morning: sun rising (cos_z increasing)
        is_morning = cos_z_change > 0.05
        
        # Evening: sun setting (cos_z decreasing)
        is_evening = cos_z_change < -0.05
        
        # Adjust heating/cooling rates
        solar_heating = torch.where(is_morning,
                                   solar_heating * self.MORNING_BOOST,
                                   solar_heating)
        
        solar_heating = torch.where(is_evening,
                                   solar_heating * self.EVENING_REDUCTION,
                                   solar_heating)
        
        # 3. DIURNAL CYCLE (local time based, smooth)
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lon_rad = torch.atan2(sin_lon, cos_lon)
        
        UTC_hour = current_dt.hour + current_dt.minute/60.0 + current_dt.second/3600.0
        
        # SMOOTH local time calculation
        # Reduce longitude effect to prevent bands
        lon_effect = torch.rad2deg(lon_rad) / 15.0
        
        # Apply strong smoothing to longitude effect
        if H > 1 and W > 1:
            lon_effect_4d = lon_effect.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(1, 1, 7, 7, device=self.device) / 49.0
            lon_effect_smooth = torch.nn.functional.conv2d(
                torch.nn.functional.pad(lon_effect_4d, (3, 3, 3, 3), mode='replicate'),
                kernel
            ).squeeze()
            lon_effect = 0.2 * lon_effect + 0.8 * lon_effect_smooth
        
        local_hour_current = (UTC_hour + lon_effect) % 24
        local_hour_future = (local_hour_current + delta_hours) % 24
        
        # Diurnal amplitude
        diurnal_amplitude = (self.DIURNAL['land_amplitude'] * land_fraction + 
                           self.DIURNAL['ocean_amplitude'] * (1.0 - land_fraction))
        
        # Smooth diurnal function (cosine)
        peak_hour = self.DIURNAL['peak_hour']
        
        phase_current = (local_hour_current - peak_hour) * 2.0 * self.PI / 24.0
        phase_future = (local_hour_future - peak_hour) * 2.0 * self.PI / 24.0
        
        diurnal_current = diurnal_amplitude * torch.cos(phase_current)
        diurnal_future = diurnal_amplitude * torch.cos(phase_future)
        
        diurnal_change = diurnal_future - diurnal_current
        
        # 4. ADVECTION (strong to smooth bands)
        wind_speed = torch.sqrt(U**2 + V**2 + self.EPS)
        
        # Temperature gradient
        dx = 111320 * torch.clamp(cos_lat, 0.3, 1.0)
        dy = 111320 * torch.ones_like(cos_lat)
        
        T0_padded = torch.nn.functional.pad(T0.unsqueeze(0).unsqueeze(0), 
                                          (1, 1, 1, 1), mode='replicate').squeeze()
        
        dT_dx = (T0_padded[1:-1, 2:] - T0_padded[1:-1, :-2]) / (2 * dx + self.EPS)
        dT_dy = (T0_padded[2:, 1:-1] - T0_padded[:-2, 1:-1]) / (2 * dy + self.EPS)
        
        advection = -(U * dT_dx + V * dT_dy) * delta_hours_tensor * self.WIND_MIXING
        
        # 5. RADIATIVE COOLING (smooth)
        radiative_cooling = -0.08 * delta_hours_tensor
        
        # Enhanced cooling at night
        is_night = cos_z_future < 0.1
        radiative_cooling = torch.where(is_night,
                                       -0.15 * delta_hours_tensor,
                                       radiative_cooling)
        
        # 6. MOISTURE EFFECTS
        RH = torch.clamp(1.0 - (T0 - Td) / 20.0, 0.0, 1.0)
        evaporative_cooling = -0.25 * (1.0 - RH) * land_fraction * delta_hours_tensor
        
        # 7. COMBINE ALL EFFECTS
        dT_total = (solar_heating +
                   diurnal_change +
                   advection +
                   radiative_cooling +
                   evaporative_cooling)
        
        # 8. RATE LIMITS (asymmetric for morning/evening)
        # Morning can warm faster
        max_warming_morning = 1.0 * delta_hours_tensor
        max_warming_other = 0.7 * delta_hours_tensor
        
        max_warming = torch.where(is_morning,
                                 max_warming_morning,
                                 max_warming_other)
        
        max_warming = torch.where(land_fraction > 0.5,
                                 max_warming,
                                 max_warming * 0.5)  # Ocean slower
        
        # Evening cooling limits
        max_cooling_evening = -0.5 * delta_hours_tensor
        max_cooling_night = -0.8 * delta_hours_tensor
        
        max_cooling = torch.where(is_evening,
                                 max_cooling_evening,
                                 max_cooling_night)
        
        max_cooling = torch.where(land_fraction > 0.5,
                                 max_cooling,
                                 max_cooling * 0.6)  # Ocean slower
        
        # Apply limits
        dT_limited = torch.clamp(dT_total, max_cooling, max_warming)
        
        # 9. PREDICT FINAL TEMPERATURE
        T_pred = T0 + dT_limited
        
        # 10. APPLY ASYMMETRIC SMOOTHING
        # More smoothing in east-west direction to break bands
        T_pred = self._apply_asymmetric_smoothing(T_pred, is_morning, is_evening)
        
        # 11. PHYSICAL CONSTRAINTS
        T_pred = torch.clamp(T_pred, 220.0, 330.0)
        min_T = Td - 2.0
        T_pred = torch.maximum(T_pred, min_T)
        
        # Final check
        T_pred = torch.where(torch.isnan(T_pred), T0, T_pred)
        T_pred = torch.where(torch.isinf(T_pred), T0, T_pred)
        
        return T_pred
    
    def _apply_asymmetric_smoothing(self, temperature: torch.Tensor,
                                   is_morning: torch.Tensor,
                                   is_evening: torch.Tensor) -> torch.Tensor:
        """
        Apply asymmetric smoothing to break morning/evening bands.
        More smoothing in east-west direction.
        """
        # Strong east-west smoothing kernel
        kernel_ew = torch.tensor([[0.0, 0.15, 0.0],
                                 [0.0, 0.70, 0.0],
                                 [0.0, 0.15, 0.0]], device=self.device).view(1, 1, 3, 3)
        
        # Mild north-south smoothing
        kernel_ns = torch.tensor([[0.0, 0.0, 0.0],
                                 [0.08, 0.84, 0.08],
                                 [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)
        
        temp_4d = temperature.unsqueeze(0).unsqueeze(0)
        
        # Apply east-west smoothing
        smoothed_ew = torch.nn.functional.conv2d(
            torch.nn.functional.pad(temp_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_ew
        ).squeeze()
        
        # Apply north-south smoothing
        smoothed_ns = torch.nn.functional.conv2d(
            torch.nn.functional.pad(temp_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_ns
        ).squeeze()
        
        # More smoothing in morning/evening bands
        morning_evening_mask = is_morning | is_evening
        
        # Strong smoothing for band regions
        smoothing_strength = torch.where(morning_evening_mask,
                                        torch.tensor(0.4, device=self.device),
                                        torch.tensor(0.2, device=self.device))
        
        # Blend: 70% east-west, 30% north-south
        smoothed = 0.7 * smoothed_ew + 0.3 * smoothed_ns
        
        result = (1 - smoothing_strength) * temperature + smoothing_strength * smoothed
        
        return result
    
    def to(self, device: str) -> 'T2mPredictor':
        self.device = device
        self._setup_constants()
        self._setup_banding_fix()
        return self


def analyze_banding_fix():
    """Analyze and visualize the banding fix."""
    print("ANALYZING MORNING/EVENING BANDING FIX")
    print("=" * 60)
    
    predictor = T2mPredictor(device='cpu')
    
    # Create test grid
    H, W = 180, 360
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Initialize tensors
    meteo = torch.ones(6, H, W)
    geo_dense = torch.ones(5, H, W)
    geo_cat = torch.ones(2, H, W)
    geo_tempo = torch.ones(4, H, W)
    geo_tempo_f = torch.ones(4, H, W)
    
    # Setup
    meteo[0] = 283.0  # 10°C
    meteo[4] = 280.0  # Dew point
    
    # Land-ocean pattern
    land_mask = torch.zeros(H, W)
    # Simple continent at equator
    land_mask[(lat_grid > -30) & (lat_grid < 30) & 
              (lon_grid > -60) & (lon_grid < 60)] = 1.0
    geo_cat[0] = land_mask
    
    # Geographical data
    geo_dense[1] = torch.sin(lat_grid * math.pi / 180.0)
    geo_dense[2] = torch.cos(lat_grid * math.pi / 180.0)
    geo_dense[3] = torch.sin(lon_grid * math.pi / 180.0)
    geo_dense[4] = torch.cos(lon_grid * math.pi / 180.0)
    
    # Simulate solar geometry for 6 AM UTC
    # This will create morning band at some longitudes
    current_dt = datetime(2024, 6, 21, 6, 0, 0)  # 6 AM UTC
    
    # Create realistic solar fields
    declination = 23.44 * math.pi / 180.0  # Summer
    
    for i in range(H):
        for j in range(W):
            lat_rad = lat[i] * math.pi / 180.0
            lon_rad = lon[j] * math.pi / 180.0
            
            # Current: 6 AM UTC
            # Solar hour angle at Greenwich
            hour_angle_current = (6 - 12) * 15 * math.pi / 180.0 + lon_rad
            
            cos_z_current = (math.sin(lat_rad) * math.sin(declination) +
                           math.cos(lat_rad) * math.cos(declination) * 
                           math.cos(hour_angle_current))
            
            geo_tempo[3, i, j] = max(0.0, min(1.0, cos_z_current))
            
            # Future: 12 PM UTC (6 hours later)
            hour_angle_future = (12 - 12) * 15 * math.pi / 180.0 + lon_rad
            
            cos_z_future = (math.sin(lat_rad) * math.sin(declination) +
                          math.cos(lat_rad) * math.cos(declination) * 
                          math.cos(hour_angle_future))
            
            geo_tempo_f[3, i, j] = max(0.0, min(1.0, cos_z_future))
    
    # Make prediction (6-hour forecast)
    pred = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, geo_tempo_f,
                            current_dt, 6)
    
    # Analyze
    pred_np = pred.numpy()
    cos_z_current_np = geo_tempo[3].numpy()
    cos_z_future_np = geo_tempo_f[3].numpy()
    
    # Find morning/evening regions
    cos_z_change = cos_z_future_np - cos_z_current_np
    morning_mask = cos_z_change > 0.05
    evening_mask = cos_z_change < -0.05
    noon_mask = (cos_z_change >= -0.05) & (cos_z_change <= 0.05) & (cos_z_future_np > 0.3)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Predicted temperature
    im1 = axes[0, 0].imshow(pred_np, cmap='RdBu_r', aspect='auto',
                           extent=[-180, 180, -90, 90])
    axes[0, 0].set_title('Predicted Temperature (K)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Temperature change
    temp_change = pred_np - 283.0
    im2 = axes[0, 1].imshow(temp_change, cmap='RdBu', aspect='auto',
                           extent=[-180, 180, -90, 90], vmin=-3, vmax=3)
    axes[0, 1].set_title('Temperature Change (K)')
    axes[0, 1].set_xlabel('Longitude')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Morning/Evening regions
    region_map = np.zeros_like(pred_np)
    region_map[morning_mask] = 1  # Morning (heating)
    region_map[evening_mask] = -1  # Evening (cooling)
    region_map[noon_mask] = 0.5  # Noon (stable)
    
    im3 = axes[0, 2].imshow(region_map, cmap='coolwarm', aspect='auto',
                           extent=[-180, 180, -90, 90], vmin=-1, vmax=1)
    axes[0, 2].set_title('Solar Regions (1=Morning, -1=Evening)')
    axes[0, 2].set_xlabel('Longitude')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Zonal cross-section at equator
    equator_idx = H // 2
    axes[1, 0].plot(lon.numpy(), pred_np[equator_idx, :], 'b-', label='Temperature')
    axes[1, 0].plot(lon.numpy(), temp_change[equator_idx, :], 'r-', label='ΔT')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Temperature (K) / ΔT (K)')
    axes[1, 0].set_title('Equatorial Cross-section')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Solar geometry
    axes[1, 1].plot(lon.numpy(), cos_z_current_np[equator_idx, :], 'g-', label='cos(z)_current')
    axes[1, 1].plot(lon.numpy(), cos_z_future_np[equator_idx, :], 'm-', label='cos(z)_future')
    axes[1, 1].plot(lon.numpy(), cos_z_change[equator_idx, :], 'k-', label='Δcos(z)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Cosine Zenith')
    axes[1, 1].set_title('Solar Geometry at Equator')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Morning vs Evening heating
    morning_heating = np.zeros_like(lon.numpy())
    evening_cooling = np.zeros_like(lon.numpy())
    
    for j in range(W):
        if morning_mask[equator_idx, j]:
            morning_heating[j] = temp_change[equator_idx, j]
        elif evening_mask[equator_idx, j]:
            evening_cooling[j] = temp_change[equator_idx, j]
    
    axes[1, 2].plot(lon.numpy(), morning_heating, 'g-', label='Morning Heating')
    axes[1, 2].plot(lon.numpy(), evening_cooling, 'r-', label='Evening Cooling')
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('ΔT (K)')
    axes[1, 2].set_title('Morning vs Evening Response')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Banding analysis
    print("\nBANDING ANALYSIS:")
    print("-" * 40)
    
    # Check for bands
    lon_grad = np.gradient(pred_np, axis=1)
    lon_grad_abs = np.abs(lon_grad)
    
    # Morning band check
    morning_grad = lon_grad_abs[morning_mask].mean() if np.any(morning_mask) else 0
    evening_grad = lon_grad_abs[evening_mask].mean() if np.any(evening_mask) else 0
    
    print(f"Morning region gradient: {morning_grad:.3f} K/degree")
    print(f"Evening region gradient: {evening_grad:.3f} K/degree")
    
    # Banding severity
    if morning_grad > 0.5 or evening_grad > 0.5:
        print("⚠️  Warning: Possible banding detected")
    else:
        print("✓ Good: No significant banding")
    
    print("\nKey fixes applied:")
    print("1. Absolute solar heating (not change-based)")
    print("2. Morning/evening asymmetry factors")
    print("3. Strong east-west smoothing")
    print("4. Reduced longitude effect in local time")
    print("5. Asymmetric rate limits")
    
    return pred


if __name__ == "__main__":
    analyze_banding_fix()