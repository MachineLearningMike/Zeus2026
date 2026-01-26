"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (H, W) in lat and lon,
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).

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

Please write a class code that predicts t2m at time t + delta hours, by fastest physical logic and not by tedious ML or numerical analysis. delta is in range: 0, 1, ..., 12
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import numpy as np
import math
from datetime import datetime
from typing import Optional, Union
import torch.nn.functional as F

class FastPhysicsT2MPredictor:
    """
    Fast physics-based temperature prediction with proper continent thermal inertia.
    FIXED: Greenland cooler bias, mid-latitude lands faster diurnal response.
    """
    
    def __init__(self, device: Optional[str] = None, return_cpu: bool = True):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_cpu = return_cpu
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize all physical constants with FIXED values."""
        # Thermal inertia (higher = slower response)
        self.INERTIA_OCEAN = torch.tensor(0.10, device=self.device)      # Ocean: fast response
        self.INERTIA_LAND = torch.tensor(0.35, device=self.device)       # FIXED: Land slower (was 0.25)
        self.INERTIA_ICE = torch.tensor(0.95, device=self.device)        # Ice: extremely slow
        
        # Albedo (reflectivity)
        self.ALBEDO_OCEAN = torch.tensor(0.08, device=self.device)
        self.ALBEDO_LAND = torch.tensor(0.20, device=self.device)
        self.ALBEDO_ICE = torch.tensor(0.85, device=self.device)         # Ice high albedo
        
        # Base temperatures (K)
        self.BASE_TROPICAL = torch.tensor(300.0, device=self.device)
        self.BASE_MIDLAT = torch.tensor(285.0, device=self.device)
        self.BASE_ARCTIC = torch.tensor(255.0, device=self.device)       # FIXED: Colder Arctic
        self.BASE_ANTARCTIC = torch.tensor(200.0, device=self.device)
        self.BASE_GREENLAND = torch.tensor(240.0, device=self.device)    # NEW: Specific Greenland base temp
        
        # Diurnal amplitudes (higher = stronger day-night cycle)
        # FIXED: Increased land amplitude for faster response
        self.AMPLITUDE_OCEAN = torch.tensor(0.8, device=self.device)
        self.AMPLITUDE_LAND = torch.tensor(2.5, device=self.device)      # FIXED: Increased (was 1.5)
        self.AMPLITUDE_MIDLAT_LAND = torch.tensor(3.0, device=self.device) # NEW: Higher for mid-latitudes
        self.AMPLITUDE_ICE = torch.tensor(0.3, device=self.device)
        
        # Elevation cooling (K per 1000m)
        self.LAPSE_RATE = torch.tensor(-6.5 / 1000.0, device=self.device)
        
        # Region thresholds
        self.ANTARCTICA_LAT = torch.tensor(-60.0, device=self.device)
        self.GREENLAND_LAT = torch.tensor(60.0, device=self.device)
        self.GREENLAND_LON_MIN = torch.tensor(-60.0, device=self.device)  # NEW: Greenland bounding box
        self.GREENLAND_LON_MAX = torch.tensor(-20.0, device=self.device)
        self.ICE_ELEVATION = torch.tensor(1000.0, device=self.device)
        
        # Mid-latitude range for special treatment
        self.MIDLAT_MIN = torch.tensor(30.0, device=self.device)
        self.MIDLAT_MAX = torch.tensor(60.0, device=self.device)
        
        # Other factors
        self.ADVECTION_FACTOR = torch.tensor(0.005, device=self.device)
        self.SMOOTHING = torch.tensor(0.15, device=self.device)
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                current_dt: datetime, geo_tempo_f: torch.Tensor,
                delta_hours: int, return_tensor: Optional[bool] = None) -> Union[torch.Tensor, np.ndarray]:
        
        delta_hours = max(0, min(12, delta_hours))
        if delta_hours == 0:
            result = meteo[0].clone()
            return self._convert_result(result, return_tensor)
        
        # Move to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T_current = meteo[0]           # Current temperature
        U = meteo[2]                   # U-wind
        V = meteo[3]                   # V-wind
        Td = meteo[4]                  # Dew point
        orograph = geo_dense[0]        # Elevation
        
        sin_lat = geo_dense[1]         # sin(latitude)
        cos_lat = geo_dense[2]         # cos(latitude)
        sin_lon = geo_dense[3]         # sin(longitude)
        cos_lon = geo_dense[4]         # cos(longitude)
        
        land_mask = geo_cat[0]         # 0=sea, 1=land
        soil_type = geo_cat[1]         # Soil type
        
        # Get latitude and longitude in degrees
        lat_rad = torch.asin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lat_deg = torch.rad2deg(lat_rad)
        
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lon_deg = torch.rad2deg(lon_rad)
        
        # Solar zenith
        cos_z_current = torch.clamp(geo_tempo[3], 0.0, 1.0)
        cos_z_future = torch.clamp(geo_tempo_f[3], 0.0, 1.0)
        
        # --- IDENTIFY SPECIAL REGIONS ---
        # FIXED: Better Greenland detection
        is_greenland = (
            (lat_deg > self.GREENLAND_LAT) & 
            (lon_deg > self.GREENLAND_LON_MIN) & 
            (lon_deg < self.GREENLAND_LON_MAX) &
            (orograph > 500)  # Greenland has high elevation
        )
        
        is_antarctica = (lat_deg < self.ANTARCTICA_LAT) & (orograph > 500)
        is_ice = is_antarctica | is_greenland
        
        # FIXED: Identify mid-latitude lands
        is_midlat = (torch.abs(lat_deg) >= self.MIDLAT_MIN) & (torch.abs(lat_deg) <= self.MIDLAT_MAX)
        is_midlat_land = is_midlat & (land_mask == 1) & (~is_ice)
        
        # Basic land/ocean masks
        is_land = (land_mask == 1) & (~is_ice) & (~is_midlat_land)  # Regular land (not mid-lat, not ice)
        is_ocean = (land_mask == 0)
        
        # --- COMPUTE REGION PROPERTIES ---
        inertia = torch.zeros_like(lat_deg, device=self.device)
        albedo = torch.zeros_like(lat_deg, device=self.device)
        amplitude = torch.zeros_like(lat_deg, device=self.device)
        T_base = torch.zeros_like(lat_deg, device=self.device)
        
        # Ocean
        inertia = torch.where(is_ocean, self.INERTIA_OCEAN, inertia)
        albedo = torch.where(is_ocean, self.ALBEDO_OCEAN, albedo)
        amplitude = torch.where(is_ocean, self.AMPLITUDE_OCEAN, amplitude)
        
        # Regular land (non-midlatitude)
        inertia = torch.where(is_land, self.INERTIA_LAND, inertia)
        albedo = torch.where(is_land, self.ALBEDO_LAND, albedo)
        amplitude = torch.where(is_land, self.AMPLITUDE_LAND, amplitude)
        
        # FIXED: Mid-latitude lands get special treatment - faster response
        inertia = torch.where(is_midlat_land, self.INERTIA_LAND * 0.8, inertia)  # 20% faster response
        albedo = torch.where(is_midlat_land, self.ALBEDO_LAND, albedo)
        amplitude = torch.where(is_midlat_land, self.AMPLITUDE_MIDLAT_LAND, amplitude)
        
        # Ice regions
        inertia = torch.where(is_ice, self.INERTIA_ICE, inertia)
        albedo = torch.where(is_ice, self.ALBEDO_ICE, albedo)
        amplitude = torch.where(is_ice, self.AMPLITUDE_ICE, amplitude)
        
        # --- BASE TEMPERATURES ---
        # FIXED: Greenland gets its own colder base temperature
        abs_lat = torch.abs(lat_deg)
        is_tropical = abs_lat < 30.0
        is_midlat_zone = (abs_lat >= 30.0) & (abs_lat < 60.0)
        is_arctic = (lat_deg >= 60.0) & (~is_greenland)  # Arctic but not Greenland
        is_antarctic_region = (lat_deg <= -60.0) & (~is_antarctica)
        
        # Apply base temperatures
        T_base = torch.where(is_tropical, self.BASE_TROPICAL, T_base)
        T_base = torch.where(is_midlat_zone, self.BASE_MIDLAT, T_base)
        T_base = torch.where(is_arctic, self.BASE_ARCTIC, T_base)
        T_base = torch.where(is_antarctic_region, self.BASE_ANTARCTIC, T_base)
        
        # FIXED: Greenland gets specific colder temperature
        T_base = torch.where(is_greenland, self.BASE_GREENLAND, T_base)
        
        # Antarctica ice sheet
        T_base = torch.where(is_antarctica, self.BASE_ANTARCTIC - 10.0, T_base)  # Even colder
        
        # FIXED: Mid-latitude lands get adjusted base temperature for faster response
        T_base = torch.where(is_midlat_land, 
                            T_base * 0.95 + T_current * 0.05,  # Slight adjustment toward current
                            T_base)
        
        # --- COMPUTE EFFECTS ---
        
        # 1. Solar effect with albedo
        # FIXED: Mid-latitude lands get stronger solar response
        solar_factor = (1.0 - albedo) * (1.0 - inertia)
        solar_change = (cos_z_future - cos_z_current) * solar_factor * 0.2
        
        # FIXED: Amplify solar effect for mid-latitude lands
        solar_amplification = torch.where(is_midlat_land, 1.5, 1.0)
        solar_effect = solar_change * amplitude * 5.0 * solar_amplification
        
        # 2. Relaxation toward base temperature
        # FIXED: Mid-latitude lands relax faster
        T_diff = T_base - T_current
        relaxation_speed = torch.where(is_midlat_land, 0.3, 0.2)  # Faster for mid-lat lands
        relaxation = (1.0 - inertia) * relaxation_speed
        base_effect = T_diff * relaxation * (delta_hours / 6.0)
        
        # 3. Elevation cooling
        elevation_effect = orograph * self.LAPSE_RATE * 0.1
        
        # FIXED: Extra elevation cooling for Greenland
        greenland_elevation_factor = torch.where(is_greenland, 1.5, 1.0)
        elevation_effect = elevation_effect * greenland_elevation_factor
        
        # 4. Advection
        advection_effect = self._compute_advection(
            T_current, U, V, cos_lat, inertia, delta_hours
        )
        
        # 5. FIXED: Wind chill effect for Greenland
        wind_speed = torch.sqrt(U**2 + V**2)
        wind_chill = torch.where(is_greenland, -wind_speed * 0.1, 0.0)
        
        # 6. Combine effects
        delta_T = solar_effect + base_effect + elevation_effect + advection_effect + wind_chill
        
        # 7. Rate limits
        # FIXED: Mid-latitude lands can change faster
        max_rate_base = torch.where(is_ice, 0.5, torch.where(is_midlat_land, 2.0, 1.0))
        max_rate = max_rate_base * (delta_hours / 6.0)
        delta_T = torch.clamp(delta_T, -max_rate, max_rate)
        
        # 8. Predict temperature
        T_pred = T_current + delta_T
        
        # 9. FIXED: Temperature bounds - Greenland should be colder
        min_T = torch.where(is_greenland, 220.0, 
                           torch.where(is_ice, 220.0, 240.0))
        max_T = torch.where(is_greenland, 270.0,
                           torch.where(is_ice, 270.0, 320.0))
        T_pred = torch.clamp(T_pred, min_T, max_T)
        T_pred = torch.maximum(T_pred, Td - 5.0)
        
        # 10. Smoothing (less smoothing for mid-latitude lands)
        smoothing_factor = torch.where(is_midlat_land, 0.05, self.SMOOTHING)
        T_pred = self._apply_smoothing_with_factor(T_pred, smoothing_factor)
        
        return self._convert_result(T_pred, return_tensor)
    
    def _compute_advection(self, T: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                          cos_lat: torch.Tensor, inertia: torch.Tensor,
                          delta_hours: int) -> torch.Tensor:
        """Compute advection effect."""
        T_left = torch.roll(T, shifts=-1, dims=1)
        T_right = torch.roll(T, shifts=1, dims=1)
        T_up = torch.roll(T, shifts=-1, dims=0)
        T_down = torch.roll(T, shifts=1, dims=0)
        
        dT_dx = (T_right - T_left) / 2.0
        dT_dy = (T_down - T_up) / 2.0
        dT_dy[0, :] = 0
        dT_dy[-1, :] = 0
        
        advection = -(U * dT_dx + V * dT_dy) * 0.001
        inertia_factor = 1.0 - inertia
        effect = advection * inertia_factor * self.ADVECTION_FACTOR * delta_hours
        
        return effect
    
    def _apply_smoothing_with_factor(self, field: torch.Tensor, smoothing_factor: torch.Tensor) -> torch.Tensor:
        """Apply variable smoothing based on region."""
        field_4d = field.unsqueeze(0).unsqueeze(0)
        smoothed = F.avg_pool2d(field_4d, kernel_size=3, stride=1, padding=1).squeeze()
        
        # Apply different smoothing factors to different regions
        result = field * (1 - smoothing_factor) + smoothed * smoothing_factor
        return result
    
    def _apply_smoothing(self, field: torch.Tensor) -> torch.Tensor:
        """Apply default smoothing."""
        return self._apply_smoothing_with_factor(field, self.SMOOTHING)
    
    def _convert_result(self, result: torch.Tensor, return_tensor: Optional[bool]):
        if return_tensor is None:
            return_tensor = not self.return_cpu
        return result if return_tensor else result.cpu().numpy()
    
    def to(self, device: str) -> 'FastPhysicsT2MPredictor':
        self.device = device
        self._setup_constants()
        return self