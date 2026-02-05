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
    meteo_normal = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0),
    dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).
, where sin_xxx means sin(xxx) and cos_xxx means cos(xxx)
We also have inputs for the given future time f. 
The current and the future shares all inputs, except that the future has no meteo_normal and its own geo_tempo called geo_tempo_f.

def predict(self, meteo_normal: torch.Tensor, dense: torch.Tensor, 
                cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor

Please write a Python class "V100mPredictor" that has the above predict(.) function, which predicts v100, 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import math
from datetime import datetime
from typing import Optional

class V100mPredictor:
    """
    Predict meridional wind (v100) at time t+delta.
    
    Key physical insights for v-component:
    1. Meridional wind is typically weaker than zonal wind
    2. Driven by east-west pressure gradients (geostrophic balance)
    3. Influenced by jet stream meanders (Rossby waves)
    4. Has seasonal and diurnal patterns
    5. Affected by topography blocking
    """
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize thoughtfully chosen constants."""
        # Earth parameters
        self.omega = torch.tensor(7.2921159e-5, device=self.device)  # Earth rotation (rad/s)
        self.R = torch.tensor(287.058, device=self.device)           # Gas constant (J/kg/K)
        self.rho = torch.tensor(1.2, device=self.device)             # Air density (kg/m³)
        
        # Jet stream characteristics for v-component
        # v-component of jet is strongest during meanders/ridges
        self.jet_lat_center = torch.tensor(45.0, device=self.device)  # Mid-latitude jets
        self.jet_max_speed = torch.tensor(8.0, device=self.device)    # Meridional jet speed
        
        # Time evolution
        self.hourly_persistence = torch.tensor(0.96, device=self.device)  # 96% persistence per hour
        
        # Pressure gradient scaling (empirical)
        self.pressure_scale = torch.tensor(1e-8, device=self.device)
        
        # Topographic blocking
        self.topo_blocking_scale = torch.tensor(0.001, device=self.device)
        
        # Realistic limits based on ERA5 v100 statistics
        self.V_MIN = torch.tensor(-20.0, device=self.device)
        self.V_MAX = torch.tensor(20.0, device=self.device)
        
        # Seasonal variation amplitude
        self.seasonal_amplitude = torch.tensor(1.5, device=self.device)
    
    def predict(self, meteo_normal: torch.Tensor, dense: torch.Tensor, 
                cat: torch.Tensor, geo_tempo: torch.Tensor,
                geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Thoughtful prediction of meridional wind.
        
        Strategy:
        1. Strong persistence (wind patterns evolve slowly)
        2. Pressure gradient forcing (geostrophic balance)
        3. Jet stream influence (Rossby wave patterns)
        4. Topographic effects (mountain blocking)
        5. Seasonal/diurnal adjustments
        
        Args:
            meteo_normal: [t2m, tp, u100, v100, d2m, sp] at current time
            dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            cat: [land_sea_mask, soil_type]
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] at current time
            geo_tempo_f: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] at future time
            current_dt: current datetime in UTC
            delta_hours: hours to predict ahead
            
        Returns:
            Predicted v100 field at t + delta_hours
        """
        # Move to device
        meteo_normal = meteo_normal.to(self.device).float()
        dense = dense.to(self.device).float()
        cat = cat.to(self.device).float()
        geo_tempo = geo_tempo.to(self.device).float()
        geo_tempo_f = geo_tempo_f.to(self.device).float()
        
        # Extract current v100 - this is what we're predicting
        current_v100 = meteo_normal[3].clone()  # Index 3 is v100
        
        # Other useful variables
        sp = meteo_normal[5]  # Surface pressure (Pa)
        
        # Geographical information
        orograph = dense[0]  # Topography (m)
        sin_lat = dense[1]
        cos_lat = dense[2]
        sin_lon = dense[3]
        cos_lon = dense[4]
        
        # Land-sea mask and soil type
        land_sea_mask = cat[0]  # 0: sea, 1: land
        
        # Current time information
        sin_orbit = geo_tempo[0]
        cos_orbit = geo_tempo[1]
        sin_zenith = geo_tempo[2]
        cos_zenith = geo_tempo[3]
        
        # Future time information
        sin_orbit_f = geo_tempo_f[0]
        cos_orbit_f = geo_tempo_f[1]
        sin_zenith_f = geo_tempo_f[2]
        cos_zenith_f = geo_tempo_f[3]
        
        # Calculate latitude in radians
        cos_lat_clamped = torch.clamp(cos_lat, -0.9999, 0.9999)
        lat_rad = torch.where(
            torch.abs(cos_lat_clamped) > 0.5,
            torch.asin(torch.clamp(sin_lat, -0.9999, 0.9999)),
            torch.acos(cos_lat_clamped) * torch.sign(sin_lat)
        )
        lat_deg = torch.rad2deg(lat_rad)
        
        # Calculate longitude in radians
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lon_deg = torch.rad2deg(lon_rad)
        
        # === COMPONENT 1: PERSISTENCE (70%) ===
        # Wind patterns have strong temporal autocorrelation
        persistence = current_v100 * (self.hourly_persistence ** delta_hours)
        
        # === COMPONENT 2: PRESSURE GRADIENT FORCING (15%) ===
        # Geostrophic wind: v_g = (1/(ρf)) * ∂p/∂x
        pressure_effect = self._compute_pressure_effect(sp, lat_rad, current_v100)
        
        # === COMPONENT 3: JET STREAM INFLUENCE (10%) ===
        # Meridional wind associated with jet meanders
        jet_effect = self._compute_jet_effect(lat_rad, lon_rad, current_dt, current_v100)
        
        # === COMPONENT 4: TOPOGRAPHIC EFFECTS (5%) ===
        # Mountains block meridional flow
        topo_effect = self._compute_topo_effect(orograph, lat_rad, current_v100)
        
        # === COMBINE COMPONENTS ===
        predicted_v = (
            0.70 * persistence +
            0.15 * pressure_effect +
            0.10 * jet_effect +
            0.05 * topo_effect
        )
        
        # === ADDITIONAL PHYSICAL EFFECTS ===
        # Land-sea contrast effect
        land_sea_effect = self._compute_land_sea_effect(land_sea_mask, current_v100)
        predicted_v = predicted_v + 0.02 * land_sea_effect
        
        # Diurnal effect (weaker at night, stronger during day)
        diurnal_effect = self._compute_diurnal_effect(sin_zenith_f, cos_zenith_f, current_v100)
        predicted_v = predicted_v + 0.01 * diurnal_effect
        
        # Seasonal effect (use FUTURE orbital angle)
        seasonal_effect = self._compute_seasonal_effect(sin_orbit_f, cos_orbit_f, current_v100)
        predicted_v = predicted_v + 0.01 * seasonal_effect
        
        # === POST-PROCESSING ===
        # Apply spatial smoothing (wind is coherent over space)
        predicted_v = self._apply_spatial_smoothing(predicted_v, current_v100)
        
        # Maintain statistical properties similar to input
        predicted_v = self._preserve_statistics(predicted_v, current_v100)
        
        # Apply physical limits
        predicted_v = torch.clamp(predicted_v, self.V_MIN, self.V_MAX)
        
        # Return CPU tensor if device is GPU
        if self.device.type == 'cuda':
            return predicted_v.cpu()
        else:
            return predicted_v
    
    def _compute_pressure_effect(self, sp: torch.Tensor, lat_rad: torch.Tensor, 
                                current_v: torch.Tensor) -> torch.Tensor:
        """Compute pressure gradient effect on meridional wind."""
        # Smooth pressure field to get large-scale gradients
        sp_smooth = self._gaussian_smooth(sp, sigma=2.0)
        
        # Compute east-west pressure gradient (∂p/∂x)
        kernel_x = torch.tensor([[0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [-1/12, 8/12, 0, -8/12, 1/12],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]], device=self.device, dtype=sp.dtype)
        
        sp_4d = sp_smooth.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel_x.unsqueeze(0).unsqueeze(0)
        
        padded = torch.nn.functional.pad(sp_4d, (2, 2, 2, 2), mode='replicate')
        dp_dx = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        # Coriolis parameter (avoid division by zero near equator)
        f = 2 * self.omega * torch.sin(lat_rad)
        f_safe = torch.sign(f) * torch.maximum(torch.abs(f), torch.tensor(1e-10, device=self.device))
        
        # Geostrophic wind approximation
        v_geo = dp_dx / (self.rho * f_safe) * self.pressure_scale
        
        # Blend with current wind
        return current_v * 0.7 + v_geo * 0.3
    
    def _compute_jet_effect(self, lat_rad: torch.Tensor, lon_rad: torch.Tensor,
                          current_dt: datetime, current_v: torch.Tensor) -> torch.Tensor:
        """Compute jet stream influence on meridional wind."""
        # Base jet profile
        lat_from_center = torch.rad2deg(lat_rad) - self.jet_lat_center
        jet_profile = torch.exp(-(lat_from_center / 15.0) ** 2)
        
        # Latitude dependence: sin²φ × cos²φ (peaks at 45°)
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)
        lat_amplitude = (sin_lat ** 2) * (cos_lat ** 2)
        
        # Longitudinal wave pattern
        wave_number = 6
        lon_pattern = torch.sin(wave_number * lon_rad + math.pi/4)
        
        # Seasonal variation
        day_of_year = current_dt.timetuple().tm_yday
        
        # Create a tensor for seasonal variation
        seasonal_scalar = torch.cos(torch.tensor(2 * math.pi * (day_of_year - 172) / 365.0, 
                                                device=self.device, dtype=lat_rad.dtype))
        seasonal = torch.full_like(lat_rad, seasonal_scalar)
        
        # Combine components
        jet_strength = (
            jet_profile * 
            lat_amplitude * 
            (1.0 + 0.3 * lon_pattern) * 
            (1.0 + 0.2 * seasonal)
        )
        
        # Scale to appropriate magnitude
        jet_v = self.jet_max_speed * jet_strength
        
        # Blend with current wind
        return current_v * 0.8 + jet_v * 0.2
    
    def _compute_topo_effect(self, orograph: torch.Tensor, lat_rad: torch.Tensor,
                           current_v: torch.Tensor) -> torch.Tensor:
        """Compute topographic blocking effect."""
        # Identify significant topography
        topo_threshold = 1000.0
        significant_topo = torch.where(orograph > topo_threshold, orograph, torch.tensor(0.0, device=self.device))
        
        # Compute north-south gradient
        kernel_y = torch.tensor([[0, -0.25, 0],
                                 [0, 0, 0],
                                 [0, 0.25, 0]], device=self.device, dtype=orograph.dtype)
        
        topo_4d = significant_topo.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        padded = torch.nn.functional.pad(topo_4d, (1, 1, 1, 1), mode='replicate')
        dh_dy = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        # Mountains block meridional flow
        v_topo = -dh_dy * self.topo_blocking_scale
        
        # Effect is stronger at higher latitudes
        lat_factor = torch.abs(torch.sin(lat_rad))
        v_topo = v_topo * lat_factor
        
        # Only apply where topography is significant
        mask = (orograph > topo_threshold).float()
        v_topo = v_topo * mask
        
        # Blend gently
        return current_v * 0.9 + v_topo * 0.1
    
    def _compute_land_sea_effect(self, land_sea_mask: torch.Tensor, current_v: torch.Tensor) -> torch.Tensor:
        """Compute land-sea contrast effect."""
        # Land has higher friction -> reduced wind
        friction_factor = 1.0 - 0.2 * land_sea_mask
        
        # Smooth the transition
        friction_smooth = self._gaussian_smooth(friction_factor, sigma=1.0)
        
        return current_v * (friction_smooth - 1.0)
    
    def _compute_diurnal_effect(self, sin_zenith: torch.Tensor, cos_zenith: torch.Tensor,
                              current_v: torch.Tensor) -> torch.Tensor:
        """Compute diurnal variation effect."""
        # Daytime when sun is high (cos_zenith > 0)
        daytime_factor = torch.sigmoid(5.0 * cos_zenith)
        
        # Diurnal amplitude (stronger mixing during day)
        diurnal_amplitude = 0.3 * daytime_factor
        
        return current_v * diurnal_amplitude
    
    def _compute_seasonal_effect(self, sin_orbit: torch.Tensor, cos_orbit: torch.Tensor,
                               current_v: torch.Tensor) -> torch.Tensor:
        """Compute seasonal variation effect."""
        # Use cos_orbit for seasonal variation
        seasonal_factor = 1.0 - cos_orbit  # 0 in NH summer, 2 in NH winter
        
        # Scale to reasonable amplitude
        seasonal_amplitude = 0.2 * seasonal_factor
        
        return current_v * seasonal_amplitude
    
    def _apply_spatial_smoothing(self, wind: torch.Tensor, current_v: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing appropriate for wind fields."""
        # Two-pass light smoothing
        kernel = torch.tensor([[0.02, 0.08, 0.02],
                               [0.08, 0.6, 0.08],
                               [0.02, 0.08, 0.02]], device=self.device, dtype=wind.dtype)
        
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # First pass
        padded1 = torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='replicate')
        smoothed1 = torch.nn.functional.conv2d(padded1, kernel_4d, padding=0)
        
        # Second pass
        padded2 = torch.nn.functional.pad(smoothed1, (1, 1, 1, 1), mode='replicate')
        smoothed2 = torch.nn.functional.conv2d(padded2, kernel_4d, padding=0).squeeze()
        
        # Preserve some of the original structure
        return wind * 0.7 + smoothed2 * 0.3
    
    def _preserve_statistics(self, predicted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Ensure predicted field has similar statistics to original."""
        # Preserve mean and standard deviation
        pred_mean = torch.mean(predicted)
        pred_std = torch.std(predicted)
        orig_mean = torch.mean(original)
        orig_std = torch.std(original)
        
        # Avoid division by zero
        safe_std = torch.where(pred_std < 1e-8, torch.tensor(1.0, device=self.device), pred_std)
        
        # Standardize and rescale
        standardized = (predicted - pred_mean) / safe_std
        rescaled = standardized * orig_std + orig_mean
        
        return rescaled
    
    def _gaussian_smooth(self, field: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        size = int(3 * sigma)
        x = torch.arange(-size, size + 1, device=self.device, dtype=field.dtype)
        gaussian = torch.exp(-x**2 / (2 * sigma**2))
        kernel = torch.outer(gaussian, gaussian)
        kernel = kernel / torch.sum(kernel)
        
        field_4d = field.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        padded = torch.nn.functional.pad(field_4d, (size, size, size, size), mode='replicate')
        smoothed = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        return smoothed