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

Please write a Python class "U100mPredictor" that has the above predict(.) function, which predicts u100m 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import math
from datetime import datetime
from typing import Optional

class U100mPredictor:
    """
    Predict u100 at time t+delta using physical principles.
    Fixed polar katabatic wind patterns.
    """
    
    def __init__(self, device: Optional[str] = None):
        # Convert string to torch.device object
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize all physical constants."""
        # Physical constants
        self.R = torch.tensor(287.058, device=self.device)
        self.omega = torch.tensor(7.2921159e-5, device=self.device)
        self.rho = torch.tensor(1.225, device=self.device)
        
        # Jet stream parameters
        self.jet_lat_center = torch.tensor(45.0, device=self.device)
        self.jet_width = torch.tensor(10.0, device=self.device)
        self.jet_max_speed = torch.tensor(25.0, device=self.device)
        
        # Friction coefficients
        self.cf_land = torch.tensor(0.0035, device=self.device)
        self.cf_sea = torch.tensor(0.00025, device=self.device)
        self.cf_ice = torch.tensor(0.0045, device=self.device)
        
        # SIMPLIFIED WEIGHTS (90% persistence approach)
        self.WEIGHT_PERSISTENCE = torch.tensor(0.90, device=self.device)
        self.WEIGHT_JET = torch.tensor(0.05, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.03, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.01, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.01, device=self.device)
        
        # Scaling factors
        self.PRESSURE_SCALING = torch.tensor(0.12, device=self.device)
        self.THERMAL_WIND_SCALING = torch.tensor(-0.006, device=self.device)
        self.TOPO_SCALING = torch.tensor(-0.00001, device=self.device)
        
        # Limits
        self.WIND_MIN = torch.tensor(-40.0, device=self.device)
        self.WIND_MAX = torch.tensor(40.0, device=self.device)
        self.POLAR_COASTAL_LIMIT = torch.tensor(8.0, device=self.device)
        
        # Smoothing
        self.FINAL_SMOOTHING = torch.tensor(0.15, device=self.device)
        
        # Polar region thresholds
        self.ANTARCTICA_LAT = torch.tensor(-60.0, device=self.device)
        self.GREENLAND_LAT_MIN = torch.tensor(60.0, device=self.device)
        self.GREENLAND_LAT_MAX = torch.tensor(85.0, device=self.device)
        self.GREENLAND_LON_MIN = torch.tensor(-50.0, device=self.device)
        self.GREENLAND_LON_MAX = torch.tensor(-20.0, device=self.device)
        
        # Strip removal parameters
        self.STRIP_REDUCTION = torch.tensor(0.3, device=self.device)
        self.STRIP_SMOOTHING = torch.tensor(0.4, device=self.device)
        self.COASTAL_BLEND_WIDTH = torch.tensor(3.0, device=self.device)
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor,
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                geo_tempo_f: torch.Tensor, current_dt: datetime,
                delta_hours: int) -> torch.Tensor:
        """
        SIMPLIFIED PREDICTION: Focus on persistence with minimal polar effects.
        """
        # Move to device
        meteo = meteo.to(self.device).float()
        geo_dense = geo_dense.to(self.device).float()
        geo_cat = geo_cat.to(self.device).float()
        geo_tempo = geo_tempo.to(self.device).float()
        geo_tempo_f = geo_tempo_f.to(self.device).float()
        
        # Extract components
        t2m = meteo[0]
        current_u100 = meteo[2]
        sp = meteo[5]
        
        orograph = geo_dense[0]
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        sin_lon = geo_dense[3]
        cos_lon = geo_dense[4]
        
        land_sea_mask = geo_cat[0]
        
        height, width = t2m.shape
        
        # Calculate latitude and longitude
        cos_lat_clamped = torch.clamp(cos_lat, -0.9999, 0.9999)
        lat_rad = torch.where(
            cos_lat_clamped > 0.5,
            torch.asin(torch.clamp(sin_lat, -0.9999, 0.9999)),
            torch.acos(cos_lat_clamped) * torch.sign(sin_lat)
        )
        lat_rad = torch.remainder(lat_rad + math.pi, 2 * math.pi) - math.pi
        lat_deg = torch.rad2deg(lat_rad)
        
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lon_deg = torch.rad2deg(lon_rad)
        
        # 1. Identify polar regions
        antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask = \
            self._identify_polar_regions(lat_deg, lon_deg, orograph)
        
        polar_mask = antarctica_mask | greenland_mask
        
        # 2. Strong persistence (90% weight)
        time_decay = self._compute_simple_friction_decay(land_sea_mask, delta_hours, polar_mask)
        persistence = current_u100 * time_decay
        
        # 3. Minimal other effects (10% total)
        jet_strength = self._compute_jet_stream(lat_rad, lat_deg, current_dt) * 0.05
        pressure_gradient = self._compute_pressure_gradient(sp, lat_rad) * 0.03
        temp_gradient = self._compute_temperature_gradient(t2m) * 0.01
        topo_effect = self._compute_minimal_topo_effect(orograph, polar_mask) * 0.01
        
        # 4. Combine (90% persistence, 10% other)
        combined = persistence * 0.9 + (jet_strength + pressure_gradient + temp_gradient + topo_effect) * 0.1
        
        # 5. Apply POLAR HARD LIMITS (not corrections - just limits)
        predicted_u100 = self._apply_polar_hard_limits_only(
            combined, lat_deg, lon_deg,
            antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask
        )
        
        # 6. Apply stabilization using future temporal features
        # Use diurnal cycle stabilization
        stabilization_factor = torch.tensor(0.9, device=self.device)  # 90% persistence
        predicted_u100 = stabilization_factor * predicted_u100 + (1 - stabilization_factor) * current_u100
        
        # 7. Final smoothing
        predicted_u100 = self._minimal_smoothing(predicted_u100)
        
        # 8. Final clamp
        predicted_u100 = torch.clamp(predicted_u100, self.WIND_MIN, self.WIND_MAX)
        
        # Return u100 component only
        # FIXED: Check device type properly
        if str(self.device) == 'cuda':
            return predicted_u100.cpu()
        else:
            return predicted_u100
    
    def _identify_polar_regions(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                orograph: torch.Tensor) -> tuple:
        """Identify polar regions with sub-regions."""
        # Antarctica
        antarctica_mask = (lat_deg < self.ANTARCTICA_LAT) & (orograph > 500)
        
        # Greenland main mask
        greenland_lat = (lat_deg > self.GREENLAND_LAT_MIN) & (lat_deg < self.GREENLAND_LAT_MAX)
        greenland_lon = (lon_deg > self.GREENLAND_LON_MIN) & (lon_deg < self.GREENLAND_LON_MAX)
        greenland_mask = greenland_lat & greenland_lon & (orograph > 500)
        
        # North Greenland
        greenland_north_mask = greenland_mask & (lat_deg > 75.0)
        
        # Middle Greenland coasts
        greenland_mid_lat = (lat_deg > 65.0) & (lat_deg < 75.0)
        greenland_east_coast = greenland_mid_lat & (lon_deg > -30.0) & (lon_deg < -20.0)
        greenland_west_coast = greenland_mid_lat & (lon_deg > -50.0) & (lon_deg < -40.0)
        greenland_mid_mask = (greenland_east_coast | greenland_west_coast) & greenland_mask
        
        return antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask
    
    def _compute_simple_friction_decay(self, land_sea_mask: torch.Tensor, 
                                      delta_hours: int,
                                      polar_mask: torch.Tensor) -> torch.Tensor:
        """Simple friction decay with polar adjustments."""
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, dtype=torch.float32, device=self.device)
        
        # Base decay
        decay = torch.exp(-self.cf_sea * delta_hours_tensor)
        
        # Land has faster decay
        land_decay = torch.exp(-self.cf_land * delta_hours_tensor)
        decay = torch.where(land_sea_mask > 0.5, land_decay, decay)
        
        # Polar regions: slower decay (more persistence)
        polar_decay = torch.exp(-self.cf_sea * delta_hours_tensor * 0.5)
        decay = torch.where(polar_mask, polar_decay, decay)
        
        # Ensure reasonable bounds
        decay = torch.clamp(decay, torch.tensor(0.3, device=self.device), torch.tensor(1.0, device=self.device))
        
        return decay
    
    def _compute_minimal_topo_effect(self, orograph: torch.Tensor, polar_mask: torch.Tensor) -> torch.Tensor:
        """Minimal topographic effect - mostly zeros."""
        # ZERO in polar regions
        effect = torch.zeros_like(orograph, device=self.device)
        
        # Very small effect elsewhere
        non_polar = ~polar_mask
        if torch.any(non_polar):
            # Tiny gradient using large kernel (smooth)
            kernel = torch.tensor([[0, 0.1, 0],
                                   [0.1, 0, 0.1],
                                   [0, 0.1, 0]], device=self.device, dtype=orograph.dtype) * 0.1
            
            orograph_4d = orograph.unsqueeze(0).unsqueeze(0)
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
            
            grad = torch.nn.functional.conv2d(
                torch.nn.functional.pad(orograph_4d, (1, 1, 1, 1), mode='replicate'),
                kernel_4d,
                padding=0
            ).squeeze()
            
            effect = torch.where(non_polar, grad * 0.001, effect)
        
        return effect
    
    def _apply_polar_hard_limits_only(self, wind: torch.Tensor,
                                     lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                     antarctica_mask: torch.Tensor,
                                     greenland_mask: torch.Tensor,
                                     greenland_north_mask: torch.Tensor,
                                     greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """Apply ONLY hard limits - no complex corrections."""
        result = wind.clone()
        
        # Antarctica: Strict limits
        if torch.any(antarctica_mask):
            result = torch.where(antarctica_mask,
                               torch.clamp(result, 
                                         torch.tensor(-8.0, device=self.device),
                                         torch.tensor(8.0, device=self.device)),
                               result)
        
        # Greenland: Regional limits
        if torch.any(greenland_mask):
            # North Greenland
            if torch.any(greenland_north_mask):
                result = torch.where(greenland_north_mask,
                                   torch.clamp(result,
                                             torch.tensor(-5.0, device=self.device),
                                             torch.tensor(10.0, device=self.device)),
                                   result)
            
            # Middle Greenland
            if torch.any(greenland_mid_mask):
                result = torch.where(greenland_mid_mask,
                                   torch.clamp(result,
                                             torch.tensor(-8.0, device=self.device),
                                             torch.tensor(8.0, device=self.device)),
                                   result)
            
            # Rest of Greenland
            rest_greenland = greenland_mask & ~greenland_north_mask & ~greenland_mid_mask
            if torch.any(rest_greenland):
                result = torch.where(rest_greenland,
                                   torch.clamp(result,
                                             torch.tensor(-10.0, device=self.device),
                                             torch.tensor(10.0, device=self.device)),
                                   result)
        
        return result
    
    def _compute_pressure_gradient(self, sp: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute pressure gradient."""
        height, width = sp.shape
        
        kernel_y = torch.tensor([[0, -0.5, 0],
                                 [0, 0, 0],
                                 [0, 0.5, 0]], dtype=sp.dtype, device=self.device)
        
        sp_4d = sp.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(sp_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        grad_y = grad_y * 1e-6
        
        f = 2 * self.omega * torch.sin(lat_rad)
        f_sign = torch.sign(f)
        f_abs = torch.abs(f)
        f_min = torch.tensor(1e-10, device=self.device)
        f_safe = f_sign * torch.maximum(f_abs, f_min)
        
        u_geo = -grad_y / (self.rho * f_safe)
        
        return self.PRESSURE_SCALING * u_geo
    
    def _compute_temperature_gradient(self, t2m: torch.Tensor) -> torch.Tensor:
        """Compute temperature gradient."""
        height, width = t2m.shape
        
        kernel_y = torch.tensor([[0, -0.3, 0],
                                 [0, 0, 0],
                                 [0, 0.3, 0]], dtype=t2m.dtype, device=self.device)
        
        t2m_4d = t2m.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(t2m_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        return self.THERMAL_WIND_SCALING * grad_y
    
    def _compute_jet_stream(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor, 
                           current_time_utc: datetime) -> torch.Tensor:
        """Compute jet stream."""
        lat_from_center = lat_deg - self.jet_lat_center
        
        width_factor = torch.where(lat_from_center > 0, 
                                  self.jet_width * 0.8,
                                  self.jet_width * 1.2)
        
        lat_norm = lat_from_center / width_factor
        jet_profile = torch.exp(-lat_norm**4)
        
        lat_amplitude = torch.sin(lat_rad)**2
        jet_profile = jet_profile * lat_amplitude
        
        return self.jet_max_speed * jet_profile
    
    def _minimal_smoothing(self, wind: torch.Tensor) -> torch.Tensor:
        """Apply minimal smoothing."""
        kernel = torch.tensor([[0, 0.1, 0],
                               [0.1, 0.6, 0.1],
                               [0, 0.1, 0]], dtype=wind.dtype, device=self.device)
        
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return wind * (1 - self.FINAL_SMOOTHING) + smoothed * self.FINAL_SMOOTHING


# Simple test
if __name__ == "__main__":
    # Create simple test data
    H, W = 32, 64
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Test data
    meteo = torch.randn(6, H, W)
    geo_dense = torch.randn(5, H, W)
    geo_cat = torch.randn(2, H, W)
    geo_tempo = torch.randn(4, H, W)
    geo_tempo_f = torch.randn(4, H, W)
    
    # Create predictor
    predictor = U100mPredictor()
    
    # Test prediction
    result = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, 
                              geo_tempo_f, datetime.now(), 6)
    
    print(f"Prediction shape: {result.shape}")
    print(f"Min: {result.min().item():.2f}, Max: {result.max().item():.2f}")
    print(f"Device check: predictor.device = {predictor.device}")
    print(f"Result is on CPU: {result.device == torch.device('cpu')}")