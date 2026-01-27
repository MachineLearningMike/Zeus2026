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
    Physics-based 100m wind speed (u100) predictor.
    Following the exact same logic as the original code, with proper API.
    """
    
    def __init__(self, device: Optional[str] = None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize all constants as tensors on the correct device
        self._setup_constants()
        
        print(f"U100mPredictor initialized on device: {self.device}")
    
    def _setup_constants(self):
        """Initialize all physical constants as tensors."""
        # Physical constants as tensors
        self.R = torch.tensor(287.058, device=self.device)
        self.omega = torch.tensor(7.2921159e-5, device=self.device)
        self.rho = torch.tensor(1.225, device=self.device)
        
        # Jet stream parameters
        self.jet_lat_center = torch.tensor(45.0, device=self.device)
        self.jet_width = torch.tensor(15.0, device=self.device)
        self.jet_max_speed = torch.tensor(25.0, device=self.device)
        
        # Friction coefficients (per hour) as tensors
        self.cf_land = torch.tensor(0.0035, device=self.device)
        self.cf_sea = torch.tensor(0.00025, device=self.device)
        self.cf_ice = torch.tensor(0.0045, device=self.device)
        
        # Effect weights (match your simplified weights) as tensors
        self.WEIGHT_PERSISTENCE = torch.tensor(0.90, device=self.device)  # 90% persistence
        self.WEIGHT_JET = torch.tensor(0.05, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.03, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.01, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.01, device=self.device)  # Only 1%
        
        # Scaling factors as tensors
        self.PRESSURE_SCALING = torch.tensor(0.12, device=self.device)
        self.THERMAL_WIND_SCALING = torch.tensor(-0.006, device=self.device)
        self.TOPO_SCALING = torch.tensor(-0.00001, device=self.device)  # Almost zero as in your code
        
        # Limits as tensors
        self.WIND_MIN = torch.tensor(-40.0, device=self.device)
        self.WIND_MAX = torch.tensor(40.0, device=self.device)
        
        # Polar region thresholds as tensors
        self.ANTARCTICA_LAT = torch.tensor(-60.0, device=self.device)
        self.GREENLAND_LAT_MIN = torch.tensor(60.0, device=self.device)
        self.GREENLAND_LAT_MAX = torch.tensor(85.0, device=self.device)
        self.GREENLAND_LON_MIN = torch.tensor(-50.0, device=self.device)
        self.GREENLAND_LON_MAX = torch.tensor(-20.0, device=self.device)
        
        # Smoothing as tensor
        self.FINAL_SMOOTHING = torch.tensor(0.15, device=self.device)
    
    def _safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor,
                    eps: float = 1e-10) -> torch.Tensor:
        """Safe division to avoid NaN and extreme values at equator."""
        eps_tensor = torch.tensor(eps, device=self.device)
        # For Coriolis parameter near equator, use minimum value
        f_sign = torch.sign(denominator)
        f_abs = torch.abs(denominator)
        f_safe = f_sign * torch.maximum(f_abs, eps_tensor)
        return numerator / f_safe
    
    def _compute_pressure_gradient_wind(self, sp: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """
        Compute pressure gradient wind with safe handling for equator.
        Same logic as your _compute_pressure_gradient.
        """
        height, width = sp.shape
        
        # Use convolution for gradient like your code
        kernel_y = torch.tensor([[0, -0.5, 0],
                                 [0, 0, 0],
                                 [0, 0.5, 0]], dtype=sp.dtype, device=self.device)
        
        sp_4d = sp.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        # Replicate padding like your code
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(sp_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        # Scale like your code - convert to tensor
        scale_factor = torch.tensor(1e-6, device=self.device)
        grad_y = grad_y * scale_factor
        
        # Coriolis parameter
        f = 2 * self.omega * torch.sin(lat_rad)
        
        # Safe division to avoid extreme values at equator
        u_geo = -self._safe_divide(grad_y, self.rho * f)
        
        # Apply scaling like your code
        return self.PRESSURE_SCALING * u_geo
    
    def _compute_temperature_gradient_wind(self, t2m: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature gradient wind.
        Same logic as your _compute_temperature_gradient.
        """
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
    
    def _compute_jet_stream(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
        """
        Compute jet stream with sin²*cos² pattern.
        Same logic as your _compute_jet_stream.
        """
        lat_from_center = lat_deg - self.jet_lat_center
        
        # Use width_factor like your code
        width_factor = torch.where(lat_from_center > 0, 
                                  self.jet_width * 0.8,
                                  self.jet_width * 1.2)
        
        lat_norm = lat_from_center / width_factor
        jet_profile = torch.exp(-lat_norm**4)  # Use **4 like your code
        
        # sin²(lat) amplitude like your code
        lat_amplitude = torch.sin(lat_rad)**2
        jet_profile = jet_profile * lat_amplitude
        
        return self.jet_max_speed * jet_profile
    
    def _compute_minimal_topo_effect(self, orograph: torch.Tensor, polar_mask: torch.Tensor) -> torch.Tensor:
        """
        Minimal topographic effect - mostly zeros.
        Same logic as your _compute_minimal_topo_effect.
        """
        # ZERO in polar regions like your code
        effect = torch.zeros_like(orograph, device=self.device)
        
        # Very small effect elsewhere
        non_polar = ~polar_mask
        if torch.any(non_polar):
            # Tiny gradient using kernel like your code
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
            
            # Convert scalar to tensor
            scaling = torch.tensor(0.001, device=self.device)
            effect = torch.where(non_polar, grad * scaling, effect)
        
        return effect
    
    def _compute_simple_friction_decay(self, land_sea_mask: torch.Tensor, 
                                     delta_hours: int,
                                     polar_mask: torch.Tensor) -> torch.Tensor:
        """
        Simple friction decay with polar adjustments.
        Same logic as your _compute_simple_friction_decay.
        """
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, dtype=torch.float32, device=self.device)
        
        # Base decay using sea value as base
        decay = torch.exp(-self.cf_sea * delta_hours_tensor)
        
        # Land has faster decay
        land_decay = torch.exp(-self.cf_land * delta_hours_tensor)
        decay = torch.where(land_sea_mask > 0.5, land_decay, decay)
        
        # Polar regions: slower decay (more persistence)
        polar_decay = torch.exp(-self.cf_sea * delta_hours_tensor * 0.5)  # 50% slower decay
        decay = torch.where(polar_mask, polar_decay, decay)
        
        # Ensure reasonable bounds like your code
        decay_min = torch.tensor(0.3, device=self.device)
        decay_max = torch.tensor(1.0, device=self.device)
        decay = torch.clamp(decay, decay_min, decay_max)
        
        return decay
    
    def _identify_polar_regions(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                              orograph: torch.Tensor) -> tuple:
        """
        Identify polar regions with sub-regions.
        Same logic as your _identify_polar_regions.
        """
        # Convert thresholds to same dtype as lat_deg
        ant_lat_threshold = self.ANTARCTICA_LAT.to(lat_deg.dtype)
        greenland_lat_min = self.GREENLAND_LAT_MIN.to(lat_deg.dtype)
        greenland_lat_max = self.GREENLAND_LAT_MAX.to(lat_deg.dtype)
        greenland_lon_min = self.GREENLAND_LON_MIN.to(lat_deg.dtype)
        greenland_lon_max = self.GREENLAND_LON_MAX.to(lat_deg.dtype)
        
        # Antarctica
        orograph_threshold = torch.tensor(500, dtype=orograph.dtype, device=self.device)
        antarctica_mask = (lat_deg < ant_lat_threshold) & (orograph > orograph_threshold)
        
        # Greenland main mask
        greenland_lat = (lat_deg > greenland_lat_min) & (lat_deg < greenland_lat_max)
        greenland_lon = (lon_deg > greenland_lon_min) & (lon_deg < greenland_lon_max)
        greenland_mask = greenland_lat & greenland_lon & (orograph > orograph_threshold)
        
        # North Greenland
        north_threshold = torch.tensor(75.0, dtype=lat_deg.dtype, device=self.device)
        greenland_north_mask = greenland_mask & (lat_deg > north_threshold)
        
        # Middle Greenland coasts
        mid_lat_min = torch.tensor(65.0, dtype=lat_deg.dtype, device=self.device)
        mid_lat_max = torch.tensor(75.0, dtype=lat_deg.dtype, device=self.device)
        east_lon_min = torch.tensor(-30.0, dtype=lon_deg.dtype, device=self.device)
        east_lon_max = torch.tensor(-20.0, dtype=lon_deg.dtype, device=self.device)
        west_lon_min = torch.tensor(-50.0, dtype=lon_deg.dtype, device=self.device)
        west_lon_max = torch.tensor(-40.0, dtype=lon_deg.dtype, device=self.device)
        
        greenland_mid_lat = (lat_deg > mid_lat_min) & (lat_deg < mid_lat_max)
        greenland_east_coast = greenland_mid_lat & (lon_deg > east_lon_min) & (lon_deg < east_lon_max)
        greenland_west_coast = greenland_mid_lat & (lon_deg > west_lon_min) & (lon_deg < west_lon_max)
        greenland_mid_mask = (greenland_east_coast | greenland_west_coast) & greenland_mask
        
        return antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask
    
    def _apply_polar_hard_limits_only(self, wind: torch.Tensor,
                                    lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                    antarctica_mask: torch.Tensor,
                                    greenland_mask: torch.Tensor,
                                    greenland_north_mask: torch.Tensor,
                                    greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply ONLY hard limits - no complex corrections.
        Same logic as your _apply_polar_hard_limits_only.
        """
        result = wind.clone()
        
        # Use same dtype as wind for limits
        dtype = wind.dtype
        
        # Antarctica: Strict limits like your code
        if torch.any(antarctica_mask):
            ant_min = torch.tensor(-8.0, dtype=dtype, device=self.device)
            ant_max = torch.tensor(8.0, dtype=dtype, device=self.device)
            result = torch.where(antarctica_mask,
                               torch.clamp(result, ant_min, ant_max),
                               result)
        
        # Greenland: Regional limits like your code
        if torch.any(greenland_mask):
            # North Greenland
            if torch.any(greenland_north_mask):
                north_min = torch.tensor(-5.0, dtype=dtype, device=self.device)
                north_max = torch.tensor(10.0, dtype=dtype, device=self.device)
                result = torch.where(greenland_north_mask,
                                   torch.clamp(result, north_min, north_max),
                                   result)
            
            # Middle Greenland
            if torch.any(greenland_mid_mask):
                mid_min = torch.tensor(-8.0, dtype=dtype, device=self.device)
                mid_max = torch.tensor(8.0, dtype=dtype, device=self.device)
                result = torch.where(greenland_mid_mask,
                                   torch.clamp(result, mid_min, mid_max),
                                   result)
            
            # Rest of Greenland
            rest_greenland = greenland_mask & ~greenland_north_mask & ~greenland_mid_mask
            if torch.any(rest_greenland):
                rest_min = torch.tensor(-10.0, dtype=dtype, device=self.device)
                rest_max = torch.tensor(10.0, dtype=dtype, device=self.device)
                result = torch.where(rest_greenland,
                                   torch.clamp(result, rest_min, rest_max),
                                   result)
        
        return result
    
    def _minimal_smoothing(self, wind: torch.Tensor) -> torch.Tensor:
        """
        Apply minimal smoothing.
        Same logic as your _minimal_smoothing.
        """
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
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor,
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                geo_tempo_f: torch.Tensor, current_dt: datetime,
                delta_hours: int) -> torch.Tensor:
        """
        Predict u100m for future time t + delta_hours.
        Following the exact same logic as your predict method.
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
        
        # Calculate latitude and longitude like your code
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
        
        # 1. Identify polar regions like your code
        antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask = \
            self._identify_polar_regions(lat_deg, lon_deg, orograph)
        
        polar_mask = antarctica_mask | greenland_mask
        
        # **SIMPLIFIED APPROACH: 90% persistence, 10% other effects**
        # Exactly like your weights: 0.90 persistence
        
        # 2. Strong persistence (90% weight) like your code
        time_decay = self._compute_simple_friction_decay(land_sea_mask, delta_hours, polar_mask)
        persistence = current_u100 * time_decay
        
        # 3. Minimal other effects (10% total) like your code
        jet_strength = self._compute_jet_stream(lat_rad, lat_deg) * self.WEIGHT_JET
        pressure_gradient = self._compute_pressure_gradient_wind(sp, lat_rad) * self.WEIGHT_PRESSURE
        temp_gradient = self._compute_temperature_gradient_wind(t2m) * self.WEIGHT_THERMAL
        topo_effect = self._compute_minimal_topo_effect(orograph, polar_mask) * self.WEIGHT_TOPO
        
        # 4. Combine (90% persistence, 10% other) like your code
        combined = persistence * self.WEIGHT_PERSISTENCE + \
                  (jet_strength + pressure_gradient + temp_gradient + topo_effect)
        
        # 5. Apply POLAR HARD LIMITS (not corrections - just limits) like your code
        predicted_u100 = self._apply_polar_hard_limits_only(
            combined, lat_deg, lon_deg,
            antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask
        )
        
        # 6. Final smoothing like your code
        predicted_u100 = self._minimal_smoothing(predicted_u100)
        
        # 7. Final clamp like your code
        wind_min = self.WIND_MIN.to(predicted_u100.dtype)
        wind_max = self.WIND_MAX.to(predicted_u100.dtype)
        predicted_u100 = torch.clamp(predicted_u100, wind_min, wind_max)
        
        # Return on CPU if using GPU
        if self.device.type == 'cuda':
            return predicted_u100.cpu()
        else:
            return predicted_u100


def test_predictor():
    """Test the predictor with simple data."""
    # Create test data
    H, W = 64, 128
    
    # Create test data with equator included
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Meteo data
    t2m = torch.ones(H, W) * 300.0  # Warm at equator
    tp = torch.zeros(H, W)
    u100_current = torch.randn(H, W) * 3.0  # Small current winds
    v100_current = torch.randn(H, W) * 3.0
    d2m = t2m - torch.rand(H, W) * 5.0
    sp = torch.ones(H, W) * 101325.0
    
    meteo = torch.stack([t2m, tp, u100_current, v100_current, d2m, sp], dim=0)
    
    # Geographic data
    sin_lat = torch.sin(torch.deg2rad(lat_grid))
    cos_lat = torch.cos(torch.deg2rad(lat_grid))
    sin_lon = torch.sin(torch.deg2rad(lon_grid))
    cos_lon = torch.cos(torch.deg2rad(lon_grid))
    orograph = torch.zeros(H, W)  # Flat terrain
    
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0)
    
    # Categorical data
    land_sea_mask = torch.ones(H, W)  # All land
    soil_type = torch.ones(H, W) * 2
    
    geo_cat = torch.stack([land_sea_mask.float(), soil_type.float()], dim=0)
    
    # Temporal data
    sin_orbit = torch.zeros(H, W)
    cos_orbit = torch.ones(H, W)
    sin_zenith = torch.zeros(H, W)
    cos_zenith = torch.ones(H, W)
    
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0)
    geo_tempo_f = geo_tempo.clone()
    
    # Create predictor
    predictor = U100mPredictor()
    
    # Make prediction
    current_time = datetime.now()
    u_pred = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo,
                             geo_tempo_f, current_time, delta_hours=6)
    
    # Check results
    print(f"Prediction shape: {u_pred.shape}")
    print(f"Min: {u_pred.min().item():.2f} m/s")
    print(f"Max: {u_pred.max().item():.2f} m/s")
    print(f"Mean: {u_pred.mean().item():.2f} m/s")
    print(f"Std: {u_pred.std().item():.2f} m/s")
    
    # Check for extreme values at equator (latitudes near 0)
    equator_mask = (torch.abs(lat_grid) < 5.0)
    if torch.any(equator_mask):
        equator_values = u_pred[equator_mask]
        print(f"\nEquator region (|lat| < 5°):")
        print(f"  Min: {equator_values.min().item():.2f} m/s")
        print(f"  Max: {equator_values.max().item():.2f} m/s")
        print(f"  Mean: {equator_values.mean().item():.2f} m/s")
        
        # Check for extreme values
        extreme_threshold = 50.0
        extreme_count = torch.sum(torch.abs(equator_values) > extreme_threshold).item()
        if extreme_count > 0:
            print(f"  WARNING: {extreme_count} extreme values (> {extreme_threshold} m/s) at equator!")
        else:
            print(f"  OK: No extreme values at equator")
    
    # Check for invalid values
    nan_count = torch.isnan(u_pred).sum().item()
    inf_count = torch.isinf(u_pred).sum().item()
    
    if nan_count == 0 and inf_count == 0:
        print("✓ No NaN or Inf values")
    else:
        print(f"✗ Found {nan_count} NaN and {inf_count} Inf values")
    
    return u_pred


if __name__ == "__main__":
    print("Testing U100mPredictor with focus on equator...")
    result = test_predictor()
    print("Test completed successfully!")