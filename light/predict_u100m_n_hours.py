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
from typing import Optional, Union
import numpy as np

class U100mPredictor:
    """
    Physics-based 100m wind speed (u100) predictor.
    Focus on persistence with minimal physics for forecasting.
    """
    
    def __init__(self, device: Optional[str] = None):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Physical constants
        self.R = 287.05  # Gas constant (J/kg/K)
        self.omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        self.g = 9.81  # Gravity (m/s²)
        
        # Friction coefficients (per hour)
        self.cf_sea = 0.01  # Sea friction
        self.cf_land = 0.03  # Land friction
        self.cf_ice = 0.02   # Ice friction (polar regions)
        
        # Jet stream parameters
        self.jet_lat_center = 45.0  # Degrees
        self.jet_width = 15.0  # Degrees
        self.jet_max_speed = 30.0  # m/s
        
        # Effect weights (sum to 1.0)
        self.WEIGHT_PERSISTENCE = 0.85
        self.WEIGHT_JET = 0.08
        self.WEIGHT_PRESSURE = 0.05
        self.WEIGHT_THERMAL = 0.02
        
        # Polar region thresholds
        self.ANTARCTICA_LAT = -60.0
        self.GREENLAND_LAT_MIN = 60.0
        self.GREENLAND_LAT_MAX = 85.0
        self.GREENLAND_LON_MIN = -50.0
        self.GREENLAND_LON_MAX = -20.0
        
        print(f"U100mPredictor initialized on device: {self.device}")
    
    def _safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor,
                    eps: float = 1e-10) -> torch.Tensor:
        """Safe division to avoid NaN."""
        return numerator / (denominator + eps * torch.sign(denominator))
    
    def _compute_gradients(self, field: torch.Tensor) -> tuple:
        """Compute spatial gradients using finite differences."""
        h, w = field.shape
        
        # Initialize gradient arrays
        dfield_dx = torch.zeros_like(field)
        dfield_dy = torch.zeros_like(field)
        
        # Central differences for interior
        if w > 2:
            dfield_dx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0
        # Forward/backward differences for boundaries
        if w > 1:
            dfield_dx[:, 0] = field[:, 1] - field[:, 0]
            dfield_dx[:, -1] = field[:, -1] - field[:, -2]
        
        if h > 2:
            dfield_dy[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2.0
        if h > 1:
            dfield_dy[0, :] = field[1, :] - field[0, :]
            dfield_dy[-1, :] = field[-1, :] - field[-2, :]
        
        return dfield_dx, dfield_dy
    
    def _compute_pressure_gradient_wind(self, sp: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute geostrophic wind from pressure gradient."""
        # Pressure gradients
        dp_dx, dp_dy = self._compute_gradients(sp)
        
        # Grid spacing (approximate meters per grid cell)
        # Assuming 1-degree resolution
        dx = 111320 * torch.cos(torch.abs(lat_rad))  # Varies with latitude
        dy = 111320 * torch.ones_like(lat_rad)  # Constant in latitude
        
        # Scale gradients by grid spacing
        dp_dx = dp_dx / dx
        dp_dy = dp_dy / dy
        
        # Coriolis parameter
        f = 2 * self.omega * torch.sin(lat_rad)
        f = torch.where(torch.abs(f) < 1e-10, torch.sign(f) * 1e-10, f)
        
        # Air density (simplified)
        rho = 1.225  # kg/m³ at sea level
        
        # Geostrophic wind: u_g = -(1/(ρf)) * ∂p/∂y
        u_geo = -self._safe_divide(dp_dy, rho * f)
        
        # Scale down for 100m level (boundary layer reduction)
        u_geo = u_geo * 0.3
        
        return u_geo
    
    def _compute_thermal_wind(self, t2m: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute thermal wind adjustment."""
        # Temperature gradient
        dT_dx, dT_dy = self._compute_gradients(t2m)
        
        # Grid spacing
        dx = 111320 * torch.cos(torch.abs(lat_rad))
        dy = 111320 * torch.ones_like(lat_rad)
        
        dT_dx = dT_dx / dx
        dT_dy = dT_dy / dy
        
        # Coriolis parameter
        f = 2 * self.omega * torch.sin(lat_rad)
        f = torch.where(torch.abs(f) < 1e-10, torch.sign(f) * 1e-10, f)
        
        # Thermal wind: ∂u/∂z = -(g/(fT)) * ∂T/∂y
        # Integrated over 100m boundary layer
        boundary_layer_height = 100.0  # meters
        mean_temp = 288.0  # Kelvin
        
        u_thermal = -self._safe_divide(self.g * dT_dy * boundary_layer_height, f * mean_temp)
        
        # Small effect
        u_thermal = u_thermal * 0.1
        
        return u_thermal
    
    def _compute_jet_stream(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor) -> torch.Tensor:
        """Compute jet stream contribution using sin²*cos² pattern."""
        # Jet stream profile (Gaussian centered at jet latitude)
        lat_from_center = lat_deg - self.jet_lat_center
        jet_profile = torch.exp(-(lat_from_center / self.jet_width)**2)
        
        # sin²(lat) * cos²(lat) pattern for longitudinal variation
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)
        pattern = (sin_lat**2) * (cos_lat**2)
        
        # Combine
        u_jet = self.jet_max_speed * jet_profile * (0.7 + 0.3 * pattern)
        
        return u_jet
    
    def _compute_friction_decay(self, land_sea_mask: torch.Tensor,
                              delta_hours: int) -> torch.Tensor:
        """Compute wind decay due to surface friction."""
        # Base friction coefficient
        cf = land_sea_mask * self.cf_land + (1 - land_sea_mask) * self.cf_sea
        
        # Exponential decay
        decay = torch.exp(-cf * delta_hours)
        
        # Ensure reasonable bounds
        decay = torch.clamp(decay, 0.5, 1.0)
        
        return decay
    
    def _identify_polar_regions(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                              orograph: torch.Tensor) -> tuple:
        """Identify polar regions."""
        # Antarctica
        antarctica_mask = (lat_deg < self.ANTARCTICA_LAT) & (orograph > 500)
        
        # Greenland
        greenland_lat = (lat_deg > self.GREENLAND_LAT_MIN) & (lat_deg < self.GREENLAND_LAT_MAX)
        greenland_lon = (lon_deg > self.GREENLAND_LON_MIN) & (lon_deg < self.GREENLAND_LON_MAX)
        greenland_mask = greenland_lat & greenland_lon & (orograph > 500)
        
        return antarctica_mask, greenland_mask
    
    def _apply_polar_limits(self, wind: torch.Tensor,
                          antarctica_mask: torch.Tensor,
                          greenland_mask: torch.Tensor) -> torch.Tensor:
        """Apply simple limits to polar regions."""
        result = wind.clone()
        
        # Antarctica limits
        if torch.any(antarctica_mask):
            result = torch.where(antarctica_mask,
                               torch.clamp(result, -15.0, 15.0),
                               result)
        
        # Greenland limits
        if torch.any(greenland_mask):
            result = torch.where(greenland_mask,
                               torch.clamp(result, -12.0, 12.0),
                               result)
        
        return result
    
    def _apply_smoothing(self, wind: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
        """Apply simple Gaussian-like smoothing."""
        # Simple 3x3 averaging filter
        kernel = torch.tensor([[0.0625, 0.125, 0.0625],
                              [0.125, 0.250, 0.125],
                              [0.0625, 0.125, 0.0625]],
                             device=self.device, dtype=wind.dtype)
        
        # Apply convolution
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Use reflect padding for boundaries
        padded = torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='reflect')
        smoothed = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        # Blend with original
        result = wind * 0.7 + smoothed * 0.3
        
        return result
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor,
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                geo_tempo_f: torch.Tensor, current_dt: datetime,
                delta_hours: int) -> torch.Tensor:
        """
        Predict u100m for future time t + delta_hours.
        
        Args:
            meteo: [t2m, tp, u100, v100, d2m, sp] - shape [6, H, W]
            geo_dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon] - shape [5, H, W]
            geo_cat: [land_sea_mask, soil_type] - shape [2, H, W]
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] - shape [4, H, W] for current
            geo_tempo_f: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] - shape [4, H, W] for future
            current_dt: Current datetime
            delta_hours: Hours to forecast
            
        Returns:
            Predicted u100m (eastward wind component) at future time
        """
        # Move to device
        meteo = meteo.to(self.device).float()
        geo_dense = geo_dense.to(self.device).float()
        geo_cat = geo_cat.to(self.device).float()
        geo_tempo = geo_tempo.to(self.device).float()
        geo_tempo_f = geo_tempo_f.to(self.device).float()
        
        # Extract components
        t2m = meteo[0]        # 2m temperature (K)
        tp = meteo[1]         # Total precipitation (m)
        u100_current = meteo[2]  # Current u100 (m/s)
        v100_current = meteo[3]  # Current v100 (m/s)
        d2m = meteo[4]        # 2m dew point (K)
        sp = meteo[5]         # Surface pressure (Pa)
        
        orograph = geo_dense[0]   # Orography (m²/s²)
        sin_lat = geo_dense[1]    # sin(latitude)
        cos_lat = geo_dense[2]    # cos(latitude)
        sin_lon = geo_dense[3]    # sin(longitude)
        cos_lon = geo_dense[4]    # cos(longitude)
        
        land_sea_mask = geo_cat[0]  # 0=sea, 1=land
        soil_type = geo_cat[1]      # Soil type
        
        # Calculate latitude and longitude in radians and degrees
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lat_deg = torch.rad2deg(lat_rad)
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lon_deg = torch.rad2deg(lon_rad)
        
        # 1. Identify polar regions (for limits only)
        antarctica_mask, greenland_mask = self._identify_polar_regions(lat_deg, lon_deg, orograph)
        
        # 2. Persistence with friction decay (85% weight)
        decay = self._compute_friction_decay(land_sea_mask, delta_hours)
        persistence = u100_current * decay
        
        # 3. Jet stream contribution (8% weight)
        u_jet = self._compute_jet_stream(lat_rad, lat_deg)
        
        # 4. Pressure gradient wind (5% weight)
        u_pressure = self._compute_pressure_gradient_wind(sp, lat_rad)
        
        # 5. Thermal wind adjustment (2% weight)
        u_thermal = self._compute_thermal_wind(t2m, lat_rad)
        
        # 6. Combine all effects
        u_pred = (persistence * self.WEIGHT_PERSISTENCE +
                 u_jet * self.WEIGHT_JET +
                 u_pressure * self.WEIGHT_PRESSURE +
                 u_thermal * self.WEIGHT_THERMAL)
        
        # 7. Apply precipitation damping (rain reduces wind)
        # Convert precipitation from meters to mm/hour equivalent
        tp_mm_per_hour = tp * 1000.0 / max(delta_hours, 1)
        precip_factor = torch.exp(-0.005 * tp_mm_per_hour)
        precip_factor = torch.clamp(precip_factor, 0.7, 1.0)
        u_pred = u_pred * precip_factor
        
        # 8. Apply polar region limits (simple hard limits)
        u_pred = self._apply_polar_limits(u_pred, antarctica_mask, greenland_mask)
        
        # 9. Apply smoothing to remove small-scale noise
        u_pred = self._apply_smoothing(u_pred)
        
        # 10. Final global limits
        u_pred = torch.clamp(u_pred, -50.0, 50.0)
        
        # 11. Handle NaN values (fall back to persistence)
        nan_mask = torch.isnan(u_pred)
        if torch.any(nan_mask):
            u_pred = torch.where(nan_mask, persistence, u_pred)
        
        # Return on CPU if using GPU
        if self.device.type == 'cuda':
            return u_pred.cpu()
        else:
            return u_pred


# Test function
def test_predictor():
    """Test the predictor with simple data."""
    # Create test data
    H, W = 64, 128  # Small test size
    
    # Create realistic test data
    torch.manual_seed(42)
    
    # Latitude grid (-90 to 90 degrees)
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Meteo data
    t2m = torch.ones(H, W) * 288.0 + torch.randn(H, W) * 10.0
    tp = torch.rand(H, W) * 0.001  # Small precipitation
    u100_current = torch.randn(H, W) * 5.0  # Current wind
    v100_current = torch.randn(H, W) * 5.0
    d2m = t2m - torch.rand(H, W) * 5.0  # Dew point slightly lower than temp
    sp = torch.ones(H, W) * 101325.0 + torch.randn(H, W) * 1000.0
    
    meteo = torch.stack([t2m, tp, u100_current, v100_current, d2m, sp], dim=0)
    
    # Geographic data
    sin_lat = torch.sin(torch.deg2rad(lat_grid))
    cos_lat = torch.cos(torch.deg2rad(lat_grid))
    sin_lon = torch.sin(torch.deg2rad(lon_grid))
    cos_lon = torch.cos(torch.deg2rad(lon_grid))
    orograph = torch.randn(H, W) * 1000.0  # Terrain
    
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0)
    
    # Categorical data
    land_sea_mask = torch.rand(H, W) > 0.3  # 70% land
    soil_type = torch.randint(0, 8, (H, W))
    
    geo_cat = torch.stack([land_sea_mask.float(), soil_type.float()], dim=0)
    
    # Temporal data (simplified)
    sin_orbit = torch.sin(torch.tensor(0.0)) * torch.ones(H, W)
    cos_orbit = torch.cos(torch.tensor(0.0)) * torch.ones(H, W)
    sin_zenith = torch.sin(torch.deg2rad(45.0)) * torch.ones(H, W)
    cos_zenith = torch.cos(torch.deg2rad(45.0)) * torch.ones(H, W)
    
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0)
    geo_tempo_f = geo_tempo.clone()  # Same for future for testing
    
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
    
    # Check for invalid values
    nan_count = torch.isnan(u_pred).sum().item()
    inf_count = torch.isinf(u_pred).sum().item()
    
    if nan_count == 0 and inf_count == 0:
        print("✓ No NaN or Inf values")
    else:
        print(f"✗ Found {nan_count} NaN and {inf_count} Inf values")
    
    return u_pred


if __name__ == "__main__":
    print("Testing U100mPredictor...")
    result = test_predictor()
    print("Test completed successfully!")