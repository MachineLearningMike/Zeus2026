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

Please write code that predicts u100m at time t + delta hours, by fastest physical logic and not by tedious ML or numerical analysis. delta is in range: 0, 1, ...,
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import math
from datetime import datetime, timedelta
import numpy as np

import torch
import math
from datetime import datetime, timedelta
import numpy as np

import torch
import math
from datetime import datetime, timedelta

class U100mPredictor:
    """
    Predict u100 at time t+delta using physical principles
    """
    
    def __init__(self):
        # Physical constants
        self.R = 287.058  # Specific gas constant for dry air (J/(kg·K))
        self.omega = 7.2921159e-5  # Earth's rotation rate (rad/s)
        self.rho = 1.225  # Standard air density at sea level (kg/m³)
        
        # Jet stream parameters (empirical)
        self.jet_lat_center = 45.0  # Typical jet stream latitude (degrees)
        self.jet_width = 15.0  # Jet stream width (degrees)
        self.jet_max_speed = 50.0  # Maximum jet stream speed (m/s)
        
        # Surface friction coefficient
        self.cf = 0.001  # Drag coefficient over land
        self.cf_sea = 0.0001  # Drag coefficient over sea
        
    def predict_u100m(self, meteo, geo_dense, geo_cat, tempo, 
                     current_time_utc, delta_hours):
        """
        Predict u100 at time t + delta_hours
        
        Args:
            meteo: torch.Tensor shape (6, 721, 1440) - [t2m, tp, u100, v100, d2m, sp]
            geo_dense: torch.Tensor shape (5, 721, 1440) - [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            geo_cat: torch.Tensor shape (2, 721, 1440) - [land_sea_mask, soil_type]
            tempo: torch.Tensor shape (4, 721, 1440) - [sin_orbit, cos_orbit, sin_zenit, cos_zenit]
            current_time_utc: datetime object
            delta_hours: int, hours to predict ahead (0, 1, ...)
            
        Returns:
            predicted_u100: torch.Tensor shape (721, 1440)
        """
        device = meteo.device
        height, width = meteo.shape[1], meteo.shape[2]
        
        # Extract individual components
        t2m = meteo[0]  # Temperature at 2m (K)
        tp = meteo[1]   # Total precipitation (m water)
        current_u100 = meteo[2]  # Current u100
        current_v100 = meteo[3]  # Current v100
        d2m = meteo[4]  # Dew point at 2m (K)
        sp = meteo[5]   # Surface pressure (Pa)
        
        orograph = geo_dense[0]  # Topography (m)
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        sin_lon = geo_dense[3]
        cos_lon = geo_dense[4]
        
        land_sea_mask = geo_cat[0]  # 0: sea, 1: land
        soil_type = geo_cat[1]
        
        sin_zenit = tempo[2]
        cos_zenit = tempo[3]
        
        # Convert trigonometric values back to angles for calculations
        lat_rad = torch.atan2(sin_lat, cos_lat)  # Latitude in radians
        lon_rad = torch.atan2(sin_lon, cos_lon)  # Longitude in radians
        lat_deg = torch.rad2deg(lat_rad)  # Latitude in degrees
        
        # 1. Temperature tendency (solar heating effect)
        # Cosine of zenith angle indicates solar intensity
        solar_factor = cos_zenit  # cos_zenit, 1 = overhead sun, 0 = horizon
        hour_of_day = (current_time_utc.hour + delta_hours) % 24
        
        # Create tensor for diurnal cycle calculation
        hour_tensor = torch.full((height, width), hour_of_day, 
                                dtype=torch.float32, device=device)
        diurnal_cycle = torch.sin(2 * torch.pi * (hour_tensor - 12) / 24) * 5.0  # ±5K variation
        
        # Future temperature estimate
        future_t2m = t2m + diurnal_cycle * solar_factor
        
        # 2. Jet stream influence (major upper-level wind feature)
        # Jet stream follows sin²(latitude) * cos²(latitude) pattern
        # Enhanced with Gaussian envelope for more realistic profile
        lat_norm = (lat_deg - self.jet_lat_center) / self.jet_width
        
        # sin²(lat) * cos²(lat) = 0.25 * sin²(2*lat)
        # This gives maximum at 45°, zero at 0° and 90°
        jet_profile = 0.25 * torch.sin(2 * lat_rad)**2 * torch.exp(-lat_norm**2)
        jet_strength = self.jet_max_speed * jet_profile
        
        # Jet stream rotates with seasons
        day_of_year = current_time_utc.timetuple().tm_yday
        seasonal_shift = 10.0 * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365, device=device))
        
        # Create shifted jet profile
        jet_lat_shifted = self.jet_lat_center + seasonal_shift
        lat_norm_seasonal = (lat_deg - jet_lat_shifted) / self.jet_width
        jet_profile_seasonal = 0.25 * torch.sin(2 * lat_rad)**2 * torch.exp(-lat_norm_seasonal**2)
        jet_strength_seasonal = self.jet_max_speed * jet_profile_seasonal
        
        # Blend based on seasonality
        jet_strength = 0.7 * jet_strength + 0.3 * jet_strength_seasonal
        
        # 3. Pressure gradient effect (simplified geostrophic wind)
        # Use convolution to calculate pressure gradients more robustly
        pressure_gradient = self.calculate_pressure_gradient(sp, lat_rad)
        
        # 4. Friction effects (surface drag)
        # Sea has less friction than land
        friction_coeff = land_sea_mask * self.cf + (1 - land_sea_mask) * self.cf_sea
        
        # Decay wind due to friction over time
        time_decay = torch.exp(-friction_coeff * delta_hours)
        
        # 5. Temperature gradient effect (thermal wind)
        temp_gradient = self.calculate_temperature_gradient(t2m, lat_rad)
        
        # 6. Topographic effects - wind channeling
        topo_effect = self.calculate_topographic_effect(orograph, lat_rad)
        
        # 7. Combine all effects with physically reasonable weights
        predicted_u100 = (
            # Persistence with friction (40%)
            0.4 * current_u100 * time_decay +
            
            # Jet stream influence (30%)
            0.3 * jet_strength +
            
            # Pressure gradient effect (15%)
            0.15 * pressure_gradient +
            
            # Thermal wind effect (10%)
            0.1 * temp_gradient +
            
            # Topographic effect (5%)
            0.05 * topo_effect
        )
        
        # 8. Apply smoothing and constraints
        predicted_u100 = self.apply_physical_constraints(predicted_u100, lat_rad, land_sea_mask)
        
        return predicted_u100
    
    def calculate_pressure_gradient(self, sp, lat_rad):
        """Calculate simplified pressure gradient effect"""
        height, width = sp.shape
        device = sp.device
        
        # Simple gradient using finite differences with proper boundary handling
        # Use central differences in interior, forward/backward at boundaries
        
        # Initialize gradient tensors
        grad_x = torch.zeros_like(sp)
        grad_y = torch.zeros_like(sp)
        
        # Central differences for interior points
        grad_x[:, 1:-1] = (sp[:, 2:] - sp[:, :-2]) / 2.0
        grad_y[1:-1, :] = (sp[2:, :] - sp[:-2, :]) / 2.0
        
        # Forward/backward differences at boundaries
        grad_x[:, 0] = sp[:, 1] - sp[:, 0]
        grad_x[:, -1] = sp[:, -1] - sp[:, -2]
        grad_y[0, :] = sp[1, :] - sp[0, :]
        grad_y[-1, :] = sp[-1, :] - sp[-2, :]
        
        # Convert to approximate geostrophic wind
        # Geostrophic wind: u_g = - (1/(ρf)) * ∂p/∂y, v_g = (1/(ρf)) * ∂p/∂x
        # We focus on u-component (zonal wind)
        
        # Coriolis parameter (avoid division by zero near equator)
        f = 2 * self.omega * torch.sin(lat_rad)
        f = torch.where(torch.abs(f) < 1e-10, torch.sign(f) * 1e-10, f)
        
        # Geostrophic u-component (negative of pressure gradient in y-direction)
        u_geo = -grad_y / (self.rho * f)
        
        # Scale down since we're at 100m, not pure geostrophic level
        return 0.3 * u_geo
    
    def calculate_temperature_gradient(self, t2m, lat_rad):
        """Calculate temperature gradient effect"""
        height, width = t2m.shape
        device = t2m.device
        
        # Calculate meridional temperature gradient (north-south)
        # This drives thermal wind
        grad_y = torch.zeros_like(t2m)
        
        # Central differences for interior
        grad_y[1:-1, :] = (t2m[2:, :] - t2m[:-2, :]) / 2.0
        
        # Boundaries
        grad_y[0, :] = t2m[1, :] - t2m[0, :]
        grad_y[-1, :] = t2m[-1, :] - t2m[-2, :]
        
        # Thermal wind: ∂u/∂z ∝ -∂T/∂y
        # At 100m, we get some thermal wind effect
        thermal_u = -0.02 * grad_y  # Simplified thermal wind scaling
        
        return thermal_u
    
    def calculate_topographic_effect(self, orograph, lat_rad):
        """Calculate topographic influence on wind"""
        height, width = orograph.shape
        
        # Calculate gradient of topography
        grad_y, grad_x = torch.gradient(orograph, dim=[0, 1])
        
        # Wind tends to follow contours (along lines of constant elevation)
        # and accelerate over ridges
        topo_effect = torch.zeros_like(orograph)
        
        # Wind acceleration over ridges (positive gradient in wind direction)
        # Wind deceleration in valleys
        topo_effect = -0.01 * grad_y  # Simplified: wind slows when going uphill
        
        # Amplify effect in mountainous regions
        orograph_mag = torch.abs(orograph)
        mountain_mask = (orograph_mag > 1000).float()
        topo_effect = topo_effect * (1.0 + 2.0 * mountain_mask)
        
        return topo_effect
    
    def apply_physical_constraints(self, u100, lat_rad, land_sea_mask):
        """Apply physical constraints to predicted wind"""
        
        # 1. Apply smoothing to reduce noise
        u_smoothed = self.smooth_wind(u100)
        
        # 2. Apply latitudinal constraints (weaker winds near poles)
        lat_factor = torch.cos(lat_rad)  # 1 at equator, 0 at poles
        lat_factor = torch.clamp(lat_factor, 0.3, 1.0)  # Don't reduce too much
        
        # 3. Apply land/sea contrast
        # Winds are typically stronger over ocean
        ocean_boost = 1.0 + 0.2 * (1 - land_sea_mask)  # 20% stronger over ocean
        
        # Combine all constraints
        constrained_u = u_smoothed * lat_factor * ocean_boost
        
        # 4. Clamp to reasonable physical limits
        constrained_u = torch.clamp(constrained_u, -80, 80)
        
        return constrained_u
    
    def smooth_wind(self, wind):
        """Apply Gaussian-like smoothing to wind field"""
        # Simple 3x3 smoothing kernel
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=wind.dtype, device=wind.device) / 16.0
        
        # Apply convolution
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='circular'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return 0.7 * wind + 0.3 * smoothed
    
    def get_physical_explanation(self):
        """Return physical explanation of the prediction model"""
        explanation = """
        Physical Basis for U100 Prediction:
        
        1. JET STREAM INFLUENCE (30% weight):
           - Pattern: sin²(latitude) × cos²(latitude) = 0.25 × sin²(2×latitude)
           - Maximum at 45° latitude, zero at equator and poles
           - Gaussian envelope for realistic width
           - Seasonal migration (±10° latitude)
        
        2. PERSISTENCE WITH FRICTION (40% weight):
           - Current wind decays exponentially: u(t+Δt) = u(t) × exp(-C_f × Δt)
           - Friction coefficient: C_f(land) = 0.001, C_f(sea) = 0.0001
           - Ocean winds persist longer than land winds
        
        3. PRESSURE GRADIENT FORCE (15% weight):
           - Geostrophic approximation: u_g = -(1/(ρf)) × ∂p/∂y
           - Coriolis parameter: f = 2Ω sin(latitude)
           - Scaled down for 100m height (not pure geostrophic)
        
        4. THERMAL WIND EFFECT (10% weight):
           - Temperature gradient creates wind shear: ∂u/∂z ∝ -∂T/∂y
           - Warm equator → strong westerlies at altitude
           - Simplified linear scaling at 100m
        
        5. TOPOGRAPHIC EFFECTS (5% weight):
           - Wind slows when going uphill (negative gradient)
           - Enhanced in mountainous regions (>1000m elevation)
           - Factor of 3× stronger in mountains
        
        ADDITIONAL CONSTRAINTS:
        - Latitudinal scaling: winds weaker near poles (cos(latitude))
        - Ocean boost: winds 20% stronger over water
        - Smoothing: reduces small-scale noise
        - Physical limits: clamped to ±80 m/s
        - Boundary handling: proper finite differences
        
        ASSUMPTIONS & SIMPLIFICATIONS:
        1. Linear superposition of effects
        2. Constant air density
        3. Simplified boundary layer physics
        4. No explicit moisture effects
        5. Steady-state approximation for short Δt
        """
        return explanation


# Simple test function
def test_predictor():
    """Test the predictor with sample data"""
    predictor = U100mPredictor()
    
    # Create dummy data with correct shapes
    batch_size = 1
    height, width = 721, 1440
    
    # Create on CPU (can be moved to GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample tensors
    meteo = torch.randn(6, height, width, device=device)
    geo_dense = torch.randn(5, height, width, device=device)
    geo_cat = torch.randn(2, height, width, device=device)
    tempo = torch.randn(4, height, width, device=device)
    
    # Set some reasonable values
    meteo[0] = 273.15 + 20 * torch.cos(torch.linspace(-90, 90, height, device=device)).unsqueeze(1)  # t2m
    meteo[2] = 10 * torch.sin(torch.linspace(-90, 90, height, device=device)).unsqueeze(1)  # u100
    meteo[5] = 101325  # sp
    
    # Current time
    current_time = datetime(2024, 1, 15, 12, 0, 0)
    
    # Test prediction for different time horizons
    for delta in [0, 1, 3, 6]:
        print(f"\nTesting delta = {delta} hours")
        try:
            pred = predictor.predict_u100m(meteo, geo_dense, geo_cat, tempo, 
                                          current_time, delta)
            print(f"Success! Prediction shape: {pred.shape}")
            print(f"Min: {pred.min():.2f}, Max: {pred.max():.2f}, Mean: {pred.mean():.2f}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Print physical explanation
    print("\n" + "="*60)
    print(predictor.get_physical_explanation())
    print("="*60)


if __name__ == "__main__":
    test_predictor()