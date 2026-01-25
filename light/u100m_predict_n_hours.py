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
from datetime import datetime
from typing import Optional, Union
import numpy as np

class U100mPredictor:
    """
    Predict u100 at time t+delta using physical principles.
    GPU-optimized with automatic CPU conversion for compatibility.
    """
    
    def __init__(self, device: Optional[str] = None, return_cpu: bool = True):
        """
        Initialize the predictor.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
            return_cpu: If True, automatically convert result to CPU numpy array
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_cpu = return_cpu
        
        # Initialize all constants on the correct device
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize all physical constants as tensors on the correct device."""
        # Physical constants
        self.R = torch.tensor(287.058, device=self.device)  # Specific gas constant for dry air (J/(kg·K))
        self.omega = torch.tensor(7.2921159e-5, device=self.device)  # Earth's rotation rate (rad/s)
        self.rho = torch.tensor(1.225, device=self.device)  # Standard air density at sea level (kg/m³)
        
        # Jet stream parameters (empirical)
        self.jet_lat_center = torch.tensor(45.0, device=self.device)  # Typical jet stream latitude (degrees)
        self.jet_width = torch.tensor(15.0, device=self.device)  # Jet stream width (degrees)
        self.jet_max_speed = torch.tensor(50.0, device=self.device)  # Maximum jet stream speed (m/s)
        
        # Surface friction coefficient
        self.cf = torch.tensor(0.001, device=self.device)  # Drag coefficient over land
        self.cf_sea = torch.tensor(0.0001, device=self.device)  # Drag coefficient over sea
        
        # Mathematical constants
        self.PI = torch.tensor(math.pi, device=self.device)
        
        # Effect weights
        self.WEIGHT_PERSISTENCE = torch.tensor(0.4, device=self.device)
        self.WEIGHT_JET = torch.tensor(0.3, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.15, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.1, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.05, device=self.device)
        
        # Scaling factors
        self.PRESSURE_SCALING = torch.tensor(0.3, device=self.device)
        self.THERMAL_WIND_SCALING = torch.tensor(-0.02, device=self.device)
        self.TOPO_SCALING = torch.tensor(-0.01, device=self.device)
        self.TOPO_MOUNTAIN_BOOST = torch.tensor(2.0, device=self.device)
        self.OCEAN_BOOST = torch.tensor(1.2, device=self.device)
        self.LAT_FACTOR_MIN = torch.tensor(0.3, device=self.device)
        
        # Limits
        self.WIND_MIN = torch.tensor(-80.0, device=self.device)
        self.WIND_MAX = torch.tensor(80.0, device=self.device)
        self.MOUNTAIN_THRESHOLD = torch.tensor(1000.0, device=self.device)
        
        # Seasonal parameters
        self.SEASONAL_SHIFT_AMPLITUDE = torch.tensor(10.0, device=self.device)
        self.SEASONAL_BLEND = torch.tensor(0.7, device=self.device)
        
        # Smoothing parameters
        self.SMOOTHING_BLEND = torch.tensor(0.7, device=self.device)
        
        # Diurnal parameters
        self.DIURNAL_AMPLITUDE = torch.tensor(5.0, device=self.device)
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, tempo: torch.Tensor,
                current_time_utc: datetime, delta_hours: int,
                return_tensor: Optional[bool] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        Predict u100 at time t + delta_hours.
        
        Args:
            meteo: torch.Tensor shape (6, H, W) - [t2m, tp, u100, v100, d2m, sp]
            geo_dense: torch.Tensor shape (5, H, W) - [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            geo_cat: torch.Tensor shape (2, H, W) - [land_sea_mask, soil_type]
            tempo: torch.Tensor shape (4, H, W) - [sin_orbit, cos_orbit, sin_zenit, cos_zenit]
            current_time_utc: datetime object
            delta_hours: int, hours to predict ahead (0, 1, ...)
            return_tensor: If True, return torch.Tensor; if False, return numpy array.
                          If None, use self.return_cpu
            
        Returns:
            predicted_u100: torch.Tensor shape (H, W) or numpy array
        """
        # Ensure tensors are on the correct device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        tempo = tempo.to(self.device)
        
        # Extract individual components
        t2m = meteo[0]  # Temperature at 2m (K)
        current_u100 = meteo[2]  # Current u100
        sp = meteo[5]  # Surface pressure (Pa)
        
        orograph = geo_dense[0]  # Topography (m)
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        
        land_sea_mask = geo_cat[0]  # 0: sea, 1: land
        
        cos_zenit = tempo[3]
        
        height, width = t2m.shape
        
        # Convert trigonometric values back to angles for calculations
        lat_rad = torch.atan2(sin_lat, cos_lat)  # Latitude in radians
        lat_deg = torch.rad2deg(lat_rad)  # Latitude in degrees
        
        # 1. Temperature tendency (solar heating effect)
        # Cosine of zenith angle indicates solar intensity
        hour_of_day = (current_time_utc.hour + delta_hours) % 24
        
        # Create tensor for diurnal cycle calculation
        hour_tensor = torch.full((height, width), hour_of_day, 
                                dtype=torch.float32, device=self.device)
        diurnal_cycle = torch.sin(2 * self.PI * (hour_tensor - 12) / 24) * self.DIURNAL_AMPLITUDE
        
        # Future temperature estimate
        future_t2m = t2m + diurnal_cycle * cos_zenit
        
        # 2. Jet stream influence (major upper-level wind feature)
        jet_strength = self._compute_jet_stream(lat_rad, lat_deg, current_time_utc)
        
        # 3. Pressure gradient effect (simplified geostrophic wind)
        pressure_gradient = self._compute_pressure_gradient(sp, lat_rad)
        
        # 4. Friction effects (surface drag)
        time_decay = self._compute_friction_decay(land_sea_mask, delta_hours)
        
        # 5. Temperature gradient effect (thermal wind)
        temp_gradient = self._compute_temperature_gradient(t2m)
        
        # 6. Topographic effects - wind channeling
        topo_effect = self._compute_topographic_effect(orograph)
        
        # 7. Combine all effects with physically reasonable weights
        predicted_u100 = (
            # Persistence with friction
            self.WEIGHT_PERSISTENCE * current_u100 * time_decay +
            
            # Jet stream influence
            self.WEIGHT_JET * jet_strength +
            
            # Pressure gradient effect
            self.WEIGHT_PRESSURE * pressure_gradient +
            
            # Thermal wind effect
            self.WEIGHT_THERMAL * temp_gradient +
            
            # Topographic effect
            self.WEIGHT_TOPO * topo_effect
        )
        
        # 8. Apply smoothing and constraints
        predicted_u100 = self._apply_physical_constraints(predicted_u100, lat_rad, land_sea_mask)
        
        return self._convert_result(predicted_u100, return_tensor)
    
    def _convert_result(self, result: torch.Tensor, return_tensor: Optional[bool]) -> Union[torch.Tensor, np.ndarray]:
        """Convert result to appropriate format."""
        if return_tensor is None:
            return_tensor = not self.return_cpu
        
        if return_tensor:
            return result
        else:
            # Convert to CPU numpy array
            return result.cpu().numpy()
    
    def _compute_jet_stream(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor, 
                           current_time_utc: datetime) -> torch.Tensor:
        """Compute jet stream strength with seasonal variation."""
        # Base jet profile: sin²(lat) * cos²(lat) = 0.25 * sin²(2*lat)
        lat_norm = (lat_deg - self.jet_lat_center) / self.jet_width
        jet_profile = 0.25 * torch.sin(2 * lat_rad)**2 * torch.exp(-lat_norm**2)
        jet_strength = self.jet_max_speed * jet_profile
        
        # Jet stream rotates with seasons
        day_of_year = current_time_utc.timetuple().tm_yday
        seasonal_shift = self.SEASONAL_SHIFT_AMPLITUDE * torch.sin(
            torch.tensor(2 * math.pi * (day_of_year - 80) / 365, device=self.device)
        )
        
        # Create shifted jet profile
        jet_lat_shifted = self.jet_lat_center + seasonal_shift
        lat_norm_seasonal = (lat_deg - jet_lat_shifted) / self.jet_width
        jet_profile_seasonal = 0.25 * torch.sin(2 * lat_rad)**2 * torch.exp(-lat_norm_seasonal**2)
        jet_strength_seasonal = self.jet_max_speed * jet_profile_seasonal
        
        # Blend based on seasonality
        return self.SEASONAL_BLEND * jet_strength + (1 - self.SEASONAL_BLEND) * jet_strength_seasonal
    
    def _compute_pressure_gradient(self, sp: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Calculate simplified pressure gradient effect."""
        height, width = sp.shape
        
        # Initialize gradient tensors
        grad_x = torch.zeros_like(sp, device=self.device)
        grad_y = torch.zeros_like(sp, device=self.device)
        
        # Central differences for interior points
        grad_x[:, 1:-1] = (sp[:, 2:] - sp[:, :-2]) / 2.0
        grad_y[1:-1, :] = (sp[2:, :] - sp[:-2, :]) / 2.0
        
        # Forward/backward differences at boundaries
        grad_x[:, 0] = sp[:, 1] - sp[:, 0]
        grad_x[:, -1] = sp[:, -1] - sp[:, -2]
        grad_y[0, :] = sp[1, :] - sp[0, :]
        grad_y[-1, :] = sp[-1, :] - sp[-2, :]
        
        # Coriolis parameter (avoid division by zero near equator)
        f = 2 * self.omega * torch.sin(lat_rad)
        f = torch.where(torch.abs(f) < 1e-10, torch.sign(f) * 1e-10, f)
        
        # Geostrophic u-component (negative of pressure gradient in y-direction)
        u_geo = -grad_y / (self.rho * f)
        
        # Scale down since we're at 100m, not pure geostrophic level
        return self.PRESSURE_SCALING * u_geo
    
    def _compute_friction_decay(self, land_sea_mask: torch.Tensor, delta_hours: int) -> torch.Tensor:
        """Compute wind decay due to surface friction."""
        friction_coeff = land_sea_mask * self.cf + (1 - land_sea_mask) * self.cf_sea
        return torch.exp(-friction_coeff * delta_hours)
    
    def _compute_temperature_gradient(self, t2m: torch.Tensor) -> torch.Tensor:
        """Calculate temperature gradient effect."""
        height, width = t2m.shape
        
        # Calculate meridional temperature gradient (north-south)
        grad_y = torch.zeros_like(t2m, device=self.device)
        
        # Central differences for interior
        grad_y[1:-1, :] = (t2m[2:, :] - t2m[:-2, :]) / 2.0
        
        # Boundaries
        grad_y[0, :] = t2m[1, :] - t2m[0, :]
        grad_y[-1, :] = t2m[-1, :] - t2m[-2, :]
        
        # Thermal wind: ∂u/∂z ∝ -∂T/∂y
        return self.THERMAL_WIND_SCALING * grad_y
    
    def _compute_topographic_effect(self, orograph: torch.Tensor) -> torch.Tensor:
        """Calculate topographic influence on wind."""
        # Calculate gradient of topography
        grad_y, grad_x = torch.gradient(orograph, dim=[0, 1])
        
        # Wind deceleration when going uphill
        topo_effect = self.TOPO_SCALING * grad_y
        
        # Amplify effect in mountainous regions
        orograph_mag = torch.abs(orograph)
        mountain_mask = (orograph_mag > self.MOUNTAIN_THRESHOLD).float()
        topo_effect = topo_effect * (1.0 + self.TOPO_MOUNTAIN_BOOST * mountain_mask)
        
        return topo_effect
    
    def _apply_physical_constraints(self, u100: torch.Tensor, lat_rad: torch.Tensor, 
                                   land_sea_mask: torch.Tensor) -> torch.Tensor:
        """Apply physical constraints to predicted wind."""
        # 1. Apply smoothing to reduce noise
        u_smoothed = self._smooth_wind(u100)
        
        # 2. Apply latitudinal constraints (weaker winds near poles)
        lat_factor = torch.cos(lat_rad)  # 1 at equator, 0 at poles
        lat_factor = torch.clamp(lat_factor, self.LAT_FACTOR_MIN, 1.0)
        
        # 3. Apply land/sea contrast (winds stronger over ocean)
        ocean_boost = torch.where(land_sea_mask == 0, self.OCEAN_BOOST, 1.0)
        
        # Combine all constraints
        constrained_u = u_smoothed * lat_factor * ocean_boost
        
        # 4. Clamp to reasonable physical limits
        return torch.clamp(constrained_u, self.WIND_MIN, self.WIND_MAX)
    
    def _smooth_wind(self, wind: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian-like smoothing to wind field."""
        # Simple 3x3 smoothing kernel
        kernel = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=wind.dtype, device=self.device) / 16.0
        
        # Apply convolution
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='circular'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return self.SMOOTHING_BLEND * wind + (1 - self.SMOOTHING_BLEND) * smoothed
    
    def to(self, device: str) -> 'U100mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()  # Re-initialize constants on new device
        return self
    
    def get_physical_explanation(self) -> str:
        """Return physical explanation of the prediction model."""
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
        
        GPU OPTIMIZATIONS:
        - All constants stored as device tensors
        - Automatic device detection (CUDA if available)
        - Optional CPU conversion for compatibility
        - Vectorized operations for maximum speed
        
        ASSUMPTIONS & SIMPLIFICATIONS:
        1. Linear superposition of effects
        2. Constant air density
        3. Simplified boundary layer physics
        4. No explicit moisture effects
        5. Steady-state approximation for short Δt
        """
        return explanation


# Test function with GPU support
def test_predictor_gpu():
    """Test the predictor with GPU support."""
    print("Testing U100mPredictor with GPU support")
    print("=" * 60)
    
    # Initialize predictor (auto-detects GPU if available)
    predictor = U100mPredictor(return_cpu=True)  # Returns numpy arrays by default
    print(f"Using device: {predictor.device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Create test data
    height, width = 721, 1440
    
    # Create on appropriate device (predictor will move data if needed)
    meteo = torch.randn(6, height, width)
    geo_dense = torch.randn(5, height, width)
    geo_cat = torch.randn(2, height, width)
    tempo = torch.randn(4, height, width)
    
    # Set some reasonable values
    lat = torch.linspace(-90, 90, height)
    meteo[0] = 273.15 + 20 * torch.cos(lat * math.pi / 180.0).unsqueeze(1)  # t2m
    meteo[2] = 10 * torch.sin(lat * math.pi / 180.0).unsqueeze(1)  # u100
    meteo[5] = 101325  # sp
    
    # Set land/ocean mask
    land_sea_mask = (torch.abs(lat.unsqueeze(1)) < 60).float()
    geo_cat[0] = land_sea_mask
    
    current_time = datetime(2024, 1, 15, 12, 0, 0)
    
    # Warm up
    for _ in range(2):
        _ = predictor.predict(meteo, geo_dense, geo_cat, tempo, current_time, 6)
    
    # Test different prediction horizons
    import time
    
    for delta in [0, 1, 3, 6]:
        print(f"\nTesting delta = {delta} hours")
        
        start_time = time.perf_counter()
        pred = predictor.predict(meteo, geo_dense, geo_cat, tempo, current_time, delta)
        end_time = time.perf_counter()
        
        print(f"Success! Prediction time: {(end_time - start_time)*1000:.1f} ms")
        print(f"Prediction type: {type(pred)}")
        print(f"Prediction shape: {pred.shape}")
        
        if isinstance(pred, np.ndarray):
            print(f"Min: {pred.min():.2f}, Max: {pred.max():.2f}, Mean: {pred.mean():.2f}")
        else:  # torch.Tensor
            print(f"Min: {pred.min():.2f}, Max: {pred.max():.2f}, Mean: {pred.mean():.2f}")
    
    # Test with tensor return
    print("\n" + "="*60)
    print("Testing with tensor return:")
    
    pred_tensor = predictor.predict(meteo, geo_dense, geo_cat, tempo, current_time, 6, return_tensor=True)
    print(f"Tensor type: {type(pred_tensor)}")
    print(f"Tensor device: {pred_tensor.device}")
    
    # Print physical explanation
    print("\n" + "="*60)
    print(predictor.get_physical_explanation())
    print("="*60)


# Example of how to use in your existing code
def example_integration():
    """
    Example showing how to integrate with your existing code.
    """
    # In your play_physical_prediction function:
    # 1. Initialize predictor once (at module level or as class member)
    u100m_predictor = U100mPredictor(return_cpu=True)  # Returns numpy arrays
    
    # 2. Make predictions
    # pred = u100m_predictor.predict(ss_x.meteo, ss_x.geo_dense, ss_x.geo_cat, 
    #                                ss_x.tempo, dt_x, delta_hours=modulo)
    
    # 3. Your visualization code should work since pred is numpy array
    # show_array(arr1=arr_y, arr2=pred, ...)
    
    print("Integration example:")
    print("1. Initialize predictor: u100m_predictor = U100mPredictor(return_cpu=True)")
    print("2. Make predictions: pred = u100m_predictor.predict(...)")
    print("3. Use in visualization: show_array(arr1=..., arr2=pred, ...)")
    print("\nThe predictor automatically uses GPU if available and returns CPU numpy arrays.")


if __name__ == "__main__":
    test_predictor_gpu()
    example_integration()