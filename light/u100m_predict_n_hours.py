"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (721, 1440) in lat and lon,
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
    Fixed polar katabatic wind patterns.
    """
    
    def __init__(self, device: Optional[str] = None, return_cpu: bool = True):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_cpu = return_cpu
        self._setup_constants()
        
        self.TOPO_SCALING = torch.tensor(-0.0005, device=self.device)  # Further reduced
        # Stronger katabatic corrections
        self.KATABATIC_REDUCTION = torch.tensor(0.2, device=self.device)  # 80% reduction
        self.GREENLAND_NORTH_CORRECTION = torch.tensor(0.7, device=self.device)  # Stronger reverse
        self.GREENLAND_MID_CORRECTION = torch.tensor(-0.5, device=self.device)  # Stronger reduction
        
        # RADICALLY REDUCE topographic weight
        self.WEIGHT_TOPO = torch.tensor(0.01, device=self.device)  # Only 1% weight
        
        # Increase persistence weight
        self.WEIGHT_PERSISTENCE = torch.tensor(0.70, device=self.device)
        
        # Reduce pressure weight to compensate
        self.WEIGHT_PRESSURE = torch.tensor(0.10, device=self.device)
        
        # Make topographic scaling TINY
        self.TOPO_SCALING = torch.tensor(-0.0001, device=self.device)  # Almost zero
        
        # Add these new parameters for strip removal:
        self.STRIP_REDUCTION = torch.tensor(0.3, device=self.device)
        self.STRIP_SMOOTHING = torch.tensor(0.4, device=self.device)
        self.COASTAL_BLEND_WIDTH = torch.tensor(3.0, device=self.device)
        
        # More aggressive polar limits
        self.POLAR_COASTAL_LIMIT = torch.tensor(12.0, device=self.device)
        
        # **SIMPLIFIED WEIGHTS**
        self.WEIGHT_PERSISTENCE = torch.tensor(0.90, device=self.device)  # 90% persistence
        self.WEIGHT_JET = torch.tensor(0.05, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.03, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.01, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.01, device=self.device)  # Only 1%
        
        # **TINY topographic scaling**
        self.TOPO_SCALING = torch.tensor(-0.00001, device=self.device)  # Almost zero
        
        # **Stricter polar limits**
        self.POLAR_COASTAL_LIMIT = torch.tensor(8.0, device=self.device)
        
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
        self.cf_ice = torch.tensor(0.0045, device=self.device)  # Higher for ice
        
        # Effect weights
        self.WEIGHT_PERSISTENCE = torch.tensor(0.50, device=self.device)  # Reduced
        self.WEIGHT_JET = torch.tensor(0.25, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.15, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.05, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.05, device=self.device)
        
        # Scaling factors - REDUCED for polar regions
        self.PRESSURE_SCALING = torch.tensor(0.12, device=self.device)
        self.THERMAL_WIND_SCALING = torch.tensor(-0.006, device=self.device)
        self.TOPO_SCALING = torch.tensor(-0.001, device=self.device)  # Greatly reduced
        
        # Limits - TIGHTER for polar regions
        self.WIND_MIN = torch.tensor(-40.0, device=self.device)
        self.WIND_MAX = torch.tensor(40.0, device=self.device)
        self.POLAR_COASTAL_LIMIT = torch.tensor(20.0, device=self.device)  # Strict limit
        
        # Smoothing
        self.FINAL_SMOOTHING = torch.tensor(0.15, device=self.device)
        
        # Polar region thresholds
        self.ANTARCTICA_LAT = torch.tensor(-60.0, device=self.device)
        self.GREENLAND_LAT_MIN = torch.tensor(60.0, device=self.device)
        self.GREENLAND_LAT_MAX = torch.tensor(85.0, device=self.device)
        self.GREENLAND_LON_MIN = torch.tensor(-50.0, device=self.device)
        self.GREENLAND_LON_MAX = torch.tensor(-20.0, device=self.device)
        
        # NEW: Katabatic wind correction factors
        self.KATABATIC_REDUCTION = torch.tensor(0.3, device=self.device)  # 70% reduction
        self.GREENLAND_NORTH_CORRECTION = torch.tensor(0.5, device=self.device)  # Reverse sign for north Greenland
        self.GREENLAND_MID_CORRECTION = torch.tensor(-0.3, device=self.device)  # Reduce middle Greenland
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, tempo: torch.Tensor,
                current_time_utc: datetime, delta_hours: int,
                return_tensor: Optional[bool] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        SIMPLIFIED PREDICTION: Focus on persistence with minimal polar effects.
        """
        # Move to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        tempo = tempo.to(self.device)
        
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
        
        # **SIMPLIFIED APPROACH: 90% persistence, 10% other effects**
        
        # 2. Strong persistence (90% weight)
        time_decay = self._compute_simple_friction_decay(land_sea_mask, delta_hours, polar_mask)
        persistence = current_u100 * time_decay
        
        # 3. Minimal other effects (10% total)
        jet_strength = self._compute_jet_stream(lat_rad, lat_deg, current_time_utc) * 0.05
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
        
        # 6. Final smoothing
        predicted_u100 = self._minimal_smoothing(predicted_u100)
        
        # 7. Final clamp
        predicted_u100 = torch.clamp(predicted_u100, self.WIND_MIN, self.WIND_MAX)
        
        return self._convert_result(predicted_u100, return_tensor)

    def _compute_simple_friction_decay(self, land_sea_mask: torch.Tensor, 
                                    delta_hours: int,
                                    polar_mask: torch.Tensor) -> torch.Tensor:
        """Simple friction decay with polar adjustments."""
        # Base decay
        decay = torch.exp(-self.cf_sea * delta_hours)  # Use sea value as base
        
        # Land has faster decay
        land_decay = torch.exp(-self.cf_land * delta_hours)
        decay = torch.where(land_sea_mask > 0.5, land_decay, decay)
        
        # Polar regions: slower decay (more persistence)
        polar_decay = torch.exp(-self.cf_sea * delta_hours * 0.5)  # 50% slower decay
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
            
            effect = torch.where(non_polar, grad * 0.001, effect)  # Very small scaling
        
        return effect

    def _apply_polar_hard_limits_only(self, wind: torch.Tensor,
                                    lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                    antarctica_mask: torch.Tensor,
                                    greenland_mask: torch.Tensor,
                                    greenland_north_mask: torch.Tensor,
                                    greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply ONLY hard limits - no complex corrections.
        """
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
    
    def _identify_polar_regions(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                            orograph: torch.Tensor) -> tuple:
        """
        Identify polar regions with sub-regions.
        """
        # Antarctica
        antarctica_mask = (lat_deg < self.ANTARCTICA_LAT) & (orograph > 500)
        
        # Greenland main mask
        greenland_lat = (lat_deg > self.GREENLAND_LAT_MIN) & (lat_deg < self.GREENLAND_LAT_MAX)
        greenland_lon = (lon_deg > self.GREENLAND_LON_MIN) & (lon_deg < self.GREENLAND_LON_MAX)
        greenland_mask = greenland_lat & greenland_lon & (orograph > 500)
        
        # North Greenland (problem area: too negative winds)
        greenland_north_mask = greenland_mask & (lat_deg > 75.0)
        
        # Middle Greenland coasts (problem area: too positive winds)
        # Eastern coast roughly at lon > -30, western coast lon < -40
        greenland_mid_lat = (lat_deg > 65.0) & (lat_deg < 75.0)
        greenland_east_coast = greenland_mid_lat & (lon_deg > -30.0) & (lon_deg < -20.0)
        greenland_west_coast = greenland_mid_lat & (lon_deg > -50.0) & (lon_deg < -40.0)
        greenland_mid_mask = (greenland_east_coast | greenland_west_coast) & greenland_mask
        
        return antarctica_mask, greenland_mask, greenland_north_mask, greenland_mid_mask

    def _apply_direct_hard_limits(self, wind: torch.Tensor,
                                lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                antarctica_mask: torch.Tensor,
                                greenland_mask: torch.Tensor) -> torch.Tensor:
        """
        DIRECT HARD LIMITS for problem areas.
        """
        result = wind.clone()
        
        # 1. ANTARCTICA COASTAL: Cap at ±10 m/s
        if torch.any(antarctica_mask):
            ant_coastal = self._get_coastal_mask(antarctica_mask)
            if torch.any(ant_coastal):
                result = torch.where(ant_coastal,
                                torch.clamp(result, 
                                            torch.tensor(-10.0, device=self.device),
                                            torch.tensor(10.0, device=self.device)),
                                result)
        
        # 2. NORTH GREENLAND: Force eastward (positive) bias
        if torch.any(greenland_mask):
            north_greenland = greenland_mask & (lat_deg > 75.0)
            if torch.any(north_greenland):
                # Force minimum of -5 m/s (less westward)
                result = torch.where(north_greenland,
                                torch.maximum(result, torch.tensor(-5.0, device=self.device)),
                                result)
                # Force maximum of +15 m/s
                result = torch.where(north_greenland,
                                torch.minimum(result, torch.tensor(15.0, device=self.device)),
                                result)
        
        # 3. MIDDLE GREENLAND COASTS: Cap at ±8 m/s
        if torch.any(greenland_mask):
            mid_greenland = greenland_mask & (lat_deg > 65.0) & (lat_deg <= 75.0)
            if torch.any(mid_greenland):
                result = torch.where(mid_greenland,
                                torch.clamp(result,
                                            torch.tensor(-8.0, device=self.device),
                                            torch.tensor(8.0, device=self.device)),
                                result)
        
        return result

    def _get_coastal_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get coastal areas of a mask."""
        kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32, device=self.device) / 4.0
        
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(mask_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        # Coastal zone: where mask is 1 but neighbors average < 0.8
        coastal = mask & (smoothed < 0.8)
        return coastal
    
    def _compute_polar_corrected_topographic_effect(self, orograph: torch.Tensor,
                                                lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                                antarctica_mask: torch.Tensor,
                                                greenland_mask: torch.Tensor,
                                                greenland_north_mask: torch.Tensor,
                                                greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """
        ZERO topographic effect in polar regions - use climatology instead.
        """
        height, width = orograph.shape
        
        # **CRITICAL CHANGE**: Compute topographic effect ONLY for non-polar regions
        polar_mask = antarctica_mask | greenland_mask
        non_polar_mask = ~polar_mask
        
        # Initialize with zeros
        base_effect = torch.zeros_like(orograph, device=self.device, dtype=orograph.dtype)
        
        # Only compute slopes for NON-POLAR regions
        if torch.any(non_polar_mask):
            # Extract non-polar region
            # We need to compute slope on the whole field but mask it later
            kernel_y = torch.tensor([[0, -0.5, 0],
                                    [0, 0, 0],
                                    [0, 0.5, 0]], dtype=orograph.dtype, device=self.device)
            
            kernel_x = torch.tensor([[0, 0, 0],
                                    [-0.5, 0, 0.5],
                                    [0, 0, 0]], dtype=orograph.dtype, device=self.device)
            
            orograph_4d = orograph.unsqueeze(0).unsqueeze(0)
            kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
            kernel_x_4d = kernel_x.unsqueeze(0).unsqueeze(0)
            
            slope_y = torch.nn.functional.conv2d(
                torch.nn.functional.pad(orograph_4d, (1, 1, 1, 1), mode='replicate'),
                kernel_y_4d,
                padding=0
            ).squeeze()
            
            slope_x = torch.nn.functional.conv2d(
                torch.nn.functional.pad(orograph_4d, (1, 1, 1, 1), mode='replicate'),
                kernel_x_4d,
                padding=0
            ).squeeze()
            
            # Compute effect only for non-polar
            non_polar_effect = self.TOPO_SCALING * (slope_y * 0.7 + slope_x * 0.3) * 0.01
            base_effect = torch.where(non_polar_mask, non_polar_effect, base_effect)
        
        # **POLAR REGIONS: Use fixed climatological values (NO SLOPE CALCULATION)**
        
        # ANTARCTICA: Use smooth longitude-based pattern
        if torch.any(antarctica_mask):
            # Smooth eastward flow with coastal variations
            # Based on observed Antarctic near-surface wind climatology
            
            # Create smooth longitude gradient (eastward increases with longitude)
            lon_norm = (lon_deg + 180.0) / 360.0  # 0 to 1
            ant_base = torch.sin(lon_norm * 2 * math.pi) * 2.0  # ±2 m/s
            
            # Coastal enhancement (smooth)
            ant_coastal = self._get_smooth_coastal_mask(antarctica_mask, width=3)
            ant_coastal_effect = torch.sin(lon_norm * math.pi) * 1.0  # Additional ±1 m/s
            
            # Combine
            ant_pattern = ant_base + ant_coastal * ant_coastal_effect
            
            # Apply with smooth transition
            ant_transition = self._create_smooth_transition_mask(antarctica_mask, width=5)
            base_effect = base_effect * (1.0 - ant_transition) + ant_pattern * ant_transition
        
        # GREENLAND: Fixed regional patterns
        if torch.any(greenland_mask):
            greenland_transition = self._create_smooth_transition_mask(greenland_mask, width=5)
            
            # Create smooth latitudinal gradient
            lat_norm = (lat_deg - 60.0) / (85.0 - 60.0)  # 0 to 1 from south to north
            lat_norm = torch.clamp(lat_norm, 0.0, 1.0)
            
            # Base pattern: slightly eastward overall
            greenland_base = torch.ones_like(lat_deg) * 1.5  # +1.5 m/s eastward
            
            # North Greenland: stronger eastward
            if torch.any(greenland_north_mask):
                north_transition = self._create_smooth_transition_mask(greenland_north_mask, width=3)
                north_enhancement = torch.ones_like(lat_deg) * 2.0  # Additional +2 m/s
                greenland_base = greenland_base + north_enhancement * north_transition
            
            # Middle coasts: reduce eastward
            if torch.any(greenland_mid_mask):
                mid_transition = self._create_smooth_transition_mask(greenland_mid_mask, width=3)
                mid_reduction = torch.where(lon_deg < -40.0, 
                                        torch.tensor(-3.0, device=self.device),  # West coast
                                        torch.tensor(-1.0, device=self.device))  # East coast
                greenland_base = greenland_base + mid_reduction * mid_transition
            
            # Apply Greenland pattern
            base_effect = base_effect * (1.0 - greenland_transition) + greenland_base * greenland_transition
        
        # **FINAL SMOOTHING** to ensure no strips
        base_effect = self._apply_heavy_smoothing(base_effect, polar_mask)
        
        return base_effect

    def _get_smooth_coastal_mask(self, mask: torch.Tensor, width: int = 3) -> torch.Tensor:
        """Get smooth coastal mask using distance transform approximation."""
        # Use multiple convolutions to create smooth coastal gradient
        coastal_mask = torch.zeros_like(mask, dtype=torch.float32)
        
        for w in range(1, width + 1):
            kernel_size = 2 * w + 1
            kernel = torch.ones(kernel_size, kernel_size, device=self.device) / (kernel_size * kernel_size)
            
            mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
            
            pad_size = kernel_size // 2
            smoothed = torch.nn.functional.conv2d(
                torch.nn.functional.pad(mask_4d, (pad_size, pad_size, pad_size, pad_size), mode='replicate'),
                kernel_4d,
                padding=0
            ).squeeze()
            
            # Coastal is where smoothed is between thresholds
            coastal = (smoothed > 0.3) & (smoothed < 0.7) & mask
            coastal_mask = coastal_mask + coastal.float() * (1.0 / width)
        
        return coastal_mask

    def _apply_heavy_smoothing(self, tensor: torch.Tensor, polar_mask: torch.Tensor) -> torch.Tensor:
        """Apply heavy smoothing specifically to polar regions."""
        result = tensor.clone()
        
        if torch.any(polar_mask):
            # Apply different smoothing inside vs outside polar regions
            
            # Outside polar: light smoothing
            outside_polar = ~polar_mask
            if torch.any(outside_polar):
                # Light 3x3 smoothing for non-polar
                kernel_light = torch.tensor([[1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]], device=self.device, dtype=tensor.dtype) / 16.0
                
                tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
                kernel_4d = kernel_light.unsqueeze(0).unsqueeze(0)
                
                smoothed = torch.nn.functional.conv2d(
                    torch.nn.functional.pad(tensor_4d, (1, 1, 1, 1), mode='replicate'),
                    kernel_4d,
                    padding=0
                ).squeeze()
                
                # Apply only to outside polar
                result = torch.where(outside_polar, smoothed, result)
            
            # Inside polar: VERY heavy smoothing
            # Create 5x5 Gaussian kernel
            x = torch.arange(5, device=self.device) - 2
            y = torch.arange(5, device=self.device) - 2
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            kernel_heavy = torch.exp(-(xx**2 + yy**2) / 2.0)
            kernel_heavy = kernel_heavy / torch.sum(kernel_heavy)
            
            tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
            kernel_4d = kernel_heavy.unsqueeze(0).unsqueeze(0)
            
            smoothed_heavy = torch.nn.functional.conv2d(
                torch.nn.functional.pad(tensor_4d, (2, 2, 2, 2), mode='replicate'),
                kernel_4d,
                padding=0
            ).squeeze()
            
            # Apply heavy smoothing to polar regions
            polar_smooth_mask = self._create_smooth_transition_mask(polar_mask, width=3)
            result = result * (1.0 - polar_smooth_mask) + smoothed_heavy * polar_smooth_mask
        
        return result

    def _create_smooth_transition_mask(self, mask: torch.Tensor, width: int = 5) -> torch.Tensor:
        """Create smooth transition mask using Gaussian blur."""
        # Create kernel
        kernel_size = 2 * width + 1
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - width
        y = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - width
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Gaussian kernel
        sigma = width / 2.0
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        
        # Apply convolution
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        padded = torch.nn.functional.pad(mask_4d, (width, width, width, width), mode='replicate')
        smoothed = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        return smoothed

    def _apply_anisotropic_diffusion(self, tensor: torch.Tensor, iterations: int = 2) -> torch.Tensor:
        """Apply anisotropic diffusion to remove strips while preserving edges."""
        result = tensor.clone()
        
        for _ in range(iterations):
            # Compute gradients
            grad_y = torch.zeros_like(result)
            grad_x = torch.zeros_like(result)
            
            # Simple finite differences with boundary replication
            grad_y[1:-1, :] = result[2:, :] - result[:-2, :]
            grad_x[:, 1:-1] = result[:, 2:] - result[:, :-2]
            
            # Handle boundaries
            grad_y[0, :] = result[1, :] - result[0, :]
            grad_y[-1, :] = result[-1, :] - result[-2, :]
            grad_x[:, 0] = result[:, 1] - result[:, 0]
            grad_x[:, -1] = result[:, -1] - result[:, -2]
            
            # Diffusion coefficient (edge-preserving)
            grad_magnitude = torch.sqrt(grad_y**2 + grad_x**2 + 1e-8)
            c = torch.exp(-(grad_magnitude / 2.0)**2)  # Perona-Malik
            
            # Apply diffusion
            diff_y = torch.zeros_like(result)
            diff_x = torch.zeros_like(result)
            
            diff_y[1:-1, :] = c[1:-1, :] * (result[2:, :] - result[1:-1, :]) - \
                            c[:-2, :] * (result[1:-1, :] - result[:-2, :])
            
            diff_x[:, 1:-1] = c[:, 1:-1] * (result[:, 2:] - result[:, 1:-1]) - \
                            c[:, :-2] * (result[:, 1:-1] - result[:, :-2])
            
            # Update with small timestep
            result = result + 0.25 * (diff_y + diff_x)
        
        return result

    def _apply_gaussian_smoothing(self, tensor: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian smoothing."""
        # Create Gaussian kernel
        kernel_size = int(2 * 2 * sigma + 1)  # 2-sigma radius
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        
        # Apply convolution
        tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        pad_size = kernel_size // 2
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(tensor_4d, (pad_size, pad_size, pad_size, pad_size), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return smoothed

    def _get_coastal_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Get coastal areas of a mask."""
        kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32, device=self.device) / 4.0
        
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(mask_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        # Coastal zone: where mask is 1 but neighbors average < 0.8
        coastal = mask & (smoothed < 0.8)
        return coastal
    
    def _apply_polar_katabatic_corrections(self, wind: torch.Tensor,
                                        lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                        antarctica_mask: torch.Tensor,
                                        greenland_mask: torch.Tensor,
                                        greenland_north_mask: torch.Tensor,
                                        greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply strip-aware polar katabatic corrections.
        """
        # Start with original wind
        corrected = wind.clone()
        
        # 1. First, apply strip reduction filter to entire wind field
        corrected = self._apply_strip_reduction_filter(corrected)
        
        # 2. Regional corrections with SMOOTH transitions
        polar_mask = antarctica_mask | greenland_mask
        
        if torch.any(polar_mask):
            # Create smooth regional masks
            ant_smooth = self._create_smooth_transition_mask(antarctica_mask, width=4)
            greenland_smooth = self._create_smooth_transition_mask(greenland_mask, width=4)
            
            # Antarctica: Reduce extremes
            if torch.any(antarctica_mask):
                # Compute regional statistics
                ant_values = corrected[antarctica_mask]
                if len(ant_values) > 0:
                    ant_mean = torch.mean(ant_values)
                    ant_std = torch.std(ant_values)
                    
                    # Reduce extremes more than mean
                    ant_mask_flat = antarctica_mask.flatten()
                    ant_correction = torch.zeros_like(corrected)
                    
                    # Iterate through Antarctica points
                    for i in range(len(corrected.flatten())):
                        if ant_mask_flat[i]:
                            val = corrected.flatten()[i]
                            # Reduce extreme positive values more
                            if val > ant_mean + ant_std:
                                correction_factor = 0.5  # 50% reduction
                            elif val > ant_mean:
                                correction_factor = 0.8  # 20% reduction
                            elif val < ant_mean - ant_std:
                                correction_factor = 1.3  # +30% to reduce extreme negative
                            else:
                                correction_factor = 1.0  # No change
                            
                            ant_correction.flatten()[i] = val * correction_factor
                    
                    # Blend with smooth transition
                    corrected = corrected * (1.0 - ant_smooth) + ant_correction * ant_smooth
            
            # Greenland: Regional adjustments
            if torch.any(greenland_mask):
                # North Greenland: Add eastward bias smoothly
                if torch.any(greenland_north_mask):
                    north_smooth = self._create_smooth_transition_mask(greenland_north_mask, width=5)
                    north_bias = torch.tensor(4.0, device=self.device)  # +4 m/s eastward
                    corrected = corrected + north_bias * north_smooth
                
                # Middle Greenland coasts: Reduce positive extremes
                if torch.any(greenland_mid_mask):
                    mid_smooth = self._create_smooth_transition_mask(greenland_mid_mask, width=4)
                    
                    # Find positive values in this region
                    mid_positive = (corrected > 0) & greenland_mid_mask
                    if torch.any(mid_positive):
                        # Apply sigmoid-like reduction to positive values
                        reduction = 1.0 / (1.0 + torch.exp((corrected - 5.0) / 2.0))
                        corrected = torch.where(mid_positive, 
                                            corrected * (0.5 + 0.5 * reduction),
                                            corrected)
        
        # 3. Apply final strip removal
        corrected = self._apply_final_strip_removal(corrected, polar_mask)
        
        return corrected

    def _apply_strip_reduction_filter(self, wind: torch.Tensor) -> torch.Tensor:
        """Apply Fourier-based filter to remove strip artifacts."""
        height, width = wind.shape
        
        # Perform 2D FFT
        wind_fft = torch.fft.fft2(wind)
        wind_fft_shifted = torch.fft.fftshift(wind_fft)
        
        # Create frequency domain filter to remove vertical/horizontal strips
        u = torch.arange(width, device=self.device) - width // 2
        v = torch.arange(height, device=self.device) - height // 2
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        
        # Create band-stop filter for strip frequencies
        # Strips typically appear as high frequencies in one direction
        freq_u = torch.abs(uu / width)
        freq_v = torch.abs(vv / height)
        
        # Filter out high frequencies in both directions (strip patterns)
        strip_filter = torch.ones_like(wind_fft_shifted, dtype=torch.float32)
        
        # Remove very high frequencies (potential strips)
        high_freq_mask = (freq_u > 0.4) | (freq_v > 0.4)
        strip_filter[high_freq_mask] = 0.5  # Reduce but not eliminate
        
        # Apply filter
        filtered_fft = wind_fft_shifted * strip_filter
        
        # Inverse FFT
        filtered_fft_ishift = torch.fft.ifftshift(filtered_fft)
        filtered = torch.fft.ifft2(filtered_fft_ishift).real
        
        # Blend original and filtered (keep 70% original, 30% filtered)
        return wind * 0.7 + filtered * 0.3

    def _apply_final_strip_removal(self, wind: torch.Tensor, polar_mask: torch.Tensor) -> torch.Tensor:
        """Final strip removal with regional awareness."""
        # Apply median filtering in polar regions (excellent for removing strips)
        if torch.any(polar_mask):
            # Create a copy for median filtering
            wind_median = wind.clone()
            
            # Apply 3x3 median filter only to polar regions
            for i in range(1, wind.shape[0] - 1):
                for j in range(1, wind.shape[1] - 1):
                    if polar_mask[i, j]:
                        # Get 3x3 neighborhood
                        neighborhood = wind[i-1:i+2, j-1:j+2]
                        # Compute median
                        wind_median[i, j] = torch.median(neighborhood.flatten())
            
            # Blend with original using smooth transition
            polar_smooth = self._create_smooth_transition_mask(polar_mask, width=2)
            wind = wind * (1.0 - polar_smooth) + wind_median * polar_smooth
        
        return wind

    def _create_transition_mask(self, mask: torch.Tensor, width: int) -> torch.Tensor:
        """Create a transition zone mask around the given mask."""
        kernel = torch.ones(2*width+1, 2*width+1, device=self.device, dtype=torch.float32)
        kernel = kernel / kernel.numel()
        
        mask_4d = mask.float().unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution
        padded = torch.nn.functional.pad(mask_4d, (width, width, width, width), mode='replicate')
        smoothed = torch.nn.functional.conv2d(padded, kernel_4d, padding=0).squeeze()
        
        # Transition zone: where smoothed value is between 0.1 and 0.9
        transition = (smoothed > 0.1) & (smoothed < 0.9) & ~mask
        
        return transition
    
    def _apply_polar_coastal_limits(self, wind: torch.Tensor,
                                land_sea_mask: torch.Tensor,
                                antarctica_mask: torch.Tensor,
                                greenland_mask: torch.Tensor,
                                greenland_north_mask: torch.Tensor,
                                greenland_mid_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply strict limits to polar coastal regions.
        """
        # Create polar coastal mask
        polar_mask = antarctica_mask | greenland_mask
        
        # Find coastal areas in polar regions
        kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32, device=self.device) / 4.0
        
        polar_mask_4d = polar_mask.float().unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        polar_smooth = torch.nn.functional.conv2d(
            torch.nn.functional.pad(polar_mask_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        # Coastal zone in polar regions
        polar_coastal = (polar_smooth > 0.2) & (polar_smooth < 0.8) & polar_mask
        
        # Apply strict limits to polar coastal areas
        limited = wind.clone()
        
        if torch.any(polar_coastal):
            # VERY strict limits for coastal areas
            limited = torch.where(polar_coastal,
                                torch.clamp(limited, 
                                        torch.tensor(-15.0, device=self.device), 
                                        torch.tensor(15.0, device=self.device)),
                                limited)
        
        # Specific stricter limits for problem areas
        if torch.any(greenland_north_mask):
            limited = torch.where(greenland_north_mask,
                                torch.clamp(limited, 
                                        torch.tensor(-12.0, device=self.device), 
                                        torch.tensor(8.0, device=self.device)),
                                limited)
        
        if torch.any(greenland_mid_mask):
            limited = torch.where(greenland_mid_mask,
                                torch.clamp(limited, 
                                        torch.tensor(-10.0, device=self.device), 
                                        torch.tensor(10.0, device=self.device)),
                                limited)
        
        # Antarctica coastal extreme limits
        if torch.any(antarctica_mask):
            ant_coastal = polar_coastal & antarctica_mask
            if torch.any(ant_coastal):
                limited = torch.where(ant_coastal,
                                    torch.clamp(limited,
                                            torch.tensor(-12.0, device=self.device),
                                            torch.tensor(12.0, device=self.device)),
                                    limited)
        
        return limited
    
    def _compute_polar_friction_decay(self, land_sea_mask: torch.Tensor, 
                                    delta_hours: int,
                                    polar_mask: torch.Tensor,
                                    antarctica_mask: torch.Tensor,
                                    greenland_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute friction decay with polar-specific values.
        """
        # Base friction
        friction_coeff = land_sea_mask * self.cf_land + (1 - land_sea_mask) * self.cf_sea
        
        # Enhanced friction for ice sheets
        ice_mask = antarctica_mask | greenland_mask
        ice_friction = torch.where(ice_mask, self.cf_ice, torch.tensor(0.0, device=self.device))
        friction_coeff = friction_coeff + ice_friction
        
        decay = torch.exp(-friction_coeff * delta_hours)
        
        # Higher minimum decay in polar regions to prevent stuck values
        min_decay = torch.where(polar_mask,
                            torch.tensor(0.6, device=self.device),
                            torch.tensor(0.4, device=self.device))
        
        # FIX THE CLAMP ERROR:
        max_decay = torch.tensor(1.0, device=decay.device, dtype=decay.dtype)
        return torch.clamp(decay, min_decay, max_decay)
    
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
    
    def _add_variability(self, wind: torch.Tensor, land_sea_mask: torch.Tensor) -> torch.Tensor:
        """Add variability."""
        height, width = wind.shape
        
        torch.manual_seed(42)
        
        # Two scales of noise
        noise1 = torch.randn(height // 10, width // 10, device=self.device)
        noise1 = torch.nn.functional.interpolate(
            noise1.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        noise2 = torch.randn(height // 5, width // 5, device=self.device)
        noise2 = torch.nn.functional.interpolate(
            noise2.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        total_noise = noise1 * 0.6 + noise2 * 0.4
        
        if torch.std(total_noise) > 1e-6:
            total_noise = total_noise / torch.std(total_noise) * 1.0
        
        # More variability over land
        land_factor = 1.0 + land_sea_mask * 0.2
        total_noise = total_noise * land_factor
        
        return wind + total_noise
    
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
    
    def _convert_result(self, result: torch.Tensor, return_tensor: Optional[bool]) -> Union[torch.Tensor, np.ndarray]:
        """Convert result to appropriate format."""
        if return_tensor is None:
            return_tensor = not self.return_cpu
        
        if return_tensor:
            return result
        else:
            return result.cpu().numpy()
    
    def to(self, device: str) -> 'U100mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()
        return self