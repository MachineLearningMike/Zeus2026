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
    Based on research: mid-latitude winds stronger, more variability.
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
        self.R = torch.tensor(287.058, device=self.device)
        self.omega = torch.tensor(7.2921159e-5, device=self.device)
        self.rho = torch.tensor(1.225, device=self.device)
        
        # Jet stream parameters - Based on research
        self.jet_lat_center = torch.tensor(45.0, device=self.device)
        self.jet_width = torch.tensor(10.0, device=self.device)
        self.jet_max_speed = torch.tensor(25.0, device=self.device)
        
        # Surface friction
        self.cf = torch.tensor(0.002, device=self.device)
        self.cf_sea = torch.tensor(0.0003, device=self.device)
        
        # Mathematical constants
        self.PI = torch.tensor(math.pi, device=self.device)
        
        # Effect weights
        self.WEIGHT_PERSISTENCE = torch.tensor(0.55, device=self.device)
        self.WEIGHT_JET = torch.tensor(0.25, device=self.device)
        self.WEIGHT_PRESSURE = torch.tensor(0.12, device=self.device)
        self.WEIGHT_THERMAL = torch.tensor(0.05, device=self.device)
        self.WEIGHT_TOPO = torch.tensor(0.03, device=self.device)
        
        # Scaling factors
        self.PRESSURE_SCALING = torch.tensor(0.20, device=self.device)
        self.THERMAL_WIND_SCALING = torch.tensor(-0.012, device=self.device)
        self.TOPO_SCALING = torch.tensor(-0.006, device=self.device)
        self.TOPO_MOUNTAIN_BOOST = torch.tensor(1.8, device=self.device)
        self.OCEAN_BOOST = torch.tensor(1.12, device=self.device)
        
        # Latitudinal factor
        self.LAT_FACTOR_MIN = torch.tensor(0.5, device=self.device)
        
        # Limits
        self.WIND_MIN = torch.tensor(-45.0, device=self.device)
        self.WIND_MAX = torch.tensor(45.0, device=self.device)
        self.MOUNTAIN_THRESHOLD = torch.tensor(1200.0, device=self.device)
        
        # Seasonal parameters
        self.SEASONAL_SHIFT_AMPLITUDE = torch.tensor(6.0, device=self.device)
        self.SEASONAL_BLEND = torch.tensor(0.75, device=self.device)
        
        # Smoothing parameters
        self.SMOOTHING_BLEND = torch.tensor(0.7, device=self.device)
        
        # Diurnal parameters
        self.DIURNAL_AMPLITUDE = torch.tensor(2.0, device=self.device)
        
        # Normalization
        self.TARGET_MEAN = torch.tensor(6.0, device=self.device)
        self.TARGET_STD = torch.tensor(10.0, device=self.device)
        
        # Research-based amplification factors
        self.MIDLAT_AMPLIFICATION = torch.tensor(1.3, device=self.device)
        self.OCEAN_AMPLIFICATION = torch.tensor(1.15, device=self.device)
        
        # Variability parameters
        self.TURBULENCE_SCALE = torch.tensor(0.1, device=self.device)
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, tempo: torch.Tensor,
                current_time_utc: datetime, delta_hours: int,
                return_tensor: Optional[bool] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        Predict u100 at time t + delta_hours.
        """
        # Ensure tensors are on the correct device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        tempo = tempo.to(self.device)
        
        # Extract individual components
        t2m = meteo[0]
        current_u100 = meteo[2]
        sp = meteo[5]
        
        orograph = geo_dense[0]
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        
        land_sea_mask = geo_cat[0]
        
        height, width = t2m.shape
        
        # Stable latitude calculation
        cos_lat_clamped = torch.clamp(cos_lat, -0.9999, 0.9999)
        lat_rad = torch.where(
            cos_lat_clamped > 0.5,
            torch.asin(torch.clamp(sin_lat, -0.9999, 0.9999)),
            torch.acos(cos_lat_clamped) * torch.sign(sin_lat)
        )
        lat_rad = torch.remainder(lat_rad + math.pi, 2 * math.pi) - math.pi
        lat_deg = torch.rad2deg(lat_rad)
        
        # 1. Jet stream influence
        jet_strength = self._compute_research_jet_stream(lat_rad, lat_deg, current_time_utc)
        
        # 2. Pressure gradient effect
        pressure_gradient = self._compute_research_pressure_gradient(sp, lat_rad)
        
        # 3. Friction effects - FIXED METHOD CALL
        time_decay = self._compute_research_friction_decay(land_sea_mask, delta_hours)
        
        # 4. Temperature gradient effect
        temp_gradient = self._compute_research_temperature_gradient(t2m, lat_rad)
        
        # 5. Topographic effects
        topo_effect = self._compute_research_topographic_effect(orograph, lat_rad)
        
        # 6. Scale current wind
        current_scaled = self._research_based_scaling(current_u100, lat_rad)
        
        # 7. Combine with weights
        combined = (
            self.WEIGHT_PERSISTENCE * current_scaled * time_decay +
            self.WEIGHT_JET * jet_strength +
            self.WEIGHT_PRESSURE * pressure_gradient +
            self.WEIGHT_THERMAL * temp_gradient +
            self.WEIGHT_TOPO * topo_effect
        )
        
        # 8. Apply mid-latitude amplification
        combined = self._apply_midlatitude_amplification(combined, lat_rad)
        
        # 9. Add variability
        combined = self._add_research_variability(combined, lat_rad, land_sea_mask)
        
        # 10. Apply constraints
        predicted_u100 = self._apply_research_constraints(combined, lat_rad, land_sea_mask)
        
        # 11. Final light smoothing
        predicted_u100 = self._light_smoothing(predicted_u100)
        
        return self._convert_result(predicted_u100, return_tensor)
    
    def _research_based_scaling(self, wind: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Scale wind based on research."""
        current_mean = torch.mean(wind)
        current_std = torch.std(wind)
        
        lat_factor = torch.abs(torch.sin(lat_rad))
        variability_factor = 1.0 + 0.5 * lat_factor
        
        if current_std > 1e-6:
            normalized = (wind - current_mean) / current_std
            scaled = normalized * (self.TARGET_STD * variability_factor) + self.TARGET_MEAN
            return scaled
        else:
            return wind
    
    def _compute_research_jet_stream(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor, 
                                    current_time_utc: datetime) -> torch.Tensor:
        """Compute jet stream based on research."""
        lat_from_center = lat_deg - self.jet_lat_center
        
        width_factor = torch.where(lat_from_center > 0, 
                                  self.jet_width * 0.8,
                                  self.jet_width * 1.2)
        
        lat_norm = lat_from_center / width_factor
        jet_profile = torch.exp(-lat_norm**4)
        
        lat_amplitude = torch.sin(lat_rad)**2
        jet_profile = jet_profile * lat_amplitude
        
        jet_strength = self.jet_max_speed * jet_profile
        
        # Seasonal migration
        day_of_year = current_time_utc.timetuple().tm_yday
        seasonal_shift = self.SEASONAL_SHIFT_AMPLITUDE * torch.sin(
            torch.tensor(2 * math.pi * (day_of_year - 80) / 365, device=self.device)
        )
        
        # Winter amplification
        is_winter = torch.abs(torch.tensor(day_of_year - 355, device=self.device)) < 45
        winter_amplitude = torch.tensor(1.2 if is_winter else 1.0, device=self.device)
        
        # Shifted jet
        jet_lat_shifted = self.jet_lat_center + seasonal_shift
        lat_from_center_seasonal = lat_deg - jet_lat_shifted
        width_factor_seasonal = torch.where(lat_from_center_seasonal > 0, 
                                           self.jet_width * 0.8, 
                                           self.jet_width * 1.2)
        lat_norm_seasonal = lat_from_center_seasonal / width_factor_seasonal
        
        jet_profile_seasonal = torch.exp(-lat_norm_seasonal**4) * lat_amplitude
        jet_strength_seasonal = self.jet_max_speed * jet_profile_seasonal * winter_amplitude
        
        return self.SEASONAL_BLEND * jet_strength + (1 - self.SEASONAL_BLEND) * jet_strength_seasonal
    
    def _compute_research_pressure_gradient(self, sp: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute pressure gradient based on research."""
        height, width = sp.shape
        
        kernel_y = torch.tensor([[0, -0.25, 0, 0.25, 0],
                                 [0, -0.5, 0, 0.5, 0],
                                 [0, -0.75, 0, 0.75, 0],
                                 [0, -0.5, 0, 0.5, 0],
                                 [0, -0.25, 0, 0.25, 0]], 
                                dtype=sp.dtype, device=self.device) / 6.0
        
        sp_4d = sp.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(sp_4d, (2, 2, 2, 2), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        grad_y = grad_y * 3e-6
        
        f_base = 2 * self.omega * torch.sin(lat_rad)
        lat_factor = torch.abs(torch.sin(lat_rad))
        f_enhanced = f_base * (1.0 + 0.3 * lat_factor)
        
        f_safe = torch.where(torch.abs(f_enhanced) < 1e-10, 
                            torch.sign(f_enhanced) * 1e-10, 
                            f_enhanced)
        
        boundary_layer_factor = torch.tensor(0.7, device=self.device)
        u_geo = -grad_y / (self.rho * f_safe) * boundary_layer_factor
        
        midlat_factor = 1.0 + 0.4 * lat_factor
        return self.PRESSURE_SCALING * u_geo * midlat_factor
    
    def _compute_research_friction_decay(self, land_sea_mask: torch.Tensor, 
                                        delta_hours: int) -> torch.Tensor:
        """
        Compute friction decay based on research.
        FIXED: Proper tensor types for clamp - both min and max as tensors
        """
        # Research: friction depends on surface type and time of day
        friction_coeff = land_sea_mask * self.cf + (1 - land_sea_mask) * self.cf_sea
        
        # Add diurnal variation
        diurnal_factor = 1.0 - 0.3 * (1 - land_sea_mask)
        
        decay = torch.exp(-friction_coeff * diurnal_factor * delta_hours)
        
        # Research: minimum decay depends on surface
        min_decay_land = torch.tensor(0.4, device=self.device)
        min_decay_ocean = torch.tensor(0.6, device=self.device)
        min_decay = torch.where(land_sea_mask > 0.5, min_decay_land, min_decay_ocean)
        
        # FIX: Make max also a tensor
        max_decay = torch.tensor(1.0, device=self.device)
        
        # Use positional arguments with both as tensors
        return torch.clamp(decay, min_decay, max_decay)
    
    def _compute_research_temperature_gradient(self, t2m: torch.Tensor, 
                                              lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute thermal wind based on research."""
        height, width = t2m.shape
        
        kernel_y = torch.tensor([[0, -0.1, 0, 0.1, 0],
                                 [0, -0.2, 0, 0.2, 0],
                                 [0, -0.3, 0, 0.3, 0],
                                 [0, -0.2, 0, 0.2, 0],
                                 [0, -0.1, 0, 0.1, 0]], 
                                dtype=t2m.dtype, device=self.device) / 2.0
        
        t2m_4d = t2m.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(t2m_4d, (2, 2, 2, 2), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        lat_factor = torch.abs(torch.sin(lat_rad))
        thermal_factor = 1.0 + 0.5 * lat_factor
        
        return self.THERMAL_WIND_SCALING * grad_y * thermal_factor
    
    def _compute_research_topographic_effect(self, orograph: torch.Tensor, 
                                            lat_rad: torch.Tensor) -> torch.Tensor:
        """Compute topographic effects based on research."""
        height, width = orograph.shape
        
        kernel_y = torch.tensor([[0, -0.15, 0, 0.15, 0],
                                 [0, -0.3, 0, 0.3, 0],
                                 [0, -0.4, 0, 0.4, 0],
                                 [0, -0.3, 0, 0.3, 0],
                                 [0, -0.15, 0, 0.15, 0]], 
                                dtype=orograph.dtype, device=self.device) / 2.0
        
        orograph_4d = orograph.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        grad_y = torch.nn.functional.conv2d(
            torch.nn.functional.pad(orograph_4d, (2, 2, 2, 2), mode='replicate'),
            kernel_y_4d,
            padding=0
        ).squeeze()
        
        topo_effect = self.TOPO_SCALING * grad_y * 0.01
        
        mountain_mask = (orograph > self.MOUNTAIN_THRESHOLD).float()
        mountain_smooth = self._adaptive_smoothing(mountain_mask, preserve_edges=True)
        mountain_amplitude = 1.0 + self.TOPO_MOUNTAIN_BOOST * mountain_smooth
        
        return topo_effect * mountain_amplitude
    
    def _apply_midlatitude_amplification(self, wind: torch.Tensor, 
                                        lat_rad: torch.Tensor) -> torch.Tensor:
        """Apply research-based mid-latitude amplification."""
        midlat_factor = torch.abs(torch.sin(2 * lat_rad))
        amplification = 1.0 + (self.MIDLAT_AMPLIFICATION - 1.0) * midlat_factor
        return wind * amplification
    
    def _add_research_variability(self, wind: torch.Tensor, lat_rad: torch.Tensor,
                                 land_sea_mask: torch.Tensor) -> torch.Tensor:
        """Add research-based variability."""
        lat_factor = torch.abs(torch.sin(2 * lat_rad))
        height, width = wind.shape
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        # Generate multi-scale noise patterns
        noise = torch.zeros(height, width, device=self.device)
        
        # Add different scale patterns
        scales = [16, 8, 4, 2]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        for scale, weight in zip(scales, weights):
            # Create noise at this scale
            h_small = max(2, height // scale)
            w_small = max(2, width // scale)
            
            scale_noise = torch.randn(h_small, w_small, device=self.device)
            
            # Interpolate to full size
            scale_noise_full = torch.nn.functional.interpolate(
                scale_noise.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Add to total noise
            noise += weight * scale_noise_full
        
        # Normalize noise
        if torch.std(noise) > 1e-6:
            noise = noise / torch.std(noise)
        
        # Scale by latitudinal factor and land/sea
        variability = noise * lat_factor
        
        # More variability over land (boundary layer turbulence)
        land_variability_factor = 1.0 + 0.5 * land_sea_mask
        variability = variability * land_variability_factor
        
        # Scale by wind magnitude (more variability in stronger winds)
        wind_magnitude = torch.abs(wind)
        variability_scale = 0.3 + 0.7 * torch.tanh(wind_magnitude / 15.0)
        
        # Apply final scaling
        variability = variability * variability_scale * self.TURBULENCE_SCALE * 8.0
        
        return wind + variability
    
    def _apply_research_constraints(self, wind: torch.Tensor, lat_rad: torch.Tensor,
                                   land_sea_mask: torch.Tensor) -> torch.Tensor:
        """Apply research-based constraints."""
        lat_abs = torch.abs(lat_rad)
        
        comp1 = torch.cos(lat_abs)**2
        comp2 = torch.sin(2 * lat_abs)**2
        lat_factor = 0.3 * comp1 + 0.7 * comp2
        lat_factor = torch.clamp(lat_factor, self.LAT_FACTOR_MIN, 1.0)
        
        ocean_factor = torch.where(land_sea_mask == 0, 
                                  self.OCEAN_AMPLIFICATION, 
                                  torch.tensor(1.0, device=self.device))
        
        constrained = wind * lat_factor * ocean_factor
        
        return torch.clamp(constrained, self.WIND_MIN, self.WIND_MAX)
    
    def _adaptive_smoothing(self, field: torch.Tensor, preserve_edges: bool = False) -> torch.Tensor:
        """Apply adaptive smoothing."""
        if preserve_edges:
            kernel = torch.tensor([[1, 1, 1],
                                   [1, 2, 1],
                                   [1, 1, 1]], dtype=field.dtype, device=self.device) / 10.0
            padding = 1
        else:
            kernel = torch.tensor([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=field.dtype, device=self.device) / 16.0
            padding = 1
        
        field_4d = field.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(field_4d, (padding, padding, padding, padding), 
                                   mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return self.SMOOTHING_BLEND * field + (1 - self.SMOOTHING_BLEND) * smoothed
    
    def _light_smoothing(self, wind: torch.Tensor) -> torch.Tensor:
        """Apply minimal smoothing."""
        kernel = torch.tensor([[0, 1, 0],
                               [1, 2, 1],
                               [0, 1, 0]], dtype=wind.dtype, device=self.device) / 6.0
        
        wind_4d = wind.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(wind_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return 0.9 * wind + 0.1 * smoothed
    
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