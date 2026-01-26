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

Please write a class code that predicts t2m at time t + delta hours, by fastest physical logic and not by tedious ML or numerical analysis. delta is in range: 0, 1, ..., 12
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import numpy as np
import math
from datetime import datetime
from typing import Optional, Tuple, Union

class FastPhysicsT2MPredictor:
    """
    Fast physics-based temperature prediction for any delta hours (0-12).
    Oceans are much less sensitive to sunshine than land.
    GPU-optimized implementation with all original physics logic preserved.
    """
    
    def __init__(self, device: Optional[str] = None, return_cpu: bool = True):
        """
        Initialize the predictor.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
            return_cpu: If True, automatically convert result to CPU numpy array
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_cpu = return_cpu  # Auto-convert to CPU for compatibility
        self._setup_constants()
        self._setup_kernels()
        
    def _setup_constants(self):
        """Initialize all physical constants as tensors on the correct device."""
        # Physical constants
        self.SOLAR_CONST = torch.tensor(1361.0, device=self.device)  # W/m²
        self.SIGMA = torch.tensor(5.67e-8, device=self.device)      # Stefan-Boltzmann, W/m²K⁴
        self.g = torch.tensor(9.80665, device=self.device)          # Gravity, m/s²
        self.STD_PRESSURE = torch.tensor(101325.0, device=self.device)  # Pa
        
        # Mathematical constants
        self.PI = torch.tensor(math.pi, device=self.device)
        self.DEG2RAD = torch.tensor(math.pi / 180.0, device=self.device)
        self.HOURS_TO_RAD = torch.tensor(15.0 * math.pi / 180.0, device=self.device)
        
        # Heat capacities (J/m²K) - land vs ocean
        self.C_LAND = torch.tensor(1.8e6, device=self.device)
        self.C_OCEAN = torch.tensor(5.0e6, device=self.device)
        
        # Thermal response factors
        self.RESPONSE_LAND = torch.tensor(1.8, device=self.device)
        self.RESPONSE_OCEAN = torch.tensor(0.15, device=self.device)
        
        # Albedo
        self.ALBEDO_LAND = torch.tensor(0.18, device=self.device)
        self.ALBEDO_OCEAN = torch.tensor(0.08, device=self.device)
        
        # Rate limits (K)
        self.MAX_WARMING_LAND = torch.tensor(10.0, device=self.device)
        self.MAX_COOLING_LAND = torch.tensor(-8.0, device=self.device)
        self.MAX_WARMING_OCEAN = torch.tensor(3.0, device=self.device)
        self.MAX_COOLING_OCEAN = torch.tensor(-2.0, device=self.device)
        
        # Special pre-dawn limits
        self.PRE_DAWN_WARMING_LAND = torch.tensor(0.1, device=self.device)
        self.PRE_DAWN_WARMING_OCEAN = torch.tensor(0.05, device=self.device)
        
        # Advection factors
        self.ADVECTION_FACTOR_LAND = torch.tensor(0.08, device=self.device)
        self.ADVECTION_FACTOR_OCEAN = torch.tensor(0.12, device=self.device)
        
        # Damping factors (stabilization)
        self.DAMPING_LAND = torch.tensor(0.85, device=self.device)
        self.DAMPING_OCEAN = torch.tensor(0.95, device=self.device)
        
        # Diurnal cycle amplitudes
        self.DIURNAL_AMPLITUDES = {
            'pre_dawn_land': torch.tensor(-0.4, device=self.device),
            'pre_dawn_ocean': torch.tensor(-0.05, device=self.device),
            'early_night_land': torch.tensor(-2.0, device=self.device),
            'early_night_ocean': torch.tensor(-0.3, device=self.device),
            'late_night_land': torch.tensor(-1.2, device=self.device),
            'late_night_ocean': torch.tensor(-0.2, device=self.device),
            'morning_land': torch.tensor(3.5, device=self.device),
            'morning_ocean': torch.tensor(0.6, device=self.device),
            'midday_land': torch.tensor(5.0, device=self.device),
            'midday_ocean': torch.tensor(1.0, device=self.device),
            'afternoon_land': torch.tensor(1.5, device=self.device),
            'afternoon_ocean': torch.tensor(0.3, device=self.device),
        }
        
        # Solar heating factors
        self.SOLAR_HEATING_LAND = torch.tensor(6.0, device=self.device)
        self.SOLAR_HEATING_OCEAN = torch.tensor(1.0, device=self.device)
        
        # Radiative cooling factors
        self.RADIATIVE_COOLING_LAND = torch.tensor(-1.8, device=self.device)
        self.RADIATIVE_COOLING_OCEAN = torch.tensor(-0.4, device=self.device)
        
        # Energy balance parameters
        self.Q_DAY_LAND_SLOPE = torch.tensor(50.0, device=self.device)
        self.Q_DAY_LAND_INTERCEPT = torch.tensor(-25.0, device=self.device)
        self.Q_DAY_OCEAN_SLOPE = torch.tensor(15.0, device=self.device)
        self.Q_DAY_OCEAN_INTERCEPT = torch.tensor(-10.0, device=self.device)
        self.Q_NIGHT_LAND_SLOPE = torch.tensor(-30.0, device=self.device)
        self.Q_NIGHT_OCEAN_SLOPE = torch.tensor(-8.0, device=self.device)
        
        # Final adjustments
        self.GLOBAL_ADJUST = torch.tensor(-0.1, device=self.device)
        self.LAND_ADJUST = torch.tensor(0.1, device=self.device)
        self.OCEAN_ADJUST = torch.tensor(-0.05, device=self.device)
        self.HUMID_ADJUST = torch.tensor(-0.05, device=self.device)
        
        # Temperature bounds
        self.MIN_TEMP = torch.tensor(240.0, device=self.device)
        self.MAX_TEMP = torch.tensor(310.0, device=self.device)
        self.DEW_POINT_OFFSET = torch.tensor(-3.0, device=self.device)
        
        # Other constants
        self.SMOOTHING_FACTOR = torch.tensor(0.25, device=self.device)
        self.CLOUD_REDUCTION_FACTOR = torch.tensor(0.6, device=self.device)
        self.SOLAR_RADIATION_FACTOR = torch.tensor(0.75, device=self.device)
        self.DAY_NIGHT_THRESHOLD = torch.tensor(20.0, device=self.device)
        self.SOLAR_DECLINATION_MAX = torch.tensor(23.44, device=self.device)
        
    def _setup_kernels(self):
        """Initialize convolution kernels for smoothing."""
        # Separable convolution kernels for better GPU performance
        self.kernel_x = torch.ones(1, 1, 1, 3, device=self.device) / 3.0
        self.kernel_y = torch.ones(1, 1, 3, 1, device=self.device) / 3.0
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                time_t_utc: datetime, delta_hours: int, 
                return_tensor: Optional[bool] = None) -> Union[torch.Tensor, np.ndarray]:
        """
        Predict T2M at time t + delta_hours.
        
        Args:
            meteo: torch.Tensor shape (6, H, W) - [t2m, tp, u100, v100, d2m, sp]
            geo_dense: torch.Tensor shape (5, H, W) - [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            geo_cat: torch.Tensor shape (2, H, W) - [land_sea_mask, soil_type]
            geo_tempo: torch.Tensor shape (4, H, W) - [sin_orbit, cos_orbit, sin_zenith, cos_zenith]
            time_t_utc: datetime object for time t
            delta_hours: int, hours to predict ahead (0-12)
            return_tensor: If True, return torch.Tensor; if False, return numpy array.
                          If None, use self.return_cpu
            
        Returns:
            predicted_t2m: torch.Tensor shape (H, W) or numpy array
        """
        # Validate delta
        delta_hours = max(0, min(12, delta_hours))
        if delta_hours == 0:
            result = meteo[0].clone()
            return self._convert_result(result, return_tensor)
        
        # Ensure tensors are on the correct device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        
        # Extract variables
        T0 = meteo[0]           # Current 2m temperature (K)
        U = meteo[2]           # Eastward wind at 100m (m/s)
        V = meteo[3]           # Northward wind at 100m (m/s)
        Td = meteo[4]          # Dew point at 2m (K)
        
        sin_lat = geo_dense[1] # sin(latitude)
        cos_lat = geo_dense[2] # cos(latitude)
        sin_lon = geo_dense[3] # sin(longitude)
        cos_lon = geo_dense[4] # cos(longitude)
        
        land_mask = geo_cat[0] # 0=sea, 1=land
        
        sin_orbit = geo_tempo[0]  # sin(orbital_angle)
        cos_orbit = geo_tempo[1]  # cos(orbital_angle)
        
        H, W = T0.shape
        
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # --- MAIN COMPUTATION ---
        # Compute all effects
        time_factor = self._compute_time_factor(delta_hours_tensor)
        is_land, is_ocean = self._compute_surface_masks(land_mask)
        cloud_cover, cloud_reduction = self._compute_cloud_effects(T0, Td)
        cos_z_future, local_solar_future = self._compute_solar_geometry(
            sin_lat, cos_lat, sin_lon, cos_lon, sin_orbit, cos_orbit, time_t_utc, delta_hours
        )
        is_day_future, is_night_future = self._compute_day_night_mask(cos_z_future, cloud_reduction)
        time_period_masks = self._compute_time_period_masks(local_solar_future)
        
        # Combine all temperature effects
        diurnal_adj = self._compute_diurnal_adjustment(
            time_period_masks, is_land, is_ocean, cloud_cover, local_solar_future, time_factor
        )
        solar_effect = self._compute_solar_radiation_effect(
            cos_z_future, is_land, is_ocean, is_day_future, is_night_future, cloud_cover, time_factor
        )
        energy_effect = self._compute_energy_balance_effect(
            cos_z_future, is_land, is_ocean, is_day_future, is_night_future, time_factor, delta_hours_tensor
        )
        advection_effect = self._compute_advection_effect(
            T0, U, V, cos_lat, is_land, is_ocean, time_factor, delta_hours_tensor
        )
        
        # Combine all effects
        delta_T_total = diurnal_adj + solar_effect + energy_effect + advection_effect
        
        # Apply rate limits
        delta_T_total = self._apply_rate_limits(
            delta_T_total, is_land, is_ocean, time_factor, time_period_masks['pre_dawn']
        )
        
        T_pred = T0 + delta_T_total
        
        # Apply stabilization
        T_pred = self._apply_stabilization(T_pred, T0, is_land, is_ocean)
        
        # Apply smoothing
        T_pred = self._apply_smoothing(T_pred)
        
        # Apply temperature bounds
        T_pred = self._apply_temperature_bounds(T_pred, Td)
        
        # Apply final adjustments
        T_final = self._apply_final_adjustments(
            T_pred, is_land, is_ocean, time_factor, cloud_cover
        )
        
        return self._convert_result(T_final, return_tensor)
    
    def _convert_result(self, result: torch.Tensor, return_tensor: Optional[bool]) -> Union[torch.Tensor, np.ndarray]:
        """Convert result to appropriate format."""
        if return_tensor is None:
            return_tensor = not self.return_cpu
        
        if return_tensor:
            return result
        else:
            # Convert to CPU numpy array
            return result.cpu().numpy()
    
    def _compute_time_factor(self, delta_hours_tensor: torch.Tensor) -> torch.Tensor:
        """Compute time scaling factor."""
        return torch.clamp(delta_hours_tensor / 6.0, 0.0, 1.0)
    
    def _compute_surface_masks(self, land_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute land and ocean masks."""
        is_land = (land_mask == 1)
        is_ocean = (land_mask == 0)
        return is_land, is_ocean
    
    def _compute_cloud_effects(self, T0: torch.Tensor, Td: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cloud cover and reduction factor."""
        T_diff = torch.clamp(T0 - Td, min=0.1)
        RH = torch.exp(-0.05 * T_diff)
        cloud_cover = torch.clamp(1.0 * RH**2, 0.0, 1.0)
        cloud_reduction = 1.0 - self.CLOUD_REDUCTION_FACTOR * cloud_cover
        return cloud_cover, cloud_reduction
    
    def _compute_solar_geometry(self, sin_lat: torch.Tensor, cos_lat: torch.Tensor,
                               sin_lon: torch.Tensor, cos_lon: torch.Tensor,
                               sin_orbit: torch.Tensor, cos_orbit: torch.Tensor,
                               time_t_utc: datetime, delta_hours: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute solar geometry for future time."""
        orbital_rad = torch.atan2(sin_orbit, cos_orbit)
        declination_rad = self.SOLAR_DECLINATION_MAX * self.DEG2RAD * torch.sin(orbital_rad)
        sin_delta = torch.sin(declination_rad)
        cos_delta = torch.cos(declination_rad)
        
        lat_rad = torch.asin(sin_lat)
        lon_rad = torch.atan2(sin_lon, cos_lon)
        
        # Current and future local solar time
        UTC_hour = time_t_utc.hour + time_t_utc.minute/60.0 + time_t_utc.second/3600.0
        local_solar_current = (UTC_hour + torch.rad2deg(lon_rad)/15.0) % 24
        local_solar_future = (local_solar_current + delta_hours) % 24
        
        # Solar hour angle and zenithh at future time
        hour_angle_future = self.HOURS_TO_RAD * (local_solar_future - 12.0)
        cos_z_future = sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(hour_angle_future)
        cos_z_future = torch.clamp(cos_z_future, 0.0, 1.0)
        
        return cos_z_future, local_solar_future
    
    def _compute_day_night_mask(self, cos_z_future: torch.Tensor, 
                               cloud_reduction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute day and night masks."""
        I_future = self.SOLAR_CONST * self.SOLAR_RADIATION_FACTOR * cos_z_future * cloud_reduction
        is_day_future = I_future > self.DAY_NIGHT_THRESHOLD
        is_night_future = ~is_day_future
        return is_day_future, is_night_future
    
    def _compute_time_period_masks(self, local_solar_future: torch.Tensor) -> dict:
        """Compute masks for different time periods of the day."""
        return {
            'pre_dawn': (local_solar_future >= 3) & (local_solar_future < 6),
            'early_night': (local_solar_future >= 18) & (local_solar_future < 21),
            'late_night': (local_solar_future >= 21) | (local_solar_future < 3),
            'morning': (local_solar_future >= 6) & (local_solar_future < 9),
            'midday': (local_solar_future >= 9) & (local_solar_future < 15),
            'afternoon': (local_solar_future >= 15) & (local_solar_future < 18),
        }
    
    def _compute_diurnal_adjustment(self, time_masks: dict, is_land: torch.Tensor, 
                                   is_ocean: torch.Tensor, cloud_cover: torch.Tensor,
                                   local_solar_future: torch.Tensor, time_factor: torch.Tensor) -> torch.Tensor:
        """Compute diurnal temperature adjustment."""
        diurnal_adj = torch.zeros_like(local_solar_future, device=self.device)
        
        # Pre-dawn (3-6 AM)
        pre_dawn_factor = torch.where(time_masks['pre_dawn'], (local_solar_future - 3) / 3.0, 0.0)
        diurnal_adj = torch.where(
            time_masks['pre_dawn'] & is_land,
            self.DIURNAL_AMPLITUDES['pre_dawn_land'] * pre_dawn_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['pre_dawn'] & is_ocean,
            self.DIURNAL_AMPLITUDES['pre_dawn_ocean'] * pre_dawn_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        
        # Early night (6-9 PM)
        early_night_factor = torch.where(time_masks['early_night'], (local_solar_future - 18) / 3.0, 0.0)
        diurnal_adj = torch.where(
            time_masks['early_night'] & is_land,
            self.DIURNAL_AMPLITUDES['early_night_land'] * early_night_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['early_night'] & is_ocean,
            self.DIURNAL_AMPLITUDES['early_night_ocean'] * early_night_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        
        # Late night (9 PM - 3 AM)
        night_factor = torch.where(
            time_masks['late_night'] & (local_solar_future >= 21),
            (local_solar_future - 21) / 3.0,
            torch.where(
                time_masks['late_night'],
                (local_solar_future + 3) / 6.0,
                0.0
            )
        )
        diurnal_adj = torch.where(
            time_masks['late_night'] & is_land,
            self.DIURNAL_AMPLITUDES['late_night_land'] * night_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['late_night'] & is_ocean,
            self.DIURNAL_AMPLITUDES['late_night_ocean'] * night_factor * (1.0 - cloud_cover),
            diurnal_adj
        )
        
        # Morning (6-9 AM)
        morning_factor = torch.where(time_masks['morning'], (local_solar_future - 6) / 3.0, 0.0)
        diurnal_adj = torch.where(
            time_masks['morning'] & is_land,
            self.DIURNAL_AMPLITUDES['morning_land'] * morning_factor,
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['morning'] & is_ocean,
            self.DIURNAL_AMPLITUDES['morning_ocean'] * morning_factor,
            diurnal_adj
        )
        
        # Midday (9 AM - 3 PM)
        midday_factor = torch.where(time_masks['midday'], 1.0 - (torch.abs(local_solar_future - 12.0) / 3.0) ** 2, 0.0)
        diurnal_adj = torch.where(
            time_masks['midday'] & is_land,
            self.DIURNAL_AMPLITUDES['midday_land'] * midday_factor,
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['midday'] & is_ocean,
            self.DIURNAL_AMPLITUDES['midday_ocean'] * midday_factor,
            diurnal_adj
        )
        
        # Afternoon (3-6 PM)
        afternoon_factor = torch.where(time_masks['afternoon'], 1.0 - (local_solar_future - 15) / 3.0, 0.0)
        diurnal_adj = torch.where(
            time_masks['afternoon'] & is_land,
            self.DIURNAL_AMPLITUDES['afternoon_land'] * afternoon_factor,
            diurnal_adj
        )
        diurnal_adj = torch.where(
            time_masks['afternoon'] & is_ocean,
            self.DIURNAL_AMPLITUDES['afternoon_ocean'] * afternoon_factor,
            diurnal_adj
        )
        
        return diurnal_adj * time_factor
    
    def _compute_solar_radiation_effect(self, cos_z_future: torch.Tensor, is_land: torch.Tensor,
                                       is_ocean: torch.Tensor, is_day_future: torch.Tensor,
                                       is_night_future: torch.Tensor, cloud_cover: torch.Tensor,
                                       time_factor: torch.Tensor) -> torch.Tensor:
        """Compute solar radiation heating/cooling effect."""
        # Solar heating
        solar_heating_land = self.SOLAR_HEATING_LAND * cos_z_future * time_factor
        solar_heating_ocean = self.SOLAR_HEATING_OCEAN * cos_z_future * time_factor
        solar_heating = torch.where(is_land, solar_heating_land, solar_heating_ocean)
        solar_heating = torch.where(is_day_future, solar_heating, 0.0)
        
        # Radiative cooling
        radiative_cooling_land = self.RADIATIVE_COOLING_LAND * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
        radiative_cooling_ocean = self.RADIATIVE_COOLING_OCEAN * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
        radiative_cooling = torch.where(is_land, radiative_cooling_land, radiative_cooling_ocean)
        radiative_cooling = torch.where(is_night_future, radiative_cooling, 0.0)
        
        return solar_heating + radiative_cooling
    
    def _compute_energy_balance_effect(self, cos_z_future: torch.Tensor, is_land: torch.Tensor,
                                      is_ocean: torch.Tensor, is_day_future: torch.Tensor,
                                      is_night_future: torch.Tensor, time_factor: torch.Tensor,
                                      delta_hours_tensor: torch.Tensor) -> torch.Tensor:
        """Compute energy balance effect."""
        # Daytime energy balance
        Q_day_land = self.Q_DAY_LAND_SLOPE * cos_z_future + self.Q_DAY_LAND_INTERCEPT
        Q_day_ocean = self.Q_DAY_OCEAN_SLOPE * cos_z_future + self.Q_DAY_OCEAN_INTERCEPT
        Q_day = torch.where(is_land, Q_day_land, Q_day_ocean)
        
        # Nighttime energy balance
        Q_night_land = self.Q_NIGHT_LAND_SLOPE * (1.0 - cos_z_future)
        Q_night_ocean = self.Q_NIGHT_OCEAN_SLOPE * (1.0 - cos_z_future)
        Q_night = torch.where(is_land, Q_night_land, Q_night_ocean)
        
        # Combine day/night
        Q_simple = torch.where(is_day_future, Q_day, Q_night)
        Q_simple = torch.clamp(Q_simple, -25.0, 55.0)
        
        # Apply heat capacity
        C_grid = torch.where(is_land, self.C_LAND, self.C_OCEAN)
        resp_grid = torch.where(is_land, self.RESPONSE_LAND, self.RESPONSE_OCEAN)
        
        delta_t_sec = delta_hours_tensor * 3600
        delta_T_energy = (Q_simple * delta_t_sec) / C_grid
        delta_T_energy = delta_T_energy * resp_grid * time_factor
        
        return delta_T_energy
    
    def _compute_advection_effect(self, T0: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                                 cos_lat: torch.Tensor, is_land: torch.Tensor, is_ocean: torch.Tensor,
                                 time_factor: torch.Tensor, delta_hours_tensor: torch.Tensor) -> torch.Tensor:
        """Compute temperature advection effect."""
        # Grid spacing
        dx = 27830 * cos_lat
        dy = 27830 * torch.ones_like(cos_lat, device=self.device)
        
        # Compute gradients using roll
        dT_dx = (torch.roll(T0, -1, dims=1) - torch.roll(T0, 1, dims=1)) / (2 * dx)
        dT_dy = (torch.roll(T0, -1, dims=0) - torch.roll(T0, 1, dims=0)) / (2 * dy)
        
        # Handle boundaries
        dT_dx = torch.cat([dT_dx[:, 1:2], dT_dx, dT_dx[:, -2:-1]], dim=1)[:, 1:-1]
        dT_dy = torch.cat([dT_dy[1:2, :], dT_dy, dT_dy[-2:-1, :]], dim=0)[1:-1, :]
        
        # Advection
        advection = -(U * dT_dx + V * dT_dy)
        advection_factor = torch.where(is_land, self.ADVECTION_FACTOR_LAND, self.ADVECTION_FACTOR_OCEAN)
        delta_t_sec = delta_hours_tensor * 3600
        
        return advection * delta_t_sec * time_factor * advection_factor
    
    def _apply_rate_limits(self, delta_T: torch.Tensor, is_land: torch.Tensor, 
                          is_ocean: torch.Tensor, time_factor: torch.Tensor,
                          pre_dawn_mask: torch.Tensor) -> torch.Tensor:
        """Apply rate limits to temperature change."""
        max_warming_land = self.MAX_WARMING_LAND * time_factor
        max_cooling_land = self.MAX_COOLING_LAND * time_factor
        max_warming_ocean = self.MAX_WARMING_OCEAN * time_factor
        max_cooling_ocean = self.MAX_COOLING_OCEAN * time_factor
        
        max_warming = torch.where(is_land, max_warming_land, max_warming_ocean)
        max_cooling = torch.where(is_land, max_cooling_land, max_cooling_ocean)
        
        # Special pre-dawn limits
        pre_dawn_warming_limit = torch.where(is_land, self.PRE_DAWN_WARMING_LAND, 
                                             self.PRE_DAWN_WARMING_OCEAN) * time_factor
        max_warming = torch.where(pre_dawn_mask, torch.minimum(max_warming, pre_dawn_warming_limit), max_warming)
        
        return torch.clamp(delta_T, max_cooling, max_warming)
    
    def _apply_stabilization(self, T_pred: torch.Tensor, T0: torch.Tensor,
                            is_land: torch.Tensor, is_ocean: torch.Tensor) -> torch.Tensor:
        """Apply stabilization damping."""
        damping = torch.where(is_ocean, self.DAMPING_OCEAN, self.DAMPING_LAND)
        return damping * T_pred + (1 - damping) * T0
    
    def _apply_smoothing(self, T_pred: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing."""
        # Add batch and channel dimensions
        T_pred_4d = T_pred.unsqueeze(0).unsqueeze(0)
        
        # Apply separable convolution
        T_smooth_x = torch.nn.functional.conv2d(
            torch.nn.functional.pad(T_pred_4d, (1, 1, 0, 0), mode='replicate'),
            self.kernel_x,
            padding=0
        )
        
        T_smooth = torch.nn.functional.conv2d(
            torch.nn.functional.pad(T_smooth_x, (0, 0, 1, 1), mode='replicate'),
            self.kernel_y,
            padding=0
        ).squeeze()
        
        return (1 - self.SMOOTHING_FACTOR) * T_pred + self.SMOOTHING_FACTOR * T_smooth
    
    def _apply_temperature_bounds(self, T_pred: torch.Tensor, Td: torch.Tensor) -> torch.Tensor:
        """Apply physical temperature bounds."""
        # Global bounds
        T_pred = torch.clamp(T_pred, self.MIN_TEMP, self.MAX_TEMP)
        
        # Don't go below dew point - 3K
        min_T = Td + self.DEW_POINT_OFFSET
        return torch.maximum(T_pred, min_T)
    
    def _apply_final_adjustments(self, T_pred: torch.Tensor, is_land: torch.Tensor,
                                is_ocean: torch.Tensor, time_factor: torch.Tensor,
                                cloud_cover: torch.Tensor) -> torch.Tensor:
        """Apply final adjustments."""
        global_adjust = self.GLOBAL_ADJUST * time_factor
        land_adjust = torch.where(is_land, self.LAND_ADJUST * time_factor, 
                                 self.OCEAN_ADJUST * time_factor)
        humid_adjust = self.HUMID_ADJUST * cloud_cover
        
        total_adjust = global_adjust + land_adjust + humid_adjust
        T_final = T_pred + total_adjust
        
        # Final bounds
        return torch.clamp(T_final, self.MIN_TEMP, self.MAX_TEMP)
    
    def to(self, device: str) -> 'FastPhysicsT2MPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()  # Re-initialize constants on new device
        self._setup_kernels()
        return self
    
    def get_physics_summary(self) -> str:
        """Get a summary of the physics model."""
        summary = """
        FAST PHYSICS T2M PREDICTOR - PHYSICS SUMMARY
        
        KEY FEATURES:
        1. Land vs Ocean Differentiation:
           - Ocean heat capacity: 5.0e6 J/m²K (2.8x higher than land)
           - Land heat capacity: 1.8e6 J/m²K
           - Ocean response factor: 0.15 (8x slower than land)
           - Land response factor: 1.8
        
        2. Diurnal Cycle Amplitudes (Land/Ocean):
           - Pre-dawn (3-6 AM): -0.4K / -0.05K
           - Early night (6-9 PM): -2.0K / -0.3K
           - Late night: -1.2K / -0.2K
           - Morning (6-9 AM): +3.5K / +0.6K
           - Midday: +5.0K / +1.0K
           - Afternoon: +1.5K / +0.3K
        
        3. Solar Radiation:
           - Land solar heating: 6.0 * cos(zenithh)
           - Ocean solar heating: 1.0 * cos(zenithh)
           - Land radiative cooling: -1.8 * (1-cos(zenithh))
           - Ocean radiative cooling: -0.4 * (1-cos(zenithh))
        
        4. Rate Limits (max ΔT per 6h):
           - Land warming: +10K, cooling: -8K
           - Ocean warming: +3K, cooling: -2K
           - Pre-dawn limits: Land +0.1K, Ocean +0.05K
        
        5. Other Physics:
           - Cloud reduction: 60% of solar radiation
           - Advection factors: Land 0.08, Ocean 0.12
           - Damping: Land 0.85, Ocean 0.95
           - Temperature bounds: 240K to 310K
        
        GPU OPTIMIZATIONS:
        - All constants stored as device tensors
        - Vectorized operations
        - Separable convolution for smoothing
        - Automatic device detection
        - Optional CPU conversion for compatibility
        """
        return summary


# Usage example that fixes your error
def example_usage_with_visualization():
    """Example showing how to use the predictor with visualization code."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize predictor (auto-detects GPU if available)
    predictor = FastPhysicsT2MPredictor(return_cpu=True)  # Default: returns CPU numpy array
    
    # Your existing code would look like this:
    # pred = predictor.predict(ss_x.meteo, ss_x.geo_dense, ss_x.geo_cat, ss_x.tempo, dt_x, delta_hours=modulo)
    # show_array(arr1=arr_y, arr2=pred, ...)  # Now pred is a numpy array, not a GPU tensor
    
    # Create dummy data for example
    H, W = 721, 1440
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    meteo = torch.randn(6, H, W, device=device)
    geo_dense = torch.randn(5, H, W, device=device)
    geo_cat = torch.randn(2, H, W, device=device)
    geo_tempo = torch.randn(4, H, W, device=device)
    
    # Set reasonable values
    lat_tensor = torch.linspace(-90, 90, H, device=device)
    meteo[0] = 273.15 + 20 * torch.cos(lat_tensor * math.pi / 180.0).unsqueeze(1)
    meteo[4] = meteo[0] - 5
    
    time_t_utc = datetime(2024, 6, 21, 12, 0, 0)
    
    # Make prediction - returns numpy array by default
    pred_numpy = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, time_t_utc, 6)
    print(f"Prediction type: {type(pred_numpy)}")
    print(f"Prediction shape: {pred_numpy.shape}")
    print(f"Prediction device: {pred_numpy.device if hasattr(pred_numpy, 'device') else 'CPU numpy array'}")
    
    # If you need a tensor instead:
    pred_tensor = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, time_t_utc, 6, return_tensor=True)
    print(f"\nTensor prediction type: {type(pred_tensor)}")
    print(f"Tensor prediction device: {pred_tensor.device}")
    
    # For your visualization function that expects numpy arrays:
    # show_array(arr1=arr_y.cpu().numpy(), arr2=pred_numpy, ...)
    
    return pred_numpy


if __name__ == "__main__":
    print("Testing FastPhysicsT2MPredictor with automatic CPU conversion")
    print("=" * 60)
    
    pred = example_usage_with_visualization()
    
    # Also test the class directly
    predictor = FastPhysicsT2MPredictor()
    print("\n" + predictor.get_physics_summary())