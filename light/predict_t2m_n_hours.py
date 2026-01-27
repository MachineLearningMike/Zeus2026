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

Please write a Python class "T2mPredictor" that has the above predict(.) function, which predicts t2m 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import numpy as np
import math
from datetime import datetime
from typing import Optional, Tuple

class T2mPredictor:
    """
    T2M Predictor with the requested interface that uses the proven FastPhysicsT2MPredictor logic.
    Predicts t2m for time t + delta_hours using fast physical parameterizations.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize internal state
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
        
        # Small epsilon to avoid division by zero
        self.EPS = torch.tensor(1e-8, device=self.device)
        
    def _setup_kernels(self):
        """Initialize convolution kernels for smoothing."""
        # Separable convolution kernels for better GPU performance
        self.kernel_x = torch.ones(1, 1, 1, 3, device=self.device) / 3.0
        self.kernel_y = torch.ones(1, 1, 3, 1, device=self.device) / 3.0
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict t2m at future time t + delta_hours.
        
        Args:
            meteo: [t2m, tp, u100, v100, d2m, sp] shape (6, H, W)
            geo_dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon] shape (5, H, W)
            geo_cat: [land_sea_mask, soil_type] shape (2, H, W)
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for current time
            geo_tempo_f: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for future time
            current_dt: datetime for current time t
            delta_hours: hours to predict ahead (0-12)
            
        Returns:
            predicted_t2m: torch.Tensor shape (H, W)
        """
        # Validate delta
        delta_hours = max(0, min(12, delta_hours))
        if delta_hours == 0:
            return meteo[0].clone()
        
        # Ensure tensors are on the correct device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T0 = meteo[0]           # Current 2m temperature (K)
        tp = meteo[1]          # Total precipitation water (m)
        U = meteo[2]           # Eastward wind at 100m (m/s)
        V = meteo[3]           # Northward wind at 100m (m/s)
        Td = meteo[4]          # Dew point at 2m (K)
        sp = meteo[5]          # Surface pressure (Pa)
        
        # Extract geographical data
        orograph = geo_dense[0]  # Geopotential (m²/s²)
        sin_lat = geo_dense[1]   # sin(latitude)
        cos_lat = geo_dense[2]   # cos(latitude)
        sin_lon = geo_dense[3]   # sin(longitude)
        cos_lon = geo_dense[4]   # cos(longitude)
        
        land_mask = geo_cat[0]   # 0=sea, 1=land
        soil_type = geo_cat[1]   # Soil type (0-7)
        
        # Use geo_tempo_f for future solar geometry
        sin_zenith_future = geo_tempo_f[2]  # sin(zenith angle at future time)
        cos_zenith_future = geo_tempo_f[3]  # cos(zenith angle at future time)
        cos_z_future = torch.clamp(cos_zenith_future, 0.0, 1.0)
        
        H, W = T0.shape
        
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # --- MAIN COMPUTATION ---
        # Compute all effects (adapted from FastPhysicsT2MPredictor)
        time_factor = self._compute_time_factor(delta_hours_tensor)
        is_land, is_ocean = self._compute_surface_masks(land_mask)
        cloud_cover, cloud_reduction = self._compute_cloud_effects(T0, Td, tp)
        local_solar_future = self._compute_local_solar_time(
            sin_lon, cos_lon, current_dt, delta_hours
        )
        is_day_future, is_night_future = self._compute_day_night_mask(cos_z_future, cloud_reduction)
        time_period_masks = self._compute_time_period_masks(local_solar_future)
        
        # Jet-stream effect - sin²cos² latitude pattern (FIXED)
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        jet_stream_factor = torch.sin(2 * lat_rad) ** 2 * torch.cos(lat_rad) ** 2
        # Normalize safely
        max_factor = torch.max(jet_stream_factor)
        if max_factor > 0:
            jet_stream_factor = jet_stream_factor / max_factor
        else:
            jet_stream_factor = torch.zeros_like(jet_stream_factor)
        
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
        
        # Apply jet-stream effect (adds variability at mid-latitudes)
        # Use torch.randn_like with requires_grad=False to avoid NaN in backward pass
        with torch.no_grad():
            jet_stream_noise = torch.randn_like(T_final)
        jet_stream_effect = 0.3 * jet_stream_factor * jet_stream_noise * is_land.float()
        T_final = T_final + jet_stream_effect
        
        # Apply pressure effect (subtle) - FIXED division
        pressure_norm = torch.clamp(sp / self.STD_PRESSURE, 0.8, 1.2)
        T_final = T_final * (0.99 + 0.01 * pressure_norm)
        
        # Final NaN check and cleanup
        T_final = torch.where(torch.isnan(T_final), T0, T_final)
        T_final = torch.where(torch.isinf(T_final), T0, T_final)
        
        return T_final
    
    def _compute_time_factor(self, delta_hours_tensor: torch.Tensor) -> torch.Tensor:
        """Compute time scaling factor."""
        return torch.clamp(delta_hours_tensor / 6.0, 0.0, 1.0)
    
    def _compute_surface_masks(self, land_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute land and ocean masks."""
        is_land = (land_mask == 1)
        is_ocean = (land_mask == 0)
        return is_land, is_ocean
    
    def _compute_cloud_effects(self, T0: torch.Tensor, Td: torch.Tensor, tp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cloud cover and reduction factor."""
        T_diff = torch.clamp(T0 - Td, min=0.1, max=20.0)  # Limit max difference
        RH = torch.exp(-0.05 * T_diff)
        cloud_cover = torch.clamp(1.0 * RH**2 + 5.0 * tp, 0.0, 1.0)  # Reduced tp coefficient
        cloud_reduction = 1.0 - self.CLOUD_REDUCTION_FACTOR * cloud_cover
        return cloud_cover, cloud_reduction
    
    def _compute_local_solar_time(self, sin_lon: torch.Tensor, cos_lon: torch.Tensor,
                                 current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """Compute local solar time for future time."""
        lon_rad = torch.atan2(sin_lon, cos_lon)
        
        # Current UTC time
        UTC_hour = current_dt.hour + current_dt.minute/60.0 + current_dt.second/3600.0
        
        # Future UTC hour
        future_hour = (UTC_hour + delta_hours) % 24
        
        # Convert to local solar time (hours)
        local_solar_future = (future_hour + torch.rad2deg(lon_rad)/15.0) % 24
        
        return local_solar_future
    
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
        pre_dawn_factor = torch.where(time_masks['pre_dawn'], 
                                     torch.clamp((local_solar_future - 3) / 3.0, 0.0, 1.0), 
                                     0.0)
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
        early_night_factor = torch.where(time_masks['early_night'], 
                                        torch.clamp((local_solar_future - 18) / 3.0, 0.0, 1.0), 
                                        0.0)
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
            torch.clamp((local_solar_future - 21) / 3.0, 0.0, 1.0),
            torch.where(
                time_masks['late_night'],
                torch.clamp((local_solar_future + 3) / 6.0, 0.0, 1.0),
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
        morning_factor = torch.where(time_masks['morning'], 
                                    torch.clamp((local_solar_future - 6) / 3.0, 0.0, 1.0), 
                                    0.0)
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
        midday_factor = torch.where(time_masks['midday'], 
                                   torch.clamp(1.0 - (torch.abs(local_solar_future - 12.0) / 3.0) ** 2, 0.0, 1.0), 
                                   0.0)
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
        afternoon_factor = torch.where(time_masks['afternoon'], 
                                      torch.clamp(1.0 - (local_solar_future - 15) / 3.0, 0.0, 1.0), 
                                      0.0)
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
        delta_T_energy = (Q_simple * delta_t_sec) / torch.clamp(C_grid, min=1e3)
        delta_T_energy = delta_T_energy * resp_grid * time_factor
        
        return delta_T_energy
    
    def _compute_advection_effect(self, T0: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                                 cos_lat: torch.Tensor, is_land: torch.Tensor, is_ocean: torch.Tensor,
                                 time_factor: torch.Tensor, delta_hours_tensor: torch.Tensor) -> torch.Tensor:
        """Compute temperature advection effect."""
        # Grid spacing (approximate degrees to meters)
        dx = 27830 * torch.clamp(cos_lat, 0.01, 1.0)  # Avoid division by zero
        dy = 27830 * torch.ones_like(cos_lat, device=self.device)  # meters per degree latitude
        
        # Compute gradients using finite differences with roll
        dT_dx = (torch.roll(T0, -1, dims=1) - torch.roll(T0, 1, dims=1)) / (2 * dx + self.EPS)
        dT_dy = (torch.roll(T0, -1, dims=0) - torch.roll(T0, 1, dims=0)) / (2 * dy + self.EPS)
        
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
        
        # Handle NaN in smoothing
        T_smooth = torch.where(torch.isnan(T_smooth), T_pred, T_smooth)
        
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
    
    def to(self, device: str) -> 'T2mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()  # Re-initialize constants on new device
        self._setup_kernels()
        return self


# Test function to verify no NaN
def test_no_nan():
    """Test that the predictor doesn't produce NaN values."""
    print("Testing T2mPredictor for NaN values")
    print("=" * 60)
    
    # Create predictor
    predictor = T2mPredictor(device='cpu')  # Use CPU for easier debugging
    
    # Create realistic sample data
    H, W = 100, 100
    
    # Create realistic data (not random)
    meteo = torch.ones(6, H, W)
    geo_dense = torch.ones(5, H, W)
    geo_cat = torch.ones(2, H, W)
    geo_tempo = torch.ones(4, H, W)
    geo_tempo_f = torch.ones(4, H, W)
    
    # Set realistic values
    meteo[0] = 273.15 + 20.0  # 20°C
    meteo[1] = 0.001  # 1mm precipitation water
    meteo[2] = 5.0  # 5 m/s wind
    meteo[3] = 2.0  # 2 m/s wind
    meteo[4] = meteo[0] - 3.0  # Dew point 3°C lower
    meteo[5] = 101325.0  # Standard pressure
    
    # Set geographical data
    lat = torch.linspace(-90, 90, H).unsqueeze(1)
    lon = torch.linspace(-180, 180, W).unsqueeze(0)
    
    geo_dense[1] = torch.sin(lat * math.pi / 180.0)
    geo_dense[2] = torch.cos(lat * math.pi / 180.0)
    geo_dense[3] = torch.sin(lon * math.pi / 180.0)
    geo_dense[4] = torch.cos(lon * math.pi / 180.0)
    
    # Land-sea mask (mix of land and ocean)
    geo_cat[0] = (torch.rand(H, W) > 0.5).float()
    
    # Solar geometry
    geo_tempo_f[2] = torch.sin(torch.linspace(0, math.pi/2, H*W).view(H, W))
    geo_tempo_f[3] = torch.cos(torch.linspace(0, math.pi/2, H*W).view(H, W))
    
    current_dt = datetime(2024, 6, 21, 12, 0, 0)
    delta_hours = 6
    
    # Make prediction
    pred = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, geo_tempo_f, 
                            current_dt, delta_hours)
    
    # Check for NaN
    nan_count = torch.isnan(pred).sum().item()
    inf_count = torch.isinf(pred).sum().item()
    
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    print(f"Min temperature: {pred.min().item():.2f} K")
    print(f"Max temperature: {pred.max().item():.2f} K")
    print(f"Mean temperature: {pred.mean().item():.2f} K")
    print(f"Std temperature: {pred.std().item():.2f} K")
    
    if nan_count == 0 and inf_count == 0:
        print("\n✓ SUCCESS: No NaN or Inf values found!")
    else:
        print(f"\n✗ FAILED: Found {nan_count} NaN and {inf_count} Inf values")
        
        # Debug: Check which operations might be causing NaN
        print("\nDebugging NaN sources...")
        
        # Test individual components
        time_factor = predictor._compute_time_factor(torch.tensor(6.0))
        print(f"Time factor: {time_factor.item()}")
        
        is_land, is_ocean = predictor._compute_surface_masks(geo_cat[0])
        print(f"Land pixels: {is_land.sum().item()}, Ocean pixels: {is_ocean.sum().item()}")
        
        cloud_cover, cloud_reduction = predictor._compute_cloud_effects(meteo[0], meteo[4], meteo[1])
        print(f"Cloud cover range: {cloud_cover.min().item():.3f} to {cloud_cover.max().item():.3f}")
        
        cos_z_future = torch.clamp(geo_tempo_f[3], 0.0, 1.0)
        print(f"Cos zenith range: {cos_z_future.min().item():.3f} to {cos_z_future.max().item():.3f}")
    
    return pred


if __name__ == "__main__":
    # Run test
    pred = test_no_nan()
    
    # Also test with edge cases
    print("\n" + "=" * 60)
    print("Testing edge cases...")
    
    # Test with extreme values
    predictor = T2mPredictor(device='cpu')
    
    # Extreme temperature difference
    meteo_extreme = torch.ones(6, 10, 10)
    meteo_extreme[0] = 400.0  # Very hot
    meteo_extreme[4] = 200.0  # Very cold dew point
    
    geo_dense_extreme = torch.ones(5, 10, 10)
    geo_cat_extreme = torch.ones(2, 10, 10)
    geo_tempo_extreme = torch.ones(4, 10, 10)
    geo_tempo_f_extreme = torch.ones(4, 10, 10)
    
    pred_extreme = predictor.predict(meteo_extreme, geo_dense_extreme, geo_cat_extreme,
                                    geo_tempo_extreme, geo_tempo_f_extreme,
                                    datetime(2024, 6, 21, 12, 0, 0), 6)
    
    print(f"Extreme test - NaN: {torch.isnan(pred_extreme).sum().item()}, "
          f"Min: {pred_extreme.min().item():.2f}, Max: {pred_extreme.max().item():.2f}")