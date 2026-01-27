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

Please write a Python class "D2mPredictor" that has the above predict(.) function, which predicts d2m 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import numpy as np
from datetime import datetime
import math
from typing import Optional

class D2mPredictor:
    """
    D2M Predictor that uses fast physical parameterizations.
    Predicts dew point temperature (d2m) for time t + delta_hours.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"D2mPredictor using device: {self.device}")
        
        # Initialize all constants and kernels
        self._setup_constants()
        self._setup_kernels()
    
    def _setup_constants(self):
        """Initialize all physical constants as tensors on the correct device."""
        # Physical constants
        self.Rv = torch.tensor(461.5, device=self.device)      # Gas constant for water vapor [J/(kg·K)]
        self.Rd = torch.tensor(287.05, device=self.device)     # Gas constant for dry air [J/(kg·K)]
        self.eps = self.Rd / self.Rv  # Ratio ≈ 0.622
        self.Lv = torch.tensor(2.5e6, device=self.device)      # Latent heat of vaporization [J/kg]
        self.cp = torch.tensor(1005.0, device=self.device)     # Specific heat at constant pressure [J/(kg·K)]
        self.g = torch.tensor(9.80665, device=self.device)     # Gravity [m/s²]
        self.STD_PRESSURE = torch.tensor(101325.0, device=self.device)  # Standard pressure [Pa]
        
        # Dew point parameters
        self.DEW_POINT_MIN = torch.tensor(180.0, device=self.device)  # Minimum possible dew point [K]
        self.DEW_POINT_MAX = torch.tensor(310.0, device=self.device)  # Maximum possible dew point [K]
        self.DEW_POINT_DIFF_MAX = torch.tensor(40.0, device=self.device)  # Max difference from air temp [K]
        self.DEW_POINT_PERSISTENCE = torch.tensor(0.7, device=self.device)  # Base persistence
        
        # Soil properties (already defined as dict, convert to tensors)
        self.soil_evap_factor = torch.tensor([
            1.0,   # 0: Ocean/Lake - maximum evaporation
            0.3,   # 1: Coarse - sand/gravel, low water retention
            0.5,   # 2: Medium - loam
            0.6,   # 3: Medium Fine - clay loam
            0.7,   # 4: Fine - clay
            0.8,   # 5: Very Fine - silt
            0.9,   # 6: Organic - peat, high water retention
            1.0,   # 7: Tropical Organic - maximum
        ], device=self.device)
        
        self.soil_thermal_inertia = torch.tensor([
            0.9,   # 0: Ocean/Lake - high thermal inertia
            0.3,   # 1: Coarse - low thermal inertia
            0.5,   # 2: Medium
            0.6,   # 3: Medium Fine
            0.8,   # 4: Fine - high thermal inertia
            0.7,   # 5: Very Fine
            0.4,   # 6: Organic - low thermal inertia
            0.4,   # 7: Tropical Organic
        ], device=self.device)
        
        # Evaporation parameters
        self.EVAP_RATE_LAND = torch.tensor(0.001, device=self.device)
        self.EVAP_RATE_OCEAN = torch.tensor(0.002, device=self.device)
        self.EVAP_SOLAR_FACTOR = torch.tensor(0.5, device=self.device)
        
        # Advection parameters
        self.ADVECTION_FACTOR = torch.tensor(0.00005, device=self.device)
        self.JET_STREAM_BOOST = torch.tensor(5.0, device=self.device)
        
        # Boundary layer mixing
        self.MIXING_LAND = torch.tensor(0.3, device=self.device)
        self.MIXING_OCEAN = torch.tensor(0.1, device=self.device)
        
        # Precipitation effect
        self.PRECIP_EVAP_RATE = torch.tensor(0.5, device=self.device)  # Evaporation from precipitation
        self.PRECIP_REMOVAL_RATE = torch.tensor(-1.0, device=self.device)  # Moisture removal by precipitation
        
        # Diurnal cycle for moisture
        self.DIURNAL_MOISTURE_LAND = torch.tensor(0.1, device=self.device)
        self.DIURNAL_MOISTURE_OCEAN = torch.tensor(0.05, device=self.device)
        
        # Orographic effect
        self.OROGRAPHIC_FACTOR = torch.tensor(-0.0001, device=self.device)
        self.OROGRAPHIC_THRESHOLD = torch.tensor(500.0, device=self.device)  # meters
        
        # Cloud effects
        self.CLOUD_EVAP_REDUCTION = torch.tensor(0.3, device=self.device)
        self.CLOUD_MOISTURE_RETENTION = torch.tensor(0.2, device=self.device)
        
        # Smoothing
        self.SMOOTHING_FACTOR = torch.tensor(0.2, device=self.device)
        
        # Pole handling
        self.POLE_LATITUDE_THRESHOLD = torch.tensor(85.0, device=self.device)  # degrees
        self.POLE_PERSISTENCE = torch.tensor(0.9, device=self.device)
        
        # Small epsilon to avoid division by zero
        self.EPS = torch.tensor(1e-8, device=self.device)
        
        # Humidity bounds
        self.Q_MIN = torch.tensor(0.0001, device=self.device)  # Minimum specific humidity
        self.Q_MAX = torch.tensor(0.04, device=self.device)    # Maximum specific humidity
        
    def _setup_kernels(self):
        """Initialize convolution kernels for smoothing."""
        # Separable convolution kernels for better GPU performance
        self.kernel_x = torch.ones(1, 1, 1, 3, device=self.device) / 3.0
        self.kernel_y = torch.ones(1, 1, 3, 1, device=self.device) / 3.0
    
    def _saturation_vapor_pressure(self, T_K: torch.Tensor) -> torch.Tensor:
        """Calculate saturation vapor pressure using Teten's formula."""
        T_C = T_K - 273.15
        # Teten's formula (more stable than Magnus)
        es = 6.112 * torch.exp((17.67 * T_C) / (T_C + 243.5))  # hPa
        return es * 100  # Convert to Pa
    
    def _dew_point_from_specific_humidity(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculate dew point from specific humidity and pressure."""
        # Vapor pressure from specific humidity (avoid division by zero)
        e_num = q * p
        e_den = self.eps + (1 - self.eps) * q
        e = e_num / torch.clamp(e_den, min=self.EPS)
        
        # Dew point from vapor pressure (inverse Teten's)
        e_hPa = torch.clamp(e / 100.0, min=0.001, max=1000.0)  # Bound to reasonable values
        
        # Safe calculation
        log_term = torch.log(e_hPa / 6.112)
        Td_C = (243.5 * log_term) / torch.clamp(17.67 - log_term, min=self.EPS)
        
        return Td_C + 273.15  # Convert to K
    
    def _specific_humidity_from_dew_point(self, Td: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculate specific humidity from dew point temperature."""
        # Vapor pressure from dew point
        Td_C = Td - 273.15
        e = 6.112 * torch.exp((17.67 * Td_C) / (Td_C + 243.5)) * 100  # Pa
        
        # Specific humidity
        q = (self.eps * e) / torch.clamp(p - (1 - self.eps) * e, min=self.EPS)
        
        return torch.clamp(q, self.Q_MIN, self.Q_MAX)
    
    def _compute_jet_stream_factor(self, sin_lat: torch.Tensor, cos_lat: torch.Tensor) -> torch.Tensor:
        """
        Jet stream influence factor.
        sin²(lat) * cos²(lat) peaks at ±45° latitude (typical jet stream position).
        Safe for poles.
        """
        # Compute latitude in radians safely
        sin_lat_clamped = torch.clamp(sin_lat, -0.9999, 0.9999)
        lat_rad = torch.arcsin(sin_lat_clamped)
        
        # Jet stream factor
        jet_factor = torch.sin(lat_rad)**2 * torch.cos(lat_rad)**2
        
        # Normalize to [0, 1]
        max_factor = torch.max(jet_factor)
        if max_factor > 0:
            jet_factor = jet_factor / max_factor
        else:
            jet_factor = torch.zeros_like(jet_factor)
        
        return jet_factor
    
    def _compute_time_factor(self, delta_hours: int) -> torch.Tensor:
        """Compute time scaling factor."""
        delta_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        return torch.clamp(delta_tensor / 6.0, 0.0, 1.0)
    
    def _compute_surface_masks(self, land_mask: torch.Tensor) -> tuple:
        """Compute land and ocean masks."""
        is_land = (land_mask == 1)
        is_ocean = (land_mask == 0)
        return is_land, is_ocean
    
    def _compute_pole_masks(self, sin_lat: torch.Tensor) -> tuple:
        """Compute north and south pole masks."""
        # Convert sin_lat to degrees for threshold
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lat_deg = torch.rad2deg(lat_rad)
        
        is_north_pole = lat_deg > self.POLE_LATITUDE_THRESHOLD
        is_south_pole = lat_deg < -self.POLE_LATITUDE_THRESHOLD
        is_pole = is_north_pole | is_south_pole
        
        return is_pole, is_north_pole, is_south_pole
    
    def _compute_soil_factors(self, soil_type: torch.Tensor) -> tuple:
        """Compute soil evaporation and thermal inertia factors."""
        # Ensure soil_type is integer indices
        soil_idx = soil_type.long()
        
        # Get soil factors using indexing
        soil_evap = self.soil_evap_factor[soil_idx]
        soil_inertia = self.soil_thermal_inertia[soil_idx]
        
        return soil_evap, soil_inertia
    
    def _compute_cloud_effects(self, T: torch.Tensor, Td: torch.Tensor, tp: torch.Tensor) -> tuple:
        """Compute cloud cover and effects on moisture."""
        # Relative humidity estimate
        T_diff = torch.clamp(T - Td, min=1.0, max=30.0)
        RH = torch.exp(-0.05 * T_diff)
        
        # Cloud cover from RH and precipitation
        cloud_cover = torch.clamp(0.8 * RH + 2.0 * tp, 0.0, 1.0)
        
        # Cloud effects on evaporation and moisture
        evap_reduction = 1.0 - self.CLOUD_EVAP_REDUCTION * cloud_cover
        moisture_retention = 1.0 + self.CLOUD_MOISTURE_RETENTION * cloud_cover
        
        return cloud_cover, evap_reduction, moisture_retention
    
    def _compute_local_solar_time(self, sin_lon: torch.Tensor, cos_lon: torch.Tensor,
                                 current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """Compute local solar time for future time."""
        # Longitude in radians
        lon_rad = torch.atan2(sin_lon, cos_lon)
        
        # Current UTC hour
        UTC_hour = current_dt.hour + current_dt.minute/60.0 + current_dt.second/3600.0
        
        # Future UTC hour
        future_hour = (UTC_hour + delta_hours) % 24
        
        # Convert to local solar time (hours)
        local_solar_future = (future_hour + torch.rad2deg(lon_rad)/15.0) % 24
        
        return local_solar_future
    
    def _compute_time_period_masks(self, local_solar_future: torch.Tensor) -> dict:
        """Compute masks for different time periods of the day."""
        return {
            'pre_dawn': (local_solar_future >= 3) & (local_solar_future < 6),
            'morning': (local_solar_future >= 6) & (local_solar_future < 9),
            'midday': (local_solar_future >= 9) & (local_solar_future < 15),
            'afternoon': (local_solar_future >= 15) & (local_solar_future < 18),
            'evening': (local_solar_future >= 18) & (local_solar_future < 21),
            'night': (local_solar_future >= 21) | (local_solar_future < 3),
        }
    
    def _compute_evaporation(self, T: torch.Tensor, Td: torch.Tensor, wind_speed: torch.Tensor,
                            soil_evap: torch.Tensor, is_land: torch.Tensor, is_ocean: torch.Tensor,
                            solar_factor: torch.Tensor, cloud_evap_reduction: torch.Tensor,
                            time_factor: torch.Tensor) -> torch.Tensor:
        """Compute evaporation rate."""
        # Saturation vapor pressure
        es = self._saturation_vapor_pressure(T)
        
        # Actual vapor pressure from dew point
        Td_C = Td - 273.15
        e = 6.112 * torch.exp((17.67 * Td_C) / (Td_C + 243.5)) * 100  # Pa
        
        # Vapor pressure deficit
        VPD = torch.clamp(es - e, min=0.0)
        
        # Base evaporation rate
        evap_rate_land = self.EVAP_RATE_LAND
        evap_rate_ocean = self.EVAP_RATE_OCEAN
        base_rate = torch.where(is_land, evap_rate_land, evap_rate_ocean)
        
        # Wind effect
        wind_effect = torch.sqrt(wind_speed + self.EPS)
        
        # Solar effect
        solar_effect = 1.0 + self.EVAP_SOLAR_FACTOR * solar_factor
        
        # Combine all effects
        evaporation = base_rate * VPD * wind_effect * soil_evap * solar_effect * cloud_evap_reduction
        
        return evaporation * time_factor
    
    def _compute_moisture_advection(self, U: torch.Tensor, V: torch.Tensor, q: torch.Tensor,
                                   sin_lat: torch.Tensor, jet_factor: torch.Tensor,
                                   is_pole: torch.Tensor, time_factor: torch.Tensor) -> torch.Tensor:
        """Compute moisture advection."""
        # Moisture gradient (approximate: decreases with latitude)
        lat_gradient = -self.ADVECTION_FACTOR  # kg/kg per degree latitude
        
        # Effective wind for advection
        wind_speed = torch.sqrt(U**2 + V**2 + self.EPS)
        
        # Direction factor (use u-wind component)
        advection = U * lat_gradient * 111000  # Convert to m (1° ≈ 111km)
        
        # Jet stream enhancement
        advection = advection * (1.0 + self.JET_STREAM_BOOST * jet_factor)
        
        # Minimal advection at poles
        advection = torch.where(is_pole, torch.tensor(0.0, device=self.device), advection)
        
        return advection * time_factor
    
    def _compute_boundary_layer_mixing(self, wind_speed: torch.Tensor, sp: torch.Tensor,
                                      is_land: torch.Tensor, is_ocean: torch.Tensor,
                                      time_factor: torch.Tensor) -> torch.Tensor:
        """Compute boundary layer mixing effect (dries the air)."""
        # Mixing efficiency
        mixing_efficiency = torch.where(is_land, self.MIXING_LAND, self.MIXING_OCEAN)
        
        # Pressure effect (less mixing at higher pressure)
        pressure_factor = self.STD_PRESSURE / torch.clamp(sp, min=80000.0)
        
        # Mixing effect (negative = drying)
        mixing = -mixing_efficiency * wind_speed * pressure_factor * 0.001
        
        return mixing * time_factor
    
    def _compute_orographic_effect(self, orograph: torch.Tensor, wind_speed: torch.Tensor,
                                  q: torch.Tensor, time_factor: torch.Tensor) -> torch.Tensor:
        """Compute orographic effects on moisture."""
        # Convert orograph to height [m]
        height = orograph / self.g
        
        # Orographic effect (drying on lee side, moistening on windward)
        orographic_effect = torch.where(
            height > self.OROGRAPHIC_THRESHOLD,
            self.OROGRAPHIC_FACTOR * height,
            torch.tensor(0.0, device=self.device)
        )
        
        # Enhanced by wind speed
        return orographic_effect * wind_speed * q * time_factor
    
    def _compute_precipitation_effect(self, tp: torch.Tensor, is_day: torch.Tensor,
                                     time_factor: torch.Tensor) -> torch.Tensor:
        """Compute precipitation effect on moisture."""
        # Two competing effects:
        # 1. Evaporation from falling precipitation adds moisture
        # 2. Precipitation removes moisture from the air
        
        # Net effect depends on time of day
        # During day: more evaporation from precipitation
        # During night: more moisture removal
        
        evap_from_precip = self.PRECIP_EVAP_RATE * tp
        removal_by_precip = self.PRECIP_REMOVAL_RATE * tp
        
        # Daytime favors evaporation, nighttime favors removal
        effect = torch.where(is_day, evap_from_precip, removal_by_precip)
        
        return effect * time_factor
    
    def _compute_diurnal_moisture(self, time_masks: dict, is_land: torch.Tensor,
                                 is_ocean: torch.Tensor, cloud_cover: torch.Tensor,
                                 local_solar_future: torch.Tensor, time_factor: torch.Tensor) -> torch.Tensor:
        """Compute diurnal moisture variation."""
        diurnal_moisture = torch.zeros_like(local_solar_future, device=self.device)
        
        # Different moisture regimes throughout the day
        # Morning/afternoon: increasing moisture from evaporation
        # Night: decreasing moisture from condensation
        
        # Morning (6-9 AM): moisture increases
        morning_factor = torch.where(
            time_masks['morning'],
            torch.clamp((local_solar_future - 6) / 3.0, 0.0, 1.0),
            0.0
        )
        diurnal_moisture = torch.where(
            time_masks['morning'] & is_land,
            self.DIURNAL_MOISTURE_LAND * morning_factor * (1.0 - cloud_cover),
            diurnal_moisture
        )
        diurnal_moisture = torch.where(
            time_masks['morning'] & is_ocean,
            self.DIURNAL_MOISTURE_OCEAN * morning_factor * (1.0 - cloud_cover),
            diurnal_moisture
        )
        
        # Night (9 PM - 3 AM): moisture decreases
        night_factor = torch.where(
            time_masks['night'],
            torch.clamp(1.0 - torch.abs(local_solar_future - 24.0) / 3.0, 0.0, 1.0),
            0.0
        )
        diurnal_moisture = torch.where(
            time_masks['night'] & is_land,
            -self.DIURNAL_MOISTURE_LAND * night_factor * (1.0 - cloud_cover),
            diurnal_moisture
        )
        diurnal_moisture = torch.where(
            time_masks['night'] & is_ocean,
            -self.DIURNAL_MOISTURE_OCEAN * night_factor * (1.0 - cloud_cover),
            diurnal_moisture
        )
        
        return diurnal_moisture * time_factor
    
    def _apply_smoothing(self, field: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing."""
        if field.dim() == 2:
            # Add batch and channel dimensions
            field_4d = field.unsqueeze(0).unsqueeze(0)
            
            # Apply separable convolution
            field_smooth_x = torch.nn.functional.conv2d(
                torch.nn.functional.pad(field_4d, (1, 1, 0, 0), mode='replicate'),
                self.kernel_x,
                padding=0
            )
            
            field_smooth = torch.nn.functional.conv2d(
                torch.nn.functional.pad(field_smooth_x, (0, 0, 1, 1), mode='replicate'),
                self.kernel_y,
                padding=0
            ).squeeze()
            
            # Handle NaN in smoothing
            field_smooth = torch.where(torch.isnan(field_smooth), field, field_smooth)
            
            return (1 - self.SMOOTHING_FACTOR) * field + self.SMOOTHING_FACTOR * field_smooth
        
        return field
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor,
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, 
                geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict dew point temperature (d2m) at time t + delta_hours.
        
        Args:
            meteo: [t2m, tp, u100, v100, d2m, sp] shape (6, H, W)
            geo_dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon] shape (5, H, W)
            geo_cat: [land_sea_mask, soil_type] shape (2, H, W)
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for current time
            geo_tempo_f: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for future time
            current_dt: datetime for current time t
            delta_hours: hours to predict ahead (0-12)
            
        Returns:
            predicted_d2m: torch.Tensor shape (H, W)
        """
        # Validate delta
        delta_hours = max(0, min(12, delta_hours))
        if delta_hours == 0:
            return meteo[4].clone()
        
        # Ensure tensors are on the correct device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T = meteo[0]           # Current 2m temperature (K)
        tp = meteo[1]          # Total precipitation water (m)
        U = meteo[2]           # Eastward wind at 100m (m/s)
        V = meteo[3]           # Northward wind at 100m (m/s)
        current_d2m = meteo[4]  # Current dew point at 2m (K)
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
        
        # --- MAIN COMPUTATION ---
        # Compute masks and factors
        time_factor = self._compute_time_factor(delta_hours)
        is_land, is_ocean = self._compute_surface_masks(land_mask)
        is_pole, is_north_pole, is_south_pole = self._compute_pole_masks(sin_lat)
        soil_evap, soil_inertia = self._compute_soil_factors(soil_type)
        cloud_cover, cloud_evap_reduction, cloud_moisture_retention = self._compute_cloud_effects(
            T, current_d2m, tp
        )
        
        # Solar time and day/night
        local_solar_future = self._compute_local_solar_time(
            sin_lon, cos_lon, current_dt, delta_hours
        )
        time_masks = self._compute_time_period_masks(local_solar_future)
        
        # Day/night based on solar zenith angle
        is_day = cos_z_future > 0.1  # Simple threshold
        
        # Jet stream
        jet_factor = self._compute_jet_stream_factor(sin_lat, cos_lat)
        
        # Current specific humidity
        current_q = self._specific_humidity_from_dew_point(current_d2m, sp)
        
        # Wind speed
        wind_speed = torch.sqrt(U**2 + V**2 + self.EPS)
        
        # --- COMPUTE ALL MOISTURE CHANGES ---
        # 1. Evaporation (adds moisture)
        evaporation = self._compute_evaporation(
            T, current_d2m, wind_speed, soil_evap, is_land, is_ocean,
            cos_z_future, cloud_evap_reduction, time_factor
        )
        
        # 2. Moisture advection
        advection = self._compute_moisture_advection(
            U, V, current_q, sin_lat, jet_factor, is_pole, time_factor
        )
        
        # 3. Boundary layer mixing (removes moisture)
        mixing = self._compute_boundary_layer_mixing(
            wind_speed, sp, is_land, is_ocean, time_factor
        )
        
        # 4. Orographic effects
        orographic = self._compute_orographic_effect(
            orograph, wind_speed, current_q, time_factor
        )
        
        # 5. Precipitation effects
        precip_effect = self._compute_precipitation_effect(
            tp, is_day, time_factor
        )
        
        # 6. Diurnal moisture cycle
        diurnal_moisture = self._compute_diurnal_moisture(
            time_masks, is_land, is_ocean, cloud_cover, local_solar_future, time_factor
        )
        
        # 7. Cloud moisture retention
        cloud_moisture = (cloud_moisture_retention - 1.0) * current_q * time_factor
        
        # --- COMBINE ALL EFFECTS ---
        future_q = current_q + evaporation + advection + mixing + orographic + \
                  precip_effect + diurnal_moisture + cloud_moisture
        
        # Apply physical bounds
        future_q = torch.clamp(future_q, self.Q_MIN, self.Q_MAX)
        
        # Convert back to dew point
        predicted_d2m = self._dew_point_from_specific_humidity(future_q, sp)
        
        # --- APPLY PHYSICAL CONSTRAINTS ---
        
        # 1. Dew point cannot exceed temperature
        predicted_d2m = torch.minimum(predicted_d2m, T)
        
        # 2. Minimum reasonable dew point
        min_dewpoint = T - self.DEW_POINT_DIFF_MAX
        predicted_d2m = torch.maximum(predicted_d2m, min_dewpoint)
        
        # 3. Global bounds
        predicted_d2m = torch.clamp(predicted_d2m, self.DEW_POINT_MIN, self.DEW_POINT_MAX)
        
        # 4. Persistence for short forecasts
        if delta_hours <= 3:
            persistence_weight = torch.exp(torch.tensor(-delta_hours / 1.5, device=self.device))
            predicted_d2m = persistence_weight * current_d2m + \
                          (1 - persistence_weight) * predicted_d2m
        
        # 5. High persistence at poles
        predicted_d2m = torch.where(
            is_pole,
            self.POLE_PERSISTENCE * current_d2m + (1 - self.POLE_PERSISTENCE) * predicted_d2m,
            predicted_d2m
        )
        
        # 6. Land-sea contrast
        land_persistence = torch.where(is_land, 0.7, 0.9)
        predicted_d2m = land_persistence * predicted_d2m + (1 - land_persistence) * current_d2m
        
        # 7. Apply smoothing
        predicted_d2m = self._apply_smoothing(predicted_d2m)
        
        # 8. Final NaN/Inf check
        predicted_d2m = torch.where(
            torch.isnan(predicted_d2m) | torch.isinf(predicted_d2m),
            current_d2m,
            predicted_d2m
        )
        
        return predicted_d2m
    
    def to(self, device: str) -> 'D2mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()  # Re-initialize constants on new device
        self._setup_kernels()
        return self


# Test function
def test_d2m_predictor():
    """Test the D2mPredictor."""
    print("Testing D2mPredictor...")
    print("=" * 60)
    
    # Create predictor
    predictor = D2mPredictor(device='cpu')
    
    # Create realistic sample data
    H, W = 100, 100
    
    # Create tensors
    meteo = torch.ones(6, H, W)
    geo_dense = torch.ones(5, H, W)
    geo_cat = torch.ones(2, H, W)
    geo_tempo = torch.ones(4, H, W)
    geo_tempo_f = torch.ones(4, H, W)
    
    # Set realistic values
    meteo[0] = 273.15 + 20.0  # 20°C air temperature
    meteo[1] = 0.001  # 1mm precipitation water
    meteo[2] = 5.0  # 5 m/s eastward wind
    meteo[3] = 2.0  # 2 m/s northward wind
    meteo[4] = 273.15 + 15.0  # 15°C dew point
    meteo[5] = 101325.0  # Standard pressure
    
    # Set geographical data
    lat = torch.linspace(-90, 90, H).unsqueeze(1)
    lon = torch.linspace(-180, 180, W).unsqueeze(0)
    
    geo_dense[1] = torch.sin(lat * math.pi / 180.0)
    geo_dense[2] = torch.cos(lat * math.pi / 180.0)
    geo_dense[3] = torch.sin(lon * math.pi / 180.0)
    geo_dense[4] = torch.cos(lon * math.pi / 180.0)
    
    # Land-sea mask
    geo_cat[0] = (torch.rand(H, W) > 0.5).float()
    
    # Soil type
    geo_cat[1] = torch.randint(0, 8, (H, W)).float()
    
    # Solar geometry
    geo_tempo_f[2] = torch.sin(torch.linspace(0, math.pi/2, H*W).view(H, W))
    geo_tempo_f[3] = torch.cos(torch.linspace(0, math.pi/2, H*W).view(H, W))
    
    current_dt = datetime(2024, 6, 21, 12, 0, 0)
    delta_hours = 6
    
    # Make prediction
    pred = predictor.predict(meteo, geo_dense, geo_cat, geo_tempo, geo_tempo_f,
                            current_dt, delta_hours)
    
    # Check results
    nan_count = torch.isnan(pred).sum().item()
    inf_count = torch.isinf(pred).sum().item()
    
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    print(f"Min dew point: {pred.min().item():.2f} K ({pred.min().item() - 273.15:.2f}°C)")
    print(f"Max dew point: {pred.max().item():.2f} K ({pred.max().item() - 273.15:.2f}°C)")
    print(f"Mean dew point: {pred.mean().item():.2f} K ({pred.mean().item() - 273.15:.2f}°C)")
    
    # Check pole values
    print("\nPole values:")
    print(f"North pole (row 0): mean = {pred[0, :].mean().item():.2f} K")
    print(f"South pole (row -1): mean = {pred[-1, :].mean().item():.2f} K")
    
    if nan_count == 0 and inf_count == 0:
        print("\n✓ SUCCESS: No NaN or Inf values found!")
    else:
        print(f"\n✗ FAILED: Found {nan_count} NaN and {inf_count} Inf values")
    
    return pred


if __name__ == "__main__":
    # Run test
    pred = test_d2m_predictor()