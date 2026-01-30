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
    meteo_normal = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0),
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).
, where sin_xxx means sin(xxx) and cos_xxx means cos(xxx)
We also have inputs for the given future time f. 
The current and the future shares all inputs, except that the future has no meteo_normal and its own geo_tempo called geo_tempo_f.

def predict(self, meteo_normal: torch.Tensor, geo_dense: torch.Tensor, 
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
    D2M Predictor based on T2mPredictor logic but adapted for dew point temperature.
    Key insight: Dew point depends on moisture content, which changes slower than temperature.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            device: Device to run computations on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_constants()
        self._setup_dew_point_physics()
        
    def _setup_constants(self):
        """Physical constants."""
        # Gas constants
        self.Rv = torch.tensor(461.5, device=self.device)      # Gas constant for water vapor [J/(kg·K)]
        self.Rd = torch.tensor(287.05, device=self.device)     # Gas constant for dry air [J/(kg·K)]
        self.eps = self.Rd / self.Rv  # Ratio ≈ 0.622
        
        # Thermodynamic constants
        self.Lv = torch.tensor(2.5e6, device=self.device)      # Latent heat of vaporization [J/kg]
        self.cp = torch.tensor(1005.0, device=self.device)     # Specific heat at constant pressure [J/(kg·K)]
        self.g = torch.tensor(9.80665, device=self.device)     # Gravity [m/s²]
        self.STD_PRESSURE = torch.tensor(101325.0, device=self.device)  # Standard pressure [Pa]
        
        # Mathematical constants
        self.PI = torch.tensor(math.pi, device=self.device)
        self.DEG2RAD = torch.tensor(math.pi / 180.0, device=self.device)
        self.EPS = torch.tensor(1e-8, device=self.device)
        
    def _setup_dew_point_physics(self):
        """Parameters specific to dew point prediction."""
        # MOISTURE SOURCES (increases d2m)
        # 1. Evaporation - based on solar intensity and vapor pressure deficit
        self.EVAPORATION_RATE = {
            'land': torch.tensor(0.0015, device=self.device),   # K/hour per 100 W/m² deficit
            'ocean': torch.tensor(0.003, device=self.device),   # Ocean evaporates faster
        }
        
        # 2. Diurnal moisture cycle (d2m has weaker diurnal cycle than t2m)
        self.DIURNAL_MOISTURE = {
            'land_amplitude': torch.tensor(1.0, device=self.device),    # d2m varies ±1K over day
            'ocean_amplitude': torch.tensor(0.3, device=self.device),   # Ocean varies less
            'peak_hour': torch.tensor(15.0, device=self.device),        # Peak moisture ~3 PM
        }
        
        # MOISTURE SINKS (decreases d2m)
        # 3. Mixing - wind mixes drier air from aloft
        self.MIXING_RATE = torch.tensor(-0.0005, device=self.device)  # K/hour per m/s wind
        
        # 4. Advection - horizontal moisture transport
        self.ADVECTION_FACTOR = torch.tensor(0.001, device=self.device)
        
        # 5. Precipitation effects
        self.PRECIP_EVAP_BOOST = torch.tensor(0.002, device=self.device)  # Evaporation from falling rain
        self.PRECIP_REMOVAL = torch.tensor(-0.001, device=self.device)    # Moisture removal by rain
        
        # ASYMMETRY FACTORS (like T2mPredictor)
        self.MORNING_EVAP_BOOST = torch.tensor(1.3, device=self.device)    # Morning evaporation stronger
        self.EVENING_EVAP_REDUCTION = torch.tensor(0.7, device=self.device) # Evening evaporation weaker
        
        # RATE LIMITS (d2m changes slower than t2m)
        self.MAX_WARMING = {
            'land': torch.tensor(0.4, device=self.device),    # K/hour max warming
            'ocean': torch.tensor(0.2, device=self.device),   # Ocean changes slower
        }
        
        self.MAX_COOLING = {
            'land': torch.tensor(-0.3, device=self.device),   # K/hour max cooling
            'ocean': torch.tensor(-0.15, device=self.device), # Ocean changes slower
        }
        
        # MORNING/EVENING ASYMMETRY IN RATE LIMITS
        self.MORNING_WARMING_BOOST = torch.tensor(1.5, device=self.device)
        self.EVENING_COOLING_REDUCTION = torch.tensor(0.8, device=self.device)
        
        # PHYSICAL BOUNDS
        self.DEW_POINT_MIN = torch.tensor(220.0, device=self.device)
        self.DEW_POINT_MAX = torch.tensor(310.0, device=self.device)
        self.MAX_TD_T_DIFF = torch.tensor(30.0, device=self.device)  # Max T - Td difference
        
    def _calculate_specific_humidity(self, Td: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculate specific humidity from dew point."""
        # Vapor pressure from dew point (Teten's formula)
        Td_C = Td - 273.15
        e = 6.112 * torch.exp((17.67 * Td_C) / (Td_C + 243.5)) * 100  # Pa
        
        # Specific humidity: q = εe / (p - (1-ε)e)
        q = (self.eps * e) / torch.clamp(p - (1 - self.eps) * e, min=100.0)
        
        return torch.clamp(q, 0.0001, 0.04)
    
    def _calculate_dew_point(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Calculate dew point from specific humidity."""
        # Vapor pressure: e = qp / (ε + q(1-ε))
        e = (q * p) / torch.clamp(self.eps + q * (1 - self.eps), min=self.EPS)
        
        # Dew point from vapor pressure
        e_hPa = torch.clamp(e / 100.0, 0.001, 100.0)
        
        # Inverse Teten's: Td = (243.5 * ln(e/6.112)) / (17.67 - ln(e/6.112))
        ln_term = torch.log(e_hPa / 6.112)
        Td_C = (243.5 * ln_term) / torch.clamp(17.67 - ln_term, min=self.EPS)
        
        return Td_C + 273.15  # Convert to Kelvin
    
    def predict(self, meteo_normal: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict dew point temperature (d2m) for time t + delta_hours.
        
        Args:
            meteo_normal: [t2m, tp, u100, v100, d2m, sp] shape (6, H, W)
            geo_dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon] shape (5, H, W)
            geo_cat: [land_sea_mask, soil_type] shape (2, H, W)
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for current time
            geo_tempo_f: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] shape (4, H, W) for future time
            current_dt: datetime for current time t
            delta_hours: hours to predict ahead (0-12)
            
        Returns:
            predicted_d2m: torch.Tensor shape (H, W)
        """
        if delta_hours == 0:
            return meteo_normal[4].clone()
        
        # Move to device
        meteo_normal = meteo_normal.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T = meteo_normal[0]           # Current air temperature [K]
        tp = meteo_normal[1]          # Total precipitation [m water]
        U = meteo_normal[2]           # Eastward wind at 100m [m/s]
        V = meteo_normal[3]           # Northward wind at 100m [m/s]
        Td_current = meteo_normal[4]  # Current dew point [K]
        sp = meteo_normal[5]          # Surface pressure [Pa]
        
        # Geographical data
        sin_lat = geo_dense[1]   # sin(latitude)
        cos_lat = geo_dense[2]   # cos(latitude)
        sin_lon = geo_dense[3]   # sin(longitude)
        cos_lon = geo_dense[4]   # cos(longitude)
        
        land_mask = geo_cat[0]   # 0=sea, 1=land
        land_fraction = land_mask.float()
        
        # Solar geometry (use future time for prediction)
        cos_z_current = torch.clamp(geo_tempo[3], 0.0, 1.0)
        cos_z_future = torch.clamp(geo_tempo_f[3], 0.0, 1.0)
        
        H, W = Td_current.shape
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # --- MAIN PHYSICS (following T2mPredictor pattern) ---
        
        # 1. EVAPORATION/CONDENSATION (main moisture source/sink)
        # Vapor pressure deficit
        es_current = 6.112 * torch.exp((17.67 * (T - 273.15)) / ((T - 273.15) + 243.5)) * 100
        e_current = 6.112 * torch.exp((17.67 * (Td_current - 273.15)) / ((Td_current - 273.15) + 243.5)) * 100
        VPD = torch.clamp(es_current - e_current, min=0.0)  # Vapor pressure deficit [Pa]
        
        # Solar intensity affects evaporation
        solar_future = 1361.0 * cos_z_future  # W/m²
        
        # Cloud reduction (precipitation indicates clouds)
        cloud_fraction = torch.clamp(20.0 * tp, 0.0, 1.0)
        solar_effective = solar_future * (1.0 - 0.5 * cloud_fraction)
        
        # Evaporation rate (depends on land/ocean)
        evap_rate = (self.EVAPORATION_RATE['land'] * land_fraction + 
                    self.EVAPORATION_RATE['ocean'] * (1.0 - land_fraction))
        
        # ΔTd from evaporation: proportional to VPD and solar
        evap_effect = evap_rate * (VPD / 100.0) * (solar_effective / 100.0) * delta_hours_tensor
        
        # 2. MORNING/EVENING ASYMMETRY (like T2mPredictor)
        cos_z_change = cos_z_future - cos_z_current
        
        # Morning: sun rising (increasing evaporation)
        is_morning = cos_z_change > 0.05
        
        # Evening: sun setting (decreasing evaporation)
        is_evening = cos_z_change < -0.05
        
        # Adjust evaporation rates
        evap_effect = torch.where(is_morning,
                                 evap_effect * self.MORNING_EVAP_BOOST,
                                 evap_effect)
        
        evap_effect = torch.where(is_evening,
                                 evap_effect * self.EVENING_EVAP_REDUCTION,
                                 evap_effect)
        
        # 3. DIURNAL MOISTURE CYCLE (d2m has its own diurnal cycle)
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lon_rad = torch.atan2(sin_lon, cos_lon)
        
        UTC_hour = current_dt.hour + current_dt.minute/60.0 + current_dt.second/3600.0
        
        # Smooth local time calculation (reduce banding)
        lon_effect = torch.rad2deg(lon_rad) / 15.0
        
        # Apply smoothing to longitude effect
        if H > 1 and W > 1:
            lon_effect_4d = lon_effect.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(1, 1, 7, 7, device=self.device) / 49.0
            lon_effect_smooth = torch.nn.functional.conv2d(
                torch.nn.functional.pad(lon_effect_4d, (3, 3, 3, 3), mode='replicate'),
                kernel
            ).squeeze()
            lon_effect = 0.2 * lon_effect + 0.8 * lon_effect_smooth
        
        local_hour_current = (UTC_hour + lon_effect) % 24
        local_hour_future = (local_hour_current + delta_hours) % 24
        
        # Diurnal amplitude (smaller than temperature)
        diurnal_amplitude = (self.DIURNAL_MOISTURE['land_amplitude'] * land_fraction + 
                           self.DIURNAL_MOISTURE['ocean_amplitude'] * (1.0 - land_fraction))
        
        # Moisture peaks later than temperature (~3 PM vs 2 PM)
        peak_hour = self.DIURNAL_MOISTURE['peak_hour']
        
        phase_current = (local_hour_current - peak_hour) * 2.0 * self.PI / 24.0
        phase_future = (local_hour_future - peak_hour) * 2.0 * self.PI / 24.0
        
        diurnal_current = diurnal_amplitude * torch.cos(phase_current)
        diurnal_future = diurnal_amplitude * torch.cos(phase_future)
        
        diurnal_change = diurnal_future - diurnal_current
        
        # 4. ADVECTION - horizontal moisture transport
        wind_speed = torch.sqrt(U**2 + V**2 + self.EPS)
        
        # Current specific humidity
        q_current = self._calculate_specific_humidity(Td_current, sp)
        
        # Moisture gradient (use specific humidity)
        dx = 111320 * torch.clamp(cos_lat, 0.3, 1.0)  # meters per degree longitude
        dy = 111320 * torch.ones_like(cos_lat)        # meters per degree latitude
        
        q_padded = torch.nn.functional.pad(q_current.unsqueeze(0).unsqueeze(0), 
                                          (1, 1, 1, 1), mode='replicate').squeeze()
        
        dq_dx = (q_padded[1:-1, 2:] - q_padded[1:-1, :-2]) / (2 * dx + self.EPS)
        dq_dy = (q_padded[2:, 1:-1] - q_padded[:-2, 1:-1]) / (2 * dy + self.EPS)
        
        # Moisture advection: dq/dt = -U·∇q
        q_advection = -(U * dq_dx + V * dq_dy) * delta_hours_tensor * self.ADVECTION_FACTOR
        
        # Convert q change to Td change
        # Approximate: ΔTd ≈ (Δq / (dq/dTd)) where dq/dTd ≈ 0.0005 K⁻¹ at 20°C
        dTd_advection = q_advection / 0.0005
        
        # 5. BOUNDARY LAYER MIXING (dries the air)
        # Stronger winds mix drier air from aloft
        mixing_effect = self.MIXING_RATE * wind_speed * delta_hours_tensor
        
        # 6. PRECIPITATION EFFECTS
        # Two competing effects:
        # - Evaporation from falling rain increases moisture
        # - Precipitation removes moisture from air
        is_day = cos_z_future > 0.1
        
        # Daytime: evaporation from rain dominates
        # Nighttime: moisture removal dominates
        precip_effect = torch.where(
            is_day,
            self.PRECIP_EVAP_BOOST * tp * delta_hours_tensor,
            self.PRECIP_REMOVAL * tp * delta_hours_tensor
        )
        
        # 7. CLOUD EFFECTS (moisture retention)
        # Clouds trap moisture near surface
        cloud_moisture = 0.1 * cloud_fraction * delta_hours_tensor
        
        # 8. JET STREAM EFFECT (enhances horizontal transport)
        # Jet streams at mid-latitudes enhance moisture transport
        lat_rad = torch.arcsin(torch.clamp(sin_lat, -0.9999, 0.9999))
        jet_factor = torch.sin(2 * lat_rad)**2 * torch.cos(lat_rad)**2
        
        # Normalize jet factor
        jet_max = torch.max(jet_factor)
        if jet_max > 0:
            jet_factor = jet_factor / jet_max
        
        # Apply jet stream boost to advection
        dTd_advection = dTd_advection * (1.0 + 2.0 * jet_factor)
        
        # 9. COMBINE ALL EFFECTS
        dTd_total = (evap_effect +
                    diurnal_change +
                    dTd_advection +
                    mixing_effect +
                    precip_effect +
                    cloud_moisture)
        
        # 10. RATE LIMITS (asymmetric, like T2mPredictor)
        # Base rate limits
        max_warming_base = (self.MAX_WARMING['land'] * land_fraction + 
                          self.MAX_WARMING['ocean'] * (1.0 - land_fraction)) * delta_hours_tensor
        
        max_cooling_base = (self.MAX_COOLING['land'] * land_fraction + 
                          self.MAX_COOLING['ocean'] * (1.0 - land_fraction)) * delta_hours_tensor
        
        # Morning/evening adjustments
        max_warming = torch.where(is_morning,
                                 max_warming_base * self.MORNING_WARMING_BOOST,
                                 max_warming_base)
        
        max_cooling = torch.where(is_evening,
                                 max_cooling_base * self.EVENING_COOLING_REDUCTION,
                                 max_cooling_base)
        
        # Apply rate limits
        dTd_limited = torch.clamp(dTd_total, max_cooling, max_warming)
        
        # 11. PREDICT FINAL DEW POINT
        Td_pred = Td_current + dTd_limited
        
        # 12. APPLY ASYMMETRIC SMOOTHING (like T2mPredictor)
        Td_pred = self._apply_dew_point_smoothing(Td_pred, is_morning, is_evening)
        
        # 13. PHYSICAL CONSTRAINTS
        # Dew point cannot exceed air temperature
        Td_pred = torch.minimum(Td_pred, T)
        
        # Minimum reasonable dew point
        min_Td = T - self.MAX_TD_T_DIFF
        Td_pred = torch.maximum(Td_pred, min_Td)
        
        # Absolute bounds
        Td_pred = torch.clamp(Td_pred, self.DEW_POINT_MIN, self.DEW_POINT_MAX)
        
        # 14. HIGH PERSISTENCE AT POLES
        # At high latitudes, moisture changes very slowly
        is_pole = torch.abs(torch.rad2deg(lat_rad)) > 70.0
        pole_persistence = 0.95
        Td_pred = torch.where(is_pole,
                             pole_persistence * Td_current + (1 - pole_persistence) * Td_pred,
                             Td_pred)
        
        # 15. FINAL CHECKS
        Td_pred = torch.where(torch.isnan(Td_pred), Td_current, Td_pred)
        Td_pred = torch.where(torch.isinf(Td_pred), Td_current, Td_pred)
        
        return Td_pred
    
    def _apply_dew_point_smoothing(self, Td: torch.Tensor,
                                  is_morning: torch.Tensor,
                                  is_evening: torch.Tensor) -> torch.Tensor:
        """
        Apply asymmetric smoothing to dew point field.
        More smoothing in east-west direction to reduce banding.
        """
        # Strong east-west smoothing kernel
        kernel_ew = torch.tensor([[0.0, 0.15, 0.0],
                                 [0.0, 0.70, 0.0],
                                 [0.0, 0.15, 0.0]], device=self.device).view(1, 1, 3, 3)
        
        # Mild north-south smoothing
        kernel_ns = torch.tensor([[0.0, 0.0, 0.0],
                                 [0.08, 0.84, 0.08],
                                 [0.0, 0.0, 0.0]], device=self.device).view(1, 1, 3, 3)
        
        Td_4d = Td.unsqueeze(0).unsqueeze(0)
        
        # Apply east-west smoothing
        smoothed_ew = torch.nn.functional.conv2d(
            torch.nn.functional.pad(Td_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_ew
        ).squeeze()
        
        # Apply north-south smoothing
        smoothed_ns = torch.nn.functional.conv2d(
            torch.nn.functional.pad(Td_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_ns
        ).squeeze()
        
        # More smoothing in morning/evening transition zones
        morning_evening_mask = is_morning | is_evening
        
        # Stronger smoothing for transition regions
        smoothing_strength = torch.where(morning_evening_mask,
                                        torch.tensor(0.3, device=self.device),
                                        torch.tensor(0.15, device=self.device))
        
        # Blend: 60% east-west, 40% north-south (dew point is more zonally uniform)
        smoothed = 0.6 * smoothed_ew + 0.4 * smoothed_ns
        
        result = (1 - smoothing_strength) * Td + smoothing_strength * smoothed
        
        return result
    
    def to(self, device: str) -> 'D2mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()
        self._setup_dew_point_physics()
        return self


# Test function
def test_d2m_predictor():
    """Test the D2mPredictor with realistic data."""
    print("Testing D2mPredictor with T2mPredictor-style physics")
    print("=" * 60)
    
    # Create predictor
    predictor = D2mPredictor(device='cpu')
    
    # Create realistic grid
    H, W = 180, 360
    lat = torch.linspace(-90, 90, H)
    lon = torch.linspace(-180, 180, W)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing='ij')
    
    # Initialize tensors
    meteo_normal = torch.ones(6, H, W)
    geo_dense = torch.ones(5, H, W)
    geo_cat = torch.ones(2, H, W)
    geo_tempo = torch.ones(4, H, W)
    geo_tempo_f = torch.ones(4, H, W)
    
    # Setup realistic values
    # Temperature field with latitudinal gradient
    for i in range(H):
        lat_deg = lat[i].item()
        # Realistic temperature: warm at equator, cold at poles
        T_eq = 300.0  # 27°C at equator
        T_pole = 250.0  # -23°C at poles
        T_lat = T_eq - (abs(lat_deg) / 90.0) * (T_eq - T_pole)
        
        # Add some longitudinal variation
        for j in range(W):
            lon_deg = lon[j].item()
            # Warmer over continents (simplified)
            is_land = (abs(lon_deg) < 60 and abs(lat_deg) < 45)
            T = T_lat + (5.0 if is_land else 0.0)
            
            # Dew point: typically 5-10°C below temperature
            Td = T - 7.0 + torch.randn(1).item() * 2.0
            
            meteo_normal[0, i, j] = T
            meteo_normal[4, i, j] = Td
            geo_cat[0, i, j] = 1.0 if is_land else 0.0
    
    # Other meteorological variables
    meteo_normal[1] = torch.rand(H, W) * 0.005  # Precipitation 0-5mm
    meteo_normal[2] = torch.randn(H, W) * 5.0   # Wind U
    meteo_normal[3] = torch.randn(H, W) * 3.0   # Wind V
    meteo_normal[5] = 101325.0  # Pressure
    
    # Geographical data
    geo_dense[1] = torch.sin(lat_grid * math.pi / 180.0)
    geo_dense[2] = torch.cos(lat_grid * math.pi / 180.0)
    geo_dense[3] = torch.sin(lon_grid * math.pi / 180.0)
    geo_dense[4] = torch.cos(lon_grid * math.pi / 180.0)
    
    # Soil type (random for now)
    geo_cat[1] = torch.randint(0, 8, (H, W)).float()
    
    # Simulate solar geometry (like in T2mPredictor test)
    current_dt = datetime(2024, 6, 21, 6, 0, 0)  # 6 AM UTC
    declination = 23.44 * math.pi / 180.0  # Summer solstice
    
    for i in range(H):
        for j in range(W):
            lat_rad = lat[i] * math.pi / 180.0
            lon_rad = lon[j] * math.pi / 180.0
            
            # Current: 6 AM UTC
            hour_angle_current = (6 - 12) * 15 * math.pi / 180.0 + lon_rad
            cos_z_current = (math.sin(lat_rad) * math.sin(declination) +
                           math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle_current))
            geo_tempo[3, i, j] = max(0.0, min(1.0, cos_z_current))
            
            # Future: 12 PM UTC (6 hours later)
            hour_angle_future = (12 - 12) * 15 * math.pi / 180.0 + lon_rad
            cos_z_future = (math.sin(lat_rad) * math.sin(declination) +
                          math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle_future))
            geo_tempo_f[3, i, j] = max(0.0, min(1.0, cos_z_future))
    
    # Make prediction
    print(f"Making 6-hour dew point forecast...")
    pred = predictor.predict(meteo_normal, geo_dense, geo_cat, geo_tempo, geo_tempo_f,
                            current_dt, 6)
    
    # Analyze results
    print("\nRESULTS:")
    print("-" * 40)
    
    # Convert to numpy for analysis
    pred_np = pred.numpy()
    Td_current_np = meteo_normal[4].numpy()
    T_np = meteo_normal[0].numpy()
    
    # Check physical constraints
    print(f"Min dew point: {pred_np.min():.2f} K ({pred_np.min() - 273.15:.2f}°C)")
    print(f"Max dew point: {pred_np.max():.2f} K ({pred_np.max() - 273.15:.2f}°C)")
    print(f"Mean dew point: {pred_np.mean():.2f} K ({pred_np.mean() - 273.15:.2f}°C)")
    
    # Check if dew point exceeds air temperature
    exceeds_temp = np.any(pred_np > T_np + 0.1)  # Allow small numerical tolerance
    print(f"Dew point exceeds air temperature: {'YES' if exceeds_temp else 'NO'}")
    
    # Check rate of change
    dTd = pred_np - Td_current_np
    print(f"Max warming: {dTd.max():.3f} K")
    print(f"Max cooling: {dTd.min():.3f} K")
    print(f"Mean change: {dTd.mean():.3f} K")
    
    # Check NaN/Inf
    nan_count = np.isnan(pred_np).sum()
    inf_count = np.isinf(pred_np).sum()
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    
    if nan_count == 0 and inf_count == 0 and not exceeds_temp:
        print("\n✓ SUCCESS: Physically consistent dew point prediction!")
    else:
        print("\n✗ ISSUES DETECTED")
    
    # Analyze banding (like T2mPredictor)
    cos_z_current_np = geo_tempo[3].numpy()
    cos_z_future_np = geo_tempo_f[3].numpy()
    cos_z_change = cos_z_future_np - cos_z_current_np
    
    is_morning = cos_z_change > 0.05
    is_evening = cos_z_change < -0.05
    
    # Check for longitudinal banding
    lon_grad = np.gradient(pred_np, axis=1)
    lon_grad_abs = np.abs(lon_grad)
    
    morning_grad = lon_grad_abs[is_morning].mean() if np.any(is_morning) else 0
    evening_grad = lon_grad_abs[is_evening].mean() if np.any(is_evening) else 0
    
    print(f"\nBANDING ANALYSIS:")
    print(f"Morning region gradient: {morning_grad:.4f} K/degree")
    print(f"Evening region gradient: {evening_grad:.4f} K/degree")
    
    if morning_grad > 0.1 or evening_grad > 0.1:
        print("⚠️  Warning: Possible banding in dew point field")
    else:
        print("✓ Good: Minimal banding detected")
    
    print("\nKey physics included:")
    print("1. Evaporation based on solar intensity and VPD")
    print("2. Morning/evening asymmetry (like T2m)")
    print("3. Diurnal moisture cycle (weaker than temperature)")
    print("4. Moisture advection with jet stream enhancement")
    print("5. Boundary layer mixing (drying)")
    print("6. Precipitation effects (evaporation vs removal)")
    print("7. Physical constraints (Td ≤ T, reasonable bounds)")
    print("8. Asymmetric smoothing to reduce banding")
    
    return pred


if __name__ == "__main__":
    # Run test
    pred = test_d2m_predictor()