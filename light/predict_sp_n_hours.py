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
    dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).
, where sin_xxx means sin(xxx) and cos_xxx means cos(xxx)
We also have inputs for the given future time f. 
The current and the future shares all inputs, except that the future has no meteo_normal and its own geo_tempo called geo_tempo_f.

def predict(self, meteo_normal: torch.Tensor, dense: torch.Tensor, 
                cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor

Please write a Python class "SpPredictor" that has the above predict(.) function, which predicts sp 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import numpy as np
import math
from datetime import datetime
from typing import Optional
import torch.nn.functional as F

class SpPredictor:
    """
    Fast physics-based surface pressure (sp) predictor.
    Uses hydrostatic balance, thermal wind, and mass conservation principles.
    """
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize all physical constants."""
        # Physical constants
        self.g = torch.tensor(9.80665, device=self.device)        # Gravity (m/s²)
        self.R = torch.tensor(287.058, device=self.device)        # Gas constant for dry air (J/(kg·K))
        self.Rd = torch.tensor(287.0, device=self.device)         # Alternative R for calculations
        self.Rv = torch.tensor(461.5, device=self.device)         # Gas constant for water vapor
        self.omega = torch.tensor(7.292115e-5, device=self.device)  # Earth rotation rate (rad/s)
        self.a = torch.tensor(6371000.0, device=self.device)      # Earth radius (m)
        
        # Reference values
        self.P0 = torch.tensor(101325.0, device=self.device)      # Sea level pressure (Pa)
        self.T0 = torch.tensor(288.15, device=self.device)        # Sea level temperature (K)
        self.L = torch.tensor(-0.0065, device=self.device)        # Temperature lapse rate (K/m)
        
        # Pressure variation scales
        self.DIURNAL_PRESSURE_AMP = torch.tensor(200.0, device=self.device)  # Pa daily variation
        self.SYSTEM_PRESSURE_AMP = torch.tensor(5000.0, device=self.device)  # Pa weather system variation
        
        # Coriolis parameter scaling
        self.CORIOLIS_FACTOR = torch.tensor(1.0, device=self.device)
        
        # Thermal wind parameters
        self.THERMAL_WIND_FACTOR = torch.tensor(0.001, device=self.device)
        
        # Jet stream parameters
        self.JET_STREAM_PRESSURE_FACTOR = torch.tensor(0.003, device=self.device)
        
        # Advection and smoothing
        self.ADVECTION_FACTOR = torch.tensor(0.002, device=self.device)
        self.SMOOTHING = torch.tensor(0.15, device=self.device)
        
        # Region-specific parameters
        self.POLAR_PRESSURE_BIAS = torch.tensor(-2000.0, device=self.device)  # Polar lows
        self.TROPICAL_PRESSURE_BIAS = torch.tensor(1000.0, device=self.device)  # Subtropical highs
        
        # Elevation pressure correction
        self.ELEVATION_PRESSURE_FACTOR = torch.tensor(0.00012, device=self.device)  # Pa/m
        
    def predict(self, meteo_normal: torch.Tensor, dense: torch.Tensor,
                cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict sp (surface pressure) for time t + delta_hours.
        
        Args:
            meteo_normal: Current meteorological data [t2m, tp, u100, v100, d2m, sp]
            dense: Dense geographical features [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            cat: Categorical geographical features [land_sea_mask, soil_type]
            geo_tempo: Current temporal features [sin_orbit, cos_orbit, sin_zenith, cos_zenith]
            geo_tempo_f: Future temporal features for prediction time
            current_dt: Current datetime
            delta_hours: Hours to predict into future
            
        Returns:
            Predicted sp map (surface pressure in Pa)
        """
        
        # Limit prediction horizon
        delta_hours = max(0, min(12, delta_hours))
        if delta_hours == 0:
            return meteo_normal[5].clone().to(self.device)  # Return current sp
        
        # Move to device
        meteo_normal = meteo_normal.to(self.device)
        dense = dense.to(self.device)
        cat = cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract current variables
        T_current = meteo_normal[0]           # Current air temperature (t2m) in K
        precipitation = meteo_normal[1]       # Total precipitation (tp) in m of water
        U = meteo_normal[2]                   # U-wind at 100m (u100) in m/s
        V = meteo_normal[3]                   # V-wind at 100m (v100) in m/s
        Td = meteo_normal[4]                  # Dew point (d2m) in K
        sp_current = meteo_normal[5]          # Current surface pressure (sp) in Pa
        
        # Extract geographical features
        orograph = dense[0]        # Orography (m²/s²) - geopotential
        sin_lat = dense[1]         # sin(latitude)
        cos_lat = dense[2]         # cos(latitude)
        sin_lon = dense[3]         # sin(longitude)
        cos_lon = dense[4]         # cos(longitude)
        
        land_mask = cat[0]         # 0=sea, 1=land
        soil_type_idx = cat[1].long()  # Soil type index
        
        # Convert orography from geopotential to approximate height in meters
        # geopotential = g * height, so height = geopotential / g
        height = orograph / self.g  # Approximate height in meters
        
        # Get latitude in radians and degrees
        lat_rad = torch.asin(torch.clamp(sin_lat, -0.9999, 0.9999))
        lat_deg = torch.rad2deg(lat_rad)
        
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lon_deg = torch.rad2deg(lon_rad)
        
        # Coriolis parameter: f = 2Ω sin(latitude)
        coriolis = 2.0 * self.omega * sin_lat
        
        # Solar zenith angles
        cos_z_current = torch.clamp(geo_tempo[3], 0.0, 1.0)  # cos_zenith current
        cos_z_future = torch.clamp(geo_tempo_f[3], 0.0, 1.0) # cos_zenith future
        
        # --- IDENTIFY REGIONS ---
        abs_lat = torch.abs(lat_deg)
        
        # Polar regions
        is_polar = (abs_lat > 60.0)
        
        # Tropical regions
        is_tropical = (abs_lat < 23.5)
        
        # Subtropical high pressure belts
        is_subtropical = (abs_lat >= 23.5) & (abs_lat <= 35.0)
        
        # Mid-latitude storm tracks
        is_midlatitude = (abs_lat > 35.0) & (abs_lat <= 60.0)
        
        # Land-sea contrast
        is_land = (land_mask == 1)
        is_ocean = (land_mask == 0)
        
        # --- HYDROSTATIC PRESSURE CORRECTION FOR ELEVATION ---
        # Scale height: H = RT/g
        scale_height = self.R * T_current / self.g
        
        # Pressure decreases exponentially with height: P = P0 * exp(-z/H)
        elevation_factor = torch.exp(-height / scale_height)
        
        # Reference pressure at sea level (inverse of hydrostatic adjustment)
        P0_est = sp_current / elevation_factor
        
        # --- THERMAL EFFECTS ON PRESSURE ---
        # Warm air = lower density = tendency for lower surface pressure
        # Cold air = higher density = tendency for higher surface pressure
        T_anomaly = T_current - self.T0
        
        # Ideal gas law: P = ρRT, so ∂P/∂T = ρR
        # For fixed mass, warmer air expands, surface pressure decreases
        thermal_pressure_effect = -T_anomaly * 100.0  # ~100 Pa per K
        
        # Land warms/cools faster than ocean
        thermal_inertia_factor = torch.where(is_land, 1.5, 1.0)
        thermal_pressure_effect = thermal_pressure_effect * thermal_inertia_factor
        
        # --- DIURNAL PRESSURE VARIATION ---
        # Solar heating causes diurnal pressure cycle
        # Over land: afternoon pressure minimum due to thermal low
        # Over ocean: smaller diurnal variation
        diurnal_phase = torch.where(is_land, 1.0, 0.3)
        diurnal_change = (cos_z_future - cos_z_current) * diurnal_phase
        
        # Pressure tends to drop when sun is highest (thermal expansion)
        diurnal_pressure_effect = -diurnal_change * self.DIURNAL_PRESSURE_AMP
        
        # --- PRESSURE ADVECTION ---
        # Wind transports pressure systems
        pressure_advection = self._compute_pressure_advection(
            sp_current, U, V, lat_rad, delta_hours
        )
        
        # --- JET STREAM EFFECTS ---
        # Jet streams create pressure gradients aloft that influence surface pressure
        jet_effect = self._compute_jet_stream_pressure_effect(
            lat_rad, U, V, T_current, sp_current, delta_hours
        )
        
        # --- THERMAL WIND EFFECT ---
        # Temperature gradients create wind shear which affects pressure patterns
        thermal_wind_effect = self._compute_thermal_wind_effect(
            T_current, lat_rad, lon_rad, delta_hours
        )
        
        # --- CORIOLIS AND CENTRIFUGAL EFFECTS ---
        # Centrifugal force from wind curvature affects pressure
        curvature_effect = self._compute_curvature_effect(
            U, V, lat_rad, sp_current
        )
        
        # --- MOISTURE EFFECTS ---
        # Water vapor is lighter than dry air: moist air = lower surface pressure
        # Compute mixing ratio
        e = self._saturation_vapor_pressure(Td)  # Vapor pressure from dew point
        mixing_ratio = 0.622 * e / (sp_current - e)
        
        # Virtual temperature effect: Tv = T(1 + 0.61q)
        virtual_temp_factor = 1.0 + 0.61 * mixing_ratio
        T_virtual = T_current * virtual_temp_factor
        
        # Moist air is less dense: P ∝ ρT, so for same P, moist air is warmer
        # Actually, for same temperature, moist air has lower density
        moisture_pressure_effect = -(T_virtual - T_current) * 150.0  # ~150 Pa per K virtual temp excess
        
        # --- LAND-SEA CONTRAST ---
        # Land heats/cools faster, creating pressure differences
        land_sea_contrast = torch.zeros_like(lat_deg, device=self.device)
        
        # Daytime: land warmer than ocean -> lower pressure over land
        # Nighttime: land cooler than ocean -> higher pressure over land
        land_sea_temp_gradient = torch.where(is_land, 1.0, -1.0) * cos_z_current
        
        # Scale by diurnal cycle
        land_sea_effect = land_sea_temp_gradient * 300.0  # ~300 Pa contrast
        
        # --- SYNOPTIC SCALE PRESSURE TENDENCIES ---
        # Weather systems move and evolve
        synoptic_tendency = self._compute_synoptic_tendency(
            sp_current, U, V, lat_rad, delta_hours
        )
        
        # --- COMBINE ALL EFFECTS ---
        # Start with current sea level pressure estimate
        P0_pred = P0_est
        
        # Apply all effects
        P0_pred = P0_pred + thermal_pressure_effect
        P0_pred = P0_pred + diurnal_pressure_effect
        P0_pred = P0_pred + pressure_advection
        P0_pred = P0_pred + jet_effect
        P0_pred = P0_pred + thermal_wind_effect
        P0_pred = P0_pred + curvature_effect
        P0_pred = P0_pred + moisture_pressure_effect
        P0_pred = P0_pred + land_sea_effect
        P0_pred = P0_pred + synoptic_tendency
        
        # --- REGIONAL BIASES ---
        # Apply known climatological pressure patterns
        P0_pred = torch.where(is_polar, P0_pred + self.POLAR_PRESSURE_BIAS, P0_pred)
        P0_pred = torch.where(is_subtropical, P0_pred + self.TROPICAL_PRESSURE_BIAS, P0_pred)
        
        # --- CONVERT BACK TO SURFACE PRESSURE AT ACTUAL ELEVATION ---
        # Use predicted temperature for scale height
        T_pred_est = T_current + (T_current - Td) * 0.05  # Simple temperature prediction
        scale_height_pred = self.R * T_pred_est / self.g
        elevation_factor_pred = torch.exp(-height / scale_height_pred)
        
        sp_pred = P0_pred * elevation_factor_pred
        
        # --- APPLY PHYSICAL CONSTRAINTS ---
        # Realistic pressure bounds
        sp_pred = torch.clamp(sp_pred, 87000.0, 108000.0)  # Realistic Earth range
        
        # Rate limiting
        max_change = 500.0 * (delta_hours / 6.0)  # Pa per 6 hours
        delta_sp = sp_pred - sp_current
        delta_sp = torch.clamp(delta_sp, -max_change, max_change)
        sp_pred = sp_current + delta_sp
        
        # Smooth pressure fields (weather systems are smooth)
        sp_pred = self._apply_smoothing(sp_pred)
        
        # Ensure pressure gradients are physically consistent
        sp_pred = self._enforce_gradient_consistency(sp_pred, lat_rad, U, V)
        
        return sp_pred
    
    def _saturation_vapor_pressure(self, T: torch.Tensor) -> torch.Tensor:
        """Compute saturation vapor pressure using Tetens formula."""
        T_c = T - 273.15
        e_s = 611.0 * torch.exp(17.27 * T_c / (T_c + 237.3))
        return e_s
    
    def _compute_pressure_advection(self, P: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                                   lat_rad: torch.Tensor, delta_hours: int) -> torch.Tensor:
        """Compute pressure advection by winds."""
        # Pressure gradients
        P_left = torch.roll(P, shifts=-1, dims=1)
        P_right = torch.roll(P, shifts=1, dims=1)
        P_up = torch.roll(P, shifts=-1, dims=0)
        P_down = torch.roll(P, shifts=1, dims=0)
        
        dP_dx = (P_right - P_left) / 2.0
        dP_dy = (P_down - P_up) / 2.0
        
        # Avoid poles
        dP_dy[0, :] = 0
        dP_dy[-1, :] = 0
        
        # Advection: - (U * ∂P/∂x + V * ∂P/∂y)
        advection = -(U * dP_dx + V * dP_dy)
        
        # Scale by time and factor
        effect = advection * self.ADVECTION_FACTOR * delta_hours / 3600.0
        
        return effect
    
    def _compute_jet_stream_pressure_effect(self, lat_rad: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                                           T: torch.Tensor, P: torch.Tensor, delta_hours: int) -> torch.Tensor:
        """
        Compute jet stream effects on surface pressure.
        Jet streams create upper-level divergence/convergence that influences surface pressure.
        """
        # Jet stream location: sin²(lat) * cos²(lat) peaks at 45°
        jet_strength = (torch.sin(lat_rad)**2) * (torch.cos(lat_rad)**2)
        
        # Jet stream speed (approximate from 100m winds scaled up)
        wind_speed = torch.sqrt(U**2 + V**2)
        jet_speed = wind_speed * 2.0  # Jet stream is faster aloft
        
        # Upper-level divergence creates surface pressure falls
        # Convergence creates surface pressure rises
        # Use wind shear as proxy for divergence
        U_shear = torch.roll(U, shifts=-1, dims=0) - torch.roll(U, shifts=1, dims=0)
        V_shear = torch.roll(V, shifts=-1, dims=1) - torch.roll(V, shifts=1, dims=1)
        
        divergence = U_shear + V_shear
        
        # Jet stream regions enhance divergence effects
        jet_divergence = divergence * jet_strength
        
        # Surface pressure tendency from upper-level divergence
        # Divergence aloft -> surface pressure fall
        # Convergence aloft -> surface pressure rise
        pressure_tendency = -jet_divergence * jet_speed
        
        effect = pressure_tendency * self.JET_STREAM_PRESSURE_FACTOR * delta_hours
        
        return effect
    
    def _compute_thermal_wind_effect(self, T: torch.Tensor, lat_rad: torch.Tensor,
                                    lon_rad: torch.Tensor, delta_hours: int) -> torch.Tensor:
        """Compute thermal wind effect on pressure."""
        # Temperature gradients create wind shear (thermal wind)
        T_left = torch.roll(T, shifts=-1, dims=1)
        T_right = torch.roll(T, shifts=1, dims=1)
        T_up = torch.roll(T, shifts=-1, dims=0)
        T_down = torch.roll(T, shifts=1, dims=0)
        
        dT_dx = (T_right - T_left) / 2.0
        dT_dy = (T_down - T_up) / 2.0
        
        # Avoid poles
        dT_dy[0, :] = 0
        dT_dy[-1, :] = 0
        
        # Thermal wind: wind shear proportional to temperature gradient
        # Stronger temperature gradients -> stronger winds aloft -> stronger pressure gradients
        thermal_wind_x = -dT_dy * self.THERMAL_WIND_FACTOR
        thermal_wind_y = dT_dx * self.THERMAL_WIND_FACTOR
        
        # Magnitude of thermal wind effect
        thermal_wind_mag = torch.sqrt(thermal_wind_x**2 + thermal_wind_y**2)
        
        # Scale by latitude (stronger effect in mid-latitudes)
        lat_factor = torch.abs(torch.sin(lat_rad))
        
        effect = thermal_wind_mag * lat_factor * 100.0 * delta_hours / 6.0
        
        return effect
    
    def _compute_curvature_effect(self, U: torch.Tensor, V: torch.Tensor,
                                 lat_rad: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Compute centrifugal force effect from wind curvature."""
        wind_speed = torch.sqrt(U**2 + V**2)
        
        # Coriolis parameter
        f = 2.0 * self.omega * torch.sin(lat_rad)
        
        # Centripetal acceleration for curved flow
        # For cyclonic curvature (low pressure), centrifugal force adds to Coriolis
        # For anticyclonic curvature (high pressure), centrifugal force opposes Coriolis
        
        # Approximate curvature from wind gradients
        U_left = torch.roll(U, shifts=-1, dims=1)
        U_right = torch.roll(U, shifts=1, dims=1)
        V_up = torch.roll(V, shifts=-1, dims=0)
        V_down = torch.roll(V, shifts=1, dims=0)
        
        curvature_x = (U_right - U_left) / 2.0
        curvature_y = (V_down - V_up) / 2.0
        
        curvature = torch.sqrt(curvature_x**2 + curvature_y**2)
        
        # Centrifugal term: V²/R ≈ V * curvature
        centrifugal = wind_speed * curvature
        
        # Effect on pressure gradient
        # Positive curvature (cyclonic) -> lower pressure needed for balance
        # Negative curvature (anticyclonic) -> higher pressure needed for balance
        curvature_effect = -centrifugal * 0.1  # Scale factor
        
        return curvature_effect
    
    def _compute_synoptic_tendency(self, P: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                                  lat_rad: torch.Tensor, delta_hours: int) -> torch.Tensor:
        """Compute synoptic scale pressure tendency."""
        # Laplacian of pressure (curvature of pressure field)
        P_laplacian = self._compute_laplacian(P)
        
        # Pressure tendency proportional to -U·∇P (advection) and diffusion
        P_left = torch.roll(P, shifts=-1, dims=1)
        P_right = torch.roll(P, shifts=1, dims=1)
        P_up = torch.roll(P, shifts=-1, dims=0)
        P_down = torch.roll(P, shifts=1, dims=0)
        
        dP_dx = (P_right - P_left) / 2.0
        dP_dy = (P_down - P_up) / 2.0
        
        advection = -(U * dP_dx + V * dP_dy)
        
        # Diffusion term (smooths extremes)
        diffusion = 0.01 * P_laplacian
        
        # Combine with time scaling
        tendency = (advection + diffusion) * delta_hours / 6.0
        
        return tendency
    
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute discrete Laplacian of a field."""
        field_4d = field.unsqueeze(0).unsqueeze(0)
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=field.dtype, device=self.device).view(1, 1, 3, 3)
        
        laplacian = F.conv2d(field_4d, laplacian_kernel, padding=1)
        
        return laplacian.squeeze()
    
    def _apply_smoothing(self, field: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to field."""
        field_4d = field.unsqueeze(0).unsqueeze(0)
        smoothed = F.avg_pool2d(field_4d, kernel_size=3, stride=1, padding=1).squeeze()
        
        result = field * (1 - self.SMOOTHING) + smoothed * self.SMOOTHING
        return result
    
    def _enforce_gradient_consistency(self, P: torch.Tensor, lat_rad: torch.Tensor,
                                     U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Ensure pressure gradients are consistent with geostrophic wind."""
        # Compute pressure gradients
        P_left = torch.roll(P, shifts=-1, dims=1)
        P_right = torch.roll(P, shifts=1, dims=1)
        P_up = torch.roll(P, shifts=-1, dims=0)
        P_down = torch.roll(P, shifts=1, dims=0)
        
        dP_dx = (P_right - P_left) / 2.0
        dP_dy = (P_down - P_up) / 2.0
        
        # Avoid poles
        dP_dy[0, :] = 0
        dP_dy[-1, :] = 0
        
        # Geostrophic wind: fU_g = -(1/ρ)∂P/∂y, fV_g = (1/ρ)∂P/∂x
        # Use average density
        rho = 1.225  # kg/m³ approximate
        
        f = 2.0 * self.omega * torch.sin(lat_rad)
        
        # Avoid division by zero near equator
        f_safe = torch.where(torch.abs(f) < 1e-10, torch.sign(f) * 1e-10, f)
        
        U_geostrophic = - (1.0 / (rho * f_safe)) * dP_dy
        V_geostrophic = (1.0 / (rho * f_safe)) * dP_dx
        
        # Blend with observed wind
        U_blend = 0.7 * U + 0.3 * U_geostrophic
        V_blend = 0.7 * V + 0.3 * V_geostrophic
        
        # Recompute pressure gradients from blended wind
        f_abs = torch.abs(f_safe)
        dP_dx_adj = rho * f_safe * V_blend
        dP_dy_adj = -rho * f_safe * U_blend
        
        # Adjust pressure field to be consistent
        P_adj = P + (dP_dx_adj - dP_dx + dP_dy_adj - dP_dy) * 0.1
        
        return P_adj
    
    def to(self, device: str) -> 'SpPredictor':
        self.device = device
        self._setup_constants()
        return self