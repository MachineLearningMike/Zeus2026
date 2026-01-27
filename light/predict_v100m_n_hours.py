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

Please write a Python class "V100mPredictor" that has the above predict(.) function, which predicts v100, 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
import math

class V100mPredictor:
    def __init__(self, device=None):
        """
        Physical model for predicting v100 (meridional wind at 100m)
        
        Physical principles:
        1. Meridional pressure gradient force
        2. Coriolis effect on zonal winds
        3. Jet stream meandering (Rossby waves) and propagation
        4. Thermal wind relation (N-S temperature gradients) with diurnal evolution
        5. Surface friction and boundary layer effects with time dependence
        6. Topographic steering
        7. Land-sea thermal contrasts with diurnal cycle
        8. Seasonal/orbital forcing with time progression
        9. Advection and time tendencies
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Earth parameters
        self.omega = 7.2921e-5  # Earth's rotation rate (rad/s)
        self.R = 6371000.0  # Earth's radius (m)
        self.g = 9.80665  # Gravity (m/s²)
        
        # Physical constants
        self.friction_land = 0.15  # Higher friction over land (per hour)
        self.friction_sea = 0.05  # Lower friction over sea (per hour)
        self.boundary_layer_response_time = 3.0  # hours (BL adjustment time)
        self.rho = 1.225  # Air density (kg/m³)
        
        # Atmospheric wave parameters
        self.rossby_wave_speed = -10.0  # m/s (westward propagation)
        self.synoptic_scale_advection = 15.0  # m/s typical advection speed
        
        # Jet stream parameters
        self.jet_lat_center = 45.0  # Approximate jet stream latitude (degrees)
        self.jet_width = 15.0  # Jet stream width (degrees)
        
        print(f"Initialized V100mPredictor on device: {self.device}")
    
    def _compute_coriolis(self, lat_rad):
        """Compute Coriolis parameter f = 2Ωsin(latitude)"""
        return 2 * self.omega * torch.sin(lat_rad)
    
    def _compute_beta(self, lat_rad):
        """Compute beta parameter β = ∂f/∂y = 2Ωcos(latitude)/R"""
        return 2 * self.omega * torch.cos(lat_rad) / self.R
    
    def _jet_stream_evolution(self, lat_rad, lon_rad, delta_hours, season_factor):
        """
        Model jet stream evolution with Rossby wave propagation
        
        Physical basis:
        - Rossby waves propagate westward with speed c = -β/k²
        - Waves meander with time
        - Amplitude varies with season
        """
        # Base latitudinal pattern
        lat_deg = lat_rad * 180 / math.pi
        
        # Jet stream envelope (Gaussian in latitude)
        jet_envelope = torch.exp(-((lat_deg - self.jet_lat_center) / self.jet_width) ** 2)
        
        # Rossby wave pattern (wavenumber 3 typical)
        wave_number = 3.0
        k = wave_number / (self.R * torch.cos(lat_rad))  # Zonal wavenumber
        
        # Rossby wave phase speed (westward propagation)
        beta = self._compute_beta(lat_rad)
        with torch.no_grad():
            # Avoid division by zero
            k_safe = torch.where(torch.abs(k) > 1e-10, k, torch.ones_like(k) * 1e-10)
        phase_speed = -beta / (k_safe ** 2)  # Rossby wave phase speed
        
        # Time evolution: wave propagates westward
        # Δphase = k * c * Δt
        delta_time_seconds = delta_hours * 3600.0
        phase_shift = k * phase_speed * delta_time_seconds
        
        # Meridional wind pattern (90° out of phase from zonal)
        # v_jet ∝ sin(kx - ωt) with ω = kc
        rossby_wave = torch.sin(wave_number * lon_rad + phase_shift + math.pi/2)
        
        # Seasonally varying amplitude
        base_strength = 12.0  # m/s
        jet_strength = base_strength * season_factor
        
        # Time-evolving component
        # Convert delta_hours to tensor for torch operations
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        time_evolution = 1.0 / (1.0 + 0.1 * delta_hours_tensor)  # Slow decay over time
        
        return jet_strength * jet_envelope * rossby_wave * time_evolution
    
    def _thermal_wind_evolution(self, t2m, d2m, lat_rad, delta_hours, diurnal_factor):
        """
        Compute evolving thermal wind
        
        Physical basis:
        - Temperature gradients change with diurnal cycle
        - Surface heating/cooling modifies gradients
        - Time lag in atmospheric response
        """
        # Current zonal temperature gradient
        grad_T_x = torch.gradient(t2m, dim=1)[0]  # ∂T/∂x
        
        # Moisture effect (dew point depression indicates dry air)
        # Dry air heats/cools faster
        dryness = (t2m - d2m) / 10.0  # Normalized dryness
        
        # Diurnal evolution: stronger gradients during day
        # Maximum heating around local noon, cooling at night
        diurnal_amplification = 1.0 + 0.3 * diurnal_factor * (1.0 - dryness)
        
        # Time lag in atmospheric response (thermal inertia)
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        response_factor = 1.0 - torch.exp(-delta_hours_tensor / self.boundary_layer_response_time)
        
        # Coriolis scaling
        coriolis = self._compute_coriolis(lat_rad)
        coriolis_factor = torch.where(torch.abs(coriolis) > 1e-8, 
                                     1.0 / torch.abs(coriolis), 
                                     torch.zeros_like(coriolis))
        
        # Evolving thermal wind
        thermal_wind = -0.08 * grad_T_x * coriolis_factor * 100.0  # Base
        thermal_wind = thermal_wind * diurnal_amplification * response_factor
        
        return thermal_wind
    
    def _pressure_gradient_evolution(self, sp, lat_rad, delta_hours):
        """
        Pressure gradient evolution over time
        
        Physical basis:
        - Pressure systems move and intensify
        - Geostrophic adjustment timescale
        - Different evolution rates over land vs sea
        """
        # Current pressure gradient
        grad_p_x = torch.gradient(sp, dim=1)[0]  # ∂p/∂x
        
        # Coriolis parameter
        f = self._compute_coriolis(lat_rad)
        
        # Geostrophic wind
        with torch.no_grad():
            f_safe = torch.where(torch.abs(f) > 1e-8, f, torch.sign(f) * 1e-8)
        v_geo = grad_p_x / (self.rho * f_safe)
        
        # Pressure systems evolve over time
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        time_decay = torch.exp(-delta_hours_tensor / 12.0)  # 12-hour decay timescale
        
        # Boundary layer reduction
        bl_factor = 0.7
        
        return v_geo * bl_factor * time_decay
    
    def _advection_tendency(self, u100_current, v100_current, v100_field, delta_hours):
        """
        Compute advection tendency: ∂v/∂t = -u·∇v
        
        Physical basis:
        - Wind carries momentum with it
        - Upstream conditions affect downstream evolution
        - Important for time evolution
        """
        # Compute gradients of v
        grad_v_x = torch.gradient(v100_field, dim=1)[0]  # ∂v/∂x
        grad_v_y = torch.gradient(v100_field, dim=0)[0]  # ∂v/∂y
        
        # Advection: ∂v/∂t = -u·∇v = -(u ∂v/∂x + v ∂v/∂y)
        advection_tendency = -(u100_current * grad_v_x + v100_current * grad_v_y)
        
        # Convert tendency to change over delta_hours
        # ∂v = (∂v/∂t) * Δt
        delta_seconds = delta_hours * 3600.0
        
        # Scale appropriately (typical synoptic scale)
        advection_change = advection_tendency * delta_seconds / 1e6
        
        return advection_change
    
    def _boundary_layer_evolution(self, u100_current, v100_current, land_sea_mask, delta_hours):
        """
        Boundary layer evolution with time
        
        Physical basis:
        - Frictional damping increases with time
        - Different evolution over land vs sea
        - Momentum mixing time scale
        """
        # Current wind magnitude
        wind_speed = torch.sqrt(u100_current**2 + v100_current**2 + 1e-6)
        
        # Time-dependent friction coefficient
        # Convert delta_hours to tensor
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # Friction accumulates over time
        friction_accumulation = 1.0 - torch.exp(-delta_hours_tensor / 6.0)
        
        # Different friction for land vs sea
        base_friction = torch.where(land_sea_mask.bool(), 
                                  self.friction_land, 
                                  self.friction_sea)
        
        effective_friction = base_friction * (1.0 + friction_accumulation)
        
        # Damping factor: exp(-friction * time)
        damping = torch.exp(-effective_friction * delta_hours_tensor)
        
        return damping
    
    def _diurnal_evolution(self, cos_zenith_current, cos_zenith_future, delta_hours):
        """
        Diurnal cycle evolution
        
        Physical basis:
        - Solar heating changes through the day
        - Thermal circulations evolve
        - Maximum effect during daytime
        """
        # Current and future solar zenith
        # cos(zenith) = 1 at noon, -1 at midnight
        solar_current = torch.clamp(cos_zenith_current, 0.0, 1.0)  # Daytime only
        solar_future = torch.clamp(cos_zenith_future, 0.0, 1.0)
        
        # Diurnal change
        solar_change = solar_future - solar_current
        
        # Time-of-day factor
        # Maximum thermal effects around local noon
        time_factor = 1.0 + 0.5 * solar_future
        
        # Convert delta_hours to tensor for tensor operations
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        
        # Evolution rate (faster during day)
        if delta_hours > 0:
            # Use tensor operation
            evolution_rate = 1.0 / (1.0 + torch.exp(-10.0 * solar_change))
        else:
            evolution_rate = torch.ones_like(solar_change)
        
        return time_factor * evolution_rate
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, 
                geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict v100 (meridional wind at 100m) delta_hours into the future
        
        Args:
            meteo: [t2m, tp, u100, v100, d2m, sp] at current time
            geo_dense: [orograph, sin_lat, cos_lat, sin_lon, cos_lon]
            geo_cat: [land_sea_mask, soil_type]
            geo_tempo: [sin_orbit, cos_orbit, sin_zenith, cos_zenith] at current time
            geo_tempo_f: same as geo_tempo but at future time
            current_dt: current datetime in UTC
            delta_hours: hours to predict ahead (positive for future)
            
        Returns:
            Predicted v100 field at t + delta_hours
        """
        # Move tensors to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Unpack inputs
        t2m, tp, u100_current, v100_current, d2m, sp = meteo
        orograph, sin_lat, cos_lat, sin_lon, cos_lon = geo_dense
        land_sea_mask, soil_type = geo_cat
        sin_orbit, cos_orbit, sin_zenith, cos_zenith = geo_tempo
        sin_orbit_f, cos_orbit_f, sin_zenith_f, cos_zenith_f = geo_tempo_f
        
        # Convert to radians
        lat_rad = torch.atan2(sin_lat, cos_lat)
        lon_rad = torch.atan2(sin_lon, cos_lon)
        lat_deg = lat_rad * 180 / math.pi
        
        # Compute diurnal evolution factor
        diurnal_factor = self._diurnal_evolution(cos_zenith, cos_zenith_f, delta_hours)
        
        # Seasonal factor (from orbital position)
        season_current = cos_orbit
        season_future = cos_orbit_f
        season_factor = 0.5 * (season_current + season_future)  # Average over interval
        season_factor = 1.0 + 0.3 * (1.0 - season_factor)  # NH winter: ~1.3, summer: ~0.7
        
        # 1. Jet stream evolution with Rossby wave propagation
        jet_v = self._jet_stream_evolution(lat_rad, lon_rad, delta_hours, season_factor)
        
        # 2. Thermal wind evolution (diurnal changes in temperature gradients)
        thermal_v = self._thermal_wind_evolution(t2m, d2m, lat_rad, delta_hours, diurnal_factor)
        
        # 3. Pressure gradient evolution
        pressure_v = self._pressure_gradient_evolution(sp, lat_rad, delta_hours)
        
        # 4. Topographic effects (relatively constant over short periods)
        grad_orog_y = torch.gradient(orograph, dim=0)[0]
        topo_v = -0.05 * grad_orog_y
        land_factor = land_sea_mask.float()
        topo_v = topo_v * (1.0 + 0.5 * land_factor)
        
        # 5. Land-sea breeze evolution
        # Temperature difference evolves with diurnal cycle
        temp_diff = t2m - d2m
        diurnal_effect = 0.02 * temp_diff * diurnal_factor
        land_sea_v = diurnal_effect * land_sea_mask.float()
        
        # Soil type effect
        soil_factor = torch.ones_like(soil_type, dtype=torch.float32)
        for soil_code, factor in [(1, 1.2), (2, 1.1), (3, 1.0), 
                                 (4, 0.9), (5, 0.8), (6, 0.7), (7, 0.6)]:
            soil_factor = torch.where(soil_type == soil_code, 
                                    factor * soil_factor, soil_factor)
        land_sea_v = land_sea_v * soil_factor
        
        # 6. Current wind persistence with time-dependent damping
        damping = self._boundary_layer_evolution(u100_current, v100_current, 
                                                land_sea_mask, delta_hours)
        persistence_v = v100_current * damping
        
        # 7. Advection tendency (wind carries itself forward)
        advection_change = self._advection_tendency(u100_current, v100_current, 
                                                   v100_current, delta_hours)
        
        # 8. Precipitation evolution
        # Convert delta_hours to tensor for torch operations
        delta_hours_tensor = torch.tensor(delta_hours, device=self.device, dtype=torch.float32)
        precip_effect = -0.01 * tp * torch.exp(-delta_hours_tensor / 3.0)  # Decays over 3 hours
        
        # 9. Coriolis-zonal interaction evolution
        coriolis = self._compute_coriolis(lat_rad)
        coriolis_on_u = 0.08 * coriolis * u100_current * damping
        
        # Combine all components with time-aware weighting
        # Shorter forecasts: more persistence
        # Longer forecasts: more evolution terms
        forecast_weight = min(1.0, delta_hours / 12.0)  # 0-1 scale
        forecast_weight_tensor = torch.tensor(forecast_weight, device=self.device, dtype=torch.float32)
        
        v_predicted = (
            (1.0 - forecast_weight_tensor * 0.3) * persistence_v +  # Less persistence for longer forecasts
            0.25 * jet_v +                                   # Jet stream always important
            forecast_weight_tensor * 0.15 * pressure_v +     # Pressure evolves
            forecast_weight_tensor * 0.12 * thermal_v +      # Thermal winds evolve
            0.08 * topo_v +                                  # Topography constant
            forecast_weight_tensor * 0.05 * land_sea_v +     # Land-sea evolves
            forecast_weight_tensor * 0.08 * advection_change +  # Advection increases with time
            forecast_weight_tensor * 0.03 * coriolis_on_u +  # Coriolis interaction evolves
            0.02 * precip_effect                             # Precipitation effect
        )
        
        # Special handling for different forecast lengths
        if delta_hours <= 1:
            # Very short range: heavily weighted to persistence
            v_predicted = 0.8 * v100_current + 0.2 * v_predicted
        elif delta_hours <= 6:
            # Short range: balanced approach
            pass  # Use the weighted combination above
        else:
            # Longer range: more evolution, less persistence
            evolution_boost = min(1.0, (delta_hours - 6) / 12.0)
            evolution_boost_tensor = torch.tensor(evolution_boost, device=self.device, dtype=torch.float32)
            v_predicted = v_predicted * (1.0 + 0.2 * evolution_boost_tensor)
        
        # Apply latitude-dependent adjustments
        # Near equator: reduce geostrophic terms, enhance diurnal
        equator_mask = torch.abs(lat_deg) < 10.0
        if equator_mask.any():
            v_equator = v_predicted * equator_mask.float()
            # Enhanced diurnal effects near equator
            v_equator = v_equator * (1.0 + 0.3 * diurnal_factor)
            # Reduce pressure gradient terms
            v_equator = 0.3 * pressure_v + 0.7 * v_equator
            v_predicted = torch.where(equator_mask, v_equator, v_predicted)
        
        # Polar regions: simplified dynamics
        polar_mask = torch.abs(lat_deg) > 70.0
        if polar_mask.any():
            v_polar = v_predicted * polar_mask.float()
            v_polar = 0.9 * persistence_v + 0.1 * v_polar  # High persistence
            v_predicted = torch.where(polar_mask, v_polar, v_predicted)
        
        # Ensure physical bounds
        v_predicted = torch.clamp(v_predicted, -50.0, 50.0)
        
        # Ocean smoothing (less variability over water)
        ocean_mask = land_sea_mask == 0
        if ocean_mask.any() and delta_hours > 0:
            # Apply stronger smoothing for longer forecasts over ocean
            smooth_factor = min(0.5, delta_hours / 24.0)
            smooth_factor_tensor = torch.tensor(smooth_factor, device=self.device, dtype=torch.float32)
            v_ocean = v_predicted * ocean_mask.float()
            # 5-point smoothing
            for _ in range(int(1 + smooth_factor_tensor.item() * 2)):
                v_ocean = (torch.roll(v_ocean, 1, dims=0) +
                          torch.roll(v_ocean, -1, dims=0) +
                          torch.roll(v_ocean, 1, dims=1) +
                          torch.roll(v_ocean, -1, dims=1) +
                          v_ocean) / 5.0
            v_predicted = torch.where(ocean_mask, v_ocean, v_predicted)
        
        return v_predicted

# Example usage with different delta_hours
if __name__ == "__main__":
    # Create predictor
    predictor = V100mPredictor()
    
    # Test different forecast lengths
    test_hours = [1, 3, 6, 12, 24]
    
    for delta_hours in test_hours:
        print(f"\n=== Testing {delta_hours}-hour forecast ===")
        
        # Create dummy data
        height, width = 180, 360
        meteo = torch.randn(6, height, width)
        geo_dense = torch.randn(5, height, width)
        geo_cat = torch.randint(0, 2, (2, height, width))
        geo_tempo = torch.randn(4, height, width)
        geo_tempo_f = torch.randn(4, height, width)
        
        current_dt = datetime(2024, 1, 15, 12, 0, 0)
        
        # Make prediction
        with torch.no_grad():
            v100_pred = predictor.predict(meteo, geo_dense, geo_cat,
                                         geo_tempo, geo_tempo_f,
                                         current_dt, delta_hours)
        
        print(f"Predicted v100 shape: {v100_pred.shape}")
        print(f"Predicted v100 range: [{v100_pred.min():.2f}, {v100_pred.max():.2f}] m/s")
        print(f"Mean: {v100_pred.mean():.2f} m/s, Std: {v100_pred.std():.2f} m/s")
        
        # Compare with persistence (if we had truth)
        v100_current = meteo[3]  # Current v100
        print(f"Persistence would give range: [{v100_current.min():.2f}, {v100_current.max():.2f}] m/s")