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
                current_dt: datetime, delta_hours: int) -> torch.Tensor] - (torch.tensor)

Please write a Python class "Wind100mPredictor" that has the above predict(.) function, 
which predicts both u100 and v100 at the same time in relation with each other, 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.
"""

import torch
import math
from datetime import datetime
from typing import Optional, Union, Tuple
import numpy as np

class Wind100mPredictor:
    """
    Predict u100 and v100 at time t+delta using physical principles.
    Jet stream model uses sin²(lat) × cos²(lat) for correct double-peaked structure.
    """
    
    def __init__(self, device: Optional[str] = None, return_cpu: bool = True):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.return_cpu = return_cpu
        self._setup_constants()
        
    def _setup_constants(self):
        """Initialize all physical constants."""
        # Physical constants
        self.R = torch.tensor(287.058, device=self.device)
        self.omega = torch.tensor(7.2921159e-5, device=self.device)
        self.rho = torch.tensor(1.225, device=self.device)
        
        # Coriolis adjustment factors
        self.BETA = torch.tensor(2.289e-11, device=self.device)
        
        # Jet stream parameters - CORRECTED FOR sin² × cos² PROFILE
        self.jet_lat_center_nh = torch.tensor(45.0, device=self.device)  # Northern hemisphere
        self.jet_lat_center_sh = torch.tensor(-45.0, device=self.device) # Southern hemisphere
        self.jet_width = torch.tensor(10.0, device=self.device)
        self.jet_max_speed_u = torch.tensor(35.0, device=self.device)  # Stronger for double peaks
        self.jet_max_speed_v = torch.tensor(8.0, device=self.device)
        
        # Friction coefficients
        self.cf_land_u = torch.tensor(0.0035, device=self.device)
        self.cf_sea_u = torch.tensor(0.00025, device=self.device)
        self.cf_land_v = torch.tensor(0.0045, device=self.device)
        self.cf_sea_v = torch.tensor(0.00035, device=self.device)
        
        # Geostrophic adjustment
        self.GEOSTROPHIC_TIMESCALE = torch.tensor(6.0, device=self.device)
        
        # Thermal wind parameters
        self.THERMAL_WIND_COEFF = torch.tensor(-0.008, device=self.device)
        
        # Diurnal cycles
        self.DIURNAL_U_AMPLITUDE = torch.tensor(2.0, device=self.device)
        self.DIURNAL_V_AMPLITUDE = torch.tensor(1.0, device=self.device)
        
        # Limits
        self.U_MIN = torch.tensor(-50.0, device=self.device)  # Allow stronger jet
        self.U_MAX = torch.tensor(50.0, device=self.device)
        self.V_MIN = torch.tensor(-25.0, device=self.device)
        self.V_MAX = torch.tensor(25.0, device=self.device)
        
        # Polar region thresholds
        self.ANTARCTICA_LAT = torch.tensor(-60.0, device=self.device)
        self.GREENLAND_LAT_MIN = torch.tensor(60.0, device=self.device)
        self.GREENLAND_LAT_MAX = torch.tensor(85.0, device=self.device)
        self.GREENLAND_LON_MIN = torch.tensor(-50.0, device=self.device)
        self.GREENLAND_LON_MAX = torch.tensor(-20.0, device=self.device)
        
        # Smoothing kernel
        self.SMOOTH_KERNEL = torch.tensor([[0.05, 0.1, 0.05],
                                           [0.1, 0.4, 0.1],
                                           [0.05, 0.1, 0.05]], device=self.device)
        
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor,
                current_dt: datetime, geo_cat_f: torch.Tensor, 
                delta_hours: int, return_tensor: Optional[bool] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                                                Tuple[np.ndarray, np.ndarray]]:
        """
        Predict both u100 and v100 at time t+delta with corrected jet stream model.
        """
        # Move to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_cat_f = geo_cat_f.to(self.device)
        
        # Extract components
        t2m = meteo[0]
        current_u100 = meteo[2]
        current_v100 = meteo[3]
        sp = meteo[5]
        
        orograph = geo_dense[0]
        sin_lat = geo_dense[1]
        cos_lat = geo_dense[2]
        sin_lon = geo_dense[3]
        cos_lon = geo_dense[4]
        
        land_sea_mask = geo_cat[0]
        
        sin_orbit = geo_tempo[0]
        cos_orbit = geo_tempo[1]
        sin_zenith = geo_tempo[2]
        cos_zenith = geo_tempo[3]
        
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
        
        # 1. Compute Coriolis parameter
        f = 2 * self.omega * torch.sin(lat_rad)
        
        # 2. Compute persistence with coupled adjustment
        persistence_u, persistence_v = self._compute_coupled_persistence(
            current_u100, current_v100, f, land_sea_mask, delta_hours
        )
        
        # 3. Compute geostrophic adjustment
        geostrophic_u, geostrophic_v = self._compute_geostrophic_adjustment(
            sp, t2m, lat_rad, current_u100, current_v100, delta_hours
        )
        
        # 4. Compute jet stream with CORRECTED sin² × cos² profile
        jet_u, jet_v = self._compute_jet_stream_sin2_cos2(
            lat_rad, lat_deg, current_dt, sin_orbit, cos_orbit
        )
        
        # 5. Compute diurnal effects
        diurnal_u, diurnal_v = self._compute_diurnal_effects(
            sin_zenith, cos_zenith, current_dt, lat_deg, lon_deg
        )
        
        # 6. Compute topographic effects
        topo_u, topo_v = self._compute_topographic_effects(
            orograph, lat_deg, lon_deg, current_u100, current_v100
        )
        
        # 7. Combine all components
        weights = self._compute_dynamic_weights(delta_hours, lat_deg, land_sea_mask)
        
        predicted_u = (weights[0] * persistence_u + 
                      weights[1] * geostrophic_u + 
                      weights[2] * jet_u + 
                      weights[3] * diurnal_u + 
                      weights[4] * topo_u)
        
        predicted_v = (weights[0] * persistence_v + 
                      weights[1] * geostrophic_v + 
                      weights[2] * jet_v + 
                      weights[3] * diurnal_v + 
                      weights[4] * topo_v)
        
        # 8. Apply boundary layer coupling
        predicted_u, predicted_v = self._apply_boundary_layer_coupling(
            predicted_u, predicted_v, current_u100, current_v100, delta_hours
        )
        
        # 9. Apply polar region adjustments
        predicted_u, predicted_v = self._apply_polar_adjustments(
            predicted_u, predicted_v, lat_deg, lon_deg, orograph
        )
        
        # 10. Apply stabilization
        land_sea_mask_future = geo_cat_f[0]
        predicted_u, predicted_v = self._apply_stabilization(
            predicted_u, predicted_v, current_u100, current_v100, land_sea_mask_future
        )
        
        # 11. Apply physical constraints
        predicted_u, predicted_v = self._apply_physical_constraints(
            predicted_u, predicted_v, lat_deg
        )
        
        # 12. Final smoothing
        predicted_u = self._smooth_field(predicted_u)
        predicted_v = self._smooth_field(predicted_v)
        
        # 13. Clamp to reasonable limits
        predicted_u = torch.clamp(predicted_u, self.U_MIN, self.U_MAX)
        predicted_v = torch.clamp(predicted_v, self.V_MIN, self.V_MAX)
        
        return self._convert_result(predicted_u, predicted_v, return_tensor)
    
    def _compute_jet_stream_sin2_cos2(self, lat_rad: torch.Tensor, lat_deg: torch.Tensor,
                                     current_time: datetime, sin_orbit: torch.Tensor,
                                     cos_orbit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CORRECTED JET STREAM MODEL: Uses sin²(latitude) × cos²(latitude)
        """
        # 1. Compute sin²(lat) × cos²(lat) profile
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)
        
        jet_profile = (sin_lat ** 2) * (cos_lat ** 2)
        
        # 2. Seasonal shifting using orbital position
        orbit_angle = torch.atan2(sin_orbit, cos_orbit)
        
        seasonal_shift_nh = 8.0 * torch.sin(orbit_angle)
        seasonal_shift_sh = -6.0 * torch.sin(orbit_angle)
        
        # 3. Create separate profiles for each hemisphere
        nh_mask = lat_deg > 0
        sh_mask = lat_deg < 0
        
        # Initialize profiles with zeros
        nh_profile = torch.zeros_like(lat_deg)
        sh_profile = torch.zeros_like(lat_deg)
        
        # Northern hemisphere profile with seasonal shift
        if torch.any(nh_mask):
            lat_offset_nh = lat_deg - (self.jet_lat_center_nh + seasonal_shift_nh)
            nh_envelope = torch.exp(-(lat_offset_nh / (self.jet_width * 1.2)) ** 2)
            nh_profile = torch.where(nh_mask, jet_profile * nh_envelope, nh_profile)
        
        # Southern hemisphere profile with seasonal shift
        if torch.any(sh_mask):
            lat_offset_sh = lat_deg - (self.jet_lat_center_sh + seasonal_shift_sh)
            sh_envelope = torch.exp(-(lat_offset_sh / (self.jet_width * 1.0)) ** 2)
            sh_profile = torch.where(sh_mask, jet_profile * sh_envelope, sh_profile)
        
        # 4. Combine hemispheres
        combined_profile = nh_profile + sh_profile
        
        # 5. U-component (zonal wind): Strong westerlies in jet core
        jet_u = self.jet_max_speed_u * combined_profile
        
        # Apply hemispheric sign: positive (westerly) in both hemispheres
        jet_u = torch.abs(jet_u)
        
        # 6. V-component (meridional wind): Weaker, related to jet meanders
        # Compute gradient of U profile to find shear zones
        combined_profile_4d = combined_profile.unsqueeze(0).unsqueeze(0)
        grad_kernel = torch.tensor([[0, -0.5, 0],
                                    [0, 0, 0],
                                    [0, 0.5, 0]], dtype=combined_profile_4d.dtype, device=self.device)
        grad_kernel_4d = grad_kernel.unsqueeze(0).unsqueeze(0)
        
        du_dlat = torch.nn.functional.conv2d(
            torch.nn.functional.pad(combined_profile_4d, (1, 1, 1, 1), mode='replicate'),
            grad_kernel_4d,
            padding=0
        ).squeeze()
        
        # V component proportional to U gradient with phase shift
        phase_shift = torch.sin(2 * lat_rad)
        jet_v = self.jet_max_speed_v * du_dlat * phase_shift
        
        # 7. Diurnal modulation
        hour_of_day = current_time.hour + current_time.minute / 60.0
        hour_tensor = torch.tensor(hour_of_day, device=self.device, dtype=lat_rad.dtype)
        diurnal_factor = 0.15 * torch.cos(2 * math.pi * (hour_tensor - 3) / 24)
        
        jet_u = jet_u * (1 + diurnal_factor)
        jet_v = jet_v * (1 - diurnal_factor * 0.5)
        
        return jet_u, jet_v
    
    def _compute_coupled_persistence(self, u_current: torch.Tensor, v_current: torch.Tensor,
                                    f: torch.Tensor, land_sea_mask: torch.Tensor,
                                    delta_hours: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute persistence with coupled Coriolis adjustment."""
        # Convert delta_hours to tensor for PyTorch operations
        delta_tensor = torch.tensor(delta_hours, device=self.device, dtype=u_current.dtype)
        
        land_decay_u = torch.exp(-self.cf_land_u * delta_tensor)
        sea_decay_u = torch.exp(-self.cf_sea_u * delta_tensor)
        land_decay_v = torch.exp(-self.cf_land_v * delta_tensor)
        sea_decay_v = torch.exp(-self.cf_sea_v * delta_tensor)
        
        decay_u = torch.where(land_sea_mask > 0.5, land_decay_u, sea_decay_u)
        decay_v = torch.where(land_sea_mask > 0.5, land_decay_v, sea_decay_v)
        
        # Coriolis-induced rotation
        inertial_period = 2 * math.pi / torch.abs(f + 1e-10)
        rotation_angle = 2 * math.pi * delta_tensor / (inertial_period * 24)
        
        u_rot = u_current * torch.cos(rotation_angle) - v_current * torch.sin(rotation_angle)
        v_rot = u_current * torch.sin(rotation_angle) + v_current * torch.cos(rotation_angle)
        
        persistence_u = u_rot * decay_u
        persistence_v = v_rot * decay_v
        
        return persistence_u, persistence_v
    
    def _compute_geostrophic_adjustment(self, sp: torch.Tensor, t2m: torch.Tensor,
                                       lat_rad: torch.Tensor, u_current: torch.Tensor,
                                       v_current: torch.Tensor, delta_hours: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute geostrophic adjustment."""
        # Convert delta_hours to tensor
        delta_tensor = torch.tensor(delta_hours, device=self.device, dtype=sp.dtype)
        
        sp_4d = sp.unsqueeze(0).unsqueeze(0)
        
        kernel_x = torch.tensor([[0, 0, 0],
                                 [-0.5, 0, 0.5],
                                 [0, 0, 0]], dtype=sp.dtype, device=self.device)
        
        kernel_y = torch.tensor([[0, -0.5, 0],
                                 [0, 0, 0],
                                 [0, 0.5, 0]], dtype=sp.dtype, device=self.device)
        
        kernel_x_4d = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        pad = (1, 1, 1, 1)
        sp_padded = torch.nn.functional.pad(sp_4d, pad, mode='replicate')
        
        dsp_dx = torch.nn.functional.conv2d(sp_padded, kernel_x_4d, padding=0).squeeze()
        dsp_dy = torch.nn.functional.conv2d(sp_padded, kernel_y_4d, padding=0).squeeze()
        
        # Temperature gradients
        t2m_4d = t2m.unsqueeze(0).unsqueeze(0)
        t2m_padded = torch.nn.functional.pad(t2m_4d, pad, mode='replicate')
        
        dt2m_dx = torch.nn.functional.conv2d(t2m_padded, kernel_x_4d, padding=0).squeeze()
        dt2m_dy = torch.nn.functional.conv2d(t2m_padded, kernel_y_4d, padding=0).squeeze()
        
        # Coriolis parameter
        f = 2 * self.omega * torch.sin(lat_rad)
        f_sign = torch.sign(f)
        f_abs = torch.abs(f)
        f_min = torch.tensor(1e-10, device=self.device)
        f_safe = f_sign * torch.maximum(f_abs, f_min)
        
        # Geostrophic winds
        u_geo = -dsp_dy / (self.rho * f_safe) * 1e-6
        v_geo = dsp_dx / (self.rho * f_safe) * 1e-6
        
        # Thermal wind correction
        u_thermal = -self.THERMAL_WIND_COEFF * dt2m_dy
        v_thermal = self.THERMAL_WIND_COEFF * dt2m_dx
        
        # Combine and relax
        geostrophic_u = u_geo + u_thermal
        geostrophic_v = v_geo + v_thermal
        
        tau = self.GEOSTROPHIC_TIMESCALE
        relaxation_factor = 1 - torch.exp(-delta_tensor / tau)
        
        adjustment_u = relaxation_factor * (geostrophic_u - u_current)
        adjustment_v = relaxation_factor * (geostrophic_v - v_current)
        
        return adjustment_u, adjustment_v
    
    def _compute_diurnal_effects(self, sin_zenith: torch.Tensor, cos_zenith: torch.Tensor,
                                current_time: datetime, lat_deg: torch.Tensor,
                                lon_deg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute diurnal wind variations."""
        zenith_angle = torch.atan2(sin_zenith, cos_zenith)
        solar_heating = torch.clamp(cos_zenith, 0, 1)
        
        lon_hour_offset = lon_deg / 15.0
        
        # Create a tensor for local hour calculation
        utc_hour = current_time.hour + current_time.minute / 60.0
        utc_hour_tensor = torch.tensor(utc_hour, device=self.device, dtype=lon_deg.dtype)
        local_hour = (utc_hour_tensor + lon_hour_offset) % 24
        
        phase = 2 * math.pi * local_hour / 24
        
        diurnal_u = self.DIURNAL_U_AMPLITUDE * solar_heating * torch.sin(phase)
        diurnal_v = self.DIURNAL_V_AMPLITUDE * solar_heating * torch.cos(phase)
        
        lat_factor = torch.cos(torch.deg2rad(lat_deg))
        diurnal_u = diurnal_u * lat_factor
        diurnal_v = diurnal_v * lat_factor
        
        return diurnal_u, diurnal_v
    
    def _compute_topographic_effects(self, orograph: torch.Tensor, lat_deg: torch.Tensor,
                                    lon_deg: torch.Tensor, u_current: torch.Tensor,
                                    v_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute topographic effects on winds."""
        orograph_4d = orograph.unsqueeze(0).unsqueeze(0)
        
        kernel_x = torch.tensor([[0, 0, 0],
                                 [-0.5, 0, 0.5],
                                 [0, 0, 0]], dtype=orograph.dtype, device=self.device)
        
        kernel_y = torch.tensor([[0, -0.5, 0],
                                 [0, 0, 0],
                                 [0, 0.5, 0]], dtype=orograph.dtype, device=self.device)
        
        kernel_x_4d = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y_4d = kernel_y.unsqueeze(0).unsqueeze(0)
        
        pad = (1, 1, 1, 1)
        oro_padded = torch.nn.functional.pad(orograph_4d, pad, mode='replicate')
        
        dh_dx = torch.nn.functional.conv2d(oro_padded, kernel_x_4d, padding=0).squeeze()
        dh_dy = torch.nn.functional.conv2d(oro_padded, kernel_y_4d, padding=0).squeeze()
        
        topo_u = -0.001 * dh_dy * torch.sign(lat_deg)
        topo_v = -0.001 * dh_dx * torch.sign(lon_deg)
        
        slope_magnitude = torch.sqrt(dh_dx**2 + dh_dy**2 + 1e-10)
        reduction_factor = torch.exp(-0.0001 * slope_magnitude)
        
        topo_u = topo_u * reduction_factor
        topo_v = topo_v * reduction_factor
        
        return topo_u, topo_v
    
    def _compute_dynamic_weights(self, delta_hours: int, lat_deg: torch.Tensor,
                                land_sea_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights for combining components.
        Returns tensor of shape [5, H, W] for proper broadcasting.
        """
        height, width = lat_deg.shape
        
        # Base weights
        if delta_hours == 0:
            base_weights = torch.tensor([0.8, 0.1, 0.05, 0.03, 0.02], device=self.device)
        elif delta_hours <= 3:
            base_weights = torch.tensor([0.6, 0.2, 0.1, 0.06, 0.04], device=self.device)
        elif delta_hours <= 6:
            base_weights = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05], device=self.device)
        else:
            base_weights = torch.tensor([0.4, 0.4, 0.1, 0.05, 0.05], device=self.device)
        
        # Create weight tensor with shape [5, H, W]
        weights = base_weights.view(-1, 1, 1).expand(5, height, width)
        
        # Jet stream stronger at mid-latitudes where sin²×cos² peaks
        lat_factor = torch.abs(torch.sin(2 * torch.deg2rad(lat_deg)))  # sin(2θ) peaks at ±45°
        
        # Expand lat_factor for weight multiplication
        weights = weights.clone()  # Create a copy to modify
        weights[2] = weights[2] * (1 + 0.8 * lat_factor)  # Boost jet contribution at jet latitudes
        
        # Adjust for land/sea
        sea_factor = (1 - land_sea_mask)
        weights[2] = weights[2] * (1 + 0.5 * sea_factor)  # Stronger jet over ocean
        
        # Normalize each spatial location independently
        weight_sum = torch.sum(weights, dim=0, keepdim=True)
        weights = weights / weight_sum
        
        return weights
    
    def _apply_boundary_layer_coupling(self, u_pred: torch.Tensor, v_pred: torch.Tensor,
                                      u_current: torch.Tensor, v_current: torch.Tensor,
                                      delta_hours: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply boundary layer physics coupling u and v."""
        # Convert delta_hours to tensor
        delta_tensor = torch.tensor(delta_hours, device=self.device, dtype=u_current.dtype)
        
        current_speed = torch.sqrt(u_current**2 + v_current**2 + 1e-10)
        pred_speed = torch.sqrt(u_pred**2 + v_pred**2 + 1e-10)
        
        current_dir = torch.atan2(v_current, u_current)
        pred_dir = torch.atan2(v_pred, u_pred)
        
        # Use tensor operations for relaxation calculations
        dir_relaxation = 1 - torch.exp(-delta_tensor / torch.tensor(12.0, device=self.device))
        speed_relaxation = 1 - torch.exp(-delta_tensor / torch.tensor(6.0, device=self.device))
        
        final_dir = current_dir + dir_relaxation * (pred_dir - current_dir)
        final_speed = current_speed + speed_relaxation * (pred_speed - current_speed)
        
        final_u = final_speed * torch.cos(final_dir)
        final_v = final_speed * torch.sin(final_dir)
        
        return final_u, final_v
    
    def _apply_polar_adjustments(self, u_pred: torch.Tensor, v_pred: torch.Tensor,
                                lat_deg: torch.Tensor, lon_deg: torch.Tensor,
                                orograph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply polar region specific adjustments."""
        antarctica_mask = (lat_deg < self.ANTARCTICA_LAT) & (orograph > 500)
        
        greenland_lat = (lat_deg > self.GREENLAND_LAT_MIN) & (lat_deg < self.GREENLAND_LAT_MAX)
        greenland_lon = (lon_deg > self.GREENLAND_LON_MIN) & (lon_deg < self.GREENLAND_LON_MAX)
        greenland_mask = greenland_lat & greenland_lon & (orograph > 500)
        
        polar_mask = antarctica_mask | greenland_mask
        
        if torch.any(polar_mask):
            reduction_factor = torch.where(polar_mask, 0.6, 1.0)
            
            # Compute topography gradient
            orograph_4d = orograph.unsqueeze(0).unsqueeze(0)
            
            # Create gradient kernels
            kernel_y = torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]], device=self.device, dtype=orograph.dtype)
            kernel_x = torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]], device=self.device, dtype=orograph.dtype)
            
            dh_dy = torch.nn.functional.conv2d(
                torch.nn.functional.pad(orograph_4d, (1, 1, 1, 1), mode='replicate'),
                kernel_y,
                padding=0
            ).squeeze()
            
            dh_dx = torch.nn.functional.conv2d(
                torch.nn.functional.pad(orograph_4d, (1, 1, 1, 1), mode='replicate'),
                kernel_x,
                padding=0
            ).squeeze()
            
            oro_grad = torch.sqrt(dh_dx**2 + dh_dy**2)
            
            katabatic_mask = polar_mask & (oro_grad > 100)
            if torch.any(katabatic_mask):
                u_pred = torch.where(katabatic_mask, u_pred * 0.5, u_pred)
                v_pred = torch.where(katabatic_mask, torch.minimum(v_pred, torch.tensor(-2.0, device=self.device)), v_pred)
            
            u_pred = u_pred * reduction_factor
            v_pred = v_pred * reduction_factor
        
        return u_pred, v_pred
    
    def _apply_stabilization(self, u_pred: torch.Tensor, v_pred: torch.Tensor,
                            u_current: torch.Tensor, v_current: torch.Tensor,
                            land_sea_mask_future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply stabilization using future land-sea mask."""
        land_damping = torch.tensor(0.8, device=self.device)
        sea_damping = torch.tensor(0.9, device=self.device)
        
        damping = torch.where(land_sea_mask_future > 0.5, land_damping, sea_damping)
        
        u_stable = damping * u_pred + (1 - damping) * u_current
        v_stable = damping * v_pred + (1 - damping) * v_current
        
        return u_stable, v_stable
    
    def _apply_physical_constraints(self, u_pred: torch.Tensor, v_pred: torch.Tensor,
                                   lat_deg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply physical constraints to wind fields."""
        # Jet stream regions: ensure strong westerlies
        jet_lat_mask = (torch.abs(lat_deg) > 20) & (torch.abs(lat_deg) < 60)
        
        if torch.any(jet_lat_mask):
            jet_profile = (torch.sin(torch.deg2rad(lat_deg)) ** 2) * \
                         (torch.cos(torch.deg2rad(lat_deg)) ** 2)
            u_boost = 0.2 * self.jet_max_speed_u * jet_profile
            u_pred = torch.where(jet_lat_mask, u_pred + u_boost, u_pred)
        
        # Ensure v is generally weaker than u
        uv_ratio = torch.abs(v_pred) / (torch.abs(u_pred) + 1e-10)
        v_reduction = torch.where(uv_ratio > 0.6, 0.6 / uv_ratio, 1.0)
        v_pred = v_pred * v_reduction
        
        return u_pred, v_pred
    
    def _smooth_field(self, field: torch.Tensor) -> torch.Tensor:
        """Apply light smoothing to wind field."""
        field_4d = field.unsqueeze(0).unsqueeze(0)
        kernel_4d = self.SMOOTH_KERNEL.unsqueeze(0).unsqueeze(0)
        
        smoothed = torch.nn.functional.conv2d(
            torch.nn.functional.pad(field_4d, (1, 1, 1, 1), mode='replicate'),
            kernel_4d,
            padding=0
        ).squeeze()
        
        return smoothed
    
    def _convert_result(self, u_result: torch.Tensor, v_result: torch.Tensor, 
                       return_tensor: Optional[bool]) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                              Tuple[np.ndarray, np.ndarray]]:
        """Convert results to appropriate format."""
        if return_tensor is None:
            return_tensor = not self.return_cpu
        
        if return_tensor:
            return u_result, v_result
        else:
            return u_result.cpu().numpy(), v_result.cpu().numpy()
    
    def to(self, device: str) -> 'Wind100mPredictor':
        """Move the predictor to a different device."""
        self.device = device
        self._setup_constants()
        return self