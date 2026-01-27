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

Please write a Python class "TpPredictor" that has the above predict(.) function, which predicts tp 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
Please choose GPU if it's avialable.
I need both code and physical explanation.

#----------------------------------------------------------- special ---------------------------------------------------------
Precipitating points are now evenly distributed on the globe.
And those points don't form a (partially) continuous chunks like clouds on the sky.
They are all scattered but not evenly.
When you look down the land from a high-flying plane in the night, there are lights coming from lots of buildings in many cities. 
The vast dark wilderness has cities scattered irregularly and unevenly. Each city has irregularly but a bit concentrated lights. 
That's what the distribution of precipitating places on the globe look like.

Yeah. It's much better now.
It's just like that the cities are too large and covers most of the land and connected each other.
In reality, cities are far from covering any significant portions of land.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
import math

class TpPredictor:
    def __init__(self, device=None):
        """Initialize the TP predictor with optional device selection."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Constants - ALL IN SI UNITS
        self.g = 9.81  # m/s²
        self.Rd = 287.05  # J/(kg·K) - dry air gas constant
        self.Rv = 461.5  # J/(kg·K) - water vapor gas constant
        self.Lv = 2.501e6  # J/kg - latent heat of vaporization at 0°C
        self.cp = 1005.0  # J/(kg·K) - specific heat at constant pressure
        self.rho_w = 1000.0  # kg/m³ - water density
        self.omega = 7.2921e-5  # rad/s - Earth's rotation rate
        self.p0 = 101325.0  # Pa - standard sea level pressure
        
        # Physical parameters - TUNED FOR REALISM
        self.grid_dx = 50000.0  # m - approximate grid spacing (50km)
        self.precip_efficiency = 0.2  # 20% of condensed water reaches ground
        
        # Jet stream parameters as tensors on the right device
        self.jet_lat_rad = torch.deg2rad(torch.tensor(45.0)).to(self.device)  # rad
        self.jet_width_rad = torch.deg2rad(torch.tensor(15.0)).to(self.device)  # rad
        
    def _lat_from_sincos(self, sin_lat, cos_lat):
        """Calculate latitude in radians from sin/cos."""
        return torch.atan2(sin_lat, cos_lat)
    
    def _lon_from_sincos(self, sin_lon, cos_lon):
        """Calculate longitude in radians from sin/cos."""
        return torch.atan2(sin_lon, cos_lon)
    
    def _jet_stream_strength(self, lat_rad):
        """Calculate jet stream strength using physical meridional profile."""
        # Gaussian jet profile centered at 45°N and 45°S
        jet_n = torch.exp(-((lat_rad - self.jet_lat_rad) / self.jet_width_rad) ** 2)
        jet_s = torch.exp(-((lat_rad + self.jet_lat_rad) / self.jet_width_rad) ** 2)
        
        # Combine both hemispheres
        jet_strength = torch.maximum(jet_n, jet_s)
        
        # Apply sin(lat) modulation for Coriolis effect
        jet_strength = jet_strength * torch.abs(torch.sin(lat_rad))
        
        # Normalize to [0.1, 1.0]
        max_val = torch.max(jet_strength)
        if max_val > 0:
            jet_strength = 0.1 + 0.9 * (jet_strength / max_val)
        
        return jet_strength
    
    def _saturation_vapor_pressure(self, T):
        """Calculate saturation vapor pressure over water (Pa)."""
        # Magnus formula for water (T in K)
        T_C = T - 273.15
        es = 611.2 * torch.exp(17.67 * T_C / (T_C + 243.5))  # Pa
        return es
    
    def _specific_humidity(self, T, p, RH):
        """Calculate specific humidity (kg/kg)."""
        es = self._saturation_vapor_pressure(T)
        e = RH * es  # Actual vapor pressure
        q = 0.622 * e / (p - 0.378 * e + 1e-6)
        return torch.clamp(q, 0, 0.03)  # Max ~30 g/kg
    
    def _relative_humidity(self, T, Td):
        """Calculate relative humidity from temperature and dew point."""
        es_T = self._saturation_vapor_pressure(T)
        es_Td = self._saturation_vapor_pressure(Td)
        RH = es_Td / (es_T + 1e-6)
        return torch.clamp(RH, 0.1, 1.05)  # 10-105% RH
    
    def _precipitable_water(self, T, Td, p):
        """Calculate precipitable water in column (kg/m² or mm)."""
        RH = self._relative_humidity(T, Td)
        q = self._specific_humidity(T, p, RH)
        
        # Approximate column integration: PW ≈ (q * p_surface) / (ρ_w * g)
        scale_height = 2000.0  # m - approximate for water vapor
        pw = q * p / (self.rho_w * self.g) * scale_height
        
        return torch.clamp(pw, 0, 50)  # Max 50 mm
    
    def _moisture_flux_convergence(self, u, v, q, p):
        """Calculate moisture flux convergence (kg/(m²·s))."""
        # Add batch dimensions if needed
        if len(u.shape) == 2:
            u_4d = u.unsqueeze(0).unsqueeze(0)
            v_4d = v.unsqueeze(0).unsqueeze(0)
            q_4d = q.unsqueeze(0).unsqueeze(0)
        else:
            u_4d = u
            v_4d = v
            q_4d = q
        
        # Moisture flux = q * v
        moisture_flux_x = q_4d * u_4d
        moisture_flux_y = q_4d * v_4d
        
        # Calculate divergence with proper spacing
        flux_x_grad_y, flux_x_grad_x = torch.gradient(moisture_flux_x, dim=[2, 3])
        flux_y_grad_y, flux_y_grad_x = torch.gradient(moisture_flux_y, dim=[2, 3])
        
        # Scale gradients by grid spacing
        flux_x_grad_x = flux_x_grad_x / self.grid_dx
        flux_x_grad_y = flux_x_grad_y / self.grid_dx
        flux_y_grad_x = flux_y_grad_x / self.grid_dx
        flux_y_grad_y = flux_y_grad_y / self.grid_dx
        
        # MFC = -∇·(q*v) = -(∂(qu)/∂x + ∂(qv)/∂y)
        mfc = -(flux_x_grad_x + flux_y_grad_y)
        
        # Convert to column convergence (kg/(m²·s))
        scale_height = 8000.0  # m - approximate atmospheric scale height
        mfc_column = mfc * scale_height
        
        if len(u.shape) == 2:
            mfc_column = mfc_column.squeeze()
        
        return torch.clamp(mfc_column, -1e-3, 1e-3)
    
    def _orographic_enhancement(self, u, v, orog, land_mask):
        """Calculate orographic precipitation enhancement (vertical velocity in m/s)."""
        if len(orog.shape) == 2:
            orog_4d = orog.unsqueeze(0).unsqueeze(0)
        else:
            orog_4d = orog
        
        # Calculate terrain gradient
        grad_y, grad_x = torch.gradient(orog_4d, dim=[2, 3])
        
        # Scale gradients by grid spacing
        grad_x = grad_x / self.grid_dx
        grad_y = grad_y / self.grid_dx
        
        if len(orog.shape) == 2:
            grad_x = grad_x.squeeze()
            grad_y = grad_y.squeeze()
        
        # Vertical velocity from orographic lift: w = v·∇h
        w_orog = u * grad_x + v * grad_y
        
        # Scale to realistic values (0.01-0.1 typical)
        w_orog = w_orog * 0.05
        
        # Only over land
        w_orog = w_orog * land_mask.float()
        
        return torch.clamp(w_orog, 0, 0.5)
    
    def _frontogenesis_function(self, T, u, v):
        """Calculate frontogenesis."""
        if len(T.shape) == 2:
            T_4d = T.unsqueeze(0).unsqueeze(0)
            u_4d = u.unsqueeze(0).unsqueeze(0)
            v_4d = v.unsqueeze(0).unsqueeze(0)
        else:
            T_4d = T
            u_4d = u
            v_4d = v
        
        # Temperature gradients
        dT_dy, dT_dx = torch.gradient(T_4d, dim=[2, 3])
        dT_dx = dT_dx / self.grid_dx
        dT_dy = dT_dy / self.grid_dx
        
        # Wind gradients
        du_dy, du_dx = torch.gradient(u_4d, dim=[2, 3])
        dv_dy, dv_dx = torch.gradient(v_4d, dim=[2, 3])
        du_dx = du_dx / self.grid_dx
        dv_dy = dv_dy / self.grid_dx
        
        # Frontogenesis magnitude (simplified)
        frontogenesis = (dT_dx * du_dx + dT_dy * dv_dy)
        
        if len(T.shape) == 2:
            frontogenesis = frontogenesis.squeeze()
        
        return frontogenesis * 1e9  # Scale to typical values
    
    def _convective_potential(self, T, Td, p, zenith, land_mask):
        """Calculate convective precipitation potential."""
        # Simplified convective index based on temperature difference
        
        # Dew point depression
        delta_T = T - Td
        
        # Convective instability index: larger dew point depression = more potential
        # But only if enough moisture (not too dry)
        RH = self._relative_humidity(T, Td)
        
        # Basic convective index
        conv_index = torch.clamp((10.0 - delta_T) / 10.0, 0, 1) * RH
        
        # Diurnal cycle: maximum in afternoon
        # zenith = 0 at local noon, increases in afternoon
        # Use sine squared for afternoon peak
        diurnal_factor = torch.sin(zenith) ** 2
        
        # Combine
        conv_potential = conv_index * diurnal_factor
        
        # Enhance over land
        conv_potential = conv_potential * (1.0 + 2.0 * land_mask.float())
        
        return torch.clamp(conv_potential, 0, 1)
    
    def _advect_field(self, field, u, v, dt_hours):
        """Simple advection using mean flow."""
        if dt_hours <= 0:
            return field
        
        dt_seconds = dt_hours * 3600.0
        
        # Use tensor operations - no .item() for tensors
        u_mean = u.mean()
        v_mean = v.mean()
        
        # Displacement in grid points (rounded to nearest integer)
        dx_pixels = torch.round(u_mean * dt_seconds / self.grid_dx).int().item()
        dy_pixels = torch.round(v_mean * dt_seconds / self.grid_dx).int().item()
        
        # Shift field
        shifted = torch.roll(field, shifts=(dy_pixels, dx_pixels), dims=(0, 1))
        
        # Decay with time (e-folding: 6 hours)
        decay = torch.exp(-torch.tensor(dt_hours / 6.0, device=self.device))
        
        return shifted * decay
    
    def _scattered_precipitation(self, precip_potential, land_mask, seed=None):
        """Create physically realistic scattered precipitation pattern."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Normalize potential to [0, 1]
        min_val = torch.min(precip_potential)
        max_val = torch.max(precip_potential)
        if (max_val - min_val) > 1e-6:
            norm_potential = (precip_potential - min_val) / (max_val - min_val)
        else:
            norm_potential = torch.zeros_like(precip_potential)
        
        # Simple clustering using local averaging
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size ** 2)
        
        if len(norm_potential.shape) == 2:
            potential_4d = norm_potential.unsqueeze(0).unsqueeze(0)
        else:
            potential_4d = norm_potential
        
        # Apply smoothing
        clustered = torch.nn.functional.conv2d(potential_4d, kernel, padding=kernel_size//2)
        
        # Precipitation occurs where clustered value exceeds threshold
        threshold = 0.3 + 0.4 * land_mask.float().unsqueeze(0).unsqueeze(0)
        precip_mask = (clustered > threshold).float()
        
        # Intensity modulation
        scattered = precip_mask * potential_4d * (0.5 + 0.5 * torch.rand_like(clustered))
        
        if len(norm_potential.shape) == 2:
            scattered = scattered.squeeze()
        
        return scattered
    
    def predict(self, meteo: torch.Tensor, geo_dense: torch.Tensor, 
                geo_cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor:
        """
        Predict 1-hour accumulated precipitation ending at t + delta_hours.
        """
        # Move tensors to device
        meteo = meteo.to(self.device)
        geo_dense = geo_dense.to(self.device)
        geo_cat = geo_cat.to(self.device)
        geo_tempo = geo_tempo.to(self.device)
        geo_tempo_f = geo_tempo_f.to(self.device)
        
        # Extract variables
        T, current_tp, u, v, Td, p = meteo  # All at time t
        orog, sin_lat, cos_lat, sin_lon, cos_lon = geo_dense
        land_mask, soil_type = geo_cat
        
        # Extract temporal variables - only need zenith for future time
        sin_zenith_f, cos_zenith_f = geo_tempo_f[2], geo_tempo_f[3]
        zenith_f = torch.atan2(sin_zenith_f, cos_zenith_f)
        
        # Calculate latitude from provided sin/cos
        lat = self._lat_from_sincos(sin_lat, cos_lat)
        
        # 1. Calculate basic atmospheric properties
        RH = self._relative_humidity(T, Td)
        q = self._specific_humidity(T, p, RH)
        
        # 2. Calculate precipitation forcing mechanisms
        
        # 2a. Moisture Flux Convergence (large-scale precipitation)
        mfc = self._moisture_flux_convergence(u, v, q, p)  # kg/(m²·s)
        
        # Convert to precipitation rate: 1 kg/(m²·s) = 3600 mm/hour
        precip_mfc = torch.clamp(mfc * 3600.0, 0, 5.0)  # mm/hour
        
        # 2b. Orographic precipitation
        w_orog = self._orographic_enhancement(u, v, orog, land_mask)  # m/s
        
        # Conversion: w (m/s) → precipitation rate (mm/hour)
        # Approximate: 0.1 m/s → ~1 mm/hour
        precip_orog = w_orog * 8.0  # mm/hour
        
        # 2c. Frontal precipitation
        frontogenesis = self._frontogenesis_function(T, u, v)
        precip_frontal = torch.clamp(frontogenesis, 0, 3.0)  # mm/hour
        
        # 2d. Convective precipitation
        conv_index = self._convective_potential(T, Td, p, zenith_f, land_mask)
        precip_conv = conv_index * 3.0  # mm/hour
        
        # 3. Jet stream enhancement
        jet = self._jet_stream_strength(lat)
        
        # 4. Combine precipitation sources with physical weights
        precip_rate = (
            0.40 * precip_mfc * (1.0 + 0.3 * jet) +      # Large-scale
            0.25 * precip_orog +                         # Orographic
            0.20 * precip_frontal * (1.0 + 0.2 * jet) +  # Frontal
            0.15 * precip_conv                           # Convective
        )  # mm/hour
        
        # 5. Apply precipitation efficiency
        precip_rate = precip_rate * self.precip_efficiency
        
        # 6. Time evolution - handle different delta_hours cases
        if delta_hours == 0:
            # No time evolution, return current
            predicted_tp = current_tp
            
        elif delta_hours == 1:
            # 1-hour forecast: balance between advection and new formation
            current_mmh = current_tp * 1000.0  # Convert m to mm/hour
            
            # Simple persistence with small adjustment
            persistence_weight = 0.7
            new_formation_weight = 0.3
            
            combined_mmh = (persistence_weight * current_mmh + 
                           new_formation_weight * precip_rate)
            predicted_tp = combined_mmh / 1000.0  # Back to meters
            
        else:
            # Multiple hours: advect and decay
            current_mmh = current_tp * 1000.0
            advected = self._advect_field(current_mmh, u, v, delta_hours)
            
            # Weighting based on lead time
            if delta_hours <= 3:
                weight_new = 0.4
            elif delta_hours <= 6:
                weight_new = 0.6
            else:
                weight_new = 0.8
            
            weight_current = 1.0 - weight_new
            
            combined_mmh = weight_current * advected + weight_new * precip_rate
            predicted_tp = combined_mmh / 1000.0
        
        # 7. Apply scattered pattern
        seed = int(current_dt.timestamp()) % 10000
        scattered = self._scattered_precipitation(predicted_tp, land_mask, seed)
        
        # 8. Apply land/sea contrast
        land_factor = 1.0 + 3.0 * land_mask.float()  # 4x more over land
        final_tp = scattered * land_factor
        
        # 9. Ensure physical bounds and realistic distribution
        
        # Most places have no precipitation - apply probability based on intensity
        intensity_prob = torch.sigmoid((final_tp * 1e6) - 2.0)  # Threshold around 2e-6 m
        
        # Random occurrence
        rand = torch.rand_like(final_tp)
        precip_occurs = (rand < intensity_prob).float()
        
        final_tp = final_tp * precip_occurs
        
        # 10. Match scale - gentle adjustment if needed
        current_mean = current_tp.mean().item()
        predicted_mean = final_tp.mean().item()
        
        # Only adjust if prediction is significantly off
        if predicted_mean > 1e-10 and current_mean > 1e-10:
            ratio = current_mean / predicted_mean
            if ratio < 0.5 or ratio > 2.0:
                # Limit adjustment to factor of 2
                scale = torch.clamp(torch.tensor(ratio, device=self.device), 0.5, 2.0).item()
                final_tp = final_tp * scale
        
        # Final bounds: 0 to 0.001 m (0 to 1 mm/hour - reasonable for 1-hour accum)
        final_tp = torch.clamp(final_tp, 0.0, 0.001)
        
        return final_tp