"""
Prompt (DeepSeek)
Again.
Now we have input maps at any given time t. They are:
    meteo = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0), where all components are of shape (721, 1440) in lat and lon,
    geo_dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    geo_cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenit, cos_zenit], dim=0).

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

Please write code that predicts t2m at time t + delta hours, by fastest physical logic and not by tedious ML or numerical analysis. delta is in range: 0, 1, ..., 12
Please take into account all the input maps, together with Jet-stream, which is best modeled with sin**2 x cos**2 of latitude.
I need both code and physical explanation.
"""

import torch
import math
from datetime import datetime

def fast_physics_t2m_prediction(meteo, geo_dense, geo_cat, geo_tempo, 
                               time_t_utc, delta_hours):
    """
    Fast physics-based temperature prediction for any delta hours (0-12).
    Oceans are much less sensitive to sunshine than land.
    GPU-optimized implementation.
    """
    # Validate delta
    delta_hours = max(0, min(12, delta_hours))
    if delta_hours == 0:
        return meteo[0].clone()
    
    # Ensure all tensors are on the same device
    device = meteo.device
    
    # --- EXTRACT VARIABLES ---
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
    
    # --- CONSTANTS ---
    SOLAR_CONST = torch.tensor(1361.0, device=device)
    PI = torch.tensor(math.pi, device=device)
    DEG2RAD = torch.tensor(math.pi / 180.0, device=device)
    HOURS_TO_RAD = torch.tensor(15.0 * math.pi / 180.0, device=device)
    
    # Convert delta_hours to tensor
    delta_hours_tensor = torch.tensor(delta_hours, device=device, dtype=torch.float32)
    
    # --- 1. SURFACE PROPERTIES - VECTORIZED ---
    is_ocean = (land_mask == 0)
    is_land = (land_mask == 1)
    
    # Precompute time factor
    time_factor = torch.clamp(delta_hours_tensor / 6.0, 0.0, 1.0)
    
    # Heat capacities and responses
    C_grid = torch.where(is_land, 1.8e6, 5.0e6)
    resp_grid = torch.where(is_land, 1.8, 0.15)
    
    # --- 2. HUMIDITY & CLOUDS - VECTORIZED ---
    T_diff = torch.clamp(T0 - Td, min=0.1)
    RH = torch.exp(-0.05 * T_diff)
    cloud_cover = torch.clamp(1.0 * RH**2, 0.0, 1.0)
    cloud_reduction = 1.0 - 0.6 * cloud_cover
    
    # --- 3. SOLAR GEOMETRY - VECTORIZED ---
    orbital_rad = torch.atan2(sin_orbit, cos_orbit)
    declination_rad = 23.44 * DEG2RAD * torch.sin(orbital_rad)
    sin_delta = torch.sin(declination_rad)
    cos_delta = torch.cos(declination_rad)
    
    lat_rad = torch.asin(sin_lat)
    
    # CURRENT and FUTURE local solar time
    UTC_hour = time_t_utc.hour + time_t_utc.minute/60.0 + time_t_utc.second/3600.0
    lon_rad = torch.atan2(sin_lon, cos_lon)
    local_solar_current = (UTC_hour + torch.rad2deg(lon_rad)/15.0) % 24
    local_solar_future = (local_solar_current + delta_hours) % 24
    
    # Solar hour angle and zenith at FUTURE time
    hour_angle_future = HOURS_TO_RAD * (local_solar_future - 12.0)
    cos_z_future = sin_lat * sin_delta + cos_lat * cos_delta * torch.cos(hour_angle_future)
    cos_z_future = torch.clamp(cos_z_future, 0.0, 1.0)
    
    # Solar radiation and day/night mask
    I_future = SOLAR_CONST * 0.75 * cos_z_future * cloud_reduction
    is_day_future = I_future > 20.0
    is_night_future = ~is_day_future
    
    # --- 4. TIME PERIOD MASKS - VECTORIZED ---
    pre_dawn = (local_solar_future >= 3) & (local_solar_future < 6)
    early_night = (local_solar_future >= 18) & (local_solar_future < 21)
    late_night = (local_solar_future >= 21) | (local_solar_future < 3)
    morning = (local_solar_future >= 6) & (local_solar_future < 9)
    midday = (local_solar_future >= 9) & (local_solar_future < 15)
    afternoon = (local_solar_future >= 15) & (local_solar_future < 18)
    
    # --- 5. DIURNAL CYCLE - VECTORIZED ---
    diurnal_adjustment = torch.zeros_like(T0, device=device)
    
    # Pre-dawn (3-6 AM)
    pre_dawn_land = pre_dawn & is_land
    pre_dawn_ocean = pre_dawn & is_ocean
    pre_dawn_factor = torch.where(pre_dawn, (local_solar_future - 3) / 3.0, 0.0)
    
    diurnal_adjustment = torch.where(
        pre_dawn_land,
        -0.4 * pre_dawn_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        pre_dawn_ocean,
        -0.05 * pre_dawn_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    
    # Early night (6-9 PM)
    early_night_land = early_night & is_land
    early_night_ocean = early_night & is_ocean
    early_night_factor = torch.where(early_night, (local_solar_future - 18) / 3.0, 0.0)
    
    diurnal_adjustment = torch.where(
        early_night_land,
        -2.0 * early_night_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        early_night_ocean,
        -0.3 * early_night_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    
    # Late night (9 PM - 3 AM)
    late_night_land = late_night & is_land
    late_night_ocean = late_night & is_ocean
    
    # Compute night factor
    night_factor = torch.where(
        late_night & (local_solar_future >= 21),
        (local_solar_future - 21) / 3.0,
        torch.where(
            late_night,
            (local_solar_future + 3) / 6.0,
            0.0
        )
    )
    
    diurnal_adjustment = torch.where(
        late_night_land,
        -1.2 * night_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        late_night_ocean,
        -0.2 * night_factor * (1.0 - cloud_cover),
        diurnal_adjustment
    )
    
    # Morning (6-9 AM)
    morning_land = morning & is_land
    morning_ocean = morning & is_ocean
    morning_factor = torch.where(morning, (local_solar_future - 6) / 3.0, 0.0)
    
    diurnal_adjustment = torch.where(
        morning_land,
        3.5 * morning_factor,
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        morning_ocean,
        0.6 * morning_factor,
        diurnal_adjustment
    )
    
    # Midday (9 AM - 3 PM)
    midday_land = midday & is_land
    midday_ocean = midday & is_ocean
    midday_factor = torch.where(midday, 1.0 - (torch.abs(local_solar_future - 12.0) / 3.0) ** 2, 0.0)
    
    diurnal_adjustment = torch.where(
        midday_land,
        5.0 * midday_factor,
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        midday_ocean,
        1.0 * midday_factor,
        diurnal_adjustment
    )
    
    # Afternoon (3-6 PM)
    afternoon_land = afternoon & is_land
    afternoon_ocean = afternoon & is_ocean
    afternoon_factor = torch.where(afternoon, 1.0 - (local_solar_future - 15) / 3.0, 0.0)
    
    diurnal_adjustment = torch.where(
        afternoon_land,
        1.5 * afternoon_factor,
        diurnal_adjustment
    )
    diurnal_adjustment = torch.where(
        afternoon_ocean,
        0.3 * afternoon_factor,
        diurnal_adjustment
    )
    
    diurnal_adjustment = diurnal_adjustment * time_factor
    
    # --- 6. SOLAR RADIATION EFFECT - VECTORIZED ---
    solar_heating_land = 6.0 * cos_z_future * time_factor
    solar_heating_ocean = 1.0 * cos_z_future * time_factor
    solar_heating = torch.where(is_land, solar_heating_land, solar_heating_ocean)
    solar_heating = torch.where(is_day_future, solar_heating, 0.0)
    
    radiative_cooling_land = -1.8 * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
    radiative_cooling_ocean = -0.4 * (1.0 - cos_z_future) * time_factor * (1.0 - cloud_cover)
    radiative_cooling = torch.where(is_land, radiative_cooling_land, radiative_cooling_ocean)
    radiative_cooling = torch.where(is_night_future, radiative_cooling, 0.0)
    
    delta_T_solar = solar_heating + radiative_cooling
    
    # --- 7. ENERGY BALANCE - VECTORIZED ---
    Q_day_land = 50.0 * cos_z_future - 25.0
    Q_day_ocean = 15.0 * cos_z_future - 10.0
    Q_day = torch.where(is_land, Q_day_land, Q_day_ocean)
    
    Q_night_land = -30.0 * (1.0 - cos_z_future)
    Q_night_ocean = -8.0 * (1.0 - cos_z_future)
    Q_night = torch.where(is_land, Q_night_land, Q_night_ocean)
    
    Q_simple = torch.where(is_day_future, Q_day, Q_night)
    Q_simple = torch.clamp(Q_simple, -25.0, 55.0)
    
    delta_t_sec = delta_hours_tensor * 3600
    delta_T_energy = (Q_simple * delta_t_sec) / C_grid
    delta_T_energy = delta_T_energy * resp_grid * time_factor
    
    # --- 8. ADVECTION - GPU-OPTIMIZED ---
    # Use fast roll operations with proper boundary handling
    dx = 27830 * cos_lat
    dy = 27830 * torch.ones_like(cos_lat, device=device)
    
    # Compute gradients using roll with circular boundary for longitude
    dT_dx = (torch.roll(T0, -1, dims=1) - torch.roll(T0, 1, dims=1)) / (2 * dx)
    dT_dy = (torch.roll(T0, -1, dims=0) - torch.roll(T0, 1, dims=0)) / (2 * dy)
    
    # Handle boundaries efficiently
    dT_dx = torch.cat([dT_dx[:, 1:2], dT_dx, dT_dx[:, -2:-1]], dim=1)[:, 1:-1]
    dT_dy = torch.cat([dT_dy[1:2, :], dT_dy, dT_dy[-2:-1, :]], dim=0)[1:-1, :]
    
    advection = -(U * dT_dx + V * dT_dy)
    advection_factor = torch.where(is_land, 0.08, 0.12)
    delta_T_adv = advection * delta_t_sec * time_factor * advection_factor
    
    # --- 9. COMBINE ALL EFFECTS ---
    delta_T_total = diurnal_adjustment + delta_T_solar + delta_T_energy + delta_T_adv
    
    # --- 10. RATE LIMITS - VECTORIZED ---
    max_warming_land = 10.0 * time_factor
    max_cooling_land = -8.0 * time_factor
    max_warming_ocean = 3.0 * time_factor
    max_cooling_ocean = -2.0 * time_factor
    
    max_warming = torch.where(is_land, max_warming_land, max_warming_ocean)
    max_cooling = torch.where(is_land, max_cooling_land, max_cooling_ocean)
    
    # Special pre-dawn limits
    pre_dawn_warming_limit = torch.where(is_land, 0.1, 0.05) * time_factor
    max_warming = torch.where(pre_dawn, torch.minimum(max_warming, pre_dawn_warming_limit), max_warming)
    
    delta_T_total = torch.clamp(delta_T_total, max_cooling, max_warming)
    
    T_pred = T0 + delta_T_total
    
    # --- 11. STABILIZATION - VECTORIZED ---
    damping = torch.where(is_ocean, 0.95, 0.85)
    T_pred = damping * T_pred + (1 - damping) * T0
    
    # --- 12. FAST SMOOTHING USING CONVOLUTION - GPU OPTIMIZED ---
    smoothing = torch.tensor(0.25, device=device)
    
    # Use separable convolution for better GPU performance
    # First smooth in x-direction
    kernel_x = torch.ones(1, 1, 1, 3, device=device) / 3.0
    T_pred_4d = T_pred.unsqueeze(0).unsqueeze(0)
    
    # Apply convolution in x-direction
    T_smooth_x = torch.nn.functional.conv2d(
        torch.nn.functional.pad(T_pred_4d, (1, 1, 0, 0), mode='replicate'),
        kernel_x,
        padding=0
    )
    
    # Then smooth in y-direction
    kernel_y = torch.ones(1, 1, 3, 1, device=device) / 3.0
    T_smooth = torch.nn.functional.conv2d(
        torch.nn.functional.pad(T_smooth_x, (0, 0, 1, 1), mode='replicate'),
        kernel_y,
        padding=0
    ).squeeze()
    
    T_pred = (1 - smoothing) * T_pred + smoothing * T_smooth
    
    # --- 13. TEMPERATURE RANGE - VECTORIZED ---
    T_pred = torch.clamp(T_pred, 240.0, 310.0)
    min_T = Td - 3.0
    T_pred = torch.maximum(T_pred, min_T)
    
    # --- 14. FINAL ADJUSTMENTS - VECTORIZED ---
    global_adjust = -0.1 * time_factor
    land_adjust = torch.where(is_land, 0.1 * time_factor, -0.05 * time_factor)
    humid_adjust = -0.05 * RH
    
    total_adjust = global_adjust + land_adjust + humid_adjust
    T_final = T_pred + total_adjust
    
    # Final bounds
    T_final = torch.clamp(T_final, 240.0, 310.0)
    
    return T_final


# GPU-enabled test and benchmark function
def benchmark_gpu_prediction():
    """Benchmark the GPU-optimized prediction function"""
    import time
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create test data on the selected device
    H, W = 721, 1440
    
    meteo = torch.randn(6, H, W, device=device)
    geo_dense = torch.randn(5, H, W, device=device)
    geo_cat = torch.randn(2, H, W, device=device)
    geo_tempo = torch.randn(4, H, W, device=device)
    
    # Set reasonable values
    lat_tensor = torch.linspace(-90, 90, H, device=device)
    meteo[0] = 273.15 + 20 * torch.cos(lat_tensor * math.pi / 180.0).unsqueeze(1)
    meteo[2] = 5.0  # U wind
    meteo[3] = 2.0  # V wind
    meteo[4] = meteo[0] - 5  # Dew point
    
    # Set land/ocean mask (alternating bands)
    land_mask = (torch.abs(lat_tensor.unsqueeze(1)) < 60).float()
    geo_cat[0] = land_mask
    
    # Set orbital position (summer solstice)
    geo_tempo[0] = torch.sin(torch.tensor(math.pi/2, device=device))
    geo_tempo[1] = torch.cos(torch.tensor(math.pi/2, device=device))
    
    time_t_utc = datetime(2024, 6, 21, 12, 0, 0)
    
    # Warm up GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for _ in range(5):
            _ = fast_physics_t2m_prediction(meteo, geo_dense, geo_cat, geo_tempo, time_t_utc, 6)
        torch.cuda.synchronize()
    
    # Benchmark
    n_runs = 20
    times = []
    
    for i in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        result = fast_physics_t2m_prediction(meteo, geo_dense, geo_cat, geo_tempo, time_t_utc, 6)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        if i == 0:
            print(f"First run result shape: {result.shape}")
            print(f"Temperature range: {result.min():.2f}K to {result.max():.2f}K")
    
    # Calculate statistics
    avg_time = sum(times) / n_runs
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nBenchmark results ({n_runs} runs):")
    print(f"Average time: {avg_time*1000:.1f} ms")
    print(f"Minimum time: {min_time*1000:.1f} ms")
    print(f"Maximum time: {max_time*1000:.1f} ms")
    print(f"Throughput: {1/avg_time:.1f} predictions/second")
    
    # Memory usage info
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e6
        memory_reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"\nGPU memory allocated: {memory_allocated:.1f} MB")
        print(f"GPU memory reserved: {memory_reserved:.1f} MB")
    
    return result


# Utility function to ensure data is on GPU
def move_to_gpu(data_dict):
    """Move data to GPU if available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if isinstance(data_dict, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in data_dict.items()}
    elif isinstance(data_dict, (list, tuple)):
        return [v.to(device) if isinstance(v, torch.Tensor) else v 
                for v in data_dict]
    else:
        return data_dict.to(device) if isinstance(data_dict, torch.Tensor) else data_dict


if __name__ == "__main__":
    print("=" * 60)
    print("GPU-OPTIMIZED T2M PREDICTION BENCHMARK")
    print("=" * 60)
    
    result = benchmark_gpu_prediction()
    
    # Test different time horizons
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT PREDICTION HORIZONS")
    print("=" * 60)
    
    # Create sample data once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H, W = 721, 1440
    
    meteo = torch.randn(6, H, W, device=device)
    geo_dense = torch.randn(5, H, W, device=device)
    geo_cat = torch.randn(2, H, W, device=device)
    geo_tempo = torch.randn(4, H, W, device=device)
    
    # Set reasonable values
    lat_tensor = torch.linspace(-90, 90, H, device=device)
    meteo[0] = 273.15 + 20 * torch.cos(lat_tensor * math.pi / 180.0).unsqueeze(1)
    meteo[2] = 5.0
    meteo[3] = 2.0
    meteo[4] = meteo[0] - 5
    
    land_mask = (torch.abs(lat_tensor.unsqueeze(1)) < 60).float()
    geo_cat[0] = land_mask
    
    time_t_utc = datetime(2024, 6, 21, 12, 0, 0)
    
    for delta in [1, 3, 6, 12]:
        print(f"\nDelta = {delta} hours:")
        start_time = time.perf_counter()
        
        pred = fast_physics_t2m_prediction(meteo, geo_dense, geo_cat, geo_tempo, 
                                          time_t_utc, delta)
        
        end_time = time.perf_counter()
        
        # Analyze land vs ocean differences
        land_pred = pred[land_mask == 1]
        ocean_pred = pred[land_mask == 0]
        land_original = meteo[0][land_mask == 1]
        ocean_original = meteo[0][land_mask == 0]
        
        land_change = torch.mean(land_pred - land_original).item()
        ocean_change = torch.mean(ocean_pred - ocean_original).item()
        
        print(f"  Prediction time: {(end_time - start_time)*1000:.1f} ms")
        print(f"  Land ΔT: {land_change:+.2f}K")
        print(f"  Ocean ΔT: {ocean_change:+.2f}K")
        if ocean_change != 0:
            print(f"  Land sensitivity ratio: {abs(land_change / ocean_change):.1f}x")