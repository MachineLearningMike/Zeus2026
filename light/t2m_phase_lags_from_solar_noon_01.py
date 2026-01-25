import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Simulate diurnal cycle
hours = np.linspace(0, 24, 241)  # 6-minute intervals
solar_noon = 12  # Solar noon hour

# 1. Solar radiation (simplified)
solar_rad = np.maximum(0, 1000 * np.cos(np.pi * (hours - solar_noon) / 12))

# 2. Different surface responses (with different lags)
def surface_temperature_response(solar_input, lag_hours, damping=0.7):
    """Simulate temperature response with lag."""
    # Simple energy balance model
    C = 1.0  # Heat capacity
    response = np.zeros_like(solar_input)
    
    # Initial condition
    T = 20  # Base temperature
    
    for i in range(len(solar_input)):
        # Net heating depends on solar minus outgoing (proportional to T)
        net_heating = solar_input[i] - damping * T
        T += net_heating / C * 0.1  # Time step factor
        response[i] = T
    
    # Apply lag by shifting
    lag_samples = int(lag_hours * 10)  # 6-min intervals
    response = np.roll(response, lag_samples)
    response[:lag_samples] = response[lag_samples]  # Handle boundary
    
    # Normalize
    response = (response - np.min(response)) / (np.max(response) - np.min(response)) * 30 + 15
    
    return response

# Generate different surface responses
t2m_desert = surface_temperature_response(solar_rad, lag_hours=2.5, damping=0.8)
t2m_vegetated = surface_temperature_response(solar_rad, lag_hours=3.5, damping=0.6)
t2m_urban = surface_temperature_response(solar_rad, lag_hours=4.0, damping=0.5)
t2m_snow = surface_temperature_response(solar_rad, lag_hours=5.0, damping=0.4)

# Plot
plt.figure(figsize=(12, 8))

# Plot solar radiation
plt.plot(hours, solar_rad/50, 'y-', linewidth=3, alpha=0.7, label='Solar Radiation (scaled)')

# Plot temperature responses
plt.plot(hours, t2m_desert, 'orange', linewidth=2.5, label='Desert (lag ≈ 2.5h)')
plt.plot(hours, t2m_vegetated, 'green', linewidth=2.5, label='Vegetated (lag ≈ 3.5h)')
plt.plot(hours, t2m_urban, 'red', linewidth=2.5, label='Urban (lag ≈ 4h)')
plt.plot(hours, t2m_snow, 'blue', linewidth=2.5, label='Snow (lag ≈ 5h)')

# Mark solar noon
plt.axvline(x=solar_noon, color='gold', linestyle='--', alpha=0.7, linewidth=2, 
           label='Solar Noon')

# Mark peak temperatures
for t2m, color, name in [(t2m_desert, 'orange', 'Desert'),
                         (t2m_vegetated, 'green', 'Vegetated'),
                         (t2m_urban, 'red', 'Urban'),
                         (t2m_snow, 'blue', 'Snow')]:
    peak_hour = hours[np.argmax(t2m)]
    peak_temp = np.max(t2m)
    plt.plot(peak_hour, peak_temp, 'o', color=color, markersize=10)
    plt.text(peak_hour, peak_temp + 0.5, f'{peak_hour:.1f}h', 
             ha='center', color=color, fontweight='bold')

plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Temperature (°C)', fontsize=14)
plt.title('Diurnal Cycle: Phase Lag Between Solar Noon and Peak t2m', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(6, 20)
plt.ylim(15, 45)

# Add phase lag diagram
plt.text(12, 16, 'Solar Radiation Peak', ha='center', color='gold', fontweight='bold')
plt.text(14.5, 16, 'Temperature Peak', ha='center', color='black', fontweight='bold')

plt.tight_layout()
plt.show()

# Quantitative analysis
print("="*70)
print("DIURNAL CYCLE PHASE LAG ANALYSIS")
print("="*70)
print("Surface Type      | Peak Hour | Lag (hours) | Amplitude (°C)")
print("-"*70)

surfaces = [
    ('Desert', t2m_desert, 2.5),
    ('Vegetated', t2m_vegetated, 3.5),
    ('Urban', t2m_urban, 4.0),
    ('Snow', t2m_snow, 5.0)
]

for name, t2m, expected_lag in surfaces:
    peak_idx = np.argmax(t2m)
    peak_hour = hours[peak_idx]
    lag = peak_hour - solar_noon
    amplitude = np.max(t2m) - np.min(t2m)
    print(f"{name:15s} | {peak_hour:9.1f} | {lag:11.1f} | {amplitude:13.1f}")

print("\n" + "="*70)
print("KEY FACTORS AFFECTING LAG:")
print("="*70)
print("1. Heat Capacity: Higher → Longer lag")
print("2. Albedo: Higher → Shorter lag (less absorption)")
print("3. Moisture Availability: More → Longer lag (evapotranspiration)")
print("4. Cloud Cover: More → Shorter lag (diffuse radiation)")
print("5. Wind Speed: Higher → Shorter lag (enhanced mixing)")