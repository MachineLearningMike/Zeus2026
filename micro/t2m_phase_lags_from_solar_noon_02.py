import numpy as np
import matplotlib.pyplot as plt

# Simulate accurate diurnal cycles
hours = np.linspace(0, 24, 241)
solar_noon = 12

# Solar radiation
solar_rad = np.maximum(0, 1000 * np.cos(np.pi * (hours - solar_noon) / 12))

def simulate_diurnal(surface_type, amplitude, lag_hours, base_temp=20):
    """Simulate diurnal temperature cycle."""
    # Phase shift
    shifted_hours = (hours - solar_noon - lag_hours) % 24
    solar_shifted = np.maximum(0, 1000 * np.cos(np.pi * shifted_hours / 12))
    
    # Temperature response (with some smoothing)
    t2m = base_temp + amplitude * solar_shifted / np.max(solar_shifted)
    
    # Add some realistic noise/features
    if surface_type == 'ocean':
        # Ocean: smoother, smaller amplitude
        from scipy.ndimage import gaussian_filter1d
        t2m = gaussian_filter1d(t2m, sigma=5)
    elif surface_type == 'desert':
        # Desert: sharper peak
        t2m = t2m * (1 + 0.1 * np.sin(np.pi * hours / 6))
    
    return t2m

# Generate curves
t2m_ocean = simulate_diurnal('ocean', amplitude=1.5, lag_hours=0.5, base_temp=18)
t2m_desert = simulate_diurnal('desert', amplitude=25, lag_hours=2.5, base_temp=20)
t2m_forest = simulate_diurnal('forest', amplitude=15, lag_hours=4.0, base_temp=19)

# Plot
plt.figure(figsize=(14, 8))

# Plot 1: Temperature comparison
plt.subplot(2, 2, 1)
plt.plot(hours, solar_rad/50, 'y-', alpha=0.5, label='Solar Radiation (scaled)', linewidth=2)
plt.plot(hours, t2m_ocean, 'blue', linewidth=3, label=f'Ocean (lag: 0.5h, ΔT: {np.ptp(t2m_ocean):.1f}°C)')
plt.plot(hours, t2m_desert, 'orange', linewidth=2, label=f'Desert (lag: 2.5h, ΔT: {np.ptp(t2m_desert):.1f}°C)')
plt.plot(hours, t2m_forest, 'green', linewidth=2, label=f'Forest (lag: 4.0h, ΔT: {np.ptp(t2m_forest):.1f}°C)')

plt.axvline(x=12, color='gold', linestyle='--', alpha=0.7, label='Solar Noon')
plt.xlabel('Hour of Day')
plt.ylabel('Temperature (°C)')
plt.title('Diurnal Temperature Cycles: Ocean vs Land')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.xlim(6, 20)

# Plot 2: Lag comparison
plt.subplot(2, 2, 2)
surfaces = ['Ocean', 'Desert', 'Forest']
lags = [0.5, 2.5, 4.0]
amplitudes = [np.ptp(t2m_ocean), np.ptp(t2m_desert), np.ptp(t2m_forest)]

bars1 = plt.bar(surfaces, lags, color=['blue', 'orange', 'green'], alpha=0.7)
plt.ylabel('Phase Lag (hours after solar noon)')
plt.title('Phase Lag Comparison')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, lag in zip(bars1, lags):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{lag}h', ha='center', va='bottom')

# Plot 3: Amplitude comparison
plt.subplot(2, 2, 3)
bars2 = plt.bar(surfaces, amplitudes, color=['blue', 'orange', 'green'], alpha=0.7)
plt.ylabel('Diurnal Amplitude (°C)')
plt.title('Diurnal Amplitude Comparison')
plt.grid(True, alpha=0.3, axis='y')

for bar, amp in zip(bars2, amplitudes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{amp:.1f}°C', ha='center', va='bottom')

# Plot 4: Lag vs Amplitude trade-off
plt.subplot(2, 2, 4)
plt.scatter(lags, amplitudes, s=200, c=['blue', 'orange', 'green'], alpha=0.8)

# Label points
for i, (lag, amp, surf) in enumerate(zip(lags, amplitudes, surfaces)):
    plt.text(lag + 0.1, amp + 0.1, surf, fontsize=10, fontweight='bold')

plt.xlabel('Phase Lag (hours)')
plt.ylabel('Diurnal Amplitude (°C)')
plt.title('Trade-off: Lag vs Amplitude')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Quantitative summary
print("="*70)
print("OCEAN DIURNAL CYCLE: CORRECTED SUMMARY")
print("="*70)
print("Property              | Ocean          | Desert         | Forest")
print("-"*70)
print(f"Phase Lag (hours)    | {lags[0]:5.1f}         | {lags[1]:5.1f}         | {lags[2]:5.1f}")
print(f"Diurnal Amplitude (°C)| {amplitudes[0]:5.1f}         | {amplitudes[1]:5.1f}         | {amplitudes[2]:5.1f}")
print(f"Peak Time            | {hours[np.argmax(t2m_ocean)]:5.1f}h       | {hours[np.argmax(t2m_desert)]:5.1f}h       | {hours[np.argmax(t2m_forest)]:5.1f}h")
print(f"Min-Max Range        | {np.min(t2m_ocean):.1f}-{np.max(t2m_ocean):.1f}°C | {np.min(t2m_desert):.1f}-{np.max(t2m_desert):.1f}°C | {np.min(t2m_forest):.1f}-{np.max(t2m_forest):.1f}°C")

print("\n" + "="*70)
print("PHYSICAL EXPLANATION")
print("="*70)
print("Ocean has MINIMAL LAG because:")
print("1. Vertical mixing distributes heat INSTANTLY through surface layer")
print("2. High thermal conductivity → rapid heat transfer")
print("3. Evaporation provides immediate cooling mechanism")
print("\nOcean has SMALL AMPLITUDE because:")
print("1. High heat capacity → needs 4× more energy than land for same ΔT")
print("2. Deep penetration of solar radiation (meters vs centimeters for land)")
print("3. Latent heat flux consumes energy without raising temperature")