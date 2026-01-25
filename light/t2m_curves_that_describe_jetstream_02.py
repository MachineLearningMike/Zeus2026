import numpy as np
import matplotlib.pyplot as plt

# Set up latitude array
latitudes = np.linspace(-90, 90, 361)
phi = np.radians(latitudes)
abs_phi = np.radians(np.abs(latitudes))

# Define curves that peak at 45°
curves = {}

# 1. Original (peaks at ~39.2°)
curves['cos³(|ϕ|)·sin²(ϕ) (peaks ~39.2°)'] = (np.cos(abs_phi)**3) * (np.sin(phi)**2)

# 2. Equal exponent m=2 (peaks at 45°)
curves['cos²(|ϕ|)·sin²(ϕ) (peaks 45°)'] = (np.cos(abs_phi)**2) * (np.sin(phi)**2)

# 3. Equal exponent m=3 (peaks at 45°)
curves['cos³(|ϕ|)·|sin(ϕ)|³ (peaks 45°)'] = (np.cos(abs_phi)**3) * (np.abs(np.sin(phi))**3)

# 4. Gaussian centered at ±45°
sigma = np.radians(12)
gaussian_nh = np.exp(-(phi - np.radians(45))**2 / (2 * sigma**2))
gaussian_sh = np.exp(-(phi + np.radians(45))**2 / (2 * sigma**2))
curves['Gaussian ±45°'] = np.maximum(gaussian_nh, gaussian_sh)

# 5. Modified 1-cos curve peaking at 45°
curves['1 - cos(2|ϕ|) (peaks 45°)'] = 1 - np.cos(2 * abs_phi)

# 6. sin(2ϕ) form (peaks at 45°)
curves['|sin(2ϕ)| (peaks 45°)'] = np.abs(np.sin(2 * phi))

# Normalize all curves
for name in curves:
    if np.max(np.abs(curves[name])) > 0:
        curves[name] = curves[name] / np.max(np.abs(curves[name]))

# Create comparison plot
plt.figure(figsize=(12, 8))

# Highlight 45° lines
plt.axvline(x=45, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
           label='Jet stream peak (45°)')
plt.axvline(x=-45, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
plt.axvline(x=39.2, color='blue', linestyle=':', alpha=0.5, linewidth=1.5,
           label='Original formula peak (39.2°)')
plt.axvline(x=-39.2, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)

# Plot each curve
colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
for idx, (name, curve) in enumerate(curves.items()):
    plt.plot(latitudes, curve, label=name, color=colors[idx], linewidth=2.5)

plt.xlabel('Latitude (°)', fontsize=14)
plt.ylabel('Normalized Influence', fontsize=14)
plt.title('Jet Stream Influence Curves with 45° Peak Correction', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.legend(loc='upper right', fontsize=10)
plt.xlim(-90, 90)
plt.ylim(-0.05, 1.05)

# Add text annotation
plt.text(45, 0.02, '45°', ha='center', color='red', fontsize=10, fontweight='bold')
plt.text(39.2, 0.02, '39.2°', ha='center', color='blue', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Print peak analysis
print("="*70)
print("PEAK LATITUDE ANALYSIS")
print("="*70)

for name, curve in curves.items():
    # Find positive peak
    pos_mask = latitudes >= 0
    pos_lats = latitudes[pos_mask]
    pos_curve = curve[pos_mask]
    
    # Find peak (handle possible multiple maxima)
    peak_value = np.max(pos_curve)
    peak_indices = np.where(pos_curve == peak_value)[0]
    
    if len(peak_indices) > 0:
        peak_lat = pos_lats[peak_indices[0]]
        # Check if flat peak
        if len(peak_indices) > 1:
            peak_range = f"[{pos_lats[peak_indices[0]]:.1f}°-{pos_lats[peak_indices[-1]]:.1f}°]"
            print(f"{name:40s} Peak: {peak_range}")
        else:
            print(f"{name:40s} Peak: {peak_lat:.1f}°")
        
        # Calculate width at half maximum
        half_max = peak_value / 2
        above_half = pos_curve >= half_max
        if np.any(above_half):
            half_min = pos_lats[above_half].min()
            half_max_lat = pos_lats[above_half].max()
            width = half_max_lat - half_min
            print(f"{'':40s} FWHM: {width:.1f}° ({half_min:.1f}° to {half_max_lat:.1f}°)")
    print("-"*70)