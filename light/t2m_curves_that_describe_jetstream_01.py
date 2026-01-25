import numpy as np
import matplotlib.pyplot as plt

# Set up latitude array
latitudes = np.linspace(-90, 90, 361)  # -90 to 90 degrees, 0.5-degree resolution
phi = np.radians(latitudes)  # Convert to radians
abs_phi = np.radians(np.abs(latitudes))  # Absolute latitude in radians

# Define all symmetric curves
curves = {}

# 1. Your discovery (made symmetric)
curves['1 - cos(|ϕ|)'] = 1 - np.cos(abs_phi)

# 2. Proposed symmetric theoretical form
curves['cos³(|ϕ|)·sin²(ϕ)'] = (np.cos(abs_phi)**3) * (np.sin(phi)**2)

# 3. Absolute sin·cos²
curves['|sin(ϕ)|·cos²(ϕ)'] = np.abs(np.sin(phi)) * (np.cos(phi)**2)

# 4. sin²·cos²
curves['sin²(ϕ)·cos²(ϕ)'] = (np.sin(phi)**2) * (np.cos(phi)**2)

# 5. Gaussian twin peaks (symmetric)
phi0 = np.radians(50)  # Peak latitude
sigma = np.radians(15)  # Width
gaussian_nh = np.exp(-(phi - phi0)**2 / (2 * sigma**2))
gaussian_sh = np.exp(-(phi + phi0)**2 / (2 * sigma**2))
curves['Gaussian twin peaks'] = np.maximum(gaussian_nh, gaussian_sh)

# 6. For comparison: Original problematic sin(ϕ)·cos²(ϕ) (NOT symmetric)
curves['sin(ϕ)·cos²(ϕ) (NOT symmetric)'] = np.sin(phi) * (np.cos(phi)**2)

# Normalize all curves to have max = 1 for easier comparison
for name in curves:
    if np.max(np.abs(curves[name])) > 0:
        curves[name] = curves[name] / np.max(np.abs(curves[name]))

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Symmetric Jet Stream Influence Curves for t2m Prediction', 
             fontsize=16, fontweight='bold')

# Color palette
colors = plt.cm.Set2(np.linspace(0, 1, len(curves)))

# Plot 1: All curves together
ax = axes[0, 0]
for idx, (name, curve) in enumerate(curves.items()):
    ax.plot(latitudes, curve, label=name, color=colors[idx], linewidth=2.5)
ax.set_xlabel('Latitude (°)', fontsize=12)
ax.set_ylabel('Normalized Weight', fontsize=12)
ax.set_title('All Curves Comparison', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-90, 90)

# Plot 2: Focus on your requested curve
ax = axes[0, 1]
requested_curve = curves['cos³(|ϕ|)·sin²(ϕ)']
ax.plot(latitudes, requested_curve, color='darkred', linewidth=3, 
        label='cos³(|ϕ|)·sin²(ϕ)')
ax.fill_between(latitudes, 0, requested_curve, alpha=0.3, color='darkred')
ax.set_xlabel('Latitude (°)', fontsize=12)
ax.set_ylabel('Normalized Weight', fontsize=12)
ax.set_title('Your Requested Curve: cos³(|ϕ|)·sin²(ϕ)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
ax.legend(loc='upper right')
ax.set_xlim(-90, 90)

# Add critical latitude markers
for lat in [30, 45, 60]:
    ax.axvline(x=lat, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=-lat, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(lat, 0.05, f'{lat}°', ha='center', fontsize=8, color='gray')
    ax.text(-lat, 0.05, f'{-lat}°', ha='center', fontsize=8, color='gray')

# Plot 3: Symmetric vs Non-symmetric comparison
ax = axes[1, 0]
symmetric_curves = ['cos³(|ϕ|)·sin²(ϕ)', '|sin(ϕ)|·cos²(ϕ)', 'sin²(ϕ)·cos²(ϕ)']
non_symmetric = 'sin(ϕ)·cos²(ϕ) (NOT symmetric)'

for name in symmetric_curves:
    ax.plot(latitudes, curves[name], label=name, linewidth=2.5)
ax.plot(latitudes, curves[non_symmetric], 'k--', label=non_symmetric, 
        linewidth=2, alpha=0.7)
ax.set_xlabel('Latitude (°)', fontsize=12)
ax.set_ylabel('Normalized Weight', fontsize=12)
ax.set_title('Symmetric vs Non-Symmetric Curves', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-90, 90)

# Plot 4: Peak latitude analysis
ax = axes[1, 1]
peak_lats = []
curve_names = []

for name, curve in curves.items():
    if name == 'sin(ϕ)·cos²(ϕ) (NOT symmetric)':
        # Handle separately for NH and SH
        nh_curve = curve[latitudes >= 0]
        nh_lats = latitudes[latitudes >= 0]
        sh_curve = curve[latitudes <= 0]
        sh_lats = latitudes[latitudes <= 0]
        
        nh_peak_lat = nh_lats[np.argmax(np.abs(nh_curve))]
        sh_peak_lat = sh_lats[np.argmax(np.abs(sh_curve))]
        peak_lats.append((nh_peak_lat, sh_peak_lat))
        curve_names.append(f'{name}\n(NH:{nh_peak_lat:.1f}°, SH:{sh_peak_lat:.1f}°)')
    else:
        # For symmetric curves, find positive peak
        pos_lats = latitudes[latitudes >= 0]
        pos_curve = curve[latitudes >= 0]
        peak_lat = pos_lats[np.argmax(pos_curve)]
        peak_lats.append((peak_lat, -peak_lat))
        curve_names.append(f'{name}\n(Peak: ±{peak_lat:.1f}°)')

# Create bar chart of peak latitudes
bar_width = 0.6
x_pos = np.arange(len(curves))
nh_peaks = [p[0] if isinstance(p, tuple) else p for p in peak_lats]

bars = ax.barh(x_pos, nh_peaks, bar_width, color=colors)
ax.set_yticks(x_pos)
ax.set_yticklabels(curve_names, fontsize=8)
ax.set_xlabel('Peak Latitude (°)', fontsize=12)
ax.set_title('Peak Latitude of Each Curve', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=45, color='red', linestyle='--', alpha=0.5, 
           label='Typical Jet Stream Max (~45°)')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 90)

# Add value labels on bars
for bar, peak in zip(bars, nh_peaks):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{peak:.1f}°', va='center', fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Additional analysis: Print key metrics
print("="*60)
print("KEY METRICS FOR cos³(|ϕ|)·sin²(ϕ) CURVE:")
print("="*60)

curve = curves['cos³(|ϕ|)·sin²(ϕ)']
# Find latitudes where curve > 0.5 (effective influence region)
effective_lats = latitudes[curve > 0.5]
if len(effective_lats) > 0:
    print(f"Effective influence region (>50% of max):")
    print(f"  NH: {effective_lats[effective_lats > 0].min():.1f}° to "
          f"{effective_lats[effective_lats > 0].max():.1f}°")
    print(f"  SH: {effective_lats[effective_lats < 0].min():.1f}° to "
          f"{effective_lats[effective_lats < 0].max():.1f}°")

# Calculate area under curve (proxy for total influence)
dx = np.radians(0.5)  # spacing in radians
area_nh = np.trapz(curve[latitudes >= 0], dx=dx)
area_sh = np.trapz(curve[latitudes <= 0], dx=dx)
print(f"\nArea under curve (normalized):")
print(f"  NH: {area_nh:.3f}")
print(f"  SH: {area_sh:.3f}")
print(f"  Symmetry ratio (|NH/SH|): {abs(area_nh/area_sh):.6f}")

# Peak information
peak_idx = np.argmax(curve[latitudes >= 0])
peak_lat = latitudes[latitudes >= 0][peak_idx]
print(f"\nPeak latitude: ±{peak_lat:.1f}°")
print(f"Value at equator: {curve[latitudes == 0][0]:.4f}")
print(f"Value at 45°: {curve[latitudes == 45][0]:.4f}")
print(f"Value at 60°: {curve[latitudes == 60][0]:.4f}")

plt.show()

# Save figure
plt.savefig('jet_stream_influence_curves.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as 'jet_stream_influence_curves.png'")