import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define the complete response matrix based on (lsm, slt)
def get_diurnal_characteristics(lsm, slt, soil_moisture=0.3):
    """
    Return diurnal cycle characteristics based on ERA5 land-sea mask and soil type.
    
    Parameters:
    -----------
    lsm : float (0-1)
        Land-sea mask (0=ocean, 1=land)
    slt : int (1-7 or np.nan)
        Soil type (only valid for lsm > 0)
    soil_moisture : float (0-1)
        Volumetric soil moisture
    """
    
    # Ocean characteristics (lsm = 0)
    if lsm == 0.0:
        return {
            'lag_hours': 0.5 + 0.5 * np.random.random(),  # 0.5-1.0 hours
            'amplitude': 0.5 + 1.5 * np.random.random(),  # 0.5-2.0°C
            'thermal_inertia': 2500,  # Very high
            'type': 'ocean'
        }
    
    # Coastal mixed cells (0 < lsm < 1)
    elif 0 < lsm < 1:
        # Blend ocean and land characteristics
        ocean_char = get_diurnal_characteristics(0.0, None, soil_moisture)
        land_char = get_diurnal_characteristics(1.0, slt, soil_moisture)
        
        return {
            'lag_hours': lsm * land_char['lag_hours'] + (1-lsm) * ocean_char['lag_hours'],
            'amplitude': lsm * land_char['amplitude'] + (1-lsm) * ocean_char['amplitude'],
            'thermal_inertia': lsm * land_char['thermal_inertia'] + (1-lsm) * ocean_char['thermal_inertia'],
            'type': f'coastal_{lsm:.2f}'
        }
    
    # Pure land (lsm = 1.0)
    elif lsm == 1.0:
        # Base soil properties (from HTESSEL)
        soil_props = {
            1: {'lambda_base': 0.3, 'C_base': 1.2e6, 'name': 'Sand'},      # Fast
            2: {'lambda_base': 0.25, 'C_base': 1.3e6, 'name': 'Loam_Sand'},
            3: {'lambda_base': 0.2, 'C_base': 1.4e6, 'name': 'Sandy_Loam'},
            4: {'lambda_base': 0.15, 'C_base': 1.5e6, 'name': 'Silt_Loam'},
            5: {'lambda_base': 0.1, 'C_base': 1.6e6, 'name': 'Clay'},      # Slow
            6: {'lambda_base': 0.05, 'C_base': 0.8e6, 'name': 'Peat'},     # Very slow
            7: {'lambda_base': 2.2, 'C_base': 1.9e6, 'name': 'Ice'}        # Special case
        }
        
        props = soil_props.get(slt, soil_props[3])  # Default to type 3
        
        # Adjust for moisture
        lambda_eff = props['lambda_base'] * (1 + 3*soil_moisture)
        C_eff = props['C_base'] * (1 + 4*soil_moisture)
        thermal_inertia = np.sqrt(C_eff * lambda_eff)
        
        # Empirical relationships
        if slt == 7:  # Ice
            lag_hours = 6.0 + 2.0 * (1 - soil_moisture)  # Ice: 6-8 hours
            amplitude = 5.0 + 10.0 * soil_moisture       # Ice: 5-15°C
        else:
            lag_hours = 12.0 / thermal_inertia * (1 + 2*soil_moisture)  # 2-6 hours
            amplitude = 30.0 / (1 + C_eff/1e6) * (1 - 0.5*soil_moisture)  # 8-25°C
        
        return {
            'lag_hours': lag_hours,
            'amplitude': amplitude,
            'thermal_inertia': thermal_inertia,
            'type': f'land_{props["name"]}'
        }
    
    else:
        raise ValueError(f"Invalid lsm value: {lsm}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Ocean-Land continuum
ax = axes[0, 0]
lsm_values = np.linspace(0, 1, 11)
colors = plt.cm.Blues_r(np.linspace(0.2, 1, len(lsm_values)))

for lsm, color in zip(lsm_values, colors):
    if lsm == 0:
        char = get_diurnal_characteristics(lsm, None, 0.0)
        label = 'Ocean'
    elif lsm == 1:
        char = get_diurnal_characteristics(lsm, 3, 0.3)  # Type 3 soil
        label = 'Land (Sandy Loam)'
    else:
        char = get_diurnal_characteristics(lsm, 3, 0.3)
        label = f'Coastal (lsm={lsm:.1f})'
    
    ax.bar(lsm, char['lag_hours'], width=0.08, color=color, alpha=0.7)
    ax.text(lsm, char['lag_hours'] + 0.1, f'{char["lag_hours"]:.1f}h', 
            ha='center', fontsize=8)

ax.set_xlabel('Land-Sea Mask (0=Ocean, 1=Land)')
ax.set_ylabel('Phase Lag (hours)')
ax.set_title('ERA5: Phase Lag vs Land-Sea Mask\n(Soil Type 3, 30% moisture)')
ax.grid(True, alpha=0.3)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(['Ocean (0.0)', 'Coastal (0.5)', 'Land (1.0)'])

# 2. Soil type matrix
ax = axes[0, 1]
soil_types = range(1, 8)
moistures = [0.1, 0.3, 0.5, 0.7]

# Create heatmap data
lag_matrix = np.zeros((len(soil_types), len(moistures)))
amp_matrix = np.zeros((len(soil_types), len(moistures)))

for i, slt in enumerate(soil_types):
    for j, moisture in enumerate(moistures):
        char = get_diurnal_characteristics(1.0, slt, moisture)
        lag_matrix[i, j] = char['lag_hours']
        amp_matrix[i, j] = char['amplitude']

# Plot lag heatmap
im = ax.imshow(lag_matrix, cmap='viridis', aspect='auto')
ax.set_xlabel('Soil Moisture')
ax.set_ylabel('Soil Type')
ax.set_title('Phase Lag Heatmap (hours)\nby Soil Type & Moisture')
ax.set_xticks(range(len(moistures)))
ax.set_xticklabels([f'{m:.0%}' for m in moistures])
ax.set_yticks(range(len(soil_types)))
ax.set_yticklabels([f'Type {s}' for s in soil_types])
plt.colorbar(im, ax=ax, label='Lag (hours)')

# Add text annotations
for i in range(len(soil_types)):
    for j in range(len(moistures)):
        ax.text(j, i, f'{lag_matrix[i, j]:.1f}', 
                ha='center', va='center', color='white', fontsize=8)

# 3. Coastal transition zones
ax = axes[0, 2]
coastal_lsms = [0.1, 0.3, 0.5, 0.7, 0.9]
soil_for_coast = 3  # Medium soil

for lsm in coastal_lsms:
    # Simulate diurnal cycle
    hours = np.linspace(0, 24, 241)
    solar_noon = 12
    solar_rad = np.maximum(0, 1000 * np.cos(np.pi * (hours - solar_noon) / 12))
    
    # Get characteristics
    char = get_diurnal_characteristics(lsm, soil_for_coast, 0.3)
    
    # Generate temperature curve
    phase_shifted = (hours - solar_noon - char['lag_hours']) % 24
    solar_shifted = np.maximum(0, 1000 * np.cos(np.pi * phase_shifted / 12))
    t2m = 20 + char['amplitude'] * solar_shifted / 1000
    
    ax.plot(hours, t2m, label=f'lsm={lsm:.1f} (lag={char["lag_hours"]:.1f}h)')

ax.set_xlabel('Hour of Day')
ax.set_ylabel('t2m (°C)')
ax.set_title('Coastal Transition: Diurnal Cycle by LSM\n(Soil Type 3, 30% moisture)')
ax.grid(True, alpha=0.3)
ax.axvline(x=12, color='gold', linestyle='--', label='Solar Noon')
ax.legend(loc='upper right', fontsize=8)
ax.set_xlim(6, 20)

# 4. Lag-Amplitude phase space
ax = axes[1, 0]
markers = ['o', 's', '^', 'v', 'D', 'p', '*']
colors = plt.cm.tab10(np.linspace(0, 1, 7))

for slt, marker, color in zip(soil_types, markers, colors):
    lags = []
    amps = []
    
    # Vary moisture
    for moisture in [0.1, 0.3, 0.5, 0.7]:
        char = get_diurnal_characteristics(1.0, slt, moisture)
        lags.append(char['lag_hours'])
        amps.append(char['amplitude'])
    
    ax.scatter(lags, amps, marker=marker, s=100, color=color, 
               label=f'Type {slt}', alpha=0.8)
    
    # Connect moisture points
    ax.plot(lags, amps, color=color, alpha=0.3, linewidth=1)
    
    # Label driest point (lowest moisture = leftmost)
    ax.text(lags[0], amps[0], f'{moistures[0]:.0%}', 
            fontsize=7, ha='right', va='bottom')
    # Label wettest point
    ax.text(lags[-1], amps[-1], f'{moistures[-1]:.0%}', 
            fontsize=7, ha='left', va='top')

# Add ocean point
ocean_char = get_diurnal_characteristics(0.0, None, 0.0)
ax.scatter(ocean_char['lag_hours'], ocean_char['amplitude'], 
          marker='X', s=200, color='darkblue', label='Ocean', zorder=5)
ax.text(ocean_char['lag_hours'], ocean_char['amplitude'], 'Ocean',
        fontsize=9, ha='left', va='bottom', fontweight='bold')

ax.set_xlabel('Phase Lag (hours)')
ax.set_ylabel('Diurnal Amplitude (°C)')
ax.set_title('ERA5: Complete Phase Space\n(Soil Types 1-7 + Ocean)')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8)

# 5. Real-world example grid
ax = axes[1, 1]
# Simulate a coastal grid with varying lsm
lsm_grid = np.array([[0.0, 0.2, 0.5, 0.8, 1.0],
                     [0.0, 0.1, 0.4, 0.7, 1.0],
                     [0.0, 0.0, 0.3, 0.6, 0.9],
                     [0.0, 0.0, 0.2, 0.5, 0.8],
                     [0.0, 0.0, 0.1, 0.4, 0.7]])

# Assume soil type 3 for all land
soil_type = 3
moisture = 0.3

# Create lag grid
lag_grid = np.zeros_like(lsm_grid)
for i in range(lsm_grid.shape[0]):
    for j in range(lsm_grid.shape[1]):
        char = get_diurnal_characteristics(lsm_grid[i, j], soil_type, moisture)
        lag_grid[i, j] = char['lag_hours']

# Plot
im = ax.imshow(lag_grid, cmap='RdYlBu_r', vmin=0, vmax=6)
ax.set_title('Example: Coastal Lag Grid\n(Soil Type 3, 30% moisture)')
ax.set_xlabel('Grid Column')
ax.set_ylabel('Grid Row')

# Add lsm values
for i in range(lsm_grid.shape[0]):
    for j in range(lsm_grid.shape[1]):
        ax.text(j, i, f'{lsm_grid[i,j]:.1f}\n{lag_grid[i,j]:.1f}h', 
                ha='center', va='center', color='black', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.colorbar(im, ax=ax, label='Lag (hours)')

# 6. Practical prediction implications
ax = axes[1, 2]
ax.axis('off')

text_content = """
ERA5 DIURNAL CYCLE: KEY PREDICTION RULES

1. LAND-SEA MASK (lsm) PRIMACY:
   • Ocean (lsm=0): Always minimal lag (0.5-1h), small ΔT (0.5-2°C)
   • Land (lsm=1): Soil type determines behavior
   • Coastal (0<lsm<1): Linear blend of ocean/land

2. SOIL TYPE (slt) HIERARCHY:
   Fast → Slow response:
   1: Sand       - Fast, large ΔT (2-4h lag)
   3: Sandy Loam - Moderate (3-5h lag)  
   5: Clay       - Slow, moderate ΔT (4-6h lag)
   6: Peat       - Very slow (5-7h lag)
   7: Ice        - Extreme lag (6-8h), small ΔT

3. SOIL MOISTURE MULTIPLIER:
   • Increases thermal inertia
   • Increases lag (longer response time)
   • Decreases amplitude (smaller ΔT)

4. PREDICTION IMPLICATION:
   t2m_diurnal = f(hour, lsm, slt, soil_moisture)
   
   Model must treat ocean/land fundamentally differently,
   then refine by soil properties for land cells.
"""

ax.text(0.05, 0.95, text_content, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.show()

# Summary statistics
print("="*90)
print("ERA5 DIURNAL CYCLE: SOIL TYPE + LAND-SEA MASK SUMMARY")
print("="*90)
print("\n1. OCEAN CELLS (lsm = 0.0):")
print("   Phase Lag: 0.5 - 1.0 hours")
print("   Amplitude: 0.5 - 2.0 °C")
print("   Soil Type: Irrelevant (undefined)")

print("\n2. LAND CELLS (lsm = 1.0):")
print("   Soil Type | Dry Lag | Wet Lag | Dry Amp | Wet Amp")
print("   " + "-"*50)
for slt in range(1, 8):
    dry = get_diurnal_characteristics(1.0, slt, 0.1)
    wet = get_diurnal_characteristics(1.0, slt, 0.7)
    print(f"   {slt:9d} | {dry['lag_hours']:7.1f}h | {wet['lag_hours']:7.1f}h | "
          f"{dry['amplitude']:7.1f}°C | {wet['amplitude']:7.1f}°C")

print("\n3. COASTAL CELLS (0.0 < lsm < 1.0):")
print("   Linear interpolation between ocean and land characteristics")
print("   Example: lsm=0.3 → 70% ocean behavior + 30% land behavior")
print("   Critical for accurate coastal t2m prediction!")

print("\n" + "="*90)
print("FOR t2m PREDICTION MODELS:")
print("-"*90)
print("1. Use lsm as primary classifier: ocean vs land vs coastal")
print("2. For land cells, use slt to determine thermal inertia")
print("3. Incorporate soil moisture (volumetric) for fine-tuning")
print("4. Coastal cells need weighted average of ocean/land physics")