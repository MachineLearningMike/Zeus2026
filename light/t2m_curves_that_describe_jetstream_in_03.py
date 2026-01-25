import numpy as np
import matplotlib.pyplot as plt

# Test the identity numerically
phi = np.linspace(-np.pi, np.pi, 1000)

# Calculate both forms
form1 = 0.25 * np.sin(2*phi)**2
form2 = np.cos(np.abs(phi))**2 * np.sin(phi)**2

# Check if they're equal (within machine precision)
print("Maximum difference:", np.max(np.abs(form1 - form2)))
print("Are they equal?", np.allclose(form1, form2))

# Visual proof
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Both functions overlap
ax = axes[0]
ax.plot(np.degrees(phi), form1, 'b-', linewidth=3, alpha=0.7, label=r'$\frac{1}{4}\sin^2(2\phi)$')
ax.plot(np.degrees(phi), form2, 'r--', linewidth=2, label=r'$\cos^2(|\phi|)\sin^2(\phi)$')
ax.set_xlabel('Latitude ϕ (degrees)')
ax.set_ylabel('w(ϕ)')
ax.set_title('Both Functions Overlap Perfectly')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Difference (should be zero)
ax = axes[1]
diff = form1 - form2
ax.plot(np.degrees(phi), diff, 'g-', linewidth=2)
ax.set_xlabel('Latitude ϕ (degrees)')
ax.set_ylabel('Difference')
ax.set_title('Difference (Machine Precision ≈ 0)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 3: Step-by-step visualization
ax = axes[2]
x = np.linspace(0, np.pi/2, 50)  # First quadrant only
ax.plot(np.degrees(x), np.sin(2*x), 'k-', label=r'$\sin(2\phi)$', alpha=0.5)
ax.plot(np.degrees(x), np.sin(x), 'b-', label=r'$\sin(\phi)$', alpha=0.7)
ax.plot(np.degrees(x), np.cos(x), 'r-', label=r'$\cos(\phi)$', alpha=0.7)
ax.plot(np.degrees(x), 0.25*np.sin(2*x)**2, 'g-', linewidth=3, 
        label=r'$\frac{1}{4}\sin^2(2\phi)$')
ax.set_xlabel('Latitude ϕ (degrees)')
ax.set_title('Components of the Identity')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()