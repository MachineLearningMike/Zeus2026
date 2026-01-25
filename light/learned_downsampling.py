# They work on a learned latent space, NOT raw wind fields
class LearnedProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Encode to latent space at high resolution
        self.encoder = nn.Sequential(
            nn.Conv2d(20, 512, 3, padding=1),  # 20 weather variables
            nn.ReLU()
        )
        
        # 2. Process in latent space at lower resolution
        self.processor = nn.Sequential(
            F.interpolate(scale_factor=0.25, mode='bilinear'),  # 0.25° → 1°
            # ... Graph neural networks or transformers
            F.interpolate(scale_factor=4, mode='bilinear'),     # 1° → 0.25°
        )
        
        # 3. Decode back to physical variables
        self.decoder = nn.Conv2d(512, 20, 3, padding=1)
        
# If you MUST use F.interpolate for weather data:

# 1. Downsample ALL correlated variables together
def downsample_weather_block(data_dict, scale_factor):
    """Downsample u, v, t, q, z together to preserve relationships"""
    downsampled = {}
    for name, tensor in data_dict.items():
        # Use same interpolation for all
        downsampled[name] = F.interpolate(
            tensor, 
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )
    return downsampled

# 2. Consider scale factor carefully
# ERA5: 0.25° → 1.0° (scale_factor=0.25) is borderline
# ERA5: 0.25° → 2.0° (scale_factor=0.125) is problematic

# 3. Add physical regularization
def physical_constraint_loss(u_down, v_down):
    """Penalize unphysical interpolations"""
    div_loss = divergence_loss(u_down, v_down)
    energy_loss = energy_spectrum_loss(u_down, v_down)
    return div_loss + energy_loss

# Don't use F.interpolate directly on wind fields!
# Instead:

# Option A: Use existing regridded datasets
# (ERA5 already available at 1°, 2° resolutions)

# Option B: Use proper regridding tools
import xesmf as xe

regridder = xe.Regridder(
    ds_in, 
    ds_out, 
    method='conservative',  # or 'bilinear' for atmosphere
    periodic=True  # Important for global data!
)

u_lowres = regridder(u_hires)