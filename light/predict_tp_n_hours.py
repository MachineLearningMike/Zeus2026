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
    meteo_normal = torch.stack([t2m, tp, u100, v100, d2m, sp], dim=0),
    dense = torch.stack([orograph, sin_lat, cos_lat, sin_lon, cos_lon], dim=0),
    cat = torch.stack([land_sea_mask, soil_type], dim=0),
    geo_tempo = torch.stack([sin_orbit, cos_orbit, sin_zenith, cos_zenith], dim=0).
, where sin_xxx means sin(xxx) and cos_xxx means cos(xxx)
We also have inputs for the given future time f. 
The current and the future shares all inputs, except that the future has no meteo_normal and its own geo_tempo called geo_tempo_f.

def predict(self, meteo_normal: torch.Tensor, dense: torch.Tensor, 
                cat: torch.Tensor, geo_tempo: torch.Tensor, geo_tempo_f: torch.Tensor,
                current_dt: datetime, delta_hours: int) -> torch.Tensor

Please write a Python class "TpPredictor" that has the above predict(.) function, which predicts tp 
for the time t + delta hours, with fast physical logic and not by tedious ML or numerical analysis.
You know that tp at time t is the accumulated precipitation during the 1 hour ending at the time t.
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
import math
from datetime import datetime

class TpPredictor:
    def __init__(self, device=None, amplification=2.5):
        """
        amplification: global factor to boost inherited precipitation
        """
        self.device = torch.device(
            "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        )
        self.amplification = amplification

    # -------------------- Core utility --------------------
    def bilinear_sample(self, field, dx, dy):
        H, W = field.shape
        y = torch.arange(H, device=field.device).view(-1,1).repeat(1,W)
        x = torch.arange(W, device=field.device).view(1,-1).repeat(H,1)
        x_sample = x - dx
        y_sample = y - dy
        x0 = x_sample.floor().clamp(0, W-1)
        x1 = (x0 + 1).clamp(0, W-1)
        y0 = y_sample.floor().clamp(0, H-1)
        y1 = (y0 + 1).clamp(0, H-1)
        wx = x_sample - x0
        wy = y_sample - y0
        f00 = field[y0.long(), x0.long()]
        f10 = field[y0.long(), x1.long()]
        f01 = field[y1.long(), x0.long()]
        f11 = field[y1.long(), x1.long()]
        return f00*(1-wx)*(1-wy) + f10*wx*(1-wy) + f01*(1-wx)*wy + f11*wx*wy

    def advect_precipitation(self, tp_now, u100, v100, delta_hours):
        spread_factor = torch.clamp(torch.sqrt(u100**2 + v100**2)/20.0, 0.1, 0.5)
        dx_total = u100*delta_hours/10.0 - v100*spread_factor
        dy_total = v100*delta_hours/10.0 + u100*spread_factor
        return self.bilinear_sample(tp_now, dx_total, dy_total)

    # -------------------- Main prediction --------------------
    def predict(self,
                meteo_normal: torch.Tensor,
                dense: torch.Tensor,
                cat: torch.Tensor,
                geo_tempo: torch.Tensor,
                geo_tempo_f: torch.Tensor,
                current_dt: datetime,
                delta_hours: int) -> torch.Tensor:

        # Move tensors to device
        meteo_normal = meteo_normal.to(self.device)

        t2m, tp_now, u100, v100, d2m, sp = meteo_normal

        # -------------------- Inherited precipitation only --------------------
        tp_advected = self.advect_precipitation(tp_now, u100, v100, delta_hours)
        decay = math.exp(-delta_hours/12.0)
        tp_future = tp_advected * decay

        # -------------------- Small uniform amplification --------------------
        tp_future = tp_future * self.amplification

        return tp_future
