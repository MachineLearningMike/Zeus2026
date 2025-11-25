from skyfield import api, almanac
from datetime import datetime, timezone
import math
import pvlib
import pandas as pd
import numpy as np
import torch
import time

# Load ephemeris
ts = api.load.timescale()
eph = api.load('de421.bsp')

class Geo_Tempo():
    lat_lon_tensor_720_1440 = None
    
    def get_season_events(year):
        t0 = ts.utc(year, 1, 1)
        t1 = ts.utc(year, 12, 31)
        times, events = almanac.find_discrete(t0, t1, almanac.seasons(eph))
        season_dict = {}
        for ti, ev in zip(times, events):
            dt = ti.utc_datetime()
            # Ensure UTC-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if ev == 0:
                season_dict['mar_equinox'] = dt
            elif ev == 1:
                season_dict['jun_solstice'] = dt
            elif ev == 2:
                season_dict['sep_equinox'] = dt
            elif ev == 3:
                season_dict['dec_solstice'] = dt
        return season_dict

    def earth_orbit_angle(dt):
        # If naive datetime, treat as UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        year = dt.year
        seasons = Geo_Tempo.get_season_events(year)
        
        season_angles = [
            (seasons['dec_solstice'], 0),
            (seasons['mar_equinox'], 90),
            (seasons['jun_solstice'], 180),
            (seasons['sep_equinox'], 270),
            (Geo_Tempo.get_season_events(year + 1)['dec_solstice'], 360)
        ]
        
        for i in range(len(season_angles) - 1):
            t0, a0 = season_angles[i]
            t1, a1 = season_angles[i+1]
            if t0 <= dt < t1:
                fraction = (dt - t0).total_seconds() / (t1 - t0).total_seconds()
                angle = a0 + fraction * (a1 - a0)
                return angle
        return None
    
    def earth_orbit_angle_tensor(dt, H, W):
        """
        Generate a tensor of Earth orbital angles (degrees 0..360) on a (H, W) grid
        by calling Geo_Tempo.earth_orbit_angle().

        Parameters:
            dt (datetime): UTC datetime
            H (int): grid height
            W (int): grid width

        Returns:
            torch.Tensor: (H, W) tensor of orbital angles in degrees
        """
        # Compute orbital angle for this datetime (scalar)
        angle_deg = Geo_Tempo.earth_orbit_angle(dt)  # scalar
        
        # Create a grid of shape (H, W) with the same value
        angle_grid = np.full((H, W), angle_deg, dtype=np.float32)
        
        # Convert to torch tensor
        angle_tensor = torch.from_numpy(angle_grid)
        
        return angle_tensor
    
    def sin_cos_earth_orbit_angle_tensor(dt_utc, height, width):
        earth_orbit_angle_tensor = Geo_Tempo.earth_orbit_angle_tensor(dt_utc, height, width)
        radian_tensor = torch.deg2rad(earth_orbit_angle_tensor)
        sin_earth_orbit_angle_tensor = torch.sin(radian_tensor)
        cos_earth_orbit_angle_tensor = torch.cos(radian_tensor)
        sin_cos = torch.stack([sin_earth_orbit_angle_tensor, cos_earth_orbit_angle_tensor], dim=0)
        return sin_cos
    
    def zenith_angle(dt_utc, lat, lon):
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        # lat: -90 <= lat <= 90 degrees
        # lon: -180 <= lon <= 180 degrees
        solpos = pvlib.solarposition.get_solarposition(pd.Timestamp(dt_utc), lat, lon)
        return solpos['zenith'].item()

    def equation_of_time_minutes(doy):
        B = math.radians(360.0 * (doy - 81) / 364.0)
        return 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    def utc_to_true_solar_hour(utc_dt, lon):
        """
        Convert UTC calendar time to Local Apparent Solar Time (decimal hour, true solar time).

        Parameters:
            utc_dt (datetime): UTC datetime (timezone-aware or naive treated as UTC)
            lon (float): Longitude in degrees (+east, -west)

        Returns:
            float: True solar time in decimal hours (0..24)
        """
        # If naive datetime, treat as UTC
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        
        t = pd.Timestamp(utc_dt)
        
        # Day of year
        doy = t.timetuple().tm_yday
        
        # Fractional day of year
        start = pd.Timestamp(year=t.year, month=1, day=1, tz=t.tzinfo)
        doy = (t - start).total_seconds() / 86400.0 + 1
        
        # UTC decimal hours
        utc_h = t.hour + t.minute/60 + t.second/3600 + t.microsecond/3.6e9
        
        # Longitude offset (hours)
        lon_h = lon / 15.0
        
        # Equation of time (hours)
        eot_h = Geo_Tempo.equation_of_time_minutes(doy) / 60.0
        
        # Local apparent solar time (0..24)
        true_solar_hour = (utc_h + lon_h + eot_h) % 24
        
        return true_solar_hour

    def utc_to_true_solar_timestamp(utc_dt, lon):
        """
        Convert a UTC datetime to a full timestamp corresponding to True Solar Time (Local Apparent Solar Time).

        Parameters:
            utc_dt (datetime): UTC datetime (timezone-aware or naive treated as UTC)
            lon (float): Longitude in degrees (+east, -west)

        Returns:
            pd.Timestamp: Timestamp in True Solar Time
        """
        # If naive datetime, treat as UTC
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)

        t = pd.Timestamp(utc_dt)

        # Compute true solar hour using the other function
        true_solar_h = Geo_Tempo.utc_to_true_solar_hour(t, lon)

        # UTC decimal hours
        utc_h = (
            t.hour
            + t.minute / 60
            + t.second / 3600
            + t.microsecond / 3.6e9
        )

        # Offset in hours
        offset_h = true_solar_h - utc_h

        # Build timestamp with the solar offset
        true_solar_ts = t + pd.to_timedelta(offset_h, unit="h")

        return true_solar_ts
    

    def utc_to_mean_solar_hour(utc_dt, lon):
        """
        Convert UTC calendar time to Local Mean Solar Time (decimal hour).

        Parameters:
            utc_dt (datetime): UTC datetime (timezone-aware or naive treated as UTC)
            lon (float): Longitude in degrees (+east, -west)

        Returns:
            float: Mean solar time in decimal hours (0..24)
        """
        # If naive datetime, treat as UTC
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
        
        t = pd.Timestamp(utc_dt)
        
        # UTC decimal hours
        utc_h = t.hour + t.minute/60 + t.second/3600 + t.microsecond/3.6e9
        
        # Longitude offset (hours)
        lon_h = lon / 15.0
        
        # Local mean solar time (0..24)
        mean_solar_hour = (utc_h + lon_h) % 24
        
        return mean_solar_hour

    def true_solar_hours_tensor(utc_dt, H, W):
        """
        Generate a tensor of Local Mean Solar Time (decimal hours) on a (H, W) grid
        by calling utc_to_mean_solar_hour().

        Parameters:
            utc_dt (datetime): UTC datetime
            H (int): grid height
            W (int): grid width

        Returns:
            torch.Tensor: (H, W) tensor of mean solar hours (0..24)
        """
        # Create a 1D array of longitudes (-180 to 180)
        step = 360.0 / W
        lon = np.linspace(-180 + step/2, 180 - step/2, W, dtype=np.float32)
        
        # Preallocate 1D array of mean solar hours for the longitudes
        mean_hours_row = np.array([Geo_Tempo.utc_to_true_solar_hour(utc_dt, lon_i) for lon_i in lon], dtype=np.float32)
        
        # Repeat along latitude to create full grid
        mean_hours_grid = np.tile(mean_hours_row, (H, 1))
        
        # Convert to torch tensor
        mean_hours_tensor = torch.from_numpy(mean_hours_grid)
        
        return mean_hours_tensor
    
    def sin_cos_true_solar_hours_tensor(dt_utc, height, width):
        true_solar_hours_tensor = Geo_Tempo.true_solar_hours_tensor(dt_utc, height, width)
        radian_tensor = torch.deg2rad(true_solar_hours_tensor * 360 / 24)
        sin_true_solar_hours_tensor = torch.sin(radian_tensor)
        cos_true_solar_hours_tensor = torch.cos(radian_tensor)
        sin_cos = torch.stack([sin_true_solar_hours_tensor, cos_true_solar_hours_tensor], dim=0)
        return sin_cos

    def lat_lon_grid_tensor(H, W):
        """
        Generate a (2, H, W) tensor of latitudes and longitudes.
        Channel 0: latitude [-90, 90]
        Channel 1: longitude [-180, 180)  (180 excluded)
        """
        
        if H == 720 and W == 1440 and Geo_Tempo.lat_lon_tensor_720_1440 is not None:
            lat_lon_tensor = Geo_Tempo.lat_lon_tensor_720_1440
        else:        
            # Step sizes
            lat_step = 180.0 / H
            lon_step = 360.0 / W

            # Latitudes: center of each row (H×1)
            lat = torch.linspace(-90 + lat_step/2, 90 - lat_step/2, H, dtype=torch.float32).unsqueeze(1)

            # Longitudes: center of each column (1×W)
            lon = torch.linspace(-180 + lon_step/2, 180 - lon_step/2, W, dtype=torch.float32).unsqueeze(0)

            
            # Broadcast to (H,W)
            lat_grid = lat.expand(H, W)
            lon_grid = lon.expand(H, W)
            
            # Stack into (2,H,W)
            lat_lon_tensor = torch.stack([lat_grid, lon_grid], dim=0)
        
        return lat_lon_tensor
    
    def cos_lat_sin_lon_tensor(H, W):
        """
        Generate a (2, H, W) tensor of (cos(lat), sin(lon)) by calling lat_lon_grid_tensor.
        
        Channel 0: cos(latitude in radians)
        Channel 1: sin(longitude in radians)
        
        Parameters:
            H (int): grid height (latitude)
            W (int): grid width (longitude)
        
        Returns:
            torch.Tensor: shape (2, H, W)
        """
        # Get the lat/lon grid first
        if H == 720 and W == 1440:
            lat_lon = Geo_Tempo.lat_lon_tensor_720_1440
        else:
            lat_lon = Geo_Tempo.lat_lon_grid_tensor(H, W)  # shape (2, H, W)
        
        lat_rad = torch.deg2rad(lat_lon[0])
        lon_rad = torch.deg2rad(lat_lon[1])

        tensor = torch.stack([torch.cos(lat_rad), torch.sin(lon_rad)], dim=0)
        return tensor
    
    def sin_lat_sin_cos_lon_tensor(H, W):
        
        if H == 720 and W == 1440:
            lat_lon = Geo_Tempo.lat_lon_tensor_720_1440
        else:
            lat_lon = Geo_Tempo.lat_lon_grid_tensor(H, W)  # shape (2, H, W)
        
        lat_rad = torch.deg2rad(lat_lon[0])
        lon_rad = torch.deg2rad(lat_lon[1])

        tensor = torch.stack([torch.sin(lat_rad), torch.sin(lon_rad), torch.cos(lon_rad)], dim=0)
        return tensor        

Geo_Tempo.lat_lon_tensor_720_1440 = Geo_Tempo.lat_lon_grid_tensor(720, 1440)


if __name__ == "__main__":
    # Examples
    angle1 = Geo_Tempo.earth_orbit_angle(datetime(2025, 11, 20, 12))  # naive treated as UTC    
    angle2 = Geo_Tempo.earth_orbit_angle(datetime(2025, 11, 20, 12, tzinfo=timezone.utc))  # aware datetime
    print(f"Earth orbital angle 1: {angle1:.2f}°")
    print(f"Earth orbital angle 2: {angle2:.2f}°")
    
    lat, lon = 40.0, -105.0
    dt_utc = datetime(2025, 11, 20, 12, tzinfo=timezone.utc)
    solar_h = Geo_Tempo.utc_to_true_solar_hour(dt_utc, lon)
    print("solar_h: ", solar_h)

    lat_py, lon_py = 39.019444, 125.738052
    dt_utc = now_utc = datetime.now(timezone.utc)
    print("utc now: ", dt_utc)
    
    oribtal_angle = Geo_Tempo.earth_orbit_angle(dt_utc)
    zeith_anglee = Geo_Tempo.zenith_angle(dt_utc, lat=lat_py, lon=lon_py)
    true_solar_h = Geo_Tempo.utc_to_true_solar_hour(dt_utc, lon=lon_py)
    mean_solar_h = Geo_Tempo.utc_to_mean_solar_hour(dt_utc, lon=lon_py)
    true_solar_ts = Geo_Tempo.utc_to_true_solar_timestamp(dt_utc, lon=lon_py)
    
    def f2h(fractional_hour):
        h = int(fractional_hour)
        m = int((fractional_hour - h)*6)
        s = ((fractional_hour - h)*6 - int((fractional_hour - h)*6)) * 6
        return str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s)
            
    print("-----------------------------------------------------------")
    print(f"orbital_angle: {oribtal_angle}, zenith_angle: {zeith_anglee},\ntrue_solar_h: {f2h(true_solar_h)}, mean_solar_h: {f2h(mean_solar_h)}, true_solar_ts: {true_solar_ts}")
    
    
    H, W = 720, 1440
    loops = 10
    s = time.time()
    for _ in range(loops):
        #------------ sin/cos earth_orbit_angle_tensor --------------
        cos_earth_orbit_angle_tensor = Geo_Tempo.sin_cos_earth_orbit_angle_tensor(dt_utc, H, W)
        
        #------------ sin/cos true_solar_hours_tensor --------------
        cos_true_solar_hours_tensor = Geo_Tempo.sin_cos_true_solar_hours_tensor(dt_utc, H, W)
        
        #------------ (lat, lon) ------------------------------------
        lat_lon_tensor = Geo_Tempo.lat_lon_grid_tensor(H, W)
        cos_lat_sin_lon_tensor = Geo_Tempo.cos_lat_sin_lon_tensor(H, W)
        
    e = time.time()
    print(f"cos_earth_orbit_angle_tensor: {cos_earth_orbit_angle_tensor.shape}, cos_true_solar_hours_tensor: {cos_true_solar_hours_tensor.shape}, time: {(e-s)/loops}")
    
    lat_lon_tensor = Geo_Tempo.lat_lon_grid_tensor(H, W)
    print(lat_lon_tensor[:, 0,0], lat_lon_tensor[:, -1, -1], lat_lon_tensor[:, 0, -1], lat_lon_tensor[:, -1, 0])
    # tensor([ -90., -180.]) tensor([ 90.0000, 179.7500]) tensor([-90.0000, 179.7500]) tensor([  90., -180.])
    
    cos_lat_sin_lon_tensor = Geo_Tempo.cos_lat_sin_lon_tensor(H, W)
    print(cos_lat_sin_lon_tensor[:, 0,0], cos_lat_sin_lon_tensor[:, -1, -1], cos_lat_sin_lon_tensor[:, 0, -1], cos_lat_sin_lon_tensor[:, -1, 0])
    # tensor([-4.3711e-08,  8.7423e-08]) tensor([-4.3711e-08,  4.3632e-03]) tensor([-4.3711e-08,  4.3632e-03]) tensor([-4.3711e-08,  8.7423e-08])
    
    print(cos_lat_sin_lon_tensor[:, H//2, W//2])
    # tensor([1., 0.])
