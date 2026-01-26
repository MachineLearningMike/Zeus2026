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
    #  -90 <= lat <= +90 [-90, +90], -180 <= lon < 180 [-180, +180) in our dataset scheme, which follows ERA5.
    # ---------------- DO NOT change -------------
    respect_north = True    # ERA5 doesn't remove lat points. But we choose to remove the south polar region if we have to remove some lat points.
    respect_east = False     # ERA5 choose to remove +180 when they remove a lon point to reflect that lontigude is cylindric.
    
    lat_lon_tensor_721_1440, lat_tensor_721, lon_tensor_1440 = None, None, None
    H_era5_grid, W_era5_grid = 721, 1440
    
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
    

    def debug_earth_orbit_angle(dt_utc):
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        y = dt_utc.year
        seasons_y = Geo_Tempo.get_season_events(y)

        if dt_utc < seasons_y['mar_equinox']:
            season_year = y - 1
        else:
            season_year = y

        s0 = Geo_Tempo.get_season_events(season_year)
        s1 = Geo_Tempo.get_season_events(season_year + 1)

        season_angles = [
            (s0['dec_solstice'], 0),
            (s1['mar_equinox'], 90),
            (s1['jun_solstice'], 180),
            (s1['sep_equinox'], 270),
            (s1['dec_solstice'], 360),
        ]

        print("dt_utc =", dt_utc)

        print("\nSeason timeline:")
        for t, a in season_angles:
            print(f"{a:3d}°  {t.isoformat()}")

        # check ordering
        print("\nOrdering checks:")
        for i in range(len(season_angles)-1):
            t0, a0 = season_angles[i]
            t1, a1 = season_angles[i+1]
            print(f"{a0}°→{a1}° : {t0 < t1}")



    def earth_orbit_angle(dt_utc):
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)

        y = dt_utc.year

        # Load events for y-1, y, y+1
        s_prev = Geo_Tempo.get_season_events(y - 1)
        s_curr = Geo_Tempo.get_season_events(y)
        s_next = Geo_Tempo.get_season_events(y + 1)

        # --- Find the December solstice immediately before dt_utc ---
        if dt_utc >= s_curr['dec_solstice']:
            solstice_before = s_curr['dec_solstice']
            solstice_after  = s_next['dec_solstice']
            eq_mar  = s_next['mar_equinox']
            eq_jun  = s_next['jun_solstice']
            eq_sep  = s_next['sep_equinox']
        else:
            solstice_before = s_prev['dec_solstice']
            solstice_after  = s_curr['dec_solstice']
            eq_mar  = s_curr['mar_equinox']
            eq_jun  = s_curr['jun_solstice']
            eq_sep  = s_curr['sep_equinox']

        season_angles = [
            (solstice_before, 0),
            (eq_mar, 90),
            (eq_jun, 180),
            (eq_sep, 270),
            (solstice_after, 360),
        ]

        for (t0, a0), (t1, a1) in zip(season_angles, season_angles[1:]):
            if t0 <= dt_utc < t1:
                f = (dt_utc - t0).total_seconds() / (t1 - t0).total_seconds()
                return a0 + f*(a1 - a0)

        raise RuntimeError("earth_orbit_angle: timestamp does not fall into seasonal intervals")


        
    def earth_orbit_angle_tensor_era5(dt_utc):
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
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        # Compute orbital angle for this datetime (scalar)
        angle_deg = Geo_Tempo.earth_orbit_angle(dt_utc)  # scalar
        
        # Create a grid of shape (H, W) with the same value
        angle_grid = np.full((Geo_Tempo.H_era5_grid, Geo_Tempo.W_era5_grid), angle_deg, dtype=np.float32)
        
        # Convert to torch tensor
        angle_tensor = torch.from_numpy(angle_grid)
        
        return angle_tensor
    
    def sin_cos_earth_orbit_angle_tensor_era5(dt_utc):
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)        
        earth_orbit_angle_tensor_era5 = Geo_Tempo.earth_orbit_angle_tensor_era5(dt_utc)
        radian_tensor = torch.deg2rad(earth_orbit_angle_tensor_era5)
        sin_earth_orbit_angle_tensor = torch.sin(radian_tensor)
        cos_earth_orbit_angle_tensor = torch.cos(radian_tensor)
        sin_cos = torch.stack([sin_earth_orbit_angle_tensor, cos_earth_orbit_angle_tensor], dim=0)
        return sin_cos
    
    def zenithh_angle(dt_utc, lat, lon):
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        solpos = pvlib.solarposition.get_solarposition(pd.Timestamp(dt_utc), lat, lon)
        return solpos['zenithh'].item()


    def get_zenithh_angle(dt_utc):
        """
        Compute zenithh angle for entire ERA5 grid.
        Warning: Slow due to pvlib's per-point computation.
        """
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        lat_lon, lat_tensor, lon_tensor = Geo_Tempo.lat_lon_grid_tensor_era5_721()
        lat_grid = lat_lon[0].numpy()
        lon_grid = lat_lon[1].numpy()
        
        # Flatten for vectorized computation
        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        
        # Create array of timestamps (all same time)
        times = pd.DatetimeIndex([pd.Timestamp(dt_utc)] * len(lat_flat))
        
        # Compute solar positions (vectorized in pvlib)
        solpos = pvlib.solarposition.get_solarposition(
            times, lat_flat, lon_flat,
            method='nrel_numpy'
        )
        
        # Reshape back to grid
        zenithh_grid = solpos['zenithh'].values.reshape(lat_grid.shape)
        
        return torch.from_numpy(zenithh_grid.astype(np.float32))

    def get_zenithh_angle_faster(dt_utc):
        """
        Wrapper for your ERA5 grid using PyTorch.
        Returns: torch.Tensor [H, W] of zenithh angles in degrees
        """
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()
        lat_grid = lat_lon[0]
        lon_grid = lat_lon[1]
        
        return Geo_Tempo.fast_zenithh_angle_map(dt_utc, lat_grid, lon_grid, use_torch=True)

    def fast_zenithh_angle_map(dt_utc, 
                            lat_grid=None, 
                            lon_grid=None, 
                            use_torch=True):
        """
        Compute zenithh angle map (degrees) for entire grid using vectorized formula.
        
        Parameters:
            dt_utc (datetime): UTC time
            lat_grid (torch.Tensor or np.ndarray): [H, W] latitudes in degrees
            lon_grid (torch.Tensor or np.ndarray): [H, W] longitudes in degrees
            use_torch (bool): Use PyTorch tensors (faster on GPU)
        
        Returns:
            zenithh_map: [H, W] zenithh angles in degrees (0-180)
        """
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        
        # Get grid if not provided
        if lat_grid is None or lon_grid is None:
            lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()
            lat_grid = lat_lon[0]
            lon_grid = lat_lon[1]
        
        # Convert to numpy if using torch (for some datetime operations)
        if use_torch:
            lat = lat_grid
            lon = lon_grid
        else:
            lat = lat_grid.numpy() if isinstance(lat_grid, torch.Tensor) else lat_grid
            lon = lon_grid.numpy() if isinstance(lon_grid, torch.Tensor) else lon_grid
        
        # 1. Compute solar declination (δ)
        t = pd.Timestamp(dt_utc)
        doy = t.timetuple().tm_yday  # day of year
        
        # Simple declination formula (accurate enough for most purposes)
        # δ = 23.45 * sin(360/365 * (doy - 81))
        B = np.radians(360 * (doy - 81) / 365)
        delta_deg = 23.45 * np.sin(B)  # solar declination in degrees
        
        # 2. Compute hour angle (h) for each longitude
        # True solar time = UTC + lon/15 + EoT
        utc_h = t.hour + t.minute/60 + t.second/3600 + t.microsecond/3.6e9
        
        # Equation of time (in hours)
        B_eot = np.radians(360 * (doy - 81) / 364)
        eot_h = (9.87 * np.sin(2*B_eot) - 7.53 * np.cos(B_eot) - 1.5 * np.sin(B_eot)) / 60
        
        # Hour angle: h = 15° * (true_solar_time - 12)
        # true_solar_time = UTC + lon/15 + EoT
        if use_torch:
            # PyTorch version
            delta_rad = torch.deg2rad(torch.tensor(delta_deg, dtype=torch.float32))
            lat_rad = torch.deg2rad(lat)
            
            # Hour angle computation
            true_solar_h = (utc_h + lon/15.0 + eot_h) % 24
            h_deg = 15.0 * (true_solar_h - 12.0)
            h_rad = torch.deg2rad(h_deg)
            
            # Spherical trig formula
            cos_z = (torch.sin(lat_rad) * torch.sin(delta_rad) + 
                    torch.cos(lat_rad) * torch.cos(delta_rad) * torch.cos(h_rad))
            
            # Clamp to avoid numerical errors
            cos_z = torch.clamp(cos_z, -1.0, 1.0)
            
            zenithh_rad = torch.acos(cos_z)
            zenithh_deg = torch.rad2deg(zenithh_rad)
            
        else:
            # NumPy version
            delta_rad = np.radians(delta_deg)
            lat_rad = np.radians(lat)
            
            # Hour angle computation
            true_solar_h = (utc_h + lon/15.0 + eot_h) % 24
            h_deg = 15.0 * (true_solar_h - 12.0)
            h_rad = np.radians(h_deg)
            
            # Spherical trig formula
            cos_z = (np.sin(lat_rad) * np.sin(delta_rad) + 
                    np.cos(lat_rad) * np.cos(delta_rad) * np.cos(h_rad))
            
            # Clamp to avoid numerical errors
            cos_z = np.clip(cos_z, -1.0, 1.0)
            
            zenithh_rad = np.arccos(cos_z)
            zenithh_deg = np.degrees(zenithh_rad)
        
        return zenithh_deg

    def sin_cos_zeith_angle_tensor_era5(dt_utc):
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
            
        zenith_angle = Geo_Tempo.get_zenithh_angle_faster(dt_utc)        
        zenith_radian_tensor = torch.deg2rad(zenith_angle)
        sin_zenith_tensor = torch.sin(zenith_radian_tensor)
        cos_zenith_tensor = torch.cos(zenith_radian_tensor)
        sin_cos = torch.stack([sin_zenith_tensor, cos_zenith_tensor], dim=0)
        return sin_cos

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

    def true_solar_hours_tensor_era5(utc_dt):
        """
        Generate a tensor of Local Mean Solar Time (decimal hours) on a (H, W) grid
        by calling utc_to_mean_solar_hour().

        Parameters:
            utc_dt (datetime): UTC datetime

        Returns:
            torch.Tensor: (H, W) tensor of mean solar hours (0..24)
        """
        _, _, lon_tensor = Geo_Tempo.lat_lon_grid_tensor_era5_721()
        lon = lon_tensor.numpy()
        
        # Preallocate 1D array of mean solar hours for the longitudes
        mean_hours_row = np.array([Geo_Tempo.utc_to_true_solar_hour(utc_dt, lon_i) for lon_i in lon], dtype=np.float32)
        
        # Repeat along latitude to create full grid
        mean_hours_grid = np.tile(mean_hours_row, (Geo_Tempo.H_era5_grid, 1))
        
        # Convert to torch tensor
        mean_hours_tensor = torch.from_numpy(mean_hours_grid)
        
        return mean_hours_tensor
    
    def sin_cos_true_solar_hours_tensor_era5(dt_utc):
        # If naive datetime, treat as UTC
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        true_solar_hours_tensor_era5 = Geo_Tempo.true_solar_hours_tensor_era5(dt_utc)
        radian_tensor = torch.deg2rad(true_solar_hours_tensor_era5 * 360 / 24)
        sin_true_solar_hours_tensor = torch.sin(radian_tensor)
        cos_true_solar_hours_tensor = torch.cos(radian_tensor)
        sin_cos = torch.stack([sin_true_solar_hours_tensor, cos_true_solar_hours_tensor], dim=0)
        return sin_cos

    def lat_lon_grid_tensor_era5_721():
        # BE CAREFUL !!!!!!!!!!! with this code, which is the foundation of geography of the whole system.

        """
        Generate a (2, 721, 1440) tensor of lat/lon at **cell centers** for ERA5 0.25° grid.
        """

        H = 721
        W = 1441

        # Step sizes
        lat_step = 180.0 / (H - 1)    # 0.25°
        lon_step = 360.0 / (W - 1)    # 0.25°

        # Latitudes: center of each cell
        lat_tensor = torch.linspace(-90.0, 90.0, H, dtype=torch.float32) # 0.25 x 721
        # print(lat_tensor[0], lat_tensor[-1], lat_tensor[1]-lat_tensor[0])
        # Longitudes: center of each cell
        lon_tensor = torch.linspace(-180.0, 180.0, W, dtype=torch.float32) # 0.25 x 1441
        # print(lon_tensor[0], lon_tensor[-1], lon_tensor[1]-lon_tensor[0])

        lat_tensor = lat_tensor[:]      # 0.25 x H
        lon_tensor = lon_tensor[:-1]    # 0.25 x (W-1), the +180, or E180, is removed. why not -180? ERA5 does that. So, west is respected.

        # Broadcast to grids
        lat_grid = lat_tensor.unsqueeze(1).expand(H, W-1)
        lon_grid = lon_tensor.unsqueeze(0).expand(H, W-1)

        # Stack into (2, H, W)
        lat_lon_tensor = torch.stack([lat_grid, lon_grid], dim=0)   # (2, 721, 1440), with a resolution of 0.25 degree

        # print(lat_lon_tensor.shape)
        # print(lat_lon_tensor[0, 360, :])
        # print(lat_lon_tensor[1, :, 0])
        
        return lat_lon_tensor, lat_tensor, lon_tensor
    
    def lat_lon_grid_tensor_era5_720():
        # BE CAREFULL !!!!
        lat_lon = Geo_Tempo.lat_lon_tensor_721_1440[:, 1:, :] if Geo_Tempo.respect_north else Geo_Tempo.lat_lon_tensor_721_1440[:, :-1, :]
        lat = Geo_Tempo.lat_tensor_721[1:] if Geo_Tempo.respect_north else Geo_Tempo.lat_tensor_721[:-1]
        lon = Geo_Tempo.lon_tensor_1440
        return lat_lon, lat, lon
    
    def cos_lat_sin_lon_tensor_era5():
        """
        Generate a (2, H, W) tensor of (cos(lat), sin(lon)) by calling lat_lon_grid_tensor.
        
        Channel 0: cos(latitude in radians)
        Channel 1: sin(longitude in radians)
        
        Parameters:
        
        Returns:
            torch.Tensor: shape (2, H, W)
        """
        # Get the lat/lon grid first
        lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()  # shape (2, H, W)
        
        lat_rad = torch.deg2rad(lat_lon[0])
        lon_rad = torch.deg2rad(lat_lon[1])

        tensor = torch.stack([torch.cos(lat_rad), torch.sin(lon_rad)], dim=0)
        return tensor
    
    def sin_lat_sin_cos_lon_tensor_era5():
        
        lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()  # shape (2, H, W)
        
        lat_rad = torch.deg2rad(lat_lon[0])
        lon_rad = torch.deg2rad(lat_lon[1])

        tensor = torch.stack([torch.sin(lat_rad), torch.sin(lon_rad), torch.cos(lon_rad)], dim=0)
        return tensor        

    def sin_cos_lat_tensor_era5():
        
        lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()  # shape (2, H, W)
        
        lat_rad = torch.deg2rad(lat_lon[0])

        tensor = torch.stack([torch.sin(lat_rad), torch.cos(lat_rad)], dim=0)
        return tensor  
    
    def sin_cos_lon_tensor_era5():
        
        lat_lon, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()  # shape (2, H, W)
        
        lon_rad = torch.deg2rad(lat_lon[1])

        tensor = torch.stack([torch.sin(lon_rad), torch.cos(lon_rad)], dim=0)
        return tensor  

Geo_Tempo.lat_lon_tensor_721_1440, Geo_Tempo.lat_tensor_721, Geo_Tempo.lon_tensor_1440 = Geo_Tempo.lat_lon_grid_tensor_era5_721()

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
    zeith_anglee = Geo_Tempo.zenithh_angle(dt_utc, lat=lat_py, lon=lon_py)
    true_solar_h = Geo_Tempo.utc_to_true_solar_hour(dt_utc, lon=lon_py)
    mean_solar_h = Geo_Tempo.utc_to_mean_solar_hour(dt_utc, lon=lon_py)
    true_solar_ts = Geo_Tempo.utc_to_true_solar_timestamp(dt_utc, lon=lon_py)
    
    def f2h(fractional_hour):
        h = int(fractional_hour)
        m = int((fractional_hour - h)*6)
        s = ((fractional_hour - h)*6 - int((fractional_hour - h)*6)) * 6
        return str(h).zfill(2) + ":" + str(m).zfill(2) + ":" + str(s)
            
    print("-----------------------------------------------------------")
    print(f"orbital_angle: {oribtal_angle}, zenithh_angle: {zeith_anglee},\ntrue_solar_h: {f2h(true_solar_h)}, mean_solar_h: {f2h(mean_solar_h)}, true_solar_ts: {true_solar_ts}")
    
    
    loops = 10
    s = time.time()
    for _ in range(loops):
        #------------ sin/cos earth_orbit_angle_tensor_era5 --------------
        cos_earth_orbit_angle_tensor = Geo_Tempo.sin_cos_earth_orbit_angle_tensor_era5(dt_utc)
        
        #------------ sin/cos true_solar_hours_tensor_era5 --------------
        cos_true_solar_hours_tensor = Geo_Tempo.sin_cos_true_solar_hours_tensor_era5(dt_utc)
        
        #------------ (lat, lon) ------------------------------------
        lat_lon_tensor, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()
        cos_lat_sin_lon_tensor_era5 = Geo_Tempo.cos_lat_sin_lon_tensor_era5()
    e = time.time()
    print(f"cos_earth_orbit_angle_tensor: {cos_earth_orbit_angle_tensor.shape}, cos_true_solar_hours_tensor: {cos_true_solar_hours_tensor.shape}, time: {(e-s)/loops}")
    
    lat_lon_tensor, _, _ = Geo_Tempo.lat_lon_grid_tensor_era5_721()
    print(lat_lon_tensor[:, 0,0], lat_lon_tensor[:, -1, -1], lat_lon_tensor[:, 0, -1], lat_lon_tensor[:, -1, 0])
    # tensor([ -90., -180.]) tensor([ 90.0000, 179.7500]) tensor([-90.0000, 179.7500]) tensor([  90., -180.])
    
    cos_lat_sin_lon_tensor_era5 = Geo_Tempo.cos_lat_sin_lon_tensor_era5()
    print(cos_lat_sin_lon_tensor_era5[:, 0,0], cos_lat_sin_lon_tensor_era5[:, -1, -1], cos_lat_sin_lon_tensor_era5[:, 0, -1], cos_lat_sin_lon_tensor_era5[:, -1, 0])
    # tensor([-4.3711e-08,  8.7423e-08]) tensor([-4.3711e-08,  4.3634e-03]) tensor([-4.3711e-08,  4.3634e-03]) tensor([-4.3711e-08,  8.7423e-08])
    
    _, H, W = cos_lat_sin_lon_tensor_era5.shape
    print(cos_lat_sin_lon_tensor_era5[:, H//2, W//2])
    # tensor([1., 0.])
