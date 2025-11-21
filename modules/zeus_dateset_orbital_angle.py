from skyfield import api, almanac
from datetime import datetime, timezone

# Load ephemeris
ts = api.load.timescale()
eph = api.load('de421.bsp')

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
    seasons = get_season_events(year)
    
    season_angles = [
        (seasons['dec_solstice'], 0),
        (seasons['mar_equinox'], 90),
        (seasons['jun_solstice'], 180),
        (seasons['sep_equinox'], 270),
        (get_season_events(year + 1)['dec_solstice'], 360)
    ]
    
    for i in range(len(season_angles) - 1):
        t0, a0 = season_angles[i]
        t1, a1 = season_angles[i+1]
        if t0 <= dt < t1:
            fraction = (dt - t0).total_seconds() / (t1 - t0).total_seconds()
            angle = a0 + fraction * (a1 - a0)
            return angle
    return None

if __name__ == "__main__":
    # Examples
    angle1 = earth_orbit_angle(datetime(2025, 11, 20, 12))  # naive treated as UTC
    angle2 = earth_orbit_angle(datetime(2025, 11, 20, 12, tzinfo=timezone.utc))  # aware datetime
    print(f"Earth orbital angle 1: {angle1:.2f}°")
    print(f"Earth orbital angle 2: {angle2:.2f}°")
