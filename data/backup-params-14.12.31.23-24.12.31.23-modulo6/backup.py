# Once training has begun, you cannot change the variable name list, including its order, because it's the signature of data for training and inference.
variable_name_list_unchangeable = ["z_50","t_50","u_50","v_50","q_50","z_100","t_100","u_100","v_100","q_100","z_150","t_150","u_150","v_150","q_150","z_200","t_200","u_200","v_200","q_200","z_250","t_250","u_250","v_250","q_250","z_300","t_300","u_300","v_300","q_300","z_400","t_400","u_400","v_400","q_400","z_500","t_500","u_500","v_500","q_500","z_600","t_600","u_600","v_600","q_600","z_700","t_700","u_700","v_700","q_700","z_850","t_850","u_850","v_850","q_850","z_925","t_925","u_925","v_925","q_925","z_1000","t_1000","u_1000","v_1000","q_1000","u10m","v10m","msl","lcc","mcc","hcc","tcc","ssr","ssrd","fdir","ttr","t2m","tp","u100m","v100m","d2m","sp"]
var_name_map_unchangeable = {
    # upper-air
    'z': 'geopotential',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',

    # surface
    't2m': '2m_temperature',
    'd2m': '2m_dewpoint_temperature',
    'sst': 'sea_surface_temperature',     # NaN, on land
    'u10m': '10m_u_component_of_wind',
    'v10m': '10m_v_component_of_wind',
    'u100m': '100m_u_component_of_wind',
    'v100m': '100m_v_component_of_wind',
    'sp' : 'surface_pressure',
    'msl': 'mean_sea_level_pressure',
    'lcc': 'low_cloud_cover',
    'mcc': 'medium_cloud_cover',
    'hcc': 'high_cloud_cover',
    'tcc': 'total_cloud_cover',
    'ssr': 'surface_solar_radiation_downwards',
    'ssrd': 'surface_solar_radiation_downwards',
    'fdir': 'total_sky_direct_solar_radiation_at_surface',
    'ttr': 'top_net_thermal_radiation',
    'tp': 'total_precipitation',
    'mdts': 'mean_direction_of_total_swell',
    'mdww': 'mean_direction_of_wind_waves',
    'mpts': 'mean_period_of_total_swell',
    'mpww': 'mean_period_of_wind_waves',
    'shts': 'significant_height_of_total_swell',
    'shww': 'significant_height_of_wind_waves'
}