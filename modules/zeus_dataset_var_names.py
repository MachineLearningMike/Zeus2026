import xarray as xr
import requests

def list_variable_of_era5_via_gcp():
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None
    )
    print("----------------------- gcp-public-data-arco-era5 data vars ----------------------------")
    print(list(ds.data_vars))        # all data variable names
    print(list(ds.coords))           # coordinate names
    print(ds)                        # full dataset summary
    print("----------------------------------------------------------------------------------------")

def list_variable_of_era5_via_cds(dataset):
    """Return all variable names for a CDS ERA5 dataset."""
    url = f"https://cds.climate.copernicus.eu/api/v2/resources/{dataset}"
    r = requests.get(url)
    r.raise_for_status()
    info = r.json()
    
    # The "variable" field contains all available parameters
    variables = info["parameters"]["variable"]["options"]
    return variables

def list_variables():    
    # ERA5 datasets
    datasets = {
        "ERA5 single levels": "reanalysis-era5-single-levels",
        "ERA5 pressure levels": "reanalysis-era5-pressure-levels",
    }

    print("------------------------------------------------------------------------")
    for name, ds in datasets.items():
        print("-----------------------------", ds)
        vars = list_variable_of_era5_via_cds(ds)
        print(f"\n{name}: {len(vars)} variables\n")
        for v in vars:
            print(v)
    print("-------------------------------------------------------------------------")

#=================================================== GPT-generated =======================================

# Upper-air variables (5 vars × 13 pressure levels)
upper_air_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
upper_air_vars = ['z', 't', 'u', 'v', 'q']

upper_air_channels = [f"{var}_{level}" for level in upper_air_levels for var in upper_air_vars]

# Surface variables (24 vars, added sp)
surface_channels = [
    't2m', 'd2m', 'sst', 
    'u10m', 'v10m', 'u100m', 'v100m', 'msl',
    'sp',
    'lcc', 'mcc', 'hcc', 'tcc',
    'ssr', 'ssrd', 'fdir', 'ttr',
    'tp', 'mdts', 'mdww', 'mpts', 'mpww', 'shts', 'shww'
]

# Full list of 89 input/output channels
fuXi2_channels = upper_air_channels + surface_channels

# Variable index dictionary
fuXi2_var_index = {var: idx for idx, var in enumerate(fuXi2_channels)}

# Grouped dictionary by variable category
fuXi2_var_groups = {
    'upper_air': [f"{var}_{level}" for level in upper_air_levels for var in upper_air_vars],
    'surface_temp': ['t2m', 'd2m', 'sst'],
    'wind_10m': ['u10m', 'v10m'],
    'wind_100m': ['u100m', 'v100m'],
    'pressure': ['msl', 'sp'],  # updated
    'clouds': ['lcc', 'mcc', 'hcc', 'tcc'],
    'radiation': ['ssr', 'ssrd', 'fdir', 'ttr'],
    'precipitation': ['tp'],
    'waves': ['mdts', 'mdww', 'mpts', 'mpww', 'shts', 'shww']
}

# Category-to-channel indices
fuXi2_category_indices = {
    'upper_air': [fuXi2_var_index[var] for var in fuXi2_var_groups['upper_air']],
    'surface_temp': [fuXi2_var_index[var] for var in fuXi2_var_groups['surface_temp']],
    'wind_10m': [fuXi2_var_index[var] for var in fuXi2_var_groups['wind_10m']],
    'wind_100m': [fuXi2_var_index[var] for var in fuXi2_var_groups['wind_100m']],
    'pressure': [fuXi2_var_index[var] for var in fuXi2_var_groups['pressure']],
    'clouds': [fuXi2_var_index[var] for var in fuXi2_var_groups['clouds']],
    'radiation': [fuXi2_var_index[var] for var in fuXi2_var_groups['radiation']],
    'precipitation': [fuXi2_var_index[var] for var in fuXi2_var_groups['precipitation']],
    'waves': [fuXi2_var_index[var] for var in fuXi2_var_groups['waves']]
}


fuXi2_to_era5 = {
    # upper-air
    'z': 'geopotential',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',

    # surface
    't2m': '2m_temperature',
    'd2m': '2m_dewpoint_temperature',
    'sst': 'sea_surface_temperature',
    'u10m': '10m_u_component_of_wind',
    'v10m': '10m_v_component_of_wind',
    'u100m': '100m_u_component_of_wind',
    'v100m': '100m_v_component_of_wind',
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

#----------------------------------------- ERA5 via GCP ------------------------------------------
data_vars_era5_via_gcp = [
"100m_u_component_of_wind",
"100m_v_component_of_wind",
"10m_u_component_of_neutral_wind",
"10m_u_component_of_wind",
"10m_v_component_of_neutral_wind",
"10m_v_component_of_wind",
"10m_wind_gust_since_previous_post_processing",
"2m_dewpoint_temperature",
"2m_temperature",
"air_density_over_the_oceans",
"angle_of_sub_gridscale_orography",
"anisotropy_of_sub_gridscale_orography",
"benjamin_feir_index",
"boundary_layer_dissipation",
"boundary_layer_height",
"charnock",
"clear_sky_direct_solar_radiation_at_surface",
"cloud_base_height",
"coefficient_of_drag_with_waves",
"convective_available_potential_energy",
"convective_inhibition",
"convective_precipitation",
"convective_rain_rate",
"convective_snowfall",
"convective_snowfall_rate_water_equivalent",
"downward_uv_radiation_at_the_surface",
"duct_base_height",
"eastward_gravity_wave_surface_stress",
"eastward_turbulent_surface_stress",
"evaporation",
"forecast_albedo",
"forecast_logarithm_of_surface_roughness_for_heat",
"forecast_surface_roughness",
"fraction_of_cloud_cover",
"free_convective_velocity_over_the_oceans",
"friction_velocity",
"geopotential",
"geopotential_at_surface",
"gravity_wave_dissipation",
"high_cloud_cover",
"high_vegetation_cover",
"ice_temperature_layer_1",
"ice_temperature_layer_2",
"ice_temperature_layer_3",
"ice_temperature_layer_4",
"instantaneous_10m_wind_gust",
"instantaneous_eastward_turbulent_surface_stress",
"instantaneous_large_scale_surface_precipitation_fraction",
"instantaneous_moisture_flux",
"instantaneous_northward_turbulent_surface_stress",
"instantaneous_surface_sensible_heat_flux",
"k_index",
"lake_bottom_temperature",
"lake_cover",
"lake_depth",
"lake_ice_depth",
"lake_ice_temperature",
"lake_mix_layer_depth",
"lake_mix_layer_temperature",
"lake_shape_factor",
"lake_total_layer_temperature",
"land_sea_mask",
"large_scale_precipitation",
"large_scale_precipitation_fraction",
"large_scale_rain_rate",
"large_scale_snowfall",
"large_scale_snowfall_rate_water_equivalent",
"leaf_area_index_high_vegetation",
"leaf_area_index_low_vegetation",
"low_cloud_cover",
"low_vegetation_cover",
"maximum_2m_temperature_since_previous_post_processing",
"maximum_individual_wave_height",
"maximum_total_precipitation_rate_since_previous_post_processing",
"mean_boundary_layer_dissipation",
"mean_convective_precipitation_rate",
"mean_convective_snowfall_rate",
"mean_direction_of_total_swell",
"mean_direction_of_wind_waves",
"mean_eastward_gravity_wave_surface_stress",
"mean_eastward_turbulent_surface_stress",
"mean_evaporation_rate",
"mean_gravity_wave_dissipation",
"mean_large_scale_precipitation_fraction",
"mean_large_scale_precipitation_rate",
"mean_large_scale_snowfall_rate",
"mean_northward_gravity_wave_surface_stress",
"mean_northward_turbulent_surface_stress",
"mean_period_of_total_swell",
"mean_period_of_wind_waves",
"mean_potential_evaporation_rate",
"mean_runoff_rate",
"mean_sea_level_pressure",
"mean_snow_evaporation_rate",
"mean_snowfall_rate",
"mean_snowmelt_rate",
"mean_square_slope_of_waves",
"mean_sub_surface_runoff_rate",
"mean_surface_direct_short_wave_radiation_flux",
"mean_surface_direct_short_wave_radiation_flux_clear_sky",
"mean_surface_downward_long_wave_radiation_flux",
"mean_surface_downward_long_wave_radiation_flux_clear_sky",
"mean_surface_downward_short_wave_radiation_flux",
"mean_surface_downward_short_wave_radiation_flux_clear_sky",
"mean_surface_downward_uv_radiation_flux",
"mean_surface_latent_heat_flux",
"mean_surface_net_long_wave_radiation_flux",
"mean_surface_net_long_wave_radiation_flux_clear_sky",
"mean_surface_net_short_wave_radiation_flux",
"mean_surface_net_short_wave_radiation_flux_clear_sky",
"mean_surface_runoff_rate",
"mean_surface_sensible_heat_flux",
"mean_top_downward_short_wave_radiation_flux",
"mean_top_net_long_wave_radiation_flux",
"mean_top_net_long_wave_radiation_flux_clear_sky",
"mean_top_net_short_wave_radiation_flux",
"mean_top_net_short_wave_radiation_flux_clear_sky",
"mean_total_precipitation_rate",
"mean_vertical_gradient_of_refractivity_inside_trapping_layer",
"mean_vertically_integrated_moisture_divergence",
"mean_wave_direction",
"mean_wave_direction_of_first_swell_partition",
"mean_wave_direction_of_second_swell_partition",
"mean_wave_direction_of_third_swell_partition",
"mean_wave_period",
"mean_wave_period_based_on_first_moment",
"mean_wave_period_based_on_first_moment_for_swell",
"mean_wave_period_based_on_first_moment_for_wind_waves",
"mean_wave_period_based_on_second_moment_for_swell",
"mean_wave_period_based_on_second_moment_for_wind_waves",
"mean_wave_period_of_first_swell_partition",
"mean_wave_period_of_second_swell_partition",
"mean_wave_period_of_third_swell_partition",
"mean_zero_crossing_wave_period",
"medium_cloud_cover",
"minimum_2m_temperature_since_previous_post_processing",
"minimum_total_precipitation_rate_since_previous_post_processing",
"minimum_vertical_gradient_of_refractivity_inside_trapping_layer",
"model_bathymetry",
"near_ir_albedo_for_diffuse_radiation",
"near_ir_albedo_for_direct_radiation",
"normalized_energy_flux_into_ocean",
"normalized_energy_flux_into_waves",
"normalized_stress_into_ocean",
"northward_gravity_wave_surface_stress",
"northward_turbulent_surface_stress",
"ocean_surface_stress_equivalent_10m_neutral_wind_direction",
"ocean_surface_stress_equivalent_10m_neutral_wind_speed",
"ozone_mass_mixing_ratio",
"peak_wave_period",
"period_corresponding_to_maximum_individual_wave_height",
"potential_evaporation",
"potential_vorticity",
"precipitation_type",
"runoff",
"sea_ice_cover",
"sea_surface_temperature",
"significant_height_of_combined_wind_waves_and_swell",
"significant_height_of_total_swell",
"significant_height_of_wind_waves",
"significant_wave_height_of_first_swell_partition",
"significant_wave_height_of_second_swell_partition",
"significant_wave_height_of_third_swell_partition",
"skin_reservoir_content",
"skin_temperature",
"slope_of_sub_gridscale_orography",
"snow_albedo",
"snow_density",
"snow_depth",
"snow_evaporation",
"snowfall",
"snowmelt",
"soil_temperature_level_1",
"soil_temperature_level_2",
"soil_temperature_level_3",
"soil_temperature_level_4",
"soil_type",
"specific_cloud_ice_water_content",
"specific_cloud_liquid_water_content",
"specific_humidity",
"standard_deviation_of_filtered_subgrid_orography",
"standard_deviation_of_orography",
"sub_surface_runoff",
"surface_latent_heat_flux",
"surface_net_solar_radiation",
"surface_net_solar_radiation_clear_sky",
"surface_net_thermal_radiation",
"surface_net_thermal_radiation_clear_sky",
"surface_pressure",
"surface_runoff",
"surface_sensible_heat_flux",
"surface_solar_radiation_downward_clear_sky",
"surface_solar_radiation_downwards",
"surface_thermal_radiation_downward_clear_sky",
"surface_thermal_radiation_downwards",
"temperature",
"temperature_of_snow_layer",
"toa_incident_solar_radiation",
"top_net_solar_radiation",
"top_net_solar_radiation_clear_sky",
"top_net_thermal_radiation",
"top_net_thermal_radiation_clear_sky",
"total_cloud_cover",
"total_column_cloud_ice_water",
"total_column_cloud_liquid_water",
"total_column_ozone",
"total_column_rain_water",
"total_column_snow_water",
"total_column_supercooled_liquid_water",
"total_column_water",
"total_column_water_vapour",
"total_precipitation",
"total_sky_direct_solar_radiation_at_surface",
"total_totals_index",
"trapping_layer_base_height",
"trapping_layer_top_height",
"type_of_high_vegetation",
"type_of_low_vegetation",
"u_component_of_wind",
"u_component_stokes_drift",
"uv_visible_albedo_for_diffuse_radiation",
"uv_visible_albedo_for_direct_radiation",
"v_component_of_wind",
"v_component_stokes_drift",
"vertical_integral_of_divergence_of_cloud_frozen_water_flux",
"vertical_integral_of_divergence_of_cloud_liquid_water_flux",
"vertical_integral_of_divergence_of_geopotential_flux",
"vertical_integral_of_divergence_of_kinetic_energy_flux",
"vertical_integral_of_divergence_of_mass_flux",
"vertical_integral_of_divergence_of_moisture_flux",
"vertical_integral_of_divergence_of_ozone_flux",
"vertical_integral_of_divergence_of_thermal_energy_flux",
"vertical_integral_of_divergence_of_total_energy_flux",
"vertical_integral_of_eastward_cloud_frozen_water_flux",
"vertical_integral_of_eastward_cloud_liquid_water_flux",
"vertical_integral_of_eastward_geopotential_flux",
"vertical_integral_of_eastward_heat_flux",
"vertical_integral_of_eastward_kinetic_energy_flux",
"vertical_integral_of_eastward_mass_flux",
"vertical_integral_of_eastward_ozone_flux",
"vertical_integral_of_eastward_total_energy_flux",
"vertical_integral_of_eastward_water_vapour_flux",
"vertical_integral_of_energy_conversion",
"vertical_integral_of_kinetic_energy",
"vertical_integral_of_mass_of_atmosphere",
"vertical_integral_of_mass_tendency",
"vertical_integral_of_northward_cloud_frozen_water_flux",
"vertical_integral_of_northward_cloud_liquid_water_flux",
"vertical_integral_of_northward_geopotential_flux",
"vertical_integral_of_northward_heat_flux",
"vertical_integral_of_northward_kinetic_energy_flux",
"vertical_integral_of_northward_mass_flux",
"vertical_integral_of_northward_ozone_flux",
"vertical_integral_of_northward_total_energy_flux",
"vertical_integral_of_northward_water_vapour_flux",
"vertical_integral_of_potential_and_internal_energy",
"vertical_integral_of_potential_internal_and_latent_energy",
"vertical_integral_of_temperature",
"vertical_integral_of_thermal_energy",
"vertical_integral_of_total_energy",
"vertical_velocity",
"vertically_integrated_moisture_divergence",
"volumetric_soil_water_layer_1",
"volumetric_soil_water_layer_2",
"volumetric_soil_water_layer_3",
"volumetric_soil_water_layer_4",
"wave_spectral_directional_width",
"wave_spectral_directional_width_for_swell",
"wave_spectral_directional_width_for_wind_waves",
"wave_spectral_kurtosis",
"wave_spectral_peakedness",
"wave_spectral_skewness",
"zero_degree_level",
]
coords_era5_via_gcp = ['latitude', 'level', 'longitude', 'time']


# FuXi list of 88 variables
fuxi2_variables = [

    # ------------------------------
    # Upper-air variables (Z, T, U, V, Q)
    # ------------------------------

    # z
    "z_50","z_100","z_150","z_200","z_250",
    "z_300","z_400","z_500","z_600","z_700",
    "z_850","z_925","z_1000",

    # t
    "t_50","t_100","t_150","t_200","t_250",
    "t_300","t_400","t_500","t_600","t_700",
    "t_850","t_925","t_1000",

    # u
    "u_50","u_100","u_150","u_200","u_250",
    "u_300","u_400","u_500","u_600","u_700",
    "u_850","u_925","u_1000",

    # v
    "v_50","v_100","v_150","v_200","v_250",
    "v_300","v_400","v_500","v_600","v_700",
    "v_850","v_925","v_1000",

    # q
    "q_50","q_100","q_150","q_200","q_250",
    "q_300","q_400","q_500","q_600","q_700",
    "q_850","q_925","q_1000",

    # ------------------------------
    # Surface variables
    # ------------------------------

    # temp/humidity
    "t2m","d2m","sst",

    # winds
    "u10","v10","u100","v100",

    # pressure
    "msl",

    # clouds
    "lcc","mcc","hcc","tcc",

    # radiation
    "ssr","ssrd","fdir","ttr","tp",

    # wave / storm variables
    "mdts","mdww","mpts","mpww","shts","shww",
]

var_name_map = {
    # upper-air
    'z': 'geopotential',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'q': 'specific_humidity',

    # surface
    't2m': '2m_temperature',
    'd2m': '2m_dewpoint_temperature',
    'sst': 'sea_surface_temperature',
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

mike1_variables = [     # exported.

    # ------------------------------
    # Upper-air variables (Z, T, U, V, Q)
    # ------------------------------

    # z
    "z_50","z_100","z_150","z_200","z_250",
    "z_300","z_400","z_500","z_600","z_700",
    "z_850","z_925","z_1000",

    # t
    "t_50","t_100","t_150","t_200","t_250",
    "t_300","t_400","t_500","t_600","t_700",
    "t_850","t_925","t_1000",

    # u
    "u_50","u_100","u_150","u_200","u_250",
    "u_300","u_400","u_500","u_600","u_700",
    "u_850","u_925","u_1000",

    # v
    "v_50","v_100","v_150","v_200","v_250",
    "v_300","v_400","v_500","v_600","v_700",
    "v_850","v_925","v_1000",

    # q
    "q_50","q_100","q_150","q_200","q_250",
    "q_300","q_400","q_500","q_600","q_700",
    "q_850","q_925","q_1000",

    # ------------------------------
    # Surface variables
    # ------------------------------

    # temp/humidity
    "t2m","d2m","sst",

    # winds
    "u10m","v10m","u100m","v100m",

    # pressure
    "msl", "sp",    # "sp" is newly added

    # clouds
    "lcc","mcc","hcc","tcc",

    # radiation
    "ssr","ssrd","fdir","ttr","tp",

    # Nov 28, 2025
    # Exclude wave / storm variable because they are defined on ocean only.
    # # wave / storm variables
    # "mdts","mdww","mpts","mpww","shts","shww",
]

print("len mike1_variables: ", len(mike1_variables))

def find_missing_vars():
    fuXi_keys = set(var_name_map.keys())
    missing = []

    for var in mike1_variables:
        base_var = var.split('_')[0]  # get prefix before underscore
        if base_var not in fuXi_keys:
            missing.append(var)

    if missing:
        print("Variables not found in var_name_map:", missing)
    else:
        print("All variables are accounted for.")

    print(len(mike1_variables))
    
find_missing_vars()


def check_for_name_consistency():
    # Example: list of official ERA5 variable names
    era5_vars = [
        'geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind',
        'specific_humidity', '2m_temperature', '2m_dewpoint_temperature',
        'sea_surface_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
        '100m_u_component_of_wind', '100m_v_component_of_wind',
        'surface_pressure', 'mean_sea_level_pressure',
        'low_cloud_cover', 'medium_cloud_cover', 'high_cloud_cover', 'total_cloud_cover',
        'surface_solar_radiation_downwards', 'total_sky_direct_solar_radiation_at_surface',
        'top_net_thermal_radiation', 'total_precipitation',
        'mean_direction_of_total_swell', 'mean_direction_of_wind_waves',
        'mean_period_of_total_swell', 'mean_period_of_wind_waves',
        'significant_height_of_total_swell', 'significant_height_of_wind_waves'
    ]

    # Check
    mapped_names = set(var_name_map.values())
    missing_in_era5 = [v for v in mapped_names if v not in era5_vars]

    if missing_in_era5:
        print("These mapped names are NOT in ERA5:", missing_in_era5)
    else:
        print("All mapped names exist in ERA5.")
        
check_for_name_consistency()