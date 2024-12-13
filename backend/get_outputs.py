import os
import joblib
import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache
from collections import defaultdict
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap

from fair_wrap.fair_utils import *

REGIONAL_MODEL_NAMES = [
    "CESM2-WACCM"
]
NUM_EMULATORS = 100
# CESM2-WACCM dimensions
NUM_LAT = 192
NUM_LON = 288

MIN_TARGET = 0.5
MAX_TARGET = 4.5
STEP_TARGET = 0.1

VAR2INFO = {
    "tas": ("Temperature in °C", "Temperature (°C)", "coolwarm", "#00FF00"),
    "p-e": ("Water Availability in mm/day", "Water Availability (mm/day)", LinearSegmentedColormap.from_list(
        "brown_to_white_to_blue", ["brown", "khaki", "white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
    "tasmin": ("Minimum Temperature in °C", "Min Temperature (°C)", "coolwarm", "#00FF00"),
    "tasmax": ("Maximum Temperature in °C", "Max Temperature (°C)", "coolwarm","#00FF00"),
    "exposure_above_40": ("Person-Days Above 40°C (Millions)", "Exposure to > 40°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_red", ["white", "red"], N=100
        ), "#00FF00"),
    "exposure_above_35": ("Person-Days Above 35°C (Millions)", "Exposure to > 35°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_red", ["white", "red"], N=100
        ), "#00FF00"),
    "exposure_below_0": ("Person-Days Below 0°C (Millions)", "Exposure to < 0°C (million person-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "#00FF00"),
    "exposure_above_10": ("Person-Days Above 10mm/day (Millions)", "Exposure to > 10mm (million mm-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
    "exposure_above_20": ("Person-Days Above 20mm/day (Millions)", "Exposure to > 20mm (million mm-days)", LinearSegmentedColormap.from_list(
            "white_to_blue", ["white", "deepskyblue", "mediumblue"], N=100
        ), "red"),
}

exposurevar2var = {
    "exposure_above_40": "tas_above_40",
    "exposure_above_35": "tas_above_35",
    "exposure_below_0": "tas_below_0",
    "exposure_above_10": "pr_above_10",
    "exposure_above_20": "pr_above_20"
}


def get_global_temp(ssp_scenario):
    # Get global temperature from FaIR
    df_emis, df_configs, df_solar, df_volcanic = get_dataframes()

    initial_conditions = None

    runner = get_runner(
        df_emis, df_configs,
        df_solar, df_volcanic,
        co2_input=None, ssp_scenario=ssp_scenario,
        initial_conditions=initial_conditions,
        calibrate_to="cesm"
    )

    runner.run(progress=False)
    preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(1850, 1900)).mean('timebounds')

    global_temp = (runner.temperature.sel(layer=0) - preindustrial_temp).mean('config')

    return global_temp


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_historical_model(var, data_dir, cache_dir):
    """Gets the unrebased historical model data for the given variable."""
    historical_cache_path = cache_dir / f"historical_{var}.nc"

    if historical_cache_path.exists():
        print(f"Found {historical_cache_path}")
        historical_model_data = xr.open_dataarray(historical_cache_path, autoclose=True)
    else:
        is_above_below = "above" in var or "below" in var
        is_exposure = "exposure" in var
        if var in ["pr", "tas", "p-e"]:
            historical_model_data = xr.open_dataarray(data_dir / var / "output_gauss-baseline.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        elif var in ["tasmin", "tasmax", "tas_above_40", "tas_above_35", "tas_below_0", "pr_above_10", "pr_above_20"]:
            historical_model_data = xr.open_dataarray(data_dir / var / "output_gauss-cmip_historical.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        elif is_exposure:
            daily_var = exposurevar2var[var]
            historical_model_data = xr.open_dataarray(data_dir / daily_var / "output_gauss-cmip_historical.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        else:
            historical_model_data = None

        if historical_model_data is not None:
            if is_above_below:
                historical_model_data = historical_model_data.where(historical_model_data > 0, 0)

            if is_above_below or var in ["pr", "p-e"]:
                historical_model_data = clip_to_land(data_dir, historical_model_data)
                if is_exposure:
                    historical_model_data = get_population_exposure(data_dir, historical_model_data)
            historical_model_data = historical_model_data.sel(time=slice(None, 2014))

            historical_model_data.to_netcdf(historical_cache_path)

    return historical_model_data


def get_historical_obs_global_mean_temp(data_dir):
    # Source: https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt
    historical_obs_data = pd.read_csv(data_dir / "Land_and_Ocean_summary.txt", skiprows=56, sep="\s+")
    historical_obs_data = historical_obs_data.rename(columns={"%": "Year", "Year,": "Temperature"})

    # Convert to xarray
    historical_obs_data = xr.DataArray(historical_obs_data["Temperature"].values, dims=('time'), coords={'time': historical_obs_data["Year"].values})

    # Rebase by datasets 1850-1900 mean
    historical_obs_data = historical_obs_data - historical_obs_data.sel(time=slice(1850, 1900)).mean()

    return historical_obs_data


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_models(model_dir):

    fair2smip_coefs, fair2smip_intercepts = [], []

    for model in REGIONAL_MODEL_NAMES:
        for i in range(NUM_EMULATORS):
            fair2smip = joblib.load(model_dir / f"fair_to_smip_{model}_{i}.joblib")
            fair2smip_coefs.append(fair2smip.coef_)
            fair2smip_intercepts.append(fair2smip.intercept_)

    fair2smip_coefs = np.concatenate(fair2smip_coefs)
    fair2smip_intercepts = np.concatenate(fair2smip_intercepts)

    fair2smip = LinearRegression(n_jobs=-1)
    fair2smip.coef_ = fair2smip_coefs
    fair2smip.intercept_ = fair2smip_intercepts

    return fair2smip


def get_smip(global_temp, fair2smip):
    # Get regional projections for each model
    smip = fair2smip.predict(global_temp.sel(timebounds=slice(2015, None)))

    smip = smip.reshape((smip.shape[0], len(REGIONAL_MODEL_NAMES), NUM_EMULATORS, NUM_LAT, NUM_LON))

    return smip


def get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, cache_dir):
    # Get regional map from global temperature
    smip = get_smip(global_temp, fair2smip)
    regional_map = xr.DataArray(smip, dims=('time', 'model', 'emulator', 'lat', 'lon'))
    regional_map.coords['time'] = np.arange(2015, SIM_END_YEAR+1)
    regional_map.coords['model'] = REGIONAL_MODEL_NAMES
    regional_map.coords['emulator'] = np.arange(NUM_EMULATORS)
    correct_lat = np.load(data_dir / "correct_lat.npy")
    regional_map.coords['lat'] = correct_lat
    regional_map.coords['lon'] = np.linspace(0, 360, NUM_LON, endpoint=False)
    regional_map = xr.Dataset({var: regional_map})
    regional_map = regional_map.sel(model="CESM2-WACCM").mean(dim='emulator')[var]

    is_above_below = "above" in var or "below" in var
    is_exposure = "exposure" in var

    # Floor the values to 0 if the variable counts number of days
    if is_above_below:
        regional_map = regional_map.where(regional_map > 0, 0)

    # Clip pr and days to land only
    if is_above_below or var in ["pr", "p-e"]:
        regional_map = clip_to_land(data_dir, regional_map)
        if is_exposure:
            regional_map = get_population_exposure(data_dir, regional_map)

    historical_map = get_historical_model(var, data_dir, cache_dir)

    ### Get historical temperature data ###
    if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
        historical_rebase = historical_map.sel(time=slice(1850, 1900)).mean('time')
        regional_map = regional_map - historical_rebase

    return regional_map


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_map(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target):
    """Returns the regional map for the given variable, SSP scenario, and temperature target."""
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_regional_{var}.nc"

    if regional_cache_path.exists():
        print(f"Found {regional_cache_path}")
        regional_map = xr.open_dataarray(regional_cache_path, autoclose=True)
    else:
        # First get FaIR global temperature
        global_temp = get_global_temp(ssp_scenario)

        is_exposure = "exposure" in var

        if is_exposure:
            daily_var = exposurevar2var[var]
            variable_dir = model_dir / daily_var
        else:
            variable_dir = model_dir / var
        fair2smip = get_regional_models(variable_dir)

        # Get regional map
        regional_map = get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, cache_dir)

    return regional_map


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_interpolator(interpolator_path):
    interpolator = xr.open_dataset(interpolator_path)
    return interpolator


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, temp_target):

    ### Get regional temperature data ###
    regional_temp = get_regional_map("tas", data_dir, model_dir, cache_dir, ssp_scenario, temp_target)

    ### Get temperature difference data ###
    global_mean = regional_temp.weighted(np.cos(np.deg2rad(regional_temp.lat))).mean(('lat', 'lon'))
    temp_diff = temp_target - global_mean

    return temp_diff


def get_regional_delta_without_rampup(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, temp_diff):

    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_delta_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_regional_delta_{var}.nc"

    if regional_delta_cache_path.exists():
        print(f"Found {regional_delta_cache_path}")
        regional_delta = xr.open_dataarray(regional_delta_cache_path, autoclose=True)

    else:
        is_above_below = "above" in var or "below" in var
        is_exposure = "exposure" in var

        if is_exposure:
            var = exposurevar2var[var]

        variable_dir = model_dir / var
        
        interpolator = get_interpolator(variable_dir / "interpolator.nc")

        regional_delta = interpolator.sel(features='slope') * temp_diff

        regional_delta = regional_delta[var]

        if is_above_below or var in ["pr", "p-e"]:
            regional_delta = clip_to_land(data_dir, regional_delta)
            if is_exposure:
                regional_delta = get_population_exposure(data_dir, regional_delta)

        regional_delta.to_netcdf(regional_delta_cache_path)

    return regional_delta


def get_regional_delta(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, ramp_up, temp_diff, start_year):

    regional_delta = get_regional_delta_without_rampup(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, temp_diff)
    # Slice the data by the start year
    # Because cached data just caches the whole period
    temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
    regional_delta = regional_delta.sel(time=slice(start_year, SIM_END_YEAR))

    # Set all values to 0 for the indices of time where temp_diff > 0
    no_sai_indices = temp_diff > 0
    regional_delta = regional_delta.where(~no_sai_indices, 0)

    if ramp_up > 0 and not no_sai_indices.all():
        # Apply a linear ramp-up from first year of > temp target for ramp_up years
        actual_sai_start_year = temp_diff.time.values[~no_sai_indices][0]
        # Ensure the ramp-up is not longer than the actual SAI period
        ramp_up = min(ramp_up, SIM_END_YEAR - actual_sai_start_year)
        # Get the indices of the ramp-up period
        ramp_up_indices = np.arange(actual_sai_start_year, actual_sai_start_year + ramp_up)
        # Multiply the delta by the ramp-up factor, going from 1/ramp_up to 1
        ramp_up_factor = np.linspace(1/ramp_up, 1, ramp_up)
        regional_delta.loc[dict(time=ramp_up_indices)] *= ramp_up_factor

    return regional_delta


def get_so2_by_latitude(interpolator, temp_diff, ramp_up):
    so2_by_latitude = interpolator.sel(features='slope') * temp_diff

    # Set all values to 0 for the indices of time where temp_diff > 0
    no_sai_indices = temp_diff > 0
    so2_by_latitude = so2_by_latitude.where(~no_sai_indices, 0)

    if ramp_up > 0 and not no_sai_indices.all():
        # Apply a linear ramp-up from first year of > temp target for ramp_up years
        actual_sai_start_year = temp_diff.time.values[~no_sai_indices][0]
        # Ensure the ramp-up is not longer than the actual SAI period
        ramp_up = min(ramp_up, SIM_END_YEAR - actual_sai_start_year)
        # Get the indices of the ramp-up period
        ramp_up_indices = np.arange(actual_sai_start_year, actual_sai_start_year + ramp_up)
        # Multiply the so2 by the ramp-up factor, going from 1/ramp_up to 1
        ramp_up_factor = np.linspace(1/ramp_up, 1, ramp_up)
        so2_by_latitude.loc[dict(time=ramp_up_indices)] *= ramp_up_factor

    return so2_by_latitude


def create_mask(xarray_dataset, gdf):
    # The geometries are -180 to 180 but the xarray dataset is 0 to 360, so convert this
    xarray_dataset = xarray_dataset.copy()
    original_lon = xarray_dataset.lon
    original_lat = xarray_dataset.lat
    xarray_dataset.coords['lon'] = xr.where(xarray_dataset.lon > 180, xarray_dataset.lon - 360, xarray_dataset.lon)
    xarray_dataset = xarray_dataset.sortby('lon')
    xarray_dataset = xarray_dataset.rio.write_crs("EPSG:4326")
    xarray_dataset = xarray_dataset.rio.set_spatial_dims("lon", "lat")
    mask = xarray_dataset.rio.clip(gdf.geometry, invert=True, drop=False)
    # Convert the mask back to 0 to 360
    mask.coords['lon'] = xr.where(mask.lon < 0, mask.lon + 360, mask.lon)
    mask = mask.sortby(['lon', 'lat'])
    mask.coords['lon'] = original_lon
    mask.coords['lat'] = original_lat

    return mask.isnull()


def clip_to_land(data_dir, regional_map):
    geojson_dir = data_dir / "geojsons"
    continental_gdf = gpd.read_file(geojson_dir / "IPCC-WGII-continental-regions.geojson")

    # Take union of all geometries
    continental_gdf = gpd.GeoDataFrame(geometry=[continental_gdf.unary_union], crs=continental_gdf.crs)
    land_mask = create_mask(regional_map, continental_gdf)
    regional_map = regional_map.where(land_mask, np.nan)
    return regional_map


def regional_aggregation(xarray_dataset, weights, op):
    xarray_dataset = xarray_dataset.weighted(weights)
    if op == "mean":
        return xarray_dataset.mean(dim=('lat', 'lon'))
    elif op == "max":
        return xarray_dataset.quantile(1, dim=('lat', 'lon'))
    elif op == "min":
        return xarray_dataset.quantile(0, dim=('lat', 'lon'))
    elif op == "sum":
        return xarray_dataset.sum(dim=('lat', 'lon'))
    else:
        raise ValueError(f"Unknown operation: {op}")
    

def apply_constraints(time_series, smooth_type, filter_width=10, filter_order=3):

    if smooth_type == "min_norm":
        # Calculate the long-term mean of the time series
        long_term_mean = np.mean(time_series)

        # Minimum norm: pad with the long-term mean
        pad_min_norm = np.pad(time_series, (filter_width, filter_width), 'constant', constant_values=(long_term_mean, long_term_mean))

    elif smooth_type == "min_slope":
        # Minimum slope: reflect the series about the boundary
        pad_min_slope = np.pad(time_series, (filter_width, filter_width), 'reflect')
    
    elif smooth_type == "min_roughness":
        # Minimum roughness: reflect about the time boundary and vertically about the y-axis
        pad_min_roughness = np.pad(time_series, (filter_width, filter_width), 'symmetric')

    else:
        raise ValueError(f"Unknown smooth type: {smooth_type}")

    # Butterworth low-pass filter setup
    b, a = butter(filter_order, 0.1)  # 0.1 is a normalized frequency; adjust as needed

    # Apply filter to the padded series
    if smooth_type == "min_norm":
        smoothed_min_norm = filtfilt(b, a, pad_min_norm)
        smoothed =  smoothed_min_norm[filter_width:-filter_width]
    elif smooth_type == "min_slope":
        smoothed_min_slope = filtfilt(b, a, pad_min_slope)
        smoothed = smoothed_min_slope[filter_width:-filter_width]
    elif smooth_type == "min_roughness":
        smoothed_min_roughness = filtfilt(b, a, pad_min_roughness)
        smoothed = smoothed_min_roughness[filter_width:-filter_width]

    # Convert to xarray DataArray
    smoothed = xr.DataArray(smoothed, dims=('time'), coords={'time': time_series.time})

    return smoothed


@lru_cache(maxsize=1)  # Caches the last call
def get_population_data(data_dir):
    # Went with the pregridded data from Dan instead of our processed data
    processed_population_path = data_dir / "Regridded_pop_num.npy"
    population_np = np.load(processed_population_path)

    return population_np


@lru_cache(maxsize=1)  # Caches the last call
def get_country_boundaries(data_dir):
    # Source: world administrative boundaries from OpenDataSoft v2.1
    country_boundaries = gpd.read_file(data_dir / "world-administrative-boundaries.geojson")
    
    return country_boundaries


def aggregate_xarray_by_geopandas(xr_data, gdf, fill_value=0):
    # Ensure CRS alignment
    if not gdf.crs:
        raise ValueError("GeoDataFrame has no CRS. Define a CRS for the GeoDataFrame.")
    if not xr_data.rio.crs:
        raise ValueError("xarray data has no CRS. Define a CRS for the xarray data.")

    gdf = gdf.to_crs(xr_data.rio.crs.to_string())

    # # Revert longitude shift (0 to 360) to (-180 to 180)
    # [180, 360] -> [-180, 0]
    # [0, 180] -> [0, 180]
    xr_data = xr_data.assign_coords(lon=(((xr_data.lon + 180) % 360) - 180)).sortby('lon')

    # For each lat/lon in xr_data, find its correspondig country and create a mask
    # where each value of the mask corresponds to the country ID
    lon, lat = np.meshgrid(xr_data.lon.values, xr_data.lat.values)
    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon.flatten(), lat.flatten()), 
                                  crs=xr_data.rio.crs.to_string())
    gdf.index = gdf.index + 1
    joined_gdf = gpd.sjoin(points_gdf, gdf, how="left", predicate='intersects')
    joined_gdf = joined_gdf.fillna(0)
    rasterized_countries = joined_gdf['index_right'].values.reshape(xr_data.lat.size, xr_data.lon.size)

    # Determine the number of time steps and the unique group IDs
    if xr_data.dims != ('time', 'lat', 'lon'):
        # Array is in the wrong order (lat, lon, time), so transpose it
        xr_data = xr_data.transpose('time', 'lat', 'lon')
    time_series_values = xr_data.values
    num_time_steps = time_series_values.shape[0]
    unique_groups = np.unique(rasterized_countries)

    # Prepare an output array with the same shape
    output_values = np.zeros_like(time_series_values)
    
    # Iterate over each time step
    for t in range(num_time_steps):
        # For each group, sum values over the group members
        for group in unique_groups:
            if group == 0:
                continue
            mask = (rasterized_countries == group)
            # Sum values where the mask is True and assign to the same locations in the output array
            try:
                group_sum = np.nansum(time_series_values[t][mask])
            except:
                breakpoint()
            output_values[t][mask] = group_sum

    # Convert the output array to an xarray DataArray
    xr_data.values = output_values

    # Revert the longitude shift back to 0 to 360
    # [-180, 0] -> [180, 360]
    # [0, 180] -> [0, 180]
    xr_data = xr_data.assign_coords(lon=((xr_data.lon + 360) % 360)).sortby('lon')

    return xr_data


def get_population_exposure(data_dir, ds, country_level=False):

    population_np = get_population_data(data_dir)
    population_xr = xr.DataArray(population_np, dims=('lon', 'lat'), coords={'lat': ds.lat, 'lon': ds.lon})
    # Shift lon values by 180
    population_xr = population_xr.assign_coords(lon=(((population_xr.lon + 180) % 360))).sortby('lon').transpose()

    exposure = ds * population_xr

    exposure /= 1e6  # Convert to millions

    if country_level:
        # Aggregate to country level by summing over lat and lon
        country_boundaries = get_country_boundaries(data_dir)

        exposure = exposure.rio.write_crs("EPSG:4326")
        exposure = exposure.rio.set_spatial_dims("lon", "lat")
        exposure = aggregate_xarray_by_geopandas(exposure, country_boundaries)

    return exposure


def get_outputs(ssp_scenario, temp_target, spatial_gdf, spatial_item,
                decade_start_year, decade_end_year, start_year, ramp_up,
                data_dir, model_dir, cache_dir, var=None):

    ### Set up directories ###
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    cache_dir = Path(cache_dir)

    ### Set up variables to process ###
    variables_to_process = [var] if var else VAR2INFO.keys()

    ### Get regional temperature data ###
    regional_temp = get_regional_map("tas", data_dir, model_dir, cache_dir, ssp_scenario, temp_target)

    ### Get temperature difference data ###
    temp_diff = get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, temp_target)

    ### Get historical observations temperature data ###
    historical_obs_data = get_historical_obs_global_mean_temp(data_dir)

    ### Set up outputs ###
    output_data = defaultdict(dict)

    for var in variables_to_process:

        is_exposure = "exposure" in var
        is_above_below = "above" in var or "below" in var

        # Load projection data
        if var == "tas":
            regional_map = regional_temp
        else:
            regional_map = get_regional_map(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target)

        # Load historical data
        historical_model = get_historical_model(var, data_dir, cache_dir)
        if historical_model is not None:
            historical_rebase = historical_model.sel(time=slice(1850, 1900)).mean('time')
            if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
                historical_model = historical_model - historical_rebase

        ### Get regional mean and delta data ###
        regional_delta = get_regional_delta(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, ramp_up, temp_diff, start_year)

        regional_mean = regional_map.sel(time=slice(decade_start_year, decade_end_year)).mean('time')
        regional_delta_mean = regional_delta.sel(time=slice(decade_start_year, decade_end_year)).mean('time')

        if spatial_gdf is not None and spatial_item is not None:
            mask = create_mask(regional_mean, spatial_gdf[spatial_gdf.name == spatial_item])
            output_data[var]["mask"] = mask

        output_data[var]["regional_mean"] = regional_mean
        output_data[var]["regional_delta_mean"] = regional_delta_mean

        #### Mean over time plot ####
        # Get period between 2015 and SAI start year, SAI start year to SAI end year, and SAI end year to 2100
        recent_regional_map = regional_map.sel(time=slice(2015, start_year-1))
        regional_map = regional_map.sel(time=slice(start_year, SIM_END_YEAR))

        # Always use mean for now
        if is_exposure:
            op = "sum"
        else:
            op = "mean"

        if is_exposure:
            # Define data array of all ones like regional_map.lat
            weights = xr.DataArray(np.ones_like(regional_map.lat), dims=('lat'), coords={'lat': regional_map.lat})
        else:
            weights = np.cos(np.deg2rad(regional_map.lat))

        if spatial_gdf is None or spatial_item is None:
            if historical_model is not None:
                # Compute historical global mean
                historical_mean = regional_aggregation(historical_model, weights, op)
            # Compute recent global mean
            recent_global_mean = regional_aggregation(recent_regional_map, weights, op)
            # Compute mean temperature with and without SAI
            mean_no_sai = regional_aggregation(regional_map, weights, op)
            regional_map_with_sai = regional_map + regional_delta
            if is_above_below:
                regional_map_with_sai = regional_map_with_sai.where(regional_map_with_sai > 0, 0)
                # Clip again because the above sets nan values to 0
                regional_map_with_sai = clip_to_land(data_dir, regional_map_with_sai)
            mean_with_sai = regional_aggregation(regional_map_with_sai, weights, op)
        else:
            if historical_model is not None:
                # Compute historical global mean averaged over the selected geometry
                if is_above_below:
                    historical_model = historical_model.where(historical_model > 0, 0)
                historical_mask = create_mask(historical_model, spatial_gdf[spatial_gdf.name == spatial_item])
                masked_historical_model = historical_model.where(historical_mask, np.nan)
                historical_mean = regional_aggregation(masked_historical_model, weights, op)
            # Compute recent global mean averaged over the selected geometry
            recent_mask = create_mask(recent_regional_map, spatial_gdf[spatial_gdf.name == spatial_item])
            masked_recent_regional_map = recent_regional_map.where(recent_mask, np.nan)
            recent_global_mean = regional_aggregation(masked_recent_regional_map, weights, op)
            # Compute mean temperature averaged over the selected geometry with and without SAI
            mask = create_mask(regional_map, spatial_gdf[spatial_gdf.name == spatial_item])
            masked_regional_map = regional_map.where(mask, np.nan)
            masked_regional_delta = regional_delta.where(mask, np.nan)
            # mean_no_sai = masked_regional_map.weighted(weights)
            mean_no_sai = regional_aggregation(masked_regional_map, weights, op)
            masked_regional_map_with_sai = masked_regional_map + masked_regional_delta
            if is_above_below:
                masked_regional_map_with_sai = masked_regional_map_with_sai.where(masked_regional_map_with_sai > 0, 0)
            mean_with_sai = regional_aggregation(masked_regional_map_with_sai, weights, op)

        if historical_model is not None:
            # Concatenate the historical model data with the recent data
            historical_model_global_mean = xr.concat([historical_mean, recent_global_mean], dim='time')
        else:
            historical_model_global_mean = recent_global_mean

        # Smooth concatenated historical and mean no SAI data
        concatenated_global_mean = xr.concat([historical_model_global_mean, mean_no_sai], dim='time')
        concatenated_global_mean = apply_constraints(concatenated_global_mean, "min_roughness")
        historical_model_global_mean = concatenated_global_mean.sel(time=slice(None, historical_model_global_mean.time[-1]))
        mean_no_sai = concatenated_global_mean.sel(time=slice(historical_model_global_mean.time[-1], None))

        #### PDF plot ####
        if spatial_gdf is None or spatial_item is None:
            regional_map = regional_map.sel(time=slice(decade_start_year, decade_end_year))
            regional_delta = regional_delta.sel(time=slice(decade_start_year, decade_end_year))
            # Get histogram of the distribution of values in the regional map
            pdf_no_sai = regional_map.values.flatten()
            pdf_with_sai = (regional_map + regional_delta).values.flatten()
            # Get PDF of historical model data from 1950-1960
            if historical_model is not None:
                pdf_historical = historical_model.sel(time=slice(1950, 1960)).values.flatten()
        else:
            masked_regional_map = masked_regional_map.sel(time=slice(decade_start_year, decade_end_year))
            masked_regional_delta = masked_regional_delta.sel(time=slice(decade_start_year, decade_end_year))
            # Get histogram of the distribution of values in the regional map
            pdf_no_sai = masked_regional_map.values.flatten()
            pdf_with_sai = (masked_regional_map + masked_regional_delta).values.flatten()
            # Get PDF of historical model data from 1950-1960
            if historical_model is not None:
                pdf_historical = masked_historical_model.sel(time=slice(1950, 1960)).values.flatten()
        
        ### Get the histograms ###
        pdf_no_sai = pdf_no_sai[~np.isnan(pdf_no_sai)]
        pdf_with_sai = pdf_with_sai[~np.isnan(pdf_with_sai)]

        if is_above_below:
            # Set predefined bins
            # bins = [0, 1, 5, 10, 20, 50, np.inf]
            bins = [1, 5, 10, 20, 50, np.inf]
            # Remove 0 values
            pdf_no_sai = pdf_no_sai[pdf_no_sai > 0]
            pdf_with_sai = pdf_with_sai[pdf_with_sai > 0]

        else:
            bins = 150

        no_sai_counts, no_sai_bins = np.histogram(pdf_no_sai, bins=bins, density=False)
        with_sai_counts, with_sai_bins = np.histogram(pdf_with_sai, bins=bins, density=False)

        no_sai_counts = no_sai_counts / no_sai_counts.sum()
        with_sai_counts = with_sai_counts / with_sai_counts.sum()

        if historical_model is not None:
            pdf_historical = pdf_historical[~np.isnan(pdf_historical)]
            historical_counts, historical_bins = np.histogram(pdf_historical, bins=bins, density=False)
            historical_counts = historical_counts / historical_counts.sum()

        ### Get the outputs ###
        # Mean over time
        output_data[var]["mean_over_time"] = {
            "no_sai": mean_no_sai,
            "with_sai": mean_with_sai,
            "historical_model": historical_model_global_mean
        }

        if var == 'tas':
            output_data[var]["mean_over_time"]["historical_obs"] = historical_obs_data

        # Load natural variability
        if is_exposure:
            natural_variability = np.load(data_dir / exposurevar2var[var] / "natural_variability.npy")
        else:
            natural_variability = np.load(data_dir / var / "natural_variability.npy")
        if is_exposure:
            population_np = get_population_data(data_dir)
            if spatial_gdf is None or spatial_item is None:
                # Multiply by total population
                natural_variability = natural_variability * population_np.sum() / 1e6
            else:
                # Multiply by population in the selected region
                population_xr = xr.DataArray(population_np, dims=('lon', 'lat'), coords={'lat': regional_map.lat, 'lon': regional_map.lon})
                # Shift lon values by 180
                population_xr = population_xr.assign_coords(lon=(((population_xr.lon + 180) % 360))).sortby('lon').transpose()
                mask = create_mask(population_xr, spatial_gdf[spatial_gdf.name == spatial_item])
                natural_variability = natural_variability * population_xr.where(mask).sum() / 1e6
            natural_variability = natural_variability.item()

        output_data[var]["mean_over_time"]["natural_variability"] = natural_variability

        # PDF
        output_data[var]["distribution"] = {
            "no_sai": {
                "counts": no_sai_counts.tolist(),
                "bin_edges": no_sai_bins.tolist()
            },
            "with_sai": {
                "counts": with_sai_counts.tolist(),
                "bin_edges": with_sai_bins.tolist()
            }
        }

        if historical_model is not None:
            output_data[var]["distribution"]["historical"] = {
                "counts": historical_counts.tolist(),
                "bin_edges": historical_bins.tolist()
            }

    ### Latitude vs. Tg SO2 plot ###
    temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
    interpolator = get_interpolator(model_dir / "so2" / "interpolator.nc")
    so2_by_latitude = get_so2_by_latitude(interpolator, temp_diff, ramp_up)["so2"]
    so2_by_latitude = so2_by_latitude.drop_vars('model')
    global_so2 = so2_by_latitude.sum(('lat'))

    # Store the SO2 data
    output_data["temp_diff"] = temp_diff
    output_data["so2_by_latitude"] = so2_by_latitude
    output_data["global_so2"] = global_so2

    return output_data
