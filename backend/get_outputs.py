import rasterio
import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache
from collections import defaultdict

from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR

from .constants import *
from .historical import get_historical_model, get_historical_obs_global_mean_temp
from .baseline import get_regional_map
from .delta import get_regional_delta
from .p_values import get_regional_p_values
from .population import get_population_data
from .variable import get_variable_regional_delta
from .utils import get_interpolator, create_mask, regional_aggregation, apply_constraints, clip_to_land


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, temp_target):

    ### Get regional temperature data ###
    regional_temp, _ = get_regional_map("tas", data_dir, model_dir, cache_dir, ssp_scenario)
    
    ### Get temperature difference data ###
    global_mean = regional_temp.weighted(np.cos(np.deg2rad(regional_temp.lat))).mean(('lat', 'lon'))
    temp_diff = temp_target - global_mean

    return temp_diff


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


def get_outputs(ssp_scenario, temp_target, spatial_gdf, spatial_item,
                decade_start_year, decade_end_year, start_year, ramp_up,
                data_dir, model_dir, cache_dir, var=None, variable_injection=None):

    ### Set up directories ###
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    cache_dir = Path(cache_dir)

    ### Set up variables to process ###
    if var is None:
        variables_to_process = VAR2INFO.keys()
    elif isinstance(var, str):
        variables_to_process = [var]
    else:
        variables_to_process = var

    ### Get regional temperature data ###
    regional_temp, regional_temp_no_sai_p_values = get_regional_map("tas", data_dir, model_dir, cache_dir, ssp_scenario)

    if variable_injection is None:
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
            regional_map, regional_no_sai_p_values = regional_temp, regional_temp_no_sai_p_values
        else:
            regional_map, regional_no_sai_p_values = get_regional_map(var, data_dir, model_dir, cache_dir, ssp_scenario)

        # Slice regional_no_sai_p_values by the decade
        if regional_no_sai_p_values is not None:
            regional_no_sai_p_values = regional_no_sai_p_values.sel(time=decade_end_year)

        # Load historical data
        historical_model = get_historical_model(var, data_dir, cache_dir)
        if historical_model is not None:
            historical_rebase = historical_model.sel(time=slice(1850, 1900)).mean('time')
            if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
                historical_model = historical_model - historical_rebase

        ### Get regional mean and delta data ###
        if variable_injection is None:
            regional_delta = get_regional_delta(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, ramp_up, temp_diff, start_year)
        else:
            if var != "p-e":
                regional_delta = get_variable_regional_delta(var, data_dir, cache_dir, variable_injection)
            else:
                regional_delta_p = get_variable_regional_delta("pr", data_dir, cache_dir, variable_injection)
                regional_delta_e = get_variable_regional_delta("e", data_dir, cache_dir, variable_injection)
                regional_delta = regional_delta_p - regional_delta_e

        ## T-test 
        if historical_model is not None:
            simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
            if variable_injection is None:
                regional_p_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_regional_sai_p_values_{var}.nc"
                regional_p_cache_path_exists = regional_p_cache_path.exists()
            else:
                regional_p_cache_path_exists = False
            if regional_p_cache_path_exists:
                print(f"Found {regional_p_cache_path}")
                regional_sai_p_values = xr.open_dataarray(regional_p_cache_path, autoclose=True)
            else:
                if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
                    # The regional map is already rebased, so need to add the historical rebase back before comparing
                    regional_sai_p_values = get_regional_p_values(var, data_dir, historical_rebase, regional_map + regional_delta + historical_rebase)
                else:
                    regional_sai_p_values = get_regional_p_values(var, data_dir, historical_rebase, regional_map + regional_delta)
            # Slice regional_sai_p_values by the decade
            regional_sai_p_values = regional_sai_p_values.sel(time=decade_end_year)
        else:
            regional_sai_p_values = None

        regional_mean = regional_map.sel(time=slice(decade_start_year, decade_end_year)).mean('time')
        regional_delta_mean = regional_delta.sel(time=slice(decade_start_year, decade_end_year)).mean('time')

        if spatial_gdf is not None and spatial_item is not None:
            mask = create_mask(regional_mean, spatial_gdf[spatial_gdf.name == spatial_item])
            output_data[var]["mask"] = mask

        output_data[var]["regional_mean"] = regional_mean
        output_data[var]["regional_delta_mean"] = regional_delta_mean

        output_data[var]["regional_no_sai_p_values"] = regional_no_sai_p_values
        output_data[var]["regional_sai_p_values"] = regional_sai_p_values

        #### Mean over time plot ####
        # Get period between 2015 and SAI start year, SAI start year to SAI end year, and SAI end year to 2100
        if start_year is None:
            start_year = MIN_SAI_START
        recent_regional_map = regional_map.sel(time=slice(2015, start_year-1))
        regional_map = regional_map.sel(time=slice(start_year, SIM_END_YEAR))

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
        historical_model_global_mean = concatenated_global_mean.sel(time=slice(None, historical_model_global_mean.time[-1]+1))
        mean_no_sai = concatenated_global_mean.sel(time=slice(historical_model_global_mean.time[-1], None))

        if variable_injection is not None:
            global_injection_amounts = variable_injection.sum(axis=0)
            if global_injection_amounts.max() > 0:
                # For nonzero variable injection, simply assign the values before the start year to be the same as in mean_no_sai
                # to ensure a smooth transition over the historical / projected boundary.
                first_nonzero_index = np.argmax(global_injection_amounts > 0)
                first_nonzero_year = list(range(2035, 2101))[first_nonzero_index]
                mean_with_sai.loc[dict(time=slice(None, first_nonzero_year-1))] = mean_no_sai.loc[dict(time=slice(None, first_nonzero_year-1))]
            else:
                # For zero variable injection, we can just assign the values directly to guarantee they are identical
                # (equivalent to smoothing mean_with_sai in the same way as mean_no_sai)
                mean_with_sai = mean_no_sai
        else:
            # Assign the values before the start year to be the same as in mean_no_sai 
            # to ensure a smooth transition over the historical / projected boundary.
            mean_with_sai.loc[dict(time=slice(None, start_year-1))] = mean_no_sai.loc[dict(time=slice(None, start_year-1))]

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
    if variable_injection is None:
        temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
        so2_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_so2_{var}.nc"
        if so2_cache_path.exists():
            so2_by_latitude = xr.open_dataarray(so2_cache_path, autoclose=True)
            so2_by_latitude = so2_by_latitude.sel(start_year=start_year)
        else:
            interpolator = get_interpolator(model_dir / "so2" / "interpolator.nc")
            so2_by_latitude = get_so2_by_latitude(interpolator, temp_diff, ramp_up)["so2"]
            so2_by_latitude = so2_by_latitude.drop_vars('model')
        global_so2 = so2_by_latitude.sum(('lat'))
    else:
        temp_diff = None
        # Convert to xarray DataArray
        so2_by_latitude = xr.DataArray(
            variable_injection,
            dims=('lat', 'time'),
            coords={'lat': [-60, -30, -15, 0, 15, 30, 60],
                    'time': np.arange(2035, 2101)}
        )
        global_so2 = so2_by_latitude.sum(('lat'))

    # Store the SO2 data
    output_data["temp_diff"] = temp_diff
    output_data["so2_by_latitude"] = so2_by_latitude
    output_data["global_so2"] = global_so2

    return output_data
