import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import hashlib
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import logging
import time
import struct
from datetime import datetime

from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR

from .constants import *
from .historical import get_historical_model, get_historical_obs_global_mean_temp, get_historical_obs_sea_ice_extent
from .baseline import get_regional_map
from .delta import get_regional_delta
from .p_values import get_grid_level_p_values
from .population import get_population_data
from .variable import get_variable_regional_delta
from .utils import get_interpolator, create_mask, regional_aggregation, apply_constraints, clip_to_land
from .icefrac import get_area, predict_icefrac


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


def get_outputs(
    ssp_scenario, temp_target, spatial_gdf, spatial_item,
    decade_start_year, decade_end_year, start_year, ramp_up,
    data_dir, model_dir, cache_dir, var=None,
    variable_injection=None, output_config={}):

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
    regional_temp, regional_temp_no_sai_p_values = get_regional_map(
        "tas", data_dir, model_dir, cache_dir, ssp_scenario
    )

    if variable_injection is None:
        ### Get temperature difference data ###
        temp_diff = get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, temp_target)

    ### Get historical observations data ###
    historical_obs_data = get_historical_obs_global_mean_temp(data_dir)
    historical_obs_sea_ice = get_historical_obs_sea_ice_extent(data_dir)

    ### Set up outputs ###
    output_data = defaultdict(dict)

    for var in variables_to_process:

        is_exposure = "exposure" in var
        is_above_below = "above" in var or "below" in var
        is_icefrac = var == "icefrac"
        if is_icefrac:
            area = get_area(data_dir)

        # Load projection data
        if var == "tas":
            regional_map, regional_no_sai_p_values = regional_temp, regional_temp_no_sai_p_values
        elif is_icefrac:
            regional_map = predict_icefrac(regional_temp, model_dir)
            regional_no_sai_p_values = None
        else:
            regional_map, regional_no_sai_p_values = get_regional_map(
                var, data_dir, model_dir, cache_dir, ssp_scenario
            )

        output_data[var]["regional_map"] = regional_map

        # Load historical data
        historical_model = get_historical_model(var, data_dir, cache_dir)
        historical_rebase = None
        if historical_model is not None:
            historical_rebase = historical_model.sel(time=slice(1850, 1900)).mean('time')
            if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
                historical_model = historical_model - historical_rebase

        ### Get regional mean and delta data ###
        if is_icefrac:
            # Step 1: Get temperature regional delta
            if variable_injection is None:
                temp_regional_delta = get_regional_delta(
                    "tas", data_dir, model_dir, cache_dir, ssp_scenario,
                    temp_target, ramp_up, temp_diff, start_year
                )
            else:
                temp_regional_delta = get_variable_regional_delta("tas", data_dir, cache_dir, variable_injection)
        
            # Step 2: Calculate arctic temperature anomaly for each year
            temp_with_sai = regional_temp + temp_regional_delta
            
            # Step 3: use predict_icefrac to get ICEFRAC predictions
            icefrac_with_sai = predict_icefrac(temp_with_sai, model_dir)
            
            # Initialize output array with same structure as regional_map
            regional_delta = icefrac_with_sai - regional_map

            regional_no_sai_p_values = get_grid_level_p_values(
                var, data_dir, historical_rebase, regional_map
            )
        else:            
            if variable_injection is None:
                regional_delta = get_regional_delta(
                    var, data_dir, model_dir, cache_dir, ssp_scenario, 
                    temp_target, ramp_up, temp_diff, start_year
                )
            else:
                if var != "p-e":
                    regional_delta = get_variable_regional_delta(var, data_dir, cache_dir, variable_injection)
                else:
                    regional_delta_p = get_variable_regional_delta("pr", data_dir, cache_dir, variable_injection)
                    regional_delta_e = get_variable_regional_delta("e", data_dir, cache_dir, variable_injection)
                    regional_delta = regional_delta_p - regional_delta_e

        # Slice regional_no_sai_p_values by the decade
        if regional_no_sai_p_values is not None:
            regional_no_sai_p_values = regional_no_sai_p_values.sel(time=decade_end_year)

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
                with xr.open_dataarray(regional_p_cache_path) as da:
                    regional_sai_p_values = da.load()
            else:
                if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
                    # The regional map is already rebased, so need to add the historical rebase back before comparing
                    regional_sai_p_values = get_grid_level_p_values(
                        var, data_dir, historical_rebase,
                        regional_map + regional_delta + historical_rebase
                    )
                elif var == "icefrac":
                    regional_sai_p_values = get_grid_level_p_values(
                        var, data_dir, historical_rebase, icefrac_with_sai
                    )
                else:
                    regional_sai_p_values = get_grid_level_p_values(
                        var, data_dir, historical_rebase,
                        regional_map + regional_delta
                    )
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

        ### Create GeoTIFF files ###
        if output_config.get("create_geotiffs", False):
            output_data["geotiff_paths"] = create_geotiff_files(**locals())
        if output_config.get("create_pngs", False) or output_config.get("create_svgs", False):
            output_data["image_paths"] = create_vectorized_files(**locals())

        #### Mean over time plot ####
        # Get period between 2015 and SAI start year, SAI start year to SAI end year, and SAI end year to 2100
        if start_year is None:
            start_year = MIN_SAI_START
        recent_regional_map = regional_map.sel(time=slice(2015, start_year-1))
        regional_map = regional_map.sel(time=slice(start_year, SIM_END_YEAR))

        if is_exposure or is_icefrac:
            op = "sum"
        else:
            op = "mean"

        if is_exposure or is_icefrac:
            # Define data array of all ones like regional_map.lat
            weights = xr.DataArray(np.ones_like(regional_map.lat), dims=('lat'), coords={'lat': regional_map.lat})
        else:
            weights = np.cos(np.deg2rad(regional_map.lat))

        if spatial_gdf is None or spatial_item is None:
            if historical_model is not None:
                # Compute historical global mean
                if is_icefrac:
                    historical_mean = regional_aggregation(historical_model * area.broadcast_like(historical_model), weights, op)
                else:
                    historical_mean = regional_aggregation(historical_model, weights, op)
            
            if is_icefrac:
                # Compute recent global mean
                recent_global_mean = regional_aggregation(recent_regional_map * area.broadcast_like(recent_regional_map), weights, op)
                # Compute mean temperature with and without SAI
                mean_no_sai = regional_aggregation(regional_map * area.broadcast_like(regional_map), weights, op)
            else:
                # Compute recent global mean
                recent_global_mean = regional_aggregation(recent_regional_map, weights, op)
                # Compute mean temperature with and without SAI
                mean_no_sai = regional_aggregation(regional_map, weights, op)
            regional_map_with_sai = regional_map + regional_delta
            if is_above_below:
                regional_map_with_sai = regional_map_with_sai.where(regional_map_with_sai > 0, 0)
                # Clip again because the above sets nan values to 0
                regional_map_with_sai = clip_to_land(data_dir, regional_map_with_sai)
            if is_icefrac:
                mean_with_sai = regional_aggregation(regional_map_with_sai * area.broadcast_like(regional_map_with_sai), weights, op)
            else:
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
            reg_map = regional_map
            reg_delta = regional_delta
        else:
            reg_map = masked_regional_map
            reg_delta = masked_regional_delta

        reg_map = reg_map.sel(time=slice(decade_start_year, decade_end_year))
        reg_delta = reg_delta.sel(time=slice(decade_start_year, decade_end_year))

        # Get histogram of the distribution of values in the regional map
        pdf_no_sai = reg_map.values.flatten()
        pdf_with_sai = (reg_map + reg_delta).values.flatten()

        # Get PDF of historical model data from 1950-1960
        if historical_model is not None:
            if spatial_gdf is None or spatial_item is None:
                hist_model = historical_model
            else:
                hist_model = masked_historical_model
            pdf_historical = hist_model.sel(time=slice(1950, 1959)).values.flatten()
            if is_icefrac:
                pdf_with_sai = pdf_with_sai[pdf_historical > 0]
                pdf_no_sai = pdf_no_sai[pdf_historical > 0]
                pdf_historical = pdf_historical[pdf_historical > 0]

        
        ### Get the histograms ###
        pdf_no_sai = pdf_no_sai[~np.isnan(pdf_no_sai)]
        pdf_with_sai = pdf_with_sai[~np.isnan(pdf_with_sai)]

        if is_above_below:
            # Set predefined bins
            bins = [1, 5, 10, 20, 50, np.inf]
            # Remove 0 values
            pdf_no_sai = pdf_no_sai[pdf_no_sai > 0]
            pdf_with_sai = pdf_with_sai[pdf_with_sai > 0]
        elif var == "icefrac":
            bins = np.arange(0, 1.01, 0.1)
        else:
            max_v, min_v = np.nanmax(np.concatenate([pdf_no_sai, pdf_with_sai, pdf_historical])), np.nanmin(np.concatenate([pdf_no_sai, pdf_with_sai, pdf_historical]))
            bins = np.arange(min_v, max_v+0.1, 0.1)

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
        elif var == 'icefrac':
            output_data[var]["mean_over_time"]["historical_obs"] = historical_obs_sea_ice

        # Load model internal variability
        if spatial_gdf is None or spatial_item is None:
            # --- GLOBAL / NO REGIONAL SELECTION CASE ---
            if is_exposure:
                
                # Load variability for exposure variable
                model_internal_variability = np.load(data_dir / exposurevar2var[var] / "model_internal_variability.npy")
                
                # Apply global population weighting
                population_np = get_population_data(data_dir)
                model_internal_variability = model_internal_variability * population_np.sum() / 1e6
                model_internal_variability = model_internal_variability.item()

            else:
                # Load variability for standard variable
                model_internal_variability = np.load(data_dir / var / "model_internal_variability.npy")
                
            output_data[var]["mean_over_time"]["model_internal_variability"] = model_internal_variability

        else:
            # --- REGIONAL MODE ---
            if is_exposure:
                var_for_variability = exposurevar2var[var]
                # Apply global population weighting
                population_np = get_population_data(data_dir)
                # Multiply by population in the selected region
                population_xr = xr.DataArray(population_np, dims=('lon', 'lat'), coords={'lat': regional_map.lat, 'lon': regional_map.lon})
                # Shift lon values by 180
                population_xr = population_xr.assign_coords(lon=(((population_xr.lon + 180) % 360))).sortby('lon').transpose()
                mask = create_mask(population_xr, spatial_gdf[spatial_gdf.name == spatial_item])
            else:
                var_for_variability = var
            
            # Get the variability column name
            variability_col = f"variability_{var_for_variability}"
            
            # Filter spatial_gdf to get the specific region
            region_row = spatial_gdf[spatial_gdf.name == spatial_item]
            
            # Get variability value from geojson file
            if variability_col not in region_row.columns:
                raise ValueError(f"Column {variability_col} not found in spatial_gdf. Available columns: {list(region_row.columns)}")
            
            model_internal_variability = region_row[variability_col].iloc[0]
            if is_exposure:
                model_internal_variability = model_internal_variability * population_xr.where(mask).sum() / 1e6
                model_internal_variability = model_internal_variability.item()
            
            # If value is NaN or None set 0
            if model_internal_variability is None or pd.isna(model_internal_variability):
                model_internal_variability = 0
            
            output_data[var]["mean_over_time"]["model_internal_variability"] = model_internal_variability

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

        # Generate NetCDF file for this variable if requested
        if output_config.get("create_netcdf", False):
            # Get model internal variability
            if is_exposure:
                var_internal_variability = np.load(data_dir / exposurevar2var[var] / "model_internal_variability.npy")
            else:
                var_internal_variability = np.load(data_dir / var / "model_internal_variability.npy")

            generate_sai_netcdf(
                var=var,
                recent_regional_map=recent_regional_map,
                regional_map=regional_map,
                regional_delta=regional_delta,
                historical_model=historical_model,
                historical_rebase=historical_rebase,
                model_internal_variability=var_internal_variability,
                ssp_scenario=ssp_scenario,
                temp_target=temp_target,
                decade_start_year=decade_start_year,
                decade_end_year=decade_end_year,
                start_year=start_year,
                ramp_up=ramp_up,
                variable_injection=variable_injection,
                output_config=output_config
            )

    ### Latitude vs. Tg SO2 plot ###
    if variable_injection is None:
        temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
        so2_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_so2_{var}.nc"
        if so2_cache_path.exists():
            with xr.open_dataarray(so2_cache_path) as da:
                so2_by_latitude = da.load()
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


def _reproject_to_arctic_stereo(data_array, significance_mask, src_transform):
    """Reproject WGS84 arrays to North Polar Stereographic for polar visualization.

    Returns (dst_data, dst_sig, dst_transform, dst_crs) all in stereographic coordinates.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds, array_bounds

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_proj4(
        '+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    )

    # Clip source to northern hemisphere to avoid "Invalid latitude" warnings
    src_height, src_width = data_array.shape
    src_bounds = array_bounds(src_height, src_width, src_transform)
    if src_bounds[1] < 0:
        pixel_height = abs(src_transform.e)
        rows_to_skip = int(max(0, (0 - src_bounds[1]) / pixel_height))
        clip_row = src_height - rows_to_skip
        data_array = data_array[:clip_row, :]
        significance_mask = significance_mask[:clip_row, :]
        src_transform = from_bounds(
            src_bounds[0], 0, src_bounds[2], src_bounds[3],
            src_width, clip_row
        )

    # Extent in stereographic meters. 6 000 000 m ≈ edges at ~40°N.
    extent = 6_000_000
    dst_size = 512

    dst_transform = from_bounds(-extent, -extent, extent, extent, dst_size, dst_size)

    dst_data = np.full((dst_size, dst_size), np.nan, dtype=np.float32)
    dst_sig = np.full((dst_size, dst_size), np.nan, dtype=np.float32)

    reproject(
        source=data_array,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    reproject(
        source=significance_mask,
        destination=dst_sig,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return dst_data, dst_sig, dst_transform, dst_crs


def create_geotiff_files(var, regional_mean, regional_delta_mean,
                        regional_no_sai_p_values, regional_sai_p_values,
                        ssp_scenario, temp_target=None,
                        decade_start_year=None, decade_end_year=None,
                        start_year=None, ramp_up=None, variable_injection=None,
                        output_config={}, **kwargs):
    """
    Create GeoTIFF files for a given variable with both no_sai and with_sai scenarios.
    
    Returns:
        dict: Dictionary mapping scenario keys to file paths
    """
    geotiff_paths = {}
    
    output_dir = Path(output_config.get("output_dir", "output")) / "geotiffs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a hash of the input parameters for GeoTIFF filenames
    if variable_injection is None:
        geotiff_params = [
            ssp_scenario,
            temp_target,
            decade_start_year,
            decade_end_year,
            start_year,
            ramp_up,
        ]
    else:
        # Hash the variable_injection array to include in the cache key
        variable_hash = _hash_variable_injection(variable_injection)
        geotiff_params = [
            ssp_scenario,
            decade_start_year,
            decade_end_year,
            f"vinj_{variable_hash}"
        ]
    
    geotiff_params_str = "_".join(map(str, geotiff_params))
    geotiff_params_hash = hashlib.md5(geotiff_params_str.encode()).hexdigest()[:10]

    for data, suffix in [(regional_mean, "no_sai"), (regional_mean + regional_delta_mean, "with_sai")]:
        output_filename = f"{var}_{suffix}_{geotiff_params_hash}.tif"
        output_path = output_dir / output_filename

        # Skip if file already exists
        if output_path.exists():
            geotiff_paths[f"{var}_{suffix}"] = str(output_path)
            continue
        
        # Adjust longitude values from 0-360 to -180-180
        data_adjusted = data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
        data_adjusted = data_adjusted.sortby('lon')
        
        # Flip the data vertically and reverse the latitude coordinates
        data_adjusted = data_adjusted.isel(lat=slice(None, None, -1))
        
        # Prepare the data for GeoTIFF
        data_array = data_adjusted.values
        
        # Get the appropriate significance mask
        if suffix == "no_sai" and regional_no_sai_p_values is not None:
            # For no_sai map, adjust p-values the same way we adjusted the data
            p_values_adjusted = regional_no_sai_p_values.assign_coords(lon=(((regional_no_sai_p_values.lon + 180) % 360) - 180))
            p_values_adjusted = p_values_adjusted.sortby('lon')
            p_values_adjusted = p_values_adjusted.isel(lat=slice(None, None, -1))
            
            # 1 = not significant, NaN = either significant or not included
            significance_mask = np.where(
                (~np.isnan(p_values_adjusted.values)) & (p_values_adjusted.values >= 0.05),
                1,
                np.nan
            ).astype(np.float32)
            
        elif suffix == "with_sai" and regional_sai_p_values is not None:
            # For with_sai map, adjust p-values the same way we adjusted the data
            p_values_adjusted = regional_sai_p_values.assign_coords(lon=(((regional_sai_p_values.lon + 180) % 360) - 180))
            p_values_adjusted = p_values_adjusted.sortby('lon')
            p_values_adjusted = p_values_adjusted.isel(lat=slice(None, None, -1))

            # 1 = not significant, NaN = either significant or not included
            significance_mask = np.where(
                (~np.isnan(p_values_adjusted.values)) & (p_values_adjusted.values >= 0.05),
                1,
                np.nan
            ).astype(np.float32)
            
        else:
            # Default to all NaN (no hatching) if p-values not available
            significance_mask = np.full_like(data_array, np.nan, dtype=np.float32)
        
        # Convert to float32 if it's not already
        if data_array.dtype != np.float32:
            data_array = data_array.astype(np.float32)
        
        # Calculate the correct transform
        lon_res = (data_adjusted.lon.max() - data_adjusted.lon.min()) / (data_adjusted.lon.size - 1)
        lat_res = (data_adjusted.lat.max() - data_adjusted.lat.min()) / (data_adjusted.lat.size - 1)
        transform = rasterio.transform.from_origin(
            data_adjusted.lon.min() - lon_res/2,
            data_adjusted.lat.max() + lat_res/2,
            lon_res, lat_res
        )

        # For icefrac, reproject to Arctic LAEA for polar visualization
        geotiff_crs = '+proj=longlat +datum=WGS84 +no_defs'
        if var == 'icefrac':
            data_array, significance_mask, transform, geotiff_crs = \
                _reproject_to_arctic_stereo(data_array, significance_mask, transform)

        # Write to memory first, then save to disk atomically
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=data_array.shape[0],
                width=data_array.shape[1],
                count=2,  # Now using 2 bands
                dtype=data_array.dtype,
                crs=geotiff_crs,
                transform=transform,
                compress='deflate',
                zlevel=9,
                predictor=3,
                tiled=False,
                blockxsize=data_array.shape[1],
                blockysize=1,
            ) as dst:
                dst.write(data_array, 1)  # Data values in band 1
                dst.write(significance_mask, 2)  # Significance mask in band 2
            
            # Only write to disk if file doesn't exist
            if not output_path.exists():
                with open(output_path, 'wb') as f:
                    f.write(memfile.read())

        geotiff_paths[f"{var}_{suffix}"] = str(output_path)
    
    return geotiff_paths

@lru_cache(maxsize=32)
def _borders_50m():
    return cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="50m",
        facecolor="none",
    )

def _adjust_polar_lonlat(da: xr.DataArray) -> xr.DataArray:
    """Shift lon to [-180, 180), sort, and flip latitude (north-up)."""
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    # flip latitude order if increasing south->north
    # (your original code always flipped; this keeps it robust)
    if da.lat[0] < da.lat[-1]:
        da = da.isel(lat=slice(None, None, -1))
    return da

def _hash_variable_injection(variable_injection: np.ndarray) -> str:
    arr = np.asarray(variable_injection, dtype=np.float64)
    rows, cols = arr.shape
    payload = struct.pack('<II', rows, cols) + arr.astype('<f8', copy=False).tobytes()
    return hashlib.md5(payload).hexdigest()[:8]

def create_vectorized_files(
    var,
    regional_mean,
    regional_delta_mean,
    regional_no_sai_p_values,
    regional_sai_p_values,
    ssp_scenario,
    temp_target=None,
    decade_start_year=None,
    decade_end_year=None,
    start_year=None,
    ramp_up=None,
    variable_injection=None,
    output_config=None,
    **kwargs,
):
    """
    Create image files (PNG or SVG) for visualizing variables over arctic (e.g. sea-ice extent) data.

    This function generates polar stereographic projection plots of climate variables,
    with options for statistical significance hatching and different output formats.

    Args:
        var (str): Name of the climate variable being plotted
        regional_mean (xarray.DataArray): Regional mean values without SAI
        regional_delta_mean (xarray.DataArray): Change in regional mean values due to SAI
        regional_no_sai_p_values (xarray.DataArray, optional): P-values for no-SAI scenario
        regional_sai_p_values (xarray.DataArray, optional): P-values for SAI scenario
        ssp_scenario (str): SSP scenario identifier
        temp_target (float, optional): Temperature target in degrees Celsius
        decade_start_year (int, optional): Start year of the decade being analyzed
        decade_end_year (int, optional): End year of the decade being analyzed
        start_year (int, optional): Year SAI intervention begins
        ramp_up (int, optional): Duration of SAI ramp-up period in years
        variable_injection (numpy.ndarray, optional): Custom injection profile if used
        output_config (dict, optional): Configuration options including:
            - image_format (str): Output format, either 'png' or 'svg' (default: 'png')
            - dpi (int): Resolution in dots per inch (default: 300)
            - output_dir (str): Directory for output files (default: 'output')
        **kwargs: Additional keyword arguments passed through

    Returns:
        dict: Dictionary mapping scenario keys to file paths, with keys in format:
            "{var}_no_sai" and "{var}_with_sai"

    Features:
        - Generates both no-SAI and with-SAI visualizations
        - Uses polar stereographic projection focused on 60°N-90°N
        - Applies hatching to regions without statistical significance
        - Includes coastlines and land features
        - Caches results using hash-based filenames
        - Supports both PNG and SVG output formats

    Production-optimized: less memory, cached features, robust hashing, leak-safe figure handling.
    """
    image_paths = {}

    # ---- config (no mutable defaults) ----
    output_config = output_config or {}
    image_format = output_config.get("image_format", "png").lower()
    dpi = int(output_config.get("dpi", 50))
    output_dir = Path(output_config.get("output_dir", "output")) / f"{image_format}s"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache key (stable + cheap) ----

    if variable_injection is None:
        plot_params = [
            ssp_scenario,
            temp_target,
            decade_start_year,
            decade_end_year,
            start_year,
            ramp_up,
        ]
    else:
        # Hash the variable_injection array to include in the cache key
        variable_hash = _hash_variable_injection(variable_injection)
        plot_params = [
            ssp_scenario,
            decade_start_year,
            decade_end_year,
            f"vinj_{variable_hash}",
        ]

    plot_params_str = "_".join(map(str, plot_params))
    plot_params_hash = hashlib.md5(plot_params_str.encode()).hexdigest()[:10]

    # ---- prepare base fields once (avoid extra copies) ----
    # keep in float32 to halve memory bandwidth
    base = _adjust_polar_lonlat(regional_mean.astype(np.float32))
    with_sai = _adjust_polar_lonlat((regional_mean + regional_delta_mean).astype(np.float32))

    lon = base.lon.values
    lat = base.lat.values

    def prep_sig_mask(pvals_da: xr.DataArray | None):
        if pvals_da is None:
            return None
        pv = _adjust_polar_lonlat(pvals_da).values  # materialize as needed
        # 1 = not significant; NaN elsewhere (keeps hatch minimal)
        mask = np.where((~np.isnan(pv)) & (pv >= 0.05), 1.0, np.nan).astype(np.float32)
        return mask

    no_sai_mask = prep_sig_mask(regional_no_sai_p_values)
    with_sai_mask = prep_sig_mask(regional_sai_p_values)

    # ---- color normalization (reuse) ----
    cmap_blues = plt.cm.Blues_r
    bounds_blues = np.linspace(0, 1, 11)
    norm_blues = colors.BoundaryNorm(boundaries=bounds_blues, ncolors=cmap_blues.N, clip=True)

    for data, suffix, sigmask in [
        (base, "no_sai", no_sai_mask),
        (with_sai, "with_sai", with_sai_mask),
    ]:
        output_path = output_dir / f"{var}_{suffix}_{plot_params_hash}.{image_format}"
        if output_path.exists():
            image_paths[f"{var}_{suffix}"] = str(output_path)
            continue

        # materialize the data just-in-time (keeps xarray advantages until here)
        z = data.values  # float32

        fig = None
        try:
            # Create figure/axes
            fig = plt.figure(figsize=(10, 10), dpi=dpi)
            fig.patch.set_visible(False)
            ax = plt.subplot(111, projection=ccrs.NorthPolarStereo())
            ax.set_frame_on(False)

            # Map features (borders cached)
            # To increase the resolution of both land and borders, use 'scale="50m"' for higher-res:
            highres_land = cfeature.NaturalEarthFeature(
                category='physical',
                name='land',
                scale='50m',    # Higher resolution
                facecolor="gray"
            )
            ax.add_feature(highres_land, zorder=2)
            ax.add_feature(_borders_50m(), edgecolor="black", linewidth=1, zorder=2)
            ax.coastlines(resolution="50m")

            # Use 1D lon/lat (avoids making a meshgrid copy)
            mesh = ax.pcolormesh(
                lon,
                lat,
                z,
                transform=ccrs.PlateCarree(),
                cmap=cmap_blues,
                norm=norm_blues,
                rasterized=True,  # shrinks vector-output size; harmless for PNG
            )

            # Hatching for non-significant regions (if provided)
            if sigmask is not None and not np.all(np.isnan(sigmask)):
                ax.contourf(
                    lon,
                    lat,
                    sigmask,
                    levels=[-0.5, 0.5, 1.5],
                    hatches=["",".."], #["", "///"],
                    colors="none",
                    transform=ccrs.PlateCarree(),
                )

            # Light gridlines (no labels -> faster)
            ax.gridlines(draw_labels=False)
            ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())

            plt.tight_layout()
            fig.savefig(output_path, bbox_inches="tight", dpi=dpi, format=image_format)
            image_paths[f"{var}_{suffix}"] = str(output_path)

        finally:
            # Leak-safe: always close, even if save or plotting raises
            if fig is not None:
                plt.close(fig)

    return image_paths

def generate_sai_netcdf(
    var: str,
    recent_regional_map: xr.DataArray,
    regional_map: xr.DataArray,
    regional_delta: xr.DataArray,
    historical_model: xr.DataArray | None,
    historical_rebase: xr.DataArray | None,
    model_internal_variability: np.ndarray | float | int,
    ssp_scenario: str,
    temp_target: float,
    decade_start_year: int,
    decade_end_year: int,
    start_year: int,
    ramp_up: int,
    variable_injection: np.ndarray | None,
    model_name: str = "CESM2-WACCM",
    output_path: str = None,
    output_config: dict = None,
):
    """
    Generates a NetCDF file with historical, SAI, and No-SAI data including 
    comprehensive global and variable metadata. Output path and output directory 
    are derived from output_config if provided.
    """

    if output_config is None:
        output_config = {}

    # Determine output directory from output_config
    output_dir = Path(output_config.get("output_dir", "output")) / "netcdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compose output file name if not provided
    if output_path is None:
        # Use the same hashing approach as GeoTIFF for consistency
        if variable_injection is None:
            netcdf_params = [
                ssp_scenario,
                temp_target,
                decade_start_year,
                decade_end_year,
                start_year,
                ramp_up,
            ]
        else:
            # Hash the variable_injection array to include in the cache key
            variable_hash = _hash_variable_injection(variable_injection)
            netcdf_params = [
                ssp_scenario,
                decade_start_year,
                decade_end_year,
                f"vinj_{variable_hash}"
            ]

        netcdf_params_str = "_".join(map(str, netcdf_params))
        netcdf_params_hash = hashlib.md5(netcdf_params_str.encode()).hexdigest()[:10]
        output_filename = f"{var}_{netcdf_params_hash}.nc"
        output_path = output_dir / output_filename
    else:
        # Ensure Path object for uniformity
        output_path = Path(output_path)

        # If output_path is not absolute and not under the intended output_dir, 
        # resolve it under output_dir
        if not output_path.is_absolute() and not str(output_path).startswith(str(output_dir)):
            output_path = output_dir / output_path

    # 1. Define Metadata Mapping
    var_meta = {
        'tas': {'long_name': 'Near-Surface Air Temperature', 'units': 'K'},
        'p-e': {'long_name': 'Water Availability (Precipitation-Evaporation)', 'units': 'mm/day'},
        'tasmin': {'long_name': 'Daily Minimum Near-Surface Air Temperature', 'units': 'K'},
        'tasmax': {'long_name': 'Daily Maximum Near-Surface Air Temperature', 'units': 'K'},
        'exposure_above_40': {'long_name': 'Person-Days Above 40°C', 'units': 'Millions'},
        'exposure_above_35': {'long_name': 'Person-Days Above 35°C', 'units': 'Millions'},
        'exposure_below_0': {'long_name': 'Person-Days Below 0°C', 'units': 'Millions'},
        'exposure_above_10': {'long_name': 'Person-Days Above 10mm/day', 'units': 'Millions'},
        'exposure_above_20': {'long_name': 'Person-Days Above 20mm/day', 'units': 'Millions'},
        'icefrac': {'long_name': 'Arctic Sea Ice Fraction', 'units': 'fraction'}
    }
    
    meta = var_meta.get(var, {'long_name': var, 'units': 'unknown'})

    # 2. Perform Calculations
    # No-SAI: projection + historical rebase (1850-1900 avg)
    # For icefrac and exposure vars, don't add historical_rebase as they're already absolute values
    is_exposure = "exposure" in var
    is_icefrac = var == "icefrac"

    # Concatenate recent_regional_map and regional_map along time dimension
    concatenated_regional_map = xr.concat([recent_regional_map, regional_map], dim="time")

    if is_icefrac or is_exposure or historical_rebase is None:
        ds_nosai = concatenated_regional_map.rename(f"{var}_nosai")
        ds_sai = (regional_map + regional_delta).rename(f"{var}_sai")
    else:
        ds_nosai = (concatenated_regional_map + historical_rebase).rename(f"{var}_nosai")
        ds_sai = (regional_map + regional_delta + historical_rebase).rename(f"{var}_sai")

    # Historical
    if historical_model is not None:
        # historical_model was converted to anomalies (original - historical_rebase)
        # For consistency with nosai/sai, add historical_rebase back for absolute values
        if is_icefrac or is_exposure or historical_rebase is None:
            ds_hist = historical_model.rename(f"{var}_historical")
        else:
            ds_hist = (historical_model + historical_rebase).rename(f"{var}_historical")
    else:
        # Create a dummy historical with NaN values or skip it
        ds_hist = xr.DataArray(
            np.nan,
            coords=regional_map.coords,
            dims=regional_map.dims,
            name=f"{var}_historical"
        ).sel(time=slice(1850, 2014))  # Historical period

    # 3. Create the Dataset
    # Convert model_internal_variability scalar to DataArray
    if isinstance(model_internal_variability, (int, float, np.ndarray)):
        model_std_dev = xr.DataArray(
            model_internal_variability,
            name="model_std_dev"
        )
    else:
        model_std_dev = model_internal_variability.rename("model_std_dev")

    ds = xr.merge([ds_hist, ds_nosai, ds_sai, model_std_dev])

    # 4. Handle Optional SO2 Injection
    if variable_injection is not None:
        # variable_injection is (lat, time) with time from 2035-2101
        # latitude_bin coordinates are [-60, -30, -15, 0, 15, 30, 60]
        lat_bins = [-60, -30, -15, 0, 15, 30, 60]
        injection_time = np.arange(2035, 2101)
        inj_da = xr.DataArray(
            variable_injection,
            coords={'latitude_bin': lat_bins, 'time': injection_time},
            dims=['latitude_bin', 'time'],
            name="so2_injection_amount",
            attrs={'units': 'Tg SO2', 'long_name': 'Strategic SO2 Injection Amount'}
        )
        ds = xr.merge([ds, inj_da])

    # 5. Set Variable Attributes
    for v in [f"{var}_nosai", f"{var}_sai", f"{var}_historical"]:
        ds[v].attrs['units'] = meta['units']
        ds[v].attrs['long_name'] = meta['long_name']
        if is_icefrac or is_exposure:
            ds[v].attrs['description'] = f"Absolute values of {var}"
        elif historical_rebase is None:
            ds[v].attrs['description'] = f"Projected values of {var} without historical rebase."
        else:
            ds[v].attrs['description'] = f"Values calculated as delta from PI (1850-1900) rebased to historical mean."

    ds['model_std_dev'].attrs.update({
        'units': meta['units'],
        'long_name': f'Internal Variability Standard Deviation for {var}'
    })

    # 6. Global Attributes
    ds.attrs = {
        'title': f'Emulated Climate Projections for {meta["long_name"]}',
        'source': f'Emulated output from Reflective simulator based on {model_name}',
        'datetime_start': '1850',
        'datetime_end': '2100',
        'temporal_resolution': 'yearly',
        'spatial_resolution_lon': '1.25°',
        'spatial_resolution_lat': '0.9375°',
        'cooling_target': f'{temp_target}°C above PI' if temp_target is not None else 'N/A (custom variable injection used)',
        'scenario': ssp_scenario,
        'sai_start_year': start_year if start_year is not None else 'N/A (custom variable injection used)',
        'sai_ramp_up_years': ramp_up if ramp_up is not None else 'N/A (custom variable injection used)',
        'decade_start_year': decade_start_year,
        'decade_end_year': decade_end_year,
        'injection_strategy': 'ARISE' if variable_injection is not None else 'Standard',
        'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'paper_ref': 'Beall, Charlotte and Irvin, Jeremy A. and Dexheimer, Jake and Gruener, Dakota and Ng, Andrew Y. and Watson-Parris, Duncan and MacMartin, Douglas G. and Visioni, Daniele and Administrator, Sneak Peek, The Stratospheric Aerosol Injection (SAI) Simulator: An Open-Source Web Tool for Exploring Climate and SAI Deployment Scenarios. ONE-EARTH-D-25-00518, Available at SSRN: https://ssrn.com/abstract=5200736 or http://dx.doi.org/10.2139/ssrn.5200736',
        'data_ref': 'https://doi.org/10.5281/zenodo.15531372',
        'github_ref': 'https://github.com/reflective-org/sai-simulator',
    }

    # 7. Save to NetCDF with compression
    # Set up encoding for compression
    encoding = {}
    for var_name in ds.data_vars:
        encoding[var_name] = {
            'zlib': True,
            'complevel': 5,
            'dtype': 'float32'
        }

    ds.to_netcdf(str(output_path), engine='netcdf4', encoding=encoding)
    print(f"File saved successfully to {output_path}")
    return ds