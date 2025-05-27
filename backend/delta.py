import numpy as np
import xarray as xr

from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR
from .constants import *
from .utils import get_interpolator, clip_to_land
from .population import get_population_exposure


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
