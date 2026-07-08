"""SAI ``regional_delta`` per variable for the analytic ``temp_target`` / ``ramp_up`` path.

Computes the (lat, lon, time) delta that closes ``temp_diff`` for a given variable, using the
slope feature of the fitted SO2 → variable interpolator. Applies the standard variable-specific
clipping (floor for ``above``/``below`` count variables, land clip + population scaling for
exposure variables) and a linear ramp-up in the first ``ramp_up`` years of SAI. Caches the
no-ramp result on disk so re-runs at different ramp-up settings reuse the heavy interpolation.
"""
import xarray as xr
from pathlib import Path

from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR
from .constants import *
from .utils import get_interpolator, clip_to_land, apply_ramp_up_factor
from .utils import save_processed_data
from .population import get_population_exposure


def get_regional_delta_without_rampup(
    var: str,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    ssp_scenario: str,
    temp_target: float,
    temp_diff: xr.DataArray,
) -> xr.DataArray:
    """Compute (or load from cache) the SAI ``regional_delta`` for ``var`` before the ramp-up factor is applied.

    Multiplies the ``slope`` feature of the variable's SO2 → variable interpolator by
    ``temp_diff`` to get the per-year (lat, lon) delta, then applies the standard
    variable-specific post-processing: land clipping for precipitation/water-availability and
    ``above``/``below`` variables, population scaling for exposure variables. Substitutes the
    underlying daily variable for exposure inputs via :data:`exposurevar2var`. Results are
    cached on disk at ``cache_dir/{simple_ssp}_{temp_target:.1f}_regional_delta_{var}.nc``.
    """
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_delta_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_regional_delta_{var}.nc"

    if regional_delta_cache_path.exists():
        print(f"Found {regional_delta_cache_path}")
        with xr.open_dataarray(regional_delta_cache_path) as da:
            regional_delta = da.load()

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

        save_processed_data(regional_delta, regional_delta_cache_path, overwrite=True)

    return regional_delta


def get_regional_delta(
    var: str,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    ssp_scenario: str,
    temp_target: float,
    ramp_up: int,
    temp_diff: xr.DataArray,
    start_year: int,
) -> xr.DataArray:
    """Apply ``start_year`` slicing and the linear ramp-up to :func:`get_regional_delta_without_rampup`.

    Slices the cached no-ramp delta and ``temp_diff`` to ``[start_year, SIM_END_YEAR]``, zeroes
    out years where SAI is not needed (``temp_diff > 0``), and then routes the trimmed delta
    through :func:`apply_ramp_up_factor` so the first ``ramp_up`` years of actual SAI ramp
    linearly from ``1/ramp_up`` to ``1`` (capped at the available years to ``SIM_END_YEAR``).
    """
    regional_delta = get_regional_delta_without_rampup(var, data_dir, model_dir, cache_dir, ssp_scenario, temp_target, temp_diff)
    # Slice the data by the start year
    # Because cached data just caches the whole period
    temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
    regional_delta = regional_delta.sel(time=slice(start_year, SIM_END_YEAR))

    # Set all values to 0 for the indices of time where temp_diff > 0
    no_sai_indices = temp_diff > 0
    regional_delta = regional_delta.where(~no_sai_indices, 0)
    regional_delta = apply_ramp_up_factor(regional_delta, temp_diff, ramp_up)

    return regional_delta
