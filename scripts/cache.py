"""Precompute the on-disk cache files consumed by the SAI dashboard at request time.

For each SSP scenario this script generates the no-SAI regional map and p-values, the SO2
injection-by-latitude lookup table (keyed by ``start_year``), and the per-temperature-target
SAI regional delta + p-values. Cache files are reused by :mod:`backend.get_outputs` so
dashboard requests don't have to re-run FaIR + the FaIR→SMIP emulator.

CLI entry: ``python scripts/cache.py <data_dir> <model_dir> <output_dir> [--scenario ...]``.
"""
import sys
import fire
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# The with-SAI p-values are built through the same helpers the live request path uses, so
# the file served at the default deployment is exactly what a live request would compute.
from backend.get_outputs import (
    _compute_regional_delta,
    _load_variable_fields,
    _with_sai_comparison_field,
    get_so2_by_latitude,
    get_temp_diff,
)
from backend.baseline import get_global_temp, get_regional_models, get_regional_map_from_global_temp, get_regional_map, validate_cache_dir
from backend.historical import get_historical_model, get_historical_model_per_model
from backend.p_values import get_grid_level_p_values, no_sai_p_values_cache_path, sai_p_values_cache_path
from backend.constants import *
from backend.utils import get_interpolator
from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR

def cache_historical(data_dir: str | Path, output_dir: str | Path) -> None:
    """Pre-generate shared historical_*.nc files (safe to call before parallel runs)."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Pre-generating historical cache files...")
    for var in VAR2INFO:
        historical_model = get_historical_model(var, data_dir, output_dir)
        if historical_model is not None:
            print(f"  historical_{var}.nc ready")
        if get_historical_model_per_model(var, data_dir, output_dir) is not None:
            print(f"  historical_per_model_{var}.nc ready")
    print("Historical cache complete.")


def cache(
    data_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    scenario: str | None = None,
    historical_only: bool = False,
) -> None:
    """Generate cache files for all (or a single) SSP scenario.

    Args:
        data_dir: Path to processed data directory
        model_dir: Path to fitted models directory
        output_dir: Path to cache output directory
        scenario: Optional SSP scenario name (e.g. 'SSP1-1.9') to cache only that one.
                  If None, caches all scenarios.
        historical_only: If True, only generate shared historical_*.nc files and exit.
                         Use this before running per-scenario caching in parallel.
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    # Stamps a fresh dir with the (version, FaIR calibration) fingerprint, accepts a
    # matching resume, and refuses a legacy or mismatched dir — a skip-existing resume
    # over artifacts from another calibration would silently mix climates.
    validate_cache_dir(output_dir)

    if historical_only:
        cache_historical(data_dir, output_dir)
        return

    if scenario is not None:
        if scenario not in REVERSE_FANCY_SSP_TITLES:
            print(f"Error: Unknown scenario '{scenario}'")
            print(f"Valid scenarios: {list(REVERSE_FANCY_SSP_TITLES.keys())}")
            sys.exit(1)
        scenarios = [scenario]
    else:
        scenarios = list(REVERSE_FANCY_SSP_TITLES.keys())

    for ssp_scenario in scenarios:
        print(ssp_scenario)
        simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
        num_targets = int((MAX_TARGET - MIN_TARGET) / STEP_TARGET) + 1
        # First get FaIR global temperature
        global_temp = get_global_temp(ssp_scenario)

        # tas first: icefrac is each model's sigmoid on that model's own temperature map, and
        # the live helpers below read every variable's no-SAI maps back from this cache.
        cache_no_sai("tas", ssp_scenario, global_temp, data_dir, model_dir, output_dir)
        regional_temp, regional_temp_no_sai_p_values, regional_temp_per_model = get_regional_map(
            "tas", data_dir, model_dir, output_dir, ssp_scenario
        )
        temp_fields = dict(
            regional_temp=regional_temp,
            regional_temp_no_sai_p_values=regional_temp_no_sai_p_values,
            regional_temp_per_model=regional_temp_per_model,
        )

        for var in tqdm(VAR2INFO):
            if get_historical_model(var, data_dir, output_dir) is None:
                continue
            cache_no_sai(var, ssp_scenario, global_temp, data_dir, model_dir, output_dir, **temp_fields)

            fields = None  # loaded on first use; a resumed run with every SAI file present skips it
            for temp_target in np.linspace(MIN_TARGET, MAX_TARGET, num_targets):

                temp_diff = get_temp_diff(data_dir, model_dir, output_dir, ssp_scenario, temp_target)
                so2_cache_path = output_dir / f"{simple_ssp}_{temp_target:.1f}_so2_by_latitude.nc"
                if not so2_cache_path.exists():
                    so2_by_latitudes = []
                    start_years = list(range(2035, 2091))
                    for start_year in start_years:
                        _temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
                        interpolator = get_interpolator(model_dir / "so2" / "interpolator.nc")
                        # Assumes ramp-up is always 10 years, and caches so2 based on this.
                        # This would need to be changed if we want to support other ramp-up periods.
                        so2_by_latitude = get_so2_by_latitude(interpolator, _temp_diff, ramp_up=10)["so2"]
                        # The multi-model-mean temp_diff no longer carries a leftover
                        # 'model' coord (the old single-model .sel did) — drop if present.
                        so2_by_latitude = so2_by_latitude.drop_vars('model', errors='ignore')

                        so2_by_latitudes.append(so2_by_latitude)

                    so2_by_latitude = xr.concat(so2_by_latitudes, dim='start_year', join='outer')
                    so2_by_latitude['start_year'] = start_years
                    so2_by_latitude.to_netcdf(so2_cache_path)

                sai_p_cache_path = sai_p_values_cache_path(output_dir, simple_ssp, temp_target, var)
                if sai_p_cache_path.exists():
                    print("Found sai cache for", var, "->", sai_p_cache_path)
                    continue

                if fields is None:
                    fields = _load_variable_fields(
                        var, ssp_scenario=ssp_scenario, data_dir=data_dir, model_dir=model_dir,
                        cache_dir=output_dir, **temp_fields,
                    )
                cache_sai_p_values(
                    var, fields, ssp_scenario, temp_target, temp_diff,
                    data_dir, model_dir, output_dir, **temp_fields,
                )


def cache_no_sai(
    var: str,
    ssp_scenario: str,
    global_temp: xr.DataArray,
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    regional_temp: xr.DataArray | None = None,
    regional_temp_no_sai_p_values: xr.DataArray | None = None,
    regional_temp_per_model: xr.DataArray | None = None,
) -> None:
    """Write ``{ssp}_regional_{var}.nc``, its no-SAI p-values and (not icefrac) the per-model maps.

    icefrac needs the tas maps (pass them); its map is the mean of each model's sigmoid on
    that model's own temperature, as :func:`backend.get_outputs._load_variable_fields` builds it.
    """
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
    no_sai_cache_paths = [output_dir / f"{simple_ssp}_regional_{var}.nc",
                          no_sai_p_values_cache_path(output_dir, simple_ssp, var)]
    per_model_cache_path = output_dir / f"{simple_ssp}_regional_per_model_{var}.nc"
    if all(cache_path.exists() for cache_path in no_sai_cache_paths) and (
        var == "icefrac" or per_model_cache_path.exists()
    ):
        print("Found no sai cache for", var)
        return

    if var == "icefrac":
        fields = _load_variable_fields(
            var, ssp_scenario=ssp_scenario, data_dir=data_dir, model_dir=model_dir,
            cache_dir=output_dir, regional_temp=regional_temp,
            regional_temp_no_sai_p_values=regional_temp_no_sai_p_values,
            regional_temp_per_model=regional_temp_per_model,
        )
        regional_map = fields.regional_map
        regional_p_values = get_grid_level_p_values(
            var, data_dir, fields.significance_reference, regional_map
        )
    else:
        if "exposure" not in var:
            variable_dir = model_dir / var
        else:
            variable_dir = model_dir / exposurevar2var[var]
        fair2smip = get_regional_models(variable_dir)

        # Cache regional map and historical data (the latter is saved under the hood)
        regional_map, regional_p_values, regional_map_per_model = get_regional_map_from_global_temp(
            global_temp, fair2smip, var, data_dir, output_dir
        )
        regional_map_per_model.to_netcdf(per_model_cache_path)

    regional_map.to_netcdf(no_sai_cache_paths[0])
    regional_p_values.to_netcdf(no_sai_cache_paths[1])


def cache_sai_p_values(
    var: str,
    fields,
    ssp_scenario: str,
    temp_target: float,
    temp_diff: xr.DataArray,
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
    regional_temp: xr.DataArray,
    regional_temp_no_sai_p_values: xr.DataArray | None,
    regional_temp_per_model: xr.DataArray | None,
) -> xr.DataArray:
    """Write the with-SAI p-values for the dashboard's DEFAULT deployment; return them.

    ``get_outputs`` serves this file only for ``DEFAULT_SAI_START_YEAR`` / ``DEFAULT_RAMP_UP``
    and computes live for anything else, so it is built from the live helpers: the same
    delta (:func:`_compute_regional_delta`) and the same tested field
    (:func:`_with_sai_comparison_field`).

    The unramped delta for non-icefrac variables is cached under the hood by
    :func:`backend.delta.get_regional_delta_without_rampup`; the ramped, start-sliced delta
    returned here is NOT written back over it (every live request ramps that file itself).
    icefrac's with-SAI delta has no such under-the-hood cache; it is written for inspection.
    """
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
    delta_result = _compute_regional_delta(
        var, fields=fields, regional_temp=regional_temp, ssp_scenario=ssp_scenario,
        temp_target=temp_target, ramp_up=DEFAULT_RAMP_UP, start_year=DEFAULT_SAI_START_YEAR,
        temp_diff=temp_diff, variable_injection=None, data_dir=data_dir, model_dir=model_dir,
        cache_dir=output_dir, regional_temp_per_model=regional_temp_per_model,
    )
    comparison = _with_sai_comparison_field(
        var, fields=fields, regional_delta=delta_result.regional_delta,
        icefrac_with_sai=delta_result.icefrac_with_sai,
    )
    regional_sai_p_values = get_grid_level_p_values(
        var, data_dir, fields.significance_reference, comparison
    )
    if var == "icefrac":
        delta_result.regional_delta.to_netcdf(
            output_dir / f"{simple_ssp}_{temp_target:.1f}_regional_delta_{var}.nc"
        )
    regional_sai_p_values.to_netcdf(
        sai_p_values_cache_path(output_dir, simple_ssp, temp_target, var)
    )
    return regional_sai_p_values

if __name__ == "__main__":
    fire.Fire(cache)
