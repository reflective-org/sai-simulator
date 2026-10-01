"""Convert raw monthly CESM/GAUSS NetCDF files into the simulator's processed inputs.

Walks the input ``data_dir``, parses each filename into ``(member, experiment)`` via
:func:`identify_file`, applies variable-specific preprocessing (unit conversion, ICEFRAC
Sept-only masking, PRECC+PRECL → PRECT pre-step, etc.), bilinearly regrids to the
1.25°×0.9375° CESM2-WACCM target grid, runs the preprocessing sanity checks from
:mod:`preprocessing_sanity`, and aggregates monthly data to yearly using the right method
per variable (sum for cumulative exposure variables, min/max for tasmin/tasmax,
1850-1900-rebased mean otherwise). The final per-experiment, per-member yearly Datasets are
merged into the canonical ``output_gauss-*.nc`` files and the model internal variability
(both global and grid-level) is computed and saved.

CLI entry: ``python scripts/process_monthly_gauss.py <var> <data_dir> <output_dir> [--overwrite]``.
"""
import numpy as np
import xarray as xr
import pandas as pd
# NOTE: ``xesmf`` (and its compiled ESMF/ESMPy backend) is imported lazily
# inside the two regridding functions below, not at module top. This keeps the
# module importable — for its ``MODELS`` table, ``var2esm_var`` map, and the
# ``_group_for_var`` helper — in environments without a compiled ESMF library
# (e.g. CI unit tests), since regridding is only needed for an actual run.
# ``fire`` is similarly imported lazily inside the ``__main__`` block below so
# that unit tests of ``MODELS`` / ``var2esm_var`` / ``_group_for_var`` don't
# require ``fire`` in the CI environment.
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only seen by type checkers, never executed at runtime — keeps the
    # ``xe.Regridder`` return annotation on ``get_robust_regridder`` valid
    # without forcing the heavy compiled ESMF dependency at import time.
    import xesmf as xe

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.utils import open_xarray_datasets, save_processed_data
from version import ZARR_STORES_URI
from scripts.preprocessing_sanity import (
    SanityCheckError,
    build_monthly_cell_bounds,
    validate_monthly_cell_bounds,
    validate_no_nans,
    validate_no_negative_values,
    validate_year_continuity_strict,
    validate_year_continuity_with_max_gap,
    validate_variable_specific_ranges,
    validate_year_count_for_experiment,
)
from scripts.variability import process_geojson_variability


# --- Configuration --- #

# Define unit conversion factors here for easy maintenance
UNIT_SCALERS = {
    "PRECT": 86400 * 1000,   # m/s -> mm/day
    "QFLX": 86400,         # kg/m2/s -> mm/day
}

# Largest allowed jump in calendar ``time`` (years) after hist + SSP245 merge.
# Incomplete-year drops can leave multi-year holes at the 2014/2015 seam (e.g. gap 7).
MERGED_HIST_BASELINE_MAX_YEAR_GAP = 10

# Map user variables to aggregation methods
AGGREGATION_RULES = {
    "tasmin": "min",
    "tasmax": "max",
    "icefrac": "mean", # Special case handled in logic
    "default": "mean"  # Fallback
}

# Define the mapping between the variable name and the GAUSS variable name
var2esm_var = {
        "tas": "TREFHT",
        "pr": "PRECT",
        "aod": "AODVISstdn",
        "tasmin": "TREFHTMN",
        "tasmax": "TREFHTMX",
        "tas_above_35": "TREFHT_above_35",
        "tas_above_40": "TREFHT_above_40",
        "tas_below_0": "TREFHT_below_0",
        "pr_above_10": "PRECT_above_10",
        "pr_above_20": "PRECT_above_20",
        "p-e": "QFLX",
        "icefrac": "ICEFRAC",
    }

# Per-model processing config. Each model is processed independently (see
# process_monthly / _process_model) so its quirks stay isolated.
#
# Per model:
#   experiments       (store URL, internal label) tuples. Internal labels stay
#                     model-agnostic ("baseline", "historical", "1.5", ...) so
#                     the baseline-keyed variability/merge logic and the
#                     experiment-aware validators work the same for every model.
#                     SAI labels that aren't in EXPECTED_YEARS_BY_EXPERIMENT
#                     (e.g. "g6-1.5") simply skip the year-count check.
#   model_coord       value written to the output `model` coord.
#   output_suffix     appended to output filenames so models don't collide
#                     (empty for CESM so its filenames are unchanged).
#   has_historical    whether a historical experiment exists — gates the
#                     hist+baseline merge and the historical-derived cell bounds.
#   no_sai_experiments  the model's no-SAI reference experiment label(s). The
#                     historical run is stitched into each of them, and each
#                     supplies the upper cell bound. A model can publish two:
#                     MIROC's GeoMIP ``baseline`` is the run its G6 experiments
#                     branched from (the SAI-delta rebase reference, pinned in
#                     fit_delta.ANCHORS), while ``ssp245`` is the ScenarioMIP run
#                     the climate-change-only emulator trains on (pinned in
#                     backend.constants.REGIONAL_MODEL2PROJECTION_BASELINE_FILES).
#   experiment_missing_vars  per-experiment variables the store does not publish,
#                     skipped up front rather than failing member-by-member.
#                     Nothing uses it at present: UKESM's historical QFLX and
#                     ICEFRAC, the one live case, now come from CMIP6 ESGF.
#   variable_members  per-VARIABLE member restriction, applied in every
#                     experiment. Used where a variable is backed by a source
#                     that covers fewer members than the store, so that the
#                     series stays on one realization instead of stepping its
#                     variance where the coverage changes.
#   compute_variability  whether to emit the dashboard variability artifacts
#                     (single set, from the primary model only).
MODELS = {
    "CESM2-WACCM-MA": {
        "experiments": [
            (f"{ZARR_STORES_URI}/CESM2-WACCM-MA_baseline.zarr",   "baseline"),
            (f"{ZARR_STORES_URI}/CESM2-WACCM-MA_historical.zarr", "historical"),
            (f"{ZARR_STORES_URI}/CESM2-WACCM-MA_1.5K-SAI.zarr",   "1.5"),
            (f"{ZARR_STORES_URI}/CESM2-WACCM-MA_1.0K-SAI.zarr",   "1.0"),
            (f"{ZARR_STORES_URI}/CESM2-WACCM-MA_0.5K-SAI.zarr",   "0.5"),
        ],
        "model_coord": "CESM2-WACCM",
        "output_suffix": "",
        "has_historical": True,
        # CESM's single baseline run is both the GAUSS branch point and the
        # ScenarioMIP SSP2-4.5 projection baseline.
        "no_sai_experiments": ("baseline",),
        "experiment_missing_vars": {},
        "compute_variability": True,
    },
    "MIROC-ES2H": {
        "experiments": [
            (f"{ZARR_STORES_URI}/MIROC-ES2H_baseline.zarr",    "baseline"),
            (f"{ZARR_STORES_URI}/MIROC-ES2H_ssp245.zarr",      "ssp245"),
            (f"{ZARR_STORES_URI}/MIROC-ES2H_historical.zarr",  "historical"),
            (f"{ZARR_STORES_URI}/MIROC-ES2H_G6-1.5K-SAI.zarr", "g6-1.5"),
            (f"{ZARR_STORES_URI}/MIROC-ES2H_G6-0.5K-SAI.zarr", "g6-0.5"),
        ],
        "model_coord": "MIROC-ES2H",
        "output_suffix": "-miroc",
        "has_historical": True,
        # Two distinct SSP2-4.5 realizations: the GeoMIP ``baseline`` the G6 runs
        # branched from (SAI-delta rebase reference) and the ScenarioMIP ``ssp245``
        # run (projection baseline; 2015-2100, continuous with historical). The
        # ssp245 store's QFLX is filled from the G6 baseline run's hfls over
        # 2020-2084 only (NaN elsewhere — see the array's note attr), so ssp245
        # p-e covers 65 years; its ICEFRAC is regridded from the run's own CMOR
        # siconc (full 2015-2100 — see that array's note attr).
        "no_sai_experiments": ("baseline", "ssp245"),
        "experiment_missing_vars": {},
        "compute_variability": False,
    },
    "UKESM1-1-LL": {
        "experiments": [
            (f"{ZARR_STORES_URI}/UKESM1-1-LL_baseline.zarr",    "baseline"),
            (f"{ZARR_STORES_URI}/UKESM1-1-LL_historical.zarr",  "historical"),
            (f"{ZARR_STORES_URI}/UKESM1-1-LL_G6-1.5K-SAI.zarr", "g6-1.5"),
        ],
        "model_coord": "UKESM1-1-LL",
        "output_suffix": "-ukesm",
        "has_historical": True,
        # One het-chem SSP2-4.5 run (2015-2100, MACLOUD) is both the G6 branch
        # point and the projection baseline — CESM-style single no-SAI run.
        "no_sai_experiments": ("baseline",),
        # MACLOUD publishes neither sea ice nor evaporative flux for the
        # historical suites, but CMIP6 ESGF publishes both (siconca and hfls; see
        # the store's PARTIAL note attrs), so nothing is missing for any
        # experiment and both series stitch like any other variable.
        "experiment_missing_vars": {},
        # ESGF publishes both for ONE member, the run this store calls r3. Using
        # r3 alone in every experiment keeps each series on a single realization:
        # a three-member mean after 2015 and a one-member series before it would
        # step the variance at the stitch, and the sea-ice sigmoid trains on
        # exactly that series. tas stays a three-member mean, so the fit pairs a
        # single-member record against an ensemble-mean temperature — the forced
        # response both sides share is what it fits.
        #
        # For p-e this restricts the QFLX term ONLY. p-e is assembled below as
        # (this model's pr output) - QFLX, and that pr file stays a three-member
        # mean, so p-e pairs ensemble-mean precipitation with single-member
        # evaporation. That is deliberate rather than an oversight: both terms are
        # uniform across the 2015 stitch so there is no seam, and the three-member
        # mean is the better estimate of forced pr. It does mean p-e is not
        # "r3 throughout" in the way icefrac is.
        "variable_members": {"icefrac": ("r3",), "p-e": ("r3",)},
        "compute_variability": False,
    },
}

# Zarr group per variable family: cumulative exposure counts live under
# atmos_2d_derived; daily extremes under atmos_2d_tasminmax; everything else
# in the main atmos_2d group.
_VAR_TO_GROUP = {
    "tasmin":       "Mon/atmos_2d_tasminmax",
    "tasmax":       "Mon/atmos_2d_tasminmax",
    "tas_above_35": "Mon/atmos_2d_derived",
    "tas_above_40": "Mon/atmos_2d_derived",
    "tas_below_0":  "Mon/atmos_2d_derived",
    "pr_above_10":  "Mon/atmos_2d_derived",
    "pr_above_20":  "Mon/atmos_2d_derived",
}

# --- Helper Functions --- #


def _dataarray_to_single_var_dataset(da: xr.DataArray, name: str) -> xr.Dataset:
    """Wrap a DataArray as a single-variable Dataset, renaming to ``name`` if needed.

    Used at the boundary between :mod:`xarray` operations that strip variable names (e.g.
    arithmetic) and downstream code that addresses the variable by name.
    """
    if da.name != name:
        da = da.rename(name)
    return da.to_dataset()


def _ensure_regridded_ensemble_dataset(
    obj: xr.DataArray | xr.Dataset, esm_var: str
) -> xr.Dataset:
    """``xr.concat`` on DataArrays can return a DataArray; downstream expects a Dataset."""
    if isinstance(obj, xr.Dataset):
        return obj
    return _dataarray_to_single_var_dataset(obj, esm_var)


# Helper to check if a variable is cumulative (like "days above 30C")
def is_cumulative(var_name: str) -> bool:
    """Return ``True`` for cumulative exposure variables (``*_above_*`` / ``*_below_*``).

    These are summed across the year in :func:`get_yearly_aggregate`; everything else
    falls through to the mean/min/max rules.
    """
    return "above" in var_name or "below" in var_name

def _group_for_var(var: str) -> str:
    """Return the Zarr group path for ``var``.

    Variable families live in different subgroups of each store: cumulative
    exposure counts (``tas_above_*``, ``pr_above_*``, ``tas_below_*``) in
    ``Mon/atmos_2d_derived``; sub-daily extremes (``tasmin``/``tasmax``) in
    ``Mon/atmos_2d_tasminmax``; the main monthly fields in ``Mon/atmos_2d``.
    """
    return _VAR_TO_GROUP.get(var, "Mon/atmos_2d")


# Attrs that the Zarr stores carry (xarray reads them straight into .attrs)
# but that NetCDF treats as reserved or as a zarr-only convention. Leaving
# them in place causes ``to_netcdf`` to fail with
# ``NetCDF: String match to name in use`` because ``_FillValue`` collides
# with xarray's own encoding-driven handling.
_NETCDF_RESERVED_ATTRS = frozenset({"_FillValue", "_ARRAY_DIMENSIONS"})


def _process_member_data(
    x: xr.DataArray, esm_var: str, exp: str
) -> xr.DataArray:
    """Apply per-member processing — time slice, ICEFRAC NH-September mask,
    unit scaling, ``height`` drop, cftime→datetime64, NetCDF-reserved-attr
    strip — to one DataArray.

    Used by the Zarr-loading path in :func:`process_monthly`; callers slice
    ``ds[esm_var].sel(member=...)`` and pass the resulting DataArray here.
    Mirrors the per-file branch of :func:`process_file` so downstream
    aggregation and validation see the same shape regardless of source.
    """
    time_range = slice("1850", "2014") if "historical" in exp else slice("2015", "2099")
    x = x.sel(time=time_range)

    if esm_var == "ICEFRAC":
        is_sept_north = (x.lat > 0) & (x.time.dt.month == 9)
        x = x.where(is_sept_north, np.nan)

    if esm_var in UNIT_SCALERS:
        x = x * UNIT_SCALERS[esm_var]

    if "height" in x.coords:
        x = x.drop_vars("height")

    if x.indexes["time"].dtype == "O":
        x["time"] = x["time"].values.astype("datetime64[s]").astype("datetime64[ns]")

    x.attrs = {k: v for k, v in x.attrs.items() if k not in _NETCDF_RESERVED_ATTRS}
    for coord_name in x.coords:
        x[coord_name].attrs = {
            k: v
            for k, v in x[coord_name].attrs.items()
            if k not in _NETCDF_RESERVED_ATTRS
        }

    return x


def get_yearly_aggregate(
    ds: xr.Dataset | xr.DataArray,
    var_name: str,
    esm_var: str,
) -> xr.Dataset | xr.DataArray:
    """
    Resamples data to yearly, applies the correct aggregation, 
    and REPLACES years that do not have full 12-month coverage with NaN.
    """
    print(f" var_name: {var_name}")

    # Resample
    resampler = ds.resample(time="1YE")

    # Check Completeness (Count valid months per year)
    # This creates a boolean mask: True if year has 12 valid months, False otherwise
    counts = resampler.count(dim="time")

    # Default requirement: 12 months. 
    # EXCEPTION: ICEFRAC (which is Sept-only in your logic) should not be filtered by 12 months.
    if var_name == "icefrac":
        is_complete = (counts >= 1) # Keep if we have at least the 1 month (Sept)
    else:
        is_complete = (counts == 12)

    # --- REPORT DROPPED YEARS ---
    # If is_complete is a Dataset, we must pull out the variable's mask
    # to get a 1D array aligned with 'time'
    if isinstance(is_complete, xr.Dataset):
        # Use the variable name to get the boolean DataArray
        mask_da = is_complete[esm_var]
    else:
        # It's already a DataArray
        mask_da = is_complete

    # Reduce 3D (Time, Lat, Lon) -> 1D (Time) for logging
    # We check if the year is valid "everywhere" or just pick a reference point.
    # Since missing files affect the whole globe, selecting the first spatial point is safe and fast.
    # We use .isel to grab the first lat/lon index.
    if 'lat' in mask_da.dims and 'lon' in mask_da.dims:
        mask_1d = mask_da.isel(lat=-21, lon=-57)
    elif 'lat' in mask_da.dims: # Handle edge case of 2D data
        mask_1d = mask_da.isel(lat=-21)
    else:
        mask_1d = mask_da

    # Identify years with incomplete data
    dropped_mask = ~mask_1d.values 

    if dropped_mask.any():
        # Now we can safely index the 1D time array
        dropped_years_vals = ds.resample(time="1YE").mean().time[dropped_mask].dt.year.values
        print(f"\033[93m  [WARNING] {len(dropped_years_vals)} incomplete years for {var_name}: {dropped_years_vals}\033[0m")
    # ----------------------------

    # Determine Method & Aggregate
    if is_cumulative(var_name):
        agg = resampler.sum()
    else:
        method = AGGREGATION_RULES.get(var_name, AGGREGATION_RULES["default"])

        if method == "min":
            agg = resampler.min()
        elif method == "max":
            agg = resampler.max()
        elif method == "mean":
            agg = resampler.mean()
            # Special fix for icefrac NaN handling
            if var_name == "icefrac":
                 agg = agg.fillna(0)
        else:
            agg = resampler.mean()

    # Filter Incomplete Years
    # where(mask, drop=True) keeps where mask is True, drops the rest.
    filtered_agg = agg.where(is_complete, drop=True)
    # Optional: Warn if data was dropped (Debugging)
    dropped_years = len(agg.time) - len(filtered_agg.time)
    if dropped_years > 0:
        print(f" Dropped {dropped_years} incomplete years for {var_name}")

    return filtered_agg


def validate_continuity(ds: xr.Dataset | xr.DataArray) -> None:
    """Asserts that monthly data is continuous (diff is exactly 1 month)."""
    time_diffs = np.diff(ds.time.values.astype("datetime64[M]"))
    one_month = np.timedelta64(1, 'M')
    if not np.all(time_diffs == one_month):
        raise ValueError("\033[91mData is not temporally continuous (missing months detected).\033[0m")


def _combined_monthly_to_dataarray(
    combined: xr.DataArray | xr.Dataset, esm_var: str
) -> xr.DataArray:
    """``combine_by_coords`` on DataArrays often returns a Dataset; downstream expects one variable."""
    if isinstance(combined, xr.Dataset):
        if esm_var not in combined.data_vars:
            names = ", ".join(sorted(combined.data_vars))
            raise ValueError(
                f"Combined monthly data is a Dataset but does not contain {esm_var}. "
                f"Variables present: {names}"
            )
        return combined[esm_var]
    return combined


def merge_historical_into(
    exp2member2data_combined: dict[str, dict[str, xr.Dataset | xr.DataArray]],
    no_sai_experiments: tuple[str, ...] = ("baseline",),
) -> None:
    """
    Merges 'historical' into each no-SAI reference experiment, as a superset of members.

    A model can publish more than one no-SAI run — MIROC ships both the GeoMIP
    ``baseline`` its G6 experiments branched from and the ScenarioMIP ``ssp245`` run —
    and each needs the full 1850-onwards series its consumer expects (the delta fit
    rebases against ``baseline``; the projection emulator and the dashboard's historical
    series read ``ssp245``). The same historical members are therefore stitched into
    every entry of ``no_sai_experiments``. 'historical' is removed once all are merged.
    """
    print("Merging Historical into no-SAI experiments (Superset Mode)...")

    historical_data = exp2member2data_combined.pop('historical', {})
    if not historical_data:
        print("  Note: No historical data found. Keeping no-SAI experiments as is.")
        return

    for exp in no_sai_experiments:
        if exp not in exp2member2data_combined:
            # e.g. MIROC p-e/icefrac have no ScenarioMIP output to merge into.
            print(f"  Note: no {exp} data for this variable — nothing to merge into.")
            continue
        print(f"  -> {exp}")
        _merge_historical_into_one(historical_data, exp2member2data_combined[exp])


def _merge_historical_into_one(
    historical_data: dict[str, xr.Dataset | xr.DataArray],
    baseline_data: dict[str, xr.Dataset | xr.DataArray],
) -> None:
    """Stitch ``historical_data`` into one no-SAI experiment's member dict, in place.

    Logic:
    1. Overlapping Members (e.g. Hist 1 & Base 1): Stitched together (1850-2100).
    2. Historical-Only (e.g. Hist 2): kept as is (1850-2014).
    3. Baseline-Only (e.g. Base 7): kept as is (2015-2100).
    """
    # Identify all unique member IDs involved
    hist_ids = set(historical_data.keys())
    base_ids = set(baseline_data.keys())
    all_ids = sorted(list(hist_ids | base_ids))

    count_stitched = 0
    count_hist_only = 0
    count_base_only = 0

    for member in all_ids:
        # Case 1: Member exists in BOTH (Stitch them)
        if member in hist_ids and member in base_ids:
            ds_hist = historical_data[member].sel(time=slice(None, 2014))
            ds_base = baseline_data[member]

            try:
                # Concatenate along time
                ds_combined = xr.concat(
                    [ds_hist, ds_base],
                    dim="time",
                    join="outer"
                ).sortby("time")

                # Instead of using .dt accessor (which is unavailable for DataArray without datetime objects),
                # use values and numpy directly.
                # This fix properly extracts years even if xarray does not supply .dt
                years = ds_combined["time"].values
                # np.datetime64: convert to integer years
                # Handle if time is not datetime64
                try:
                    # This will succeed for datetimes
                    years_int = np.array([pd.Timestamp(y).year for y in years])
                except Exception:
                    try:
                        # Sometimes time values are already numeric year
                        years_int = years.astype(int)
                    except Exception:
                        years_int = np.arange(len(years))

                # Check for gaps
                if len(years_int) > 1 and np.any(np.diff(years_int) > 1):
                    print(f"\033[93m    [WARNING] Gap detected in stitched member {member}\033[0m")

                baseline_data[member] = ds_combined
                count_stitched += 1

            except Exception as e:
                print(f"\033[91m    Error stitching member {member}: {e}\033[0m")

        # Case 2: Member exists ONLY in Historical
        elif member in hist_ids:
            # Move it to baseline dict so it's included in the final ensemble
            baseline_data[member] = historical_data[member]
            count_hist_only += 1

        # Case 3: Member exists ONLY in Baseline
        elif member in base_ids:
            # It's already in baseline_data, just count it
            count_base_only += 1

    print("-" * 40)
    print(f"Merge Complete. Total Members: {len(baseline_data)}")
    print(f"  - Full Time Series (Stitched): {count_stitched}")
    print(f"  - Historical Only (1850-2014): {count_hist_only}")
    print(f"  - Baseline Only   (2015-2100): {count_base_only}")
    print("-" * 40)


def get_robust_regridder(
    ds_sample: xr.Dataset | xr.DataArray,
    target_grid: xr.Dataset | xr.DataArray,
) -> "xe.Regridder":
    """
    Initializes an xesmf regridder.
    Includes fallback logic for datasets with malformed lat/lon coordinates.
    """
    import xesmf as xe  # lazy: see module-top note on the ESMF dependency
    try:
        # Standard initialization
        return xe.Regridder(ds_sample, target_grid, 'bilinear', periodic=True)
    except Exception as e:
        print(f"\033[93m  Standard regridder failed ({e}). Attempting coordinate fix...\033[0m")
        # Fallback: Force standard lat/lon arrays if metadata is broken
        ds_fix = ds_sample.copy()
        ds_fix['lat'] = np.linspace(-90, 90, len(ds_sample.lat))
        ds_fix['lon'] = np.linspace(0, 360, len(ds_sample.lon))
        
        return xe.Regridder(ds_fix, target_grid, 'bilinear', periodic=True, ignore_degenerate=True)


def compute_and_save_variability(
    exp2member2data_combined: dict[str, dict[str, xr.Dataset | xr.DataArray]],
    var_name: str,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[float, xr.DataArray] | None:
    """
    Calculates and saves the model internal variability based on the 'baseline' experiment.
    
    Method:
    1. Global Weighted Mean (Latitudinal)
    2. Detrending (removing linear trend over time)
    3. Standard Deviation of the residuals (across time and members)
    """
    print("Computing model internal variability...")
    
    # 1. Extract Baseline Data
    if 'baseline' not in exp2member2data_combined:
        print("\033[93mWarning: No 'baseline' experiment found. Skipping variability calculation.\033[0m")
        return

    # Dictionary of {member: DataArray} -> List of DataArrays
    baseline_members = list(exp2member2data_combined['baseline'].values())
    
    if not baseline_members:
        print("Warning: Baseline experiment is empty.")
        return

    # 2. Concatenate Members
    # Result dim: (member, time, lat, lon)
    ds_baseline = xr.concat(baseline_members, dim="member", join="outer")

    # 3. Calculate Global Weighted Mean
    # Ensure we are working with the variable, not a Dataset wrapper
    if isinstance(ds_baseline, xr.Dataset):
        da = ds_baseline[var_name] # Extract DataArray
    else:
        da = ds_baseline # It is already a DataArray

    # Cosine weighting for latitude
    weights = np.cos(np.deg2rad(da.lat))
    global_mean = da.weighted(weights).mean(dim=("lat", "lon"))

    # 4. Filter Time Range (2025-2099)
    # Using slice ensures we don't crash if range is slightly smaller
    global_mean = global_mean.sel(time=slice(2025, 2099))

    # 5. Detrending (Linear)
    # We fit a degree-1 polynomial along the 'time' dimension
    print("  Detrending baseline data...")
    coeffs = global_mean.polyfit(dim='time', deg=1)
    
    # Evaluate the trend: y_trend = a*t + b
    fitted_trend = xr.polyval(global_mean['time'], coeffs.polyfit_coefficients)
    
    # Subtract trend to get residuals
    detrended_residuals = global_mean - fitted_trend

    # 6. Compute Standard Deviation
    # Taking std of all residuals across all members and time steps
    std_value = detrended_residuals.std().item()

    # 7. Save
    out_path = Path(output_dir) / "model_internal_variability.npy"
    print(f"  Internal Variability (std): {std_value:.4f}")
    if out_path.exists() and not overwrite:
        print(f"  File already exists and overwrite is False: {out_path}")
        print("  Skipping export of model internal variability.")
        return std_value, weights
    print(f"  Saving model internal variability to {out_path}")
    np.save(out_path, std_value)
    
    return std_value, weights


def compute_and_save_grid_variability(
    exp2member2data_combined: dict[str, dict[str, list[xr.Dataset]]],
    var: str,
    esm_var: str,
    output_dir: Path,
    overwrite: bool = False
) -> xr.Dataset:
    """
    Calculates and saves the GRID-LEVEL model internal variability based on the 'baseline' experiment.
    
    Method:
    1. Select Baseline Data (2025-2099)
    2. Detrend per-pixel (remove linear trend at each lat/lon)
    3. Compute Standard Deviation of residuals over the reference period (2025-2039)
    4. Export as NetCDF
    """
    print("Computing grid-level model internal variability...")
    
    # 1. Extract Baseline Data
    if 'baseline' not in exp2member2data_combined:
        print("\033[93mWarning: No 'baseline' experiment found. Skipping grid variability.\033[0m")
        return

    baseline_members = list(exp2member2data_combined['baseline'].values())
    
    if not baseline_members:
        print("Warning: Baseline experiment is empty.")
        return

    # Concatenate Members -> (member, time, lat, lon)
    ds_baseline = xr.concat(baseline_members, dim="member", join="outer")

    # Extract DataArray if needed
    if isinstance(ds_baseline, xr.Dataset):
        da = ds_baseline[esm_var]
    else:
        da = ds_baseline

    # 2. Slice Full Period for Detrending (2025-2099)
    # We use the full period to get a robust trend estimate, even if we calculate std on a shorter window later.
    regional = da.sel(time=slice(2025, 2099))
    
    print("  Calculating per-pixel linear trend (this may take a moment)...")
    
    # 3. Vectorized Detrending
    # polyfit computes the slope and intercept for every (lat, lon) simultaneously
    trend_coeffs = regional.polyfit(dim="time", deg=1)
    
    # Evaluate the trend line: y = mx + b
    fitted_trend = xr.polyval(regional["time"], trend_coeffs.polyfit_coefficients)
    
    # Subtract trend to get the "noise"
    detrended_residuals = regional - fitted_trend

    # 4. Compute Variability (Standard Deviation)
    # We focus on the near-term window (2025-2039) to estimate variability
    print("  Computing standard deviation (2025-2039)...")
    reference_window = detrended_residuals.sel(time=slice(2025, 2039))
    
    # Calculate std across BOTH time and ensemble members
    grid_variability = reference_window.std(dim=("time", "member"))

    # 5. Export
    # Convert back to Dataset for saving with metadata
    ds_out = grid_variability.to_dataset(name=var)
    
    # Add attributes for reproducibility
    ds_out[var].attrs = da.attrs
    ds_out[var].attrs['description'] = "Internal Variability (Std Dev of detrended residuals 2025-2039)"
    
    out_path = Path(output_dir) / "grid_level_model_internal_variability.nc"
    if out_path.exists() and not overwrite:
        print(f"  File already exists and overwrite is False: {out_path}")
        print("  Skipping export of grid-level model internal variability.")
        return ds_out
    print(f"  Saving grid variability to {out_path}")
    ds_out.to_netcdf(out_path)
    return ds_out


# --- Main Function --- #
def process_monthly(
    var: str,
    data_dir: str | Path = "",
    output_dir: str | Path = "data/processed",
    overwrite: bool = False,
    model: str = "all",
) -> None:
    """
    Process monthly Simulator data for one variable, per climate model.

    Each model in :data:`MODELS` is processed independently — its own
    experiment set, baseline, output-filename suffix and ``model`` metadata
    coord — so model-specific quirks (MIROC's baseline starts in 2020 leaving
    a 2015-2019 seam after the historical merge, it runs on a different native
    grid, etc.) stay isolated. Every model's outputs land in the same
    ``output_dir/<var>/`` so the fit step can pick them all up.

    Args:
        var: Variable to process.
        data_dir: Unused. Inputs come from the Zarr stores at
            :data:`version.ZARR_STORES_URI`. Kept so the existing
            ``scripts/process_all_monthly.sh`` invocation works unchanged.
        output_dir: Directory to save processed data.
        overwrite: Whether to overwrite existing files.
        model: Which model to process — a key of :data:`MODELS`, or ``"all"``
            (default) to process every model in turn.
    """
    print(f"var: {var}")
    print("=" * 100)

    if var not in var2esm_var:
        raise ValueError(f"\033[91mvar must be one of {list(var2esm_var.keys())}\033[0m")

    if model == "all":
        model_names = list(MODELS.keys())
    elif model in MODELS:
        model_names = [model]
    else:
        raise ValueError(
            f"\033[91mmodel must be 'all' or one of {list(MODELS.keys())}\033[0m"
        )

    output_dir = Path(output_dir) / var
    output_dir.mkdir(exist_ok=True, parents=True)
    esm_var = var2esm_var[var]

    for model_name in model_names:
        print(f"\n--- Processing {var} ({esm_var}) for model {model_name} ---")
        _process_model(var, esm_var, model_name, MODELS[model_name], output_dir, overwrite)

    print(f"--- Processing {var} Complete ---")


def _process_model(
    var: str,
    esm_var: str,
    model_name: str,
    model_cfg: dict,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """Process one model's experiment set into per-experiment output files.

    Extracted from the original single-model body; the ``model_cfg`` fields
    parametrise the model-specific points (which stores + labels to load, the
    ``model`` coord, the output-filename suffix, whether a historical
    experiment exists, and whether to emit dashboard variability artifacts).
    """
    experiments = model_cfg["experiments"]
    model_coord = model_cfg["model_coord"]
    output_suffix = model_cfg["output_suffix"]
    has_historical = model_cfg["has_historical"]
    no_sai_experiments = tuple(model_cfg.get("no_sai_experiments", ("baseline",)))
    experiment_missing_vars = model_cfg.get("experiment_missing_vars", {})
    variable_members = model_cfg.get("variable_members", {})
    allowed_members = variable_members.get(var)
    compute_variability = model_cfg["compute_variability"]

    # Loading from Zarr stores. Each store carries all members for one
    # experiment under the appropriate Mon/atmos_2d* subgroup; one DataArray
    # lands per (experiment, member) in exp2member2data.
    if allowed_members is not None:
        print(f"\033[93m>> {var}: restricted to member(s) {', '.join(allowed_members)} "
              f"for {model_name}\033[0m")
    print(">> Loading Zarr stores …")
    group = _group_for_var(var)
    exp2member2data: dict[str, dict[str, list[xr.DataArray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for zarr_url, exp_label in experiments:
        if var in experiment_missing_vars.get(exp_label, ()):
            # Declared absent up front (the store ships an all-NaN placeholder), so
            # skip it rather than letting every member fail the NaN check.
            print(f"\033[93m   {exp_label}: {var} not published for this experiment — skipping\033[0m")
            continue
        print(f"   {zarr_url} (group={group})")
        try:
            ds = open_xarray_datasets(zarr_url, group=group)
        except Exception as e:
            print(f"\033[91m   failed to open: {e}\033[0m")
            continue

        if esm_var not in ds.data_vars:
            print(f"\033[93m   {esm_var} not present in this store/group — skipping\033[0m")
            continue

        for m_value in ds["member"].values:
            m = str(m_value)
            if allowed_members is not None and m not in allowed_members:
                # This variable is restricted to a subset of the model's members
                # (see variable_members in the MODELS entry). Skipping here keeps
                # the restriction in one place rather than leaving members to fail
                # a downstream NaN check.
                continue
            member_da = ds[esm_var].sel(member=m_value).drop_vars(
                "member", errors="ignore"
            )
            try:
                processed = _process_member_data(member_da, esm_var, exp_label)
            except Exception as e:
                print(f"\033[91m   error processing {exp_label}/{m}: {e}\033[0m")
                continue
            exp2member2data[exp_label][m].append(processed)

    if not exp2member2data:
        print(f"\033[93mNo data loaded for {var} / {model_name} — skipping this model.\033[0m")
        return

    # Combination & Aggregation
    print(">> Combining & Aggregating...")
    # Cell bounds need a historical experiment for the temperature min bound;
    # models without one skip the check (None == no-op in the validator).
    monthly_cell_bounds = (
        build_monthly_cell_bounds(var, exp2member2data, no_sai_experiments)
        if has_historical
        else None
    )

    # Create grid for output (lat/lon) — the CESM2-WACCM target grid that all
    # models are regridded onto.
    correct_lat = np.load(output_dir.parent / "correct_lat.npy")
    common_grid = {
        'lon': np.linspace(0, 358.75, 288),
        'lat': correct_lat
    }

    exp2member2data_combined = defaultdict(lambda: defaultdict(list))
    year_count_summary = defaultdict(dict)
    failed_sanity_checks: list[tuple[str, str, str]] = []
    failed_processing: list[tuple[str, str, str]] = []
    for exp, member2data in tqdm(exp2member2data.items(), desc="Experiments"):
        for member, data_list in member2data.items():
            try:
                # A. Combine time steps
                ds_combined = _combined_monthly_to_dataarray(
                    xr.combine_by_coords(data_list, combine_attrs="drop_conflicts"),
                    esm_var,
                )

                # B. Validation
                validate_continuity(ds_combined)
                validate_monthly_cell_bounds(
                    data=ds_combined,
                    var_name=var,
                    exp=exp,
                    member=member,
                    bounds=monthly_cell_bounds,
                    no_sai_experiments=no_sai_experiments,
                )

                # C. Yearly Aggregation
                ds_yearly = get_yearly_aggregate(ds_combined, var, esm_var)

                # D. Convert Time Index (Datetime -> Year Integer)
                years = ds_yearly.time.dt.year.values
                ds_yearly = ds_yearly.assign_coords(time=years)
                year_count_summary[exp][member] = len(ds_yearly.time)

                # E. Sanity check yearly count by experiment type.
                validate_year_count_for_experiment(
                    exp=exp,
                    year_count=year_count_summary[exp][member],
                    var_name=var,
                    member=member,
                )
                validate_year_continuity_strict(
                    years=years,
                    var_name=var,
                    exp=exp,
                    member=member,
                    stage="unmerged yearly continuity",
                )

                validate_no_nans(
                    data=ds_yearly,
                    var_name=var,
                    exp=exp,
                    member=member,
                    stage="yearly aggregation",
                )
                validate_no_negative_values(
                    data=ds_yearly,
                    var_name=var,
                    exp=exp,
                    member=member,
                    stage="yearly aggregation",
                )
                validate_variable_specific_ranges(
                    data=ds_yearly,
                    var_name=var,
                    exp=exp,
                    member=member,
                    stage="yearly aggregation",
                )

                exp2member2data_combined[exp][member] = ds_yearly

            except SanityCheckError as e:
                print(f"\033[91mSanity check failed for {exp} member {member}: {e}\033[0m")
                failed_sanity_checks.append((exp, member, str(e)))
            except Exception as e:
                print(f"\033[91mIO/parsing error for {exp} member {member}: {e}\033[0m")
                failed_processing.append((exp, member, str(e)))

    # Print a compact summary of yearly counts to aid debugging.
    if year_count_summary:
        print(f"Year-count summary for {var} / {model_name}:")
        for exp in sorted(year_count_summary.keys()):
            members = year_count_summary[exp]
            detail = ", ".join(
                f"{member}:{members[member]}" for member in sorted(members.keys())
            )
            print(f"  - {exp}: {detail}")
    if failed_sanity_checks:
        print(
            f"\033[93mWarning: {len(failed_sanity_checks)} member(s) failed sanity checks:\033[0m"
        )
        for exp, member, message in failed_sanity_checks:
            print(f"  - {exp}/{member}: {message}")
    if failed_processing:
        print(
            f"\033[93mWarning: {len(failed_processing)} member(s) had IO/parsing errors:\033[0m"
        )
        for exp, member, message in failed_processing:
            print(f"  - {exp}/{member}: {message}")

    if not exp2member2data_combined:
        raise ValueError(
            f"No valid data remained for {var} / {model_name} after sanity checks. "
            "See earlier per-member validation errors."
        )

    baseline_members = exp2member2data_combined.get("baseline", {})
    if not baseline_members:
        raise ValueError(
            f"No valid baseline members remained for {var} / {model_name} after sanity "
            "checks. Baseline is required."
        )

    # Dashboard variability artifacts — emitted by the primary model only
    # (single set per variable; secondary models' baselines would overwrite).
    if compute_variability:
        variability_result = compute_and_save_variability(
            exp2member2data_combined, esm_var, output_dir, overwrite
        )
        if variability_result is None:
            raise ValueError(
                f"Could not compute model internal variability for {var}: baseline unavailable."
            )
        std_value, weights = variability_result

        grid_level_variability = compute_and_save_grid_variability(
            exp2member2data_combined, var, esm_var, output_dir, overwrite
        )
        if grid_level_variability is None:
            raise ValueError(
                f"Could not compute grid-level variability for {var}: baseline unavailable."
            )

        process_geojson_variability(
            output_dir.parent / "geojsons/original", grid_level_variability, var, weights
        )

    # Stitch Historical + Baseline (only for models that have a historical run).
    if has_historical:
        merge_historical_into(exp2member2data_combined, no_sai_experiments)
        for exp in no_sai_experiments:
            for member, ds in exp2member2data_combined.get(exp, {}).items():
                validate_year_continuity_with_max_gap(
                    years=ds.time.values,
                    var_name=var,
                    exp=exp,
                    member=member,
                    stage="merged continuity",
                    max_gap_years=MERGED_HIST_BASELINE_MAX_YEAR_GAP,
                )

    print("Regridding and creating ensemble DataArrays...")

    # Initialize Regridder ONCE
    import xesmf as xe  # lazy: see module-top note on the ESMF dependency
    first_exp = next(iter(exp2member2data_combined.values()))
    first_member_ds = next(iter(first_exp.values()))

    # Standard coords as a fallback if the native coords confuse xesmf.
    standard_lat = np.linspace(-90, 90, len(first_member_ds.lat))
    standard_lon = np.linspace(0, 360, len(first_member_ds.lon))

    use_coord_fix = False
    try:
        regridder = xe.Regridder(first_member_ds, common_grid, 'bilinear', periodic=True)
    except Exception:
        # CAUTION: this fallback *forces* a uniform -90..90 / 0..360 grid rather
        # than repairing broken coordinate metadata. It is safe only when the
        # source grid really is (near-)uniform with that extent — true for the
        # CESM2-WACCM grid this was written for. For a model whose native grid
        # is non-uniform or pole-excluding (e.g. a Gaussian latitude axis), the
        # forced coords would not match the real cell centers, silently
        # producing a spatially shifted field that still passes the NaN/range
        # checks below. If a new model trips this path, verify its grid before
        # trusting the output (MIROC's 128×256 atmos grid does NOT trip it — the
        # standard regridder handles its native coords).
        print("\033[93m  Standard regridder failed. Switching to robust mode (forcing coords).\033[0m")
        use_coord_fix = True
        ds_fix = first_member_ds.copy()
        ds_fix['lat'] = standard_lat
        ds_fix['lon'] = standard_lon
        regridder = xe.Regridder(ds_fix, common_grid, 'bilinear', periodic=True, ignore_degenerate=True)

    final_output_data = {}
    for exp, member2data in tqdm(exp2member2data_combined.items(), desc="Regridding Experiments"):
        regridded_members = []
        for member, ds in member2data.items():
            if use_coord_fix:
                ds = ds.copy()
                ds['lat'] = standard_lat
                ds['lon'] = standard_lon

            regridded = regridder(ds)
            validate_no_nans(
                data=regridded, var_name=var, exp=exp, member=member, stage="regridding",
            )
            validate_no_negative_values(
                data=regridded, var_name=var, exp=exp, member=member, stage="regridding",
            )
            validate_variable_specific_ranges(
                data=regridded, var_name=var, exp=exp, member=member, stage="regridding",
            )
            regridded = regridded.assign_coords(member=member)
            regridded_members.append(regridded)

        if regridded_members:
            ds_ensemble = xr.concat(regridded_members, dim="member", join="outer")
            ds_ensemble = _ensure_regridded_ensemble_dataset(ds_ensemble, esm_var)
            final_output_data[exp] = ds_ensemble

    print("Regridding complete.")
    print("Finalizing and saving data...")

    metadata_coords = {"ssp": ["ssp245"], "model": [model_coord]}

    for exp, ds_ensemble in final_output_data.items():

        # Save "All Members" BEFORE averaging. Every variable writes its no-SAI reference
        # experiment(s) -- the standalone variability step (scripts/variability.py) derives
        # each model's internal variability from these regridded members -- and tas/icefrac
        # additionally write their SAI experiments, which the icefrac hindcast in that
        # script trains on.
        ds_members_renamed = None
        if var in ["icefrac", "tas"] or exp in no_sai_experiments:
            ds_members_renamed = ds_ensemble.rename_vars({esm_var: var})
            if var == "p-e":
                # Per-member P-E = pr - evap from the matching pr all-members file, so the
                # file holds what its name says (the ensemble-mean path below does the
                # same). Without the pr file we would be saving evaporation under "p-e".
                pr_members_path = (
                    output_dir.parent / "pr" / f"output_gauss-{exp}{output_suffix}-all-members.nc"
                )
                if pr_members_path.exists():
                    with xr.open_dataset(pr_members_path, decode_times=True) as ds_pr_members:
                        ds_pr_members = ds_pr_members.load()
                    ds_members_renamed[var] = ds_pr_members["pr"] - ds_members_renamed[var]
                else:
                    print(
                        f"\033[93m  [WARNING] Skipping p-e all-members for {exp}: "
                        f"missing {pr_members_path}\033[0m"
                    )
                    ds_members_renamed = None

        if ds_members_renamed is not None:
            out_path_members = output_dir / f"output_gauss-{exp}{output_suffix}-all-members.nc"
            print(f"  Saving all-members data to {out_path_members.name}...")
            # No NaN check here: icefrac legitimately carries NaNs from the NH-September
            # mask, and the member/time union (join='outer') can leave NaNs for any
            # variable. (The negative-value and range guards below still apply.)
            validate_no_negative_values(
                data=ds_members_renamed[var], var_name=var, exp=exp,
                member="all-members", stage="all-members output",
            )
            validate_variable_specific_ranges(
                data=ds_members_renamed[var], var_name=var, exp=exp,
                member="all-members", stage="all-members output",
            )

            if overwrite or not out_path_members.exists():
                save_processed_data(ds_members_renamed, out_path_members, overwrite)
            else:
                print(f"    Skipping (file exists): {out_path_members.name}")

        # Compute Ensemble Mean
        ds_mean = ds_ensemble.mean(dim="member")
        ds_mean = _ensure_regridded_ensemble_dataset(ds_mean, esm_var)

        ds_mean = ds_mean.expand_dims({"ssp": 1, "model": 1})
        ds_mean = ds_mean.assign_coords(
            ssp=('ssp', metadata_coords["ssp"]),
            model=('model', metadata_coords["model"]),
            lon=(ds_mean.lon % 360)  # Ensure 0-360 longitude
        )

        ds_mean = ds_mean.rename_vars({esm_var: var})

        # P-E = pr - evap, using the matching pr output for this model/experiment.
        if var == "p-e":
            pr_path = output_dir.parent / "pr" / f"output_gauss-{exp}{output_suffix}.nc"
            if pr_path.exists():
                print(f"  Calculating P-E using {pr_path.name}...")
                try:
                    ds_pr = xr.open_dataset(pr_path, decode_times=True)
                    ds_mean[var] = ds_pr["pr"] - ds_mean[var]
                except Exception as e:
                    print(f"\033[91m  [ERROR] Failed to calculate P-E: {e}\033[0m")
            else:
                print(f"\033[93m  [WARNING] Cannot calculate P-E. Missing 'pr' file: {pr_path}\033[0m")

        validate_no_nans(
            data=ds_mean[var], var_name=var, exp=exp, member="ensemble-mean",
            stage="ensemble mean output",
        )
        validate_no_negative_values(
            data=ds_mean[var], var_name=var, exp=exp, member="ensemble-mean",
            stage="ensemble mean output",
        )
        validate_variable_specific_ranges(
            data=ds_mean[var], var_name=var, exp=exp, member="ensemble-mean",
            stage="ensemble mean output",
        )

        out_path_mean = output_dir / f"output_gauss-{exp}{output_suffix}.nc"
        print(f"  Saving ensemble mean to {out_path_mean.name}")
        if overwrite or not out_path_mean.exists():
            save_processed_data(ds_mean, out_path_mean, overwrite)
        else:
            print(f"    Skipping (file exists): {out_path_mean.name}")


if __name__ == "__main__":
    import fire  # CLI-only; kept out of module-level imports (see top of file)
    fire.Fire(process_monthly)
