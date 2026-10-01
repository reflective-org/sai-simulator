"""Compute and persist each climate model's internal-variability artifacts.

One artifact set per (variable, model), derived from the **regridded per-member** no-SAI
ensemble that ``scripts/process_monthly_gauss.py`` writes as ``*-all-members.nc`` — the same
regridding the tested fields went through — so both models are treated by one operator:

* ``grid_level_model_internal_variability_<model>.nc`` — per-cell sigma (sizes the per-cell
  t-test in :mod:`backend.p_values`).
* ``model_internal_variability_<model>.npy`` — one global-series sigma (the dashboard's
  mean-over-time band).
* a ``variability_<var>_<model>`` column in each region geojson (regional-mode band).

Two constructs, unchanged from before but now per model:

* **Field variables** (tas, pr, p-e, tasmin/max, the daily-count variables): detrend each
  member's series over 2025-2099 with a per-cell (or global-mean) linear fit; sigma is the
  std of the residuals — over 2025-2099 for the global series, over 2025-2039 for the grid.
  The members come from the model's *projection baseline* (``ssp245`` where it exists,
  resolved via ``backend.constants.REGIONAL_MODEL2PROJECTION_BASELINE_FILES``).
* **icefrac**: hindcast the model's own fitted sigmoid over the scenarios it was trained on
  (``scripts.fit_icefrac_arctic.MODEL_SCENARIOS``), subtract the simulated members, and take
  the residual std — per cell, and for the area-summed extent.

The legacy unsuffixed CESM files (native-grid, computed inside process_monthly_gauss.py)
are left in place until every reader has moved to the per-model files.

CLI entry: ``python scripts/variability.py <var_name> <data_dir> <model_dir> [--model all]``.
"""
import xarray as xr
import numpy as np
import sys
# ``fire`` is imported lazily inside the ``__main__`` block below so that
# importing this module (transitively pulled in by scripts.process_monthly_gauss
# for ``process_geojson_variability``) does not require ``fire`` in CI.
from typing import Union, Tuple
from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.icefrac import (
    create_arctic_dataset, 
    calculate_temperature_anomalies, 
    get_area, 
    load_icefrac_params,
    get_parameter_values,
    logistic
)
from backend.utils import open_xarray_datasets
from backend.utils import create_mask
from backend.constants import (
    DISPLAY_BASELINE_END,
    DISPLAY_BASELINE_START,
    REGIONAL_MODEL2PROJECTION_BASELINE_FILES,
    REGIONAL_MODEL_NAMES,
)
from scripts.fit_icefrac_arctic import MODEL_SCENARIOS

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

VAR_TEMP = "tas"
VAR_ICE = "icefrac"

# Models that get an artifact set each.
MODELS = tuple(REGIONAL_MODEL_NAMES)

# Windows of the field-variability construct (unchanged from the in-process computation
# this replaces): detrend over TREND_YEARS; the global-series sigma is the std of all
# detrended residuals over TREND_YEARS, the grid-cell sigma over GRID_STD_YEARS.
TREND_YEARS = (2025, 2099)
GRID_STD_YEARS = (2025, 2039)
# The significance test's reference period (backend.constants.DISPLAY_BASELINE_*): the
# scatter of the members over it is the reference-mean noise in the Welch test
# (backend.p_values.member_reference_noise).
REFERENCE_YEARS = (DISPLAY_BASELINE_START, DISPLAY_BASELINE_END)


def icefrac_scenarios(model: str) -> list:
    """(name, all-members filename, start, end) for the icefrac hindcast, per model.

    Mirrors ``scripts.fit_icefrac_arctic.MODEL_SCENARIOS`` — the residual variability is
    evaluated on exactly the scenarios each model's sigmoid was fit to — pointed at the
    ``-all-members`` files that carry the individual members.
    """
    return [
        (name, cfg["file"].replace(".nc", "-all-members.nc"), cfg["start"], cfg["end"])
        for name, cfg in MODEL_SCENARIOS[model].items()
    ]


def icefrac_reference_file(model: str) -> str:
    """The model's own baseline ensemble mean — the anomaly reference its sigmoid fit used."""
    return MODEL_SCENARIOS[model]["baseline"]["file"]


def resolve_all_members_file(var_dir: Path, model: str):
    """The model's projection-baseline ``*-all-members.nc`` for one variable, or ``None``.

    Same preference order as ``backend.constants.resolve_projection_baseline_file``
    (ssp245 first, branch-point baseline as the fallback), applied to the member files.
    """
    for filename in REGIONAL_MODEL2PROJECTION_BASELINE_FILES[model]:
        candidate = Path(var_dir) / filename.replace(".nc", "-all-members.nc")
        if candidate.exists():
            return candidate
    return None

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def extract_arctic_ice_data(
    ds: xr.Dataset, 
    var_name: str = 'icefrac', 
    start_year: str = '2050', 
    end_year: str = '2069'
) -> xr.DataArray:
    """
    Extracts Arctic ice fraction data for a specific time period.

    This function isolates the Arctic region using defined bands and selects
    the specified time slice.

    Args:
        ds: Input xarray Dataset.
        var_name: The name of the ice fraction variable (default: 'icefrac').
        start_year: Start year for the time slice. 
        end_year: End year for the time slice.
    
    Returns:
        xr.DataArray: The sliced Arctic data. 
                      (Use .values on this result if you strictly need a numpy array).
    
    Raises:
        KeyError: If var_name is not in the dataset.
    """
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    # 1. Create Arctic bands (Lazy operation)
    arctic_ds = create_arctic_dataset(ds)

    # 2. Slice time and select variable
    # We return the DataArray to preserve coordinates (lat_band, lon_band)
    subset = arctic_ds[var_name].sel(time=slice(start_year, end_year))

    return subset

def matching_common_time_ranges(data_dir: Path, var1: str, var2: str, scenarios: list) -> np.ndarray:
    """
    Loads two variables and returns the common time ranges, one entry per scenario.
    """
    common_time_ranges = []

    for _, filename, start_yr, end_yr in scenarios:
        file_path1 = data_dir / var1 / filename
        file_path2 = data_dir / var2 / filename
        ds1 = open_xarray_datasets(file_path1)
        ds2 = open_xarray_datasets(file_path2)
        common_time_range = xr.align(ds1, ds2, join="inner", copy=False)
        common_time_ranges.append(common_time_range[0].time.values)
    
    return common_time_ranges


def load_temperature_training_data(
    data_dir: Path,
    common_time_ranges: List[np.ndarray],
    scenarios: list,
    reference_file: str,
    members: List[str] | None = None,
) -> np.ndarray:
    """
    Loads temperature data, calculates anomalies, and flattens for training.

    This function processes multiple scenarios, calculates anomalies relative 
    to a fixed control baseline, and reshapes the result into a 3D feature matrix.

    Args:
        data_dir: Path to the root data directory.

    Returns:
        np.ndarray: Training data with shape (lon_band, lat_band, total_samples).
                    'total_samples' is the flattened combination of members and time.

    ``members`` restricts the temperature to the members the sea-ice files carry, so each
    ice sample is paired with its own realization's temperature (UKESM1-1-LL publishes its
    sea ice for one member, r3, but its temperature for three).
    """
    data_path = Path(data_dir) / VAR_TEMP

    # 1. Load the fixed reference baseline (Control Run) for anomaly calculation
    ref_path = data_path / reference_file
    ds_ref_raw = open_xarray_datasets(ref_path)
    ds_ref_arctic = create_arctic_dataset(ds_ref_raw)

    processed_arrays = []

    # 2. Iterate through scenarios
    for i, (_, filename, start_yr, end_yr) in enumerate(scenarios):
        
        file_path = data_path / filename
        
        # Load raw data and subset Arctic
        ds_raw = open_xarray_datasets(file_path)
        ds_raw = ds_raw.sel(time=common_time_ranges[i])
        if members is not None and "member" in ds_raw.dims:
            ds_raw = ds_raw.sel(member=list(members))
        ds_arctic = create_arctic_dataset(ds_raw)

        # Calculate Anomalies
        # Returns Dataset with shape (lon_band, lat_band, member, time)
        ds_anom = calculate_temperature_anomalies(
            ds_arctic, 
            ds_baseline=ds_ref_arctic, 
            start_year=start_yr, 
            end_year=end_yr
        )

        # Extract numpy array and squeeze trivial dimensions if necessary
        data_vals = ds_anom[VAR_TEMP].values.squeeze()
        processed_arrays.append(data_vals)

    # 3. Concatenate along the time axis (axis=-1)
    # Assumes inputs are (lon_band, lat_band, member, time)
    try:
        X = np.concatenate(processed_arrays, axis=-1)
    except ValueError as e:
        raise ValueError(f"Dimension mismatch during concatenation: {e}")

    # 4. Flatten for Training
    # Reshape: Keep (lon_band, lat_band), flatten (member * time)
    # Resulting shape: (lon_band, lat_band, total_samples)
    X = X.reshape(X.shape[0], X.shape[1], -1)

    return X


def load_icefrac_training_data(
    data_dir: Path, common_time_ranges: List[np.ndarray], scenarios: list, reference_file: str
) -> np.ndarray:
    """
    Loads ice fraction data, aggregates spatial bands, and maps to global grid.

    This function processes multiple scenarios, merges the Arctic spatial bands
    into a cohesive map, and places that map onto the full global latitude grid.

    Args:
        data_dir: Path to the root data directory.

    Returns:
        np.ndarray: Global ice data with shape (total_samples, lat, lon).
                    'total_samples' is the flattened combination of members and time.
                    Non-Arctic regions are filled with 0.
    """
    data_path = Path(data_dir) / VAR_ICE

    # Load Reference for Global Grid Structure
    # We need this to know what the full 'lat' and 'lon' dimensions look like
    ref_ds = open_xarray_datasets(data_path / reference_file)
    global_lat = ref_ds.lat
    global_lon = ref_ds.lon

    # Store unflattened DataArrays here
    scenario_arrays = []

    for i, (_, filename, start_yr, end_yr) in enumerate(scenarios):
        # Load raw data
        ds = open_xarray_datasets(data_path / filename)
        ds = ds.sel(time=common_time_ranges[i])

        # Extract Arctic Bands
        # Returns DataArray: (lon_band, lat_band, member, time, lat, lon)
        # Note: We use the refactored helper function from the previous step
        da_arctic = extract_arctic_ice_data(
            ds, 
            var_name=VAR_ICE, 
            start_year=str(start_yr), 
            end_year=str(end_yr)
        )

        # Aggregate Bands (Spatial Reconstruction)
        # Summing over bands collapses the sparse structure back to a real map.
        # Since bands are disjoint (padded with NaNs), 'sum' effectively stitches them.
        # Result: (member, time, lat, lon) - or similar order depending on input
        da_stitched = da_arctic.sum(dim=['lat_band', 'lon_band'], min_count=0)

        # Global Reindexing
        # Automatically places the Arctic data (e.g. 60-90N) into the correct 
        # slots of the global grid (0-90N), filling the rest with 0.
        da_global = da_stitched.reindex(
            lat=global_lat, 
            lon=global_lon, 
            fill_value=0.0
        )

        scenario_arrays.append(da_global)

    # Concatenate Scenarios along TIME
    da_combined = np.concatenate(scenario_arrays, axis=1)

    da_combined_shape = da_combined.shape

    return da_combined.reshape(da_combined_shape[0]*da_combined_shape[1], da_combined_shape[2], da_combined_shape[3])


def reconstruct_seaice_global(
    arctic_data: np.ndarray, 
    global_shape: Tuple[int, int] = (192, 288),
    sum_bands: bool = True
) -> np.ndarray:
    """
    Reconstructs a global sea ice grid from Arctic regional data.

    This function places the Arctic subset onto a global grid of zeros.
    It assumes the Arctic data corresponds to the highest latitudes 
    (top of the array).

    Args:
        arctic_data: Input array. 
            If sum_bands=True (default), expects shape (lon_band, lat_band, time, lat, lon).
            If sum_bands=False, expects shape (time, lat, lon).
        global_shape: Tuple of (n_lat, n_lon) for the target global grid. 
            Defaults to (192, 288).
        sum_bands: Whether to sum over the first two dimensions (bands) before 
            reconstruction. Defaults to True.

    Returns:
        np.ndarray: Global sea ice array with shape (time, global_lat, global_lon).
                    Non-Arctic regions are filled with 0.
    
    Raises:
        ValueError: If the Arctic latitude dimension is larger than the global latitude.
    """
    # 1. Handle Band Aggregation
    # ---------------------------------------------------------
    if sum_bands:
        # Sum over lon_band (0) and lat_band (1) to collapse sparse structure
        # Result shape: (time, arctic_lat, lon)
        ice_data = np.nansum(arctic_data, axis=(0, 1))
    else:
        ice_data = arctic_data

    # 2. Validate Dimensions
    # ---------------------------------------------------------
    # Check that input matches expected logic (Time, Lat, Lon)
    if ice_data.ndim != 3:
        raise ValueError(f"Expected 3D data after summation (time, lat, lon), got {ice_data.ndim}D.")

    n_samples, n_arctic_lat, n_lon = ice_data.shape
    global_lat, global_lon = global_shape

    if n_arctic_lat > global_lat:
        raise ValueError(f"Arctic latitudes ({n_arctic_lat}) exceed global grid size ({global_lat}).")
    
    if n_lon != global_lon:
        raise ValueError(f"Longitude mismatch: Input {n_lon} vs Target {global_lon}")

    # 3. Reconstruct Global Grid
    # ---------------------------------------------------------
    # Create output array (Time, Global_Lat, Global_Lon)
    output_shape = (n_samples, global_lat, global_lon)
    global_grid = np.zeros(output_shape, dtype=ice_data.dtype)

    # 4. Fill Arctic Region
    # ---------------------------------------------------------
    # We populate the "top" (northernmost) latitudes.
    # From index [Global_Lat - Arctic_Lat] to the end.
    start_lat_idx = global_lat - n_arctic_lat
    global_grid[:, start_lat_idx:, :] = ice_data

    return global_grid


def calculate_variability(
    var_name: str,
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    model: str = "all",
    geojsons: bool = True,
):
    """
    Compute and save each model's internal-variability artifacts for one variable.

    Args:
        var_name: Variable to analyze (any processed variable; 'icefrac' uses the hindcast).
        data_dir: Path to processed data directory.
        model_dir: Path to directory containing model parameters (icefrac sigmoids).
        model: One of ``MODELS`` or ``"all"``.
        geojsons: Also write the per-region ``variability_<var>_<model>`` columns (field
            variables only — the regional icefrac band aggregates extent by sum, and a
            region-mean of a per-cell sigma is not on that scale).
    """
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    models = list(MODELS) if model == "all" else [model]
    if any(m not in MODELS for m in models):
        raise ValueError(f"model must be 'all' or one of {list(MODELS)}, got {model!r}")

    for m in models:
        print(f"=== {var_name}: internal variability for {m} ===")
        if var_name == VAR_ICE:
            _calculate_icefrac_variability(data_path, model_path, m)
            _save_icefrac_lag1_autocorr(data_path, m)
            continue
        grid_ds, weights = _calculate_field_variability(var_name, data_path, m)
        if geojsons and grid_ds is not None:
            process_geojson_variability(
                data_path / "geojsons" / "original", grid_ds, var_name, weights,
                column_name=f"variability_{var_name}_{m}",
            )


def _detrended_residuals(da: xr.DataArray) -> xr.DataArray:
    """Residuals of a per-series linear fit along ``time`` over TREND_YEARS.

    ``polyfit`` fits each (member[, lat, lon]) series separately, so this detrends every
    member and every cell on its own.
    """
    window = da.sel(time=slice(*TREND_YEARS))
    coeffs = window.polyfit(dim="time", deg=1)
    trend = xr.polyval(window["time"], coeffs.polyfit_coefficients)
    return window - trend


def field_global_variability(members: xr.DataArray) -> Tuple[float, xr.DataArray]:
    """Sigma of the cos-lat global-mean series: detrend per member, std over members × years."""
    weights = np.cos(np.deg2rad(members.lat))
    global_mean = members.weighted(weights).mean(dim=("lat", "lon"))
    residuals = _detrended_residuals(global_mean)
    return float(residuals.std().item()), weights


def field_grid_lag1_autocorr(members: xr.DataArray, var: str) -> xr.Dataset:
    """Per-cell lag-1 autocorrelation of the detrended residuals, averaged over members.

    Feeds the effective-sample-size correction of the per-cell significance test in
    :mod:`backend.p_values`: a k-year mean of AR(1) noise has variance
    ``sigma^2 * ((1+rho)/(1-rho)) / k``. Clipped to (-0.9, 0.9); cells with no variance get 0.
    """
    residuals = _detrended_residuals(members)
    x0 = residuals.isel(time=slice(None, -1))
    x1 = residuals.isel(time=slice(1, None)).assign_coords(time=x0.time)
    num = (x0 * x1).sum("time")
    den = (residuals**2).sum("time")
    rho = (num / den.where(den > 0)).mean("member").fillna(0.0).clip(-0.9, 0.9)
    ds_out = rho.to_dataset(name=var)
    ds_out[var].attrs["description"] = (
        f"Lag-1 autocorrelation of per-member linearly detrended annual residuals, "
        f"{TREND_YEARS[0]}-{TREND_YEARS[1]}, mean over members"
    )
    return ds_out


def field_grid_variability(members: xr.DataArray, var: str) -> xr.Dataset:
    """Per-cell sigma: detrend each member's cell series, std over members × GRID_STD_YEARS.

    The member and year counts and the degrees of freedom of the estimate are stored as
    attributes (the Welch test in backend.p_values reads ``dof``).
    """
    residuals = _detrended_residuals(members).sel(time=slice(*GRID_STD_YEARS))
    grid_sigma = residuals.std(dim=("time", "member"))
    # Only members with data in the window count (CESM's daily-count files carry one
    # projection member in 2025-2039; its tasmax/tasmin files carry three historical-only
    # and three projection-only members). The NaN-skipping std above already ignores them.
    n_members = _members_with_data(residuals)
    n_years = int(residuals.sizes["time"])
    ds_out = grid_sigma.to_dataset(name=var)
    ds_out[var].attrs = dict(members.attrs)
    ds_out[var].attrs.update(
        description=(
            f"Internal variability: std of per-member linearly detrended residuals "
            f"(trend {TREND_YEARS[0]}-{TREND_YEARS[1]}, std {GRID_STD_YEARS[0]}-{GRID_STD_YEARS[1]})"
        ),
        n_members=n_members,
        n_years=n_years,
        dof=n_members * n_years - 1,
    )
    return ds_out


def _members_with_data(da: xr.DataArray) -> int:
    """Number of members with any finite value in ``da`` (members that sit out a window, e.g.
    historical-only or projection-only members, do not count as samples)."""
    other = [d for d in da.dims if d != "member"]
    return int(da.notnull().any(dim=other).sum())


def field_reference_variability(members: xr.DataArray, var: str) -> xr.Dataset | None:
    """Per-cell scatter of the members over the significance reference period.

    Each member's cell series is detrended with its own straight line WITHIN REFERENCE_YEARS
    (the period still warms), so only year-to-year noise around the model's own path is left;
    the scatter is then pooled over members and years with the trend parameters removed from
    the degrees of freedom: ``s = sqrt(Σ r² / (N·n − 2N))``. ``None`` when the members do not
    cover the period.
    """
    window = members.sel(time=slice(*REFERENCE_YEARS))
    n_years = int(window.sizes.get("time", 0))
    if n_years < REFERENCE_YEARS[1] - REFERENCE_YEARS[0] + 1:
        return None
    # Keep only the members that have data in the window (see _members_with_data).
    other = [d for d in window.dims if d != "member"]
    window = window.sel(member=window.notnull().any(dim=other))
    if window.sizes["member"] == 0:
        return None
    coeffs = window.polyfit(dim="time", deg=1)
    residuals = window - xr.polyval(window["time"], coeffs.polyfit_coefficients)
    n_members = int(residuals.sizes["member"])
    dof = n_members * n_years - 2 * n_members
    scatter = np.sqrt((residuals**2).sum(dim=("time", "member")) / dof)
    ds_out = scatter.to_dataset(name=var)
    ds_out[var].attrs.update(
        description=(
            f"Reference-period internal variability: pooled std of per-member residuals from "
            f"a linear fit within {REFERENCE_YEARS[0]}-{REFERENCE_YEARS[1]}"
        ),
        n_members=n_members,
        n_years=n_years,
        dof=dof,
    )
    return ds_out


def _save_reference_variability(members: xr.DataArray, var: str, var_dir: Path, model: str, source: str) -> None:
    ref_ds = field_reference_variability(members, var)
    if ref_ds is None:
        print(f"\033[93m  {model}: members do not cover {REFERENCE_YEARS}; no reference variability\033[0m")
        return
    ref_ds[var].attrs.update(model=model, source_file=source)
    ref_ds.to_netcdf(var_dir / f"grid_level_reference_variability_{model}.nc")
    print(f"  saved grid_level_reference_variability_{model}.nc ({ref_ds[var].attrs['n_members']} members x "
          f"{ref_ds[var].attrs['n_years']} years, median {float(ref_ds[var].median()):.4g})")


def _calculate_field_variability(var: str, data_dir: Path, model: str):
    """Field-variable handler: sigma from the model's projection-baseline members."""
    var_dir = data_dir / var
    members_path = resolve_all_members_file(var_dir, model)
    if members_path is None:
        print(
            f"\033[93m  no *-all-members.nc projection baseline for {model} in {var_dir} — "
            f"skipping (run scripts/process_monthly_gauss.py first)\033[0m"
        )
        return None, None

    ds = open_xarray_datasets(members_path)
    members = ds[var]
    for dim in ("model", "ssp"):  # not expected on member files; tolerate either way
        if dim in members.dims:
            members = members.squeeze(dim, drop=True)
    years = members.time.values
    print(f"  source: {members_path.name}  members={list(members.member.values)}  "
          f"years={int(years.min())}-{int(years.max())}")

    std_value, weights = field_global_variability(members)
    grid_ds = field_grid_variability(members, var)
    for target in (grid_ds[var],):
        target.attrs["model"] = model
        target.attrs["source_file"] = members_path.name
    print(f"  global-series sigma: {std_value:.4f}")

    np.save(var_dir / f"model_internal_variability_{model}.npy", std_value)
    grid_ds.to_netcdf(var_dir / f"grid_level_model_internal_variability_{model}.nc")
    rho_ds = field_grid_lag1_autocorr(members, var)
    rho_ds[var].attrs.update(model=model, source_file=members_path.name)
    rho_ds.to_netcdf(var_dir / f"grid_level_lag1_autocorr_{model}.nc")
    print(f"  saved model_internal_variability_{model}.npy, grid_level_model_internal_variability_{model}.nc, "
          f"grid_level_lag1_autocorr_{model}.nc (median rho {float(rho_ds[var].median()):.3f})")
    _save_reference_variability(members, var, var_dir, model, members_path.name)

    # Before/after check against the legacy unsuffixed artifacts (CESM only, native grid,
    # computed inside process_monthly_gauss.py). The regrid CESM->CESM grid should be a
    # near-identity, so these should agree closely; report rather than assume.
    legacy_npy = var_dir / "model_internal_variability.npy"
    legacy_nc = var_dir / "grid_level_model_internal_variability.nc"
    if model == "CESM2-WACCM" and legacy_npy.exists() and legacy_nc.exists():
        legacy_std = float(np.load(legacy_npy))
        with xr.open_dataarray(legacy_nc) as legacy_da:
            legacy_grid = legacy_da.load()
        diff = grid_ds[var] - legacy_grid
        rel = float(np.abs(diff).max() / np.abs(legacy_grid).max())
        print(
            f"  vs legacy (native-grid, in-process) CESM artifacts: global sigma "
            f"{legacy_std:.4f} -> {std_value:.4f} ({100 * (std_value / legacy_std - 1):+.2f}%); "
            f"grid sigma max|diff|={float(np.abs(diff).max()):.3e} "
            f"({100 * rel:.3f}% of max sigma)"
        )
    return grid_ds, weights


def _save_icefrac_lag1_autocorr(data_dir: Path, model: str) -> None:
    """Per-cell lag-1 rho for icefrac from the model's projection-baseline members (NaN -> 0 ice)."""
    var_dir = data_dir / VAR_ICE
    members_path = resolve_all_members_file(var_dir, model)
    if members_path is None:
        print(f"\033[93m  no icefrac *-all-members.nc for {model}; skipping lag-1 autocorrelation\033[0m")
        return
    members = open_xarray_datasets(members_path)[VAR_ICE].fillna(0.0)
    rho_ds = field_grid_lag1_autocorr(members, VAR_ICE)
    rho_ds[VAR_ICE].attrs.update(model=model, source_file=members_path.name)
    rho_ds.to_netcdf(var_dir / f"grid_level_lag1_autocorr_{model}.nc")
    print(f"  saved grid_level_lag1_autocorr_{model}.nc (median rho {float(rho_ds[VAR_ICE].median()):.3f})")
    _save_reference_variability(members, VAR_ICE, var_dir, model, members_path.name)


def _calculate_icefrac_variability(data_dir: Path, model_dir: Path, model: str):
    """Icefrac handler: residuals of the model's own sigmoid hindcast on its training scenarios."""

    # Load Data & Parameters
    # ------------------------------------------------------------------
    area_da = get_area(data_dir)  # Returns DataArray (lat, lon)
    params_da = load_icefrac_params(model_dir)  # (model, features, lon_band, lat_band, lat, lon)
    # The hindcast reconstructs this model's ensemble members with this model's own
    # sigmoid; the numpy broadcasting below must not carry a model dim.
    params_da = params_da.sel(model=model)
    scenarios = icefrac_scenarios(model)
    reference_file = icefrac_reference_file(model)
    print(f"  scenarios: {[name for name, *_ in scenarios]}  anomaly reference: {reference_file}")

    # Matching Common Time Ranges
    common_time_ranges = matching_common_time_ranges(data_dir, VAR_TEMP, VAR_ICE, scenarios)

    # Load processed training data. Pair each sea-ice member with that member's temperature:
    # the ice files may carry fewer members than the tas files (UKESM: r3 only).
    ice_members = [
        str(m) for m in open_xarray_datasets(data_dir / VAR_ICE / scenarios[0][1])[VAR_ICE].member.values
    ]
    # X: Temperature anomalies (lon_band, lat_band, samples)
    temp_train = load_temperature_training_data(
        data_dir, common_time_ranges, scenarios, reference_file, members=ice_members
    )
    print(f"Temperature training data shape: {temp_train.shape}")

    # Y: simulated Ice Global (samples, global_lat, global_lon)
    simulated_ice_global = load_icefrac_training_data(data_dir, common_time_ranges, scenarios, reference_file)
    print(f"Simulated ice global shape: {simulated_ice_global.shape}")

    # Prepare Parameters (Arctic Slice)
    # ------------------------------------------------------------------
    # Select Arctic latitudes (60-90) matching training logic
    params_subset = params_da.sel(lat=slice(60, 90))
    L, x0, k, b = get_parameter_values(params_subset)

    # Hindcast Prediction (Manual Broadcasting)
    # ------------------------------------------------------------------
    # We need to broadcast:
    #   Temp (X):   (lon_band, lat_band, samples)
    #   Params:     (lon_band, lat_band, arctic_lat, arctic_lon)
    # Target shape: (lon_band, lat_band, samples, arctic_lat, arctic_lon)

    # Expand X: (lon_band, lat_band, samples, 1, 1)
    X_expanded = temp_train[..., np.newaxis, np.newaxis]
    
    # Expand Params: (lon_band, lat_band, 1, arctic_lat, arctic_lon)
    # Note: We use .values to work with numpy broadcasting
    L_vals  = L.values[..., np.newaxis, :, :]
    x0_vals = x0.values[..., np.newaxis, :, :]
    k_vals  = k.values[..., np.newaxis, :, :]
    b_vals  = b.values[..., np.newaxis, :, :]

    # Calculate Logistic Response (Broadcasting happens here)
    # Result: (lon_band, lat_band, samples, arctic_lat, arctic_lon)
    predicted_bands = logistic(X_expanded, L_vals, x0_vals, k_vals, b_vals)

    # Reconstruct Global Prediction
    # ------------------------------------------------------------------
    # Sum over bands (0, 1) -> Reconstruct to global grid
    # Input shape to reconstruct: (lon_band, lat_band, samples, arctic_lat, arctic_lon)
    # We ask reconstruct to sum the first two dims for us.
    predicted_global = reconstruct_seaice_global(
        predicted_bands, 
        global_shape=simulated_ice_global.shape[1:], # (192, 288) or similar
        sum_bands=True
    )
    print(f"Predicted global shape: {predicted_global.shape}")
    
    # Calculate Variability (Residuals)
    # ------------------------------------------------------------------
    # Residuals = Observation - Prediction
    residuals = simulated_ice_global - predicted_global
    print(f"Residuals shape: {residuals.shape}")

    # Metric A: Gridcell-wise Standard Deviation
    std_gridcell = np.std(residuals, axis=0) # std over 'samples' axis

    # Metric B: Total Area Standard Deviation (Scalar)
    # Weight by cell area before summing
    print(f"Area data shape: {area_da.values.shape}")
    total_ice_simulated = np.sum(simulated_ice_global * area_da.values, axis=(1, 2))
    print(f"Total ice simulated shape: {total_ice_simulated.shape}")
    total_ice_predicted = np.sum(predicted_global * area_da.values, axis=(1, 2))
    print(f"Total ice predicted shape: {total_ice_predicted.shape}")
    global_std = np.std(total_ice_simulated - total_ice_predicted)

    # Save Outputs
    # ------------------------------------------------------------------
    output_dir = data_dir / "icefrac"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save NetCDF (Gridcell Map). std_gridcell is the std of ice-FRACTION residuals
    # (dimensionless); only the scalar below is area-weighted (million km^2).
    xr.DataArray(
        std_gridcell,
        dims=['lat', 'lon'],
        coords={'lat': area_da.lat, 'lon': area_da.lon},
        name='icefrac_std',
        attrs={
            'description': 'Arctic ice fraction residuals standard deviation (sigmoid hindcast)',
            'unit': 'fraction (dimensionless)',
            'model': model,
            # hindcast residual samples; the 4 logistic parameters are fitted per cell
            'n_samples': int(residuals.shape[0]),
            'dof': int(residuals.shape[0]) - 4,
        }
    ).to_netcdf(output_dir / f"grid_level_model_internal_variability_{model}.nc")

    # Save Scalar (Numpy): std of the area-summed extent residual, million km^2
    np.save(output_dir / f"model_internal_variability_{model}.npy", global_std.item())
    print(f"  extent sigma: {float(global_std):.4f} million km^2; saved *_{model} artifacts")


def calculate_regional_average_variability(
    variability_data: xr.Dataset, 
    region_gdf: gpd.GeoDataFrame, 
    weights: np.ndarray, var: str
) -> float:
    """
    Calculate area-weighted average variability for a specific region.

    This function computes the area-weighted mean value of a gridded variability metric
    (e.g., standard deviation or other measure) within a spatial region defined by a GeoDataFrame.
    The weights should typically be grid cell areas or cosine of latitude for equal-area weighting.

    Parameters
    ----------
    variability_data : xarray.Dataset or xarray.DataArray
        Gridded variability data with spatial dimensions (lat, lon), containing the variable of interest
        provided as the 'var' parameter. Can be an xarray.Dataset (if so, must include 'var') or DataArray.
    region_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the region geometry (should be a single-row GeoDataFrame representing the region).
    weights : np.ndarray or xarray.DataArray
        Area weights for each gridcell, with the same lat/lon grid as variability_data. For example,
        cell area or cosine of latitude for equal-area weighting.
    var : str
        Name of the variable within variability_data to compute the mean of.

    Returns
    -------
    float
        The area-weighted average variability value over the region,
        rounded to two decimal places.

    Notes
    -----
    - The region is given by the geometry of region_gdf and is rasterized to create a boolean mask
      over the grid of variability_data.
    - Only grid cells within the region geometry contribute to the mean (others are excluded).
    - The mean is computed as a weighted average over (lat, lon), using the specified weights,
      and missing values are ignored.
    - Assumes that region_gdf, variability_data, and weights all refer to the same lat/lon grid.
    """
    # Create mask for the region
    region_mask = create_mask(variability_data, region_gdf)
    masked_variability = variability_data.where(region_mask == True, np.nan)
    
    # Get area-weighted mean over the region
    regional_avg = masked_variability.weighted(weights).mean(dim=('lat', 'lon'))
    
    # Get value
    avg_value = regional_avg[var].values.item()
    
    return np.round(avg_value, 2) 



def process_geojson_variability(
    geojsons_dir: Path,
    regional_variability: xr.Dataset,
    var: str,
    weights: xr.Dataset,
    column_name: Union[str, None] = None,
) -> None:
    """
    Load in all region geojson files and add average variability columns to each.

    This function loads all geojson files from the selected input directory (see below),
    calculates the area-weighted average variability for each region defined in the geojson files,
    and saves the results as a new column. The variability values are computed using the 
    regional_variability (grid-level variability dataset averaged over one region).

    For each geojson, the output is always written to the directory one level up from geojsons_dir.
    The input directory is geojsons_dir unless the parent directory already contains geojsons
    with matching names and count, in which case input is read from the parent directory.

    Parameters:
    -----------
    geojsons_dir : pathlib.Path
        Path to the regional geojson files.
    regional_variability : xarray.Dataset
        Grid-level variability data with spatial dimensions (lat, lon) and a variable
        named according to the 'var' parameter. This is the detrended standard deviation
        computed from SSP245 over 2015-2099.
    var : str
        Variable name
    weights : np.Array
        Area weights for spatial averaging, typically cosine of latitude for
        equal-area weighting. Must have 'lat' dimension matching regional_variability.
    """
    # Determine input and output directories
    child_dir = geojsons_dir
    parent_dir = geojsons_dir.parent

    child_geojson_files = list(child_dir.glob("*.geojson"))
    child_geojson_filenames = sorted([f.name for f in child_geojson_files])

    parent_geojson_files = list(parent_dir.glob("*.geojson"))
    parent_geojson_filenames = sorted([f.name for f in parent_geojson_files])

    # If parent dir matches in count and filenames, input is parent. Otherwise, input is child.
    if (
        len(parent_geojson_filenames) == len(child_geojson_filenames)
        and set(parent_geojson_filenames) == set(child_geojson_filenames)
        and len(parent_geojson_filenames) > 0
    ):
        geojson_files = parent_geojson_files
    else:
        geojson_files = child_geojson_files

    output_dir = parent_dir

    for geojson_file in tqdm(geojson_files, desc="Processing geojson files"):
        print(f"\nProcessing {geojson_file.name}...")

        gdf = gpd.read_file(geojson_file)
        avg_col_name = column_name or f"variability_{var}"

        # Calculate variability over each region
        avg_values = []
        for idx, row in gdf.iterrows():
            region_gdf = gdf[gdf.index == idx]

            avg_value = calculate_regional_average_variability(
                regional_variability, region_gdf, weights, var
            )
            avg_values.append(avg_value)

        gdf[avg_col_name] = avg_values

        print(
            f"    Added {avg_col_name} to {geojson_file.name}; Range: "
            f"{min([v for v in avg_values if not np.isnan(v)]):.4f} to "
            f"{max([v for v in avg_values if not np.isnan(v)]):.4f}"
        )

        # Output always goes to parent directory
        out_path = output_dir / geojson_file.name
        gdf.to_file(out_path, driver='GeoJSON')


if __name__ == "__main__":
    import fire  # CLI-only; kept out of module-level imports (see top of file)
    fire.Fire(calculate_variability)
