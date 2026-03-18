"""
ICEFRAC calculation module for SAI simulator.

This module handles loading fitted sigmoid parameters and converting
arctic temperature anomalies to ICEFRAC (Arcticsea ice fraction) predictions.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, Union
from .sigmoid import logistic

@lru_cache(maxsize=1)
def get_area(data_dir: Path) -> xr.DataArray:
    """
    Loads grid cell area data from the area reference file.

    It selects the last time step from the dataset, converts units to 
    million square kilometers, and caches the result to minimize I/O.

    Args:
        data_dir: Directory path containing the 'area.nc' file.

    Returns:
        xr.DataArray: 2D DataArray containing grid cell areas in million km^2.
    
    Raises:
        FileNotFoundError: If 'area.nc' does not exist.
        KeyError: If the 'AREA' variable is missing from the file.
    """
    # Define unit conversion constants at the module level for clarity
    # Assumes input is m^2. 1e-6 * 1e-6 = 1e-12
    SQ_METERS_TO_MILLION_SQ_KM = 1e-12

    file_path = data_dir / "area.nc"
    
    with xr.open_dataset(file_path) as ds:
        if "AREA" not in ds:
            raise KeyError(f"Variable 'AREA' not found in {file_path}")

        # Use isel for positional indexing (last time step)
        # load() is called immediately to detach data from the file handler
        area = ds["AREA"].isel(time=-1).drop_vars("time") * SQ_METERS_TO_MILLION_SQ_KM
        return area.load()


def create_arctic_dataset(
    ds: xr.Dataset, 
    min_lat: float = 60.0, 
    max_lat: float = 90.0, 
    step_size: float = 5.0
) -> xr.Dataset:
    """
    bins input data into defined Arctic latitude and longitude bands.

    Note: This creates a sparse dataset (ragged array). Regions with fewer grid 
    points than the largest region will be padded with NaNs to maintain a 
    consistent rectangular shape.

    Args:
        ds: Input xarray dataset containing 'lat' and 'lon' coordinates.
        min_lat: Minimum latitude bound.
        max_lat: Maximum latitude bound.
        step_size: Size of latitude bands in degrees.

    Returns:
        xr.Dataset: Dataset with new dimensions 'lat_band' and 'lon_band'.
    """
    # Define regions as a constant (Name, lon_start, lon_end)
    ARCTIC_REGIONS = [
        ("North Atlantic", 330, 45),   # Handles 0-crossing
        ("Eurasia", 45, 165),
        ("Pacific", 165, 220),
        ("Greenland/Canada", 220, 330),
    ]

    # 1. Validation & Preprocessing
    if 'lat' not in ds.coords or 'lon' not in ds.coords:
        raise ValueError("Dataset must contain 'lat' and 'lon' coordinates.")

    # Sort lat/lon to ensure slicing works regardless of input order (Ascending/Descending)
    ds_sorted = ds.sortby(['lat', 'lon'])
    
    # 2. Define Latitude Bands
    # Create tuples of (start, end)
    lat_ranges = list(range(int(min_lat), int(max_lat), int(step_size)))
    lat_slices = []

    for start in lat_ranges:
        end = start + step_size
        # Slice lazily
        # We subtract a tiny epsilon from 'end' to avoid overlapping edges 
        # (e.g. 65.0 belonging to both 60-65 and 65-70)
        subset = ds_sorted.sel(lat=slice(start, end - 1e-6))
        
        # We assign a label but do NOT drop the original lat coord yet, 
        # as we need it for the data to make sense.
        subset = subset.assign_coords(lat_band=f"{int(start)}-{int(end)}")
        lat_slices.append(subset)

    # 3. Concatenate Latitude Bands
    # This creates the 'lat_band' dimension. 
    # Warning: This is where NaN padding occurs for non-aligned grids.
    ds_lat_binned = xr.concat(lat_slices, dim="lat_band", join="outer")

    # 4. Process Longitude Bands (Regions)
    lon_slices = []
    
    for name, start, end in ARCTIC_REGIONS:
        if start > end:
            # Handle crossing the prime meridian (e.g., 330 to 45)
            # Select [330, 360] OR [0, 45]
            mask = (ds_lat_binned.lon >= start) | (ds_lat_binned.lon < end)
            subset = ds_lat_binned.where(mask, drop=True)
        else:
            # Standard range
            subset = ds_lat_binned.sel(lon=slice(start, end - 1e-6))
            
        subset = subset.assign_coords(lon_band=name)
        lon_slices.append(subset)

    # 5. Final Concatenation
    result = xr.concat(lon_slices, dim="lon_band", join="outer")
    
    return result
  

def calculate_preindustrial_temp(
    ds: xr.Dataset, 
    lat_band: str, 
    lon_band: str, 
    var_name: str = "tas",
    start_year: int = 1850,
    end_year: int = 1900
) -> float:
    """
    Calculates the area-weighted average temperature for a specific spatiotemporal baseline.

    This function isolates a specific region (defined by lat/lon bands) and a 
    specific time period (defaulting to the 1850-1900 pre-industrial baseline),
    then computes the spatiotemporal mean.

    Args:
        ds: Input dataset (must contain defined 'lat_band' and 'lon_band' coordinates).
        lat_band: The latitude band label to select (e.g., '60-70').
        lon_band: The longitude band label to select (e.g., 'North Atlantic').
        var_name: The name of the data variable to average. Defaults to "tas".
        start_year: Start year of the baseline period (inclusive). Defaults to "1850".
        end_year: End year of the baseline period (inclusive). Defaults to "1900".

    Returns:
        float: The single scalar average value for the region and period. 
               Returns np.nan if data is missing or empty.

    Raises:
        KeyError: If the specified bands or variable name are not found in the dataset.
    """
    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    # 1. Selection & Slicing
    # Optimization: Slice time FIRST. This reduces the data volume significantly 
    # before performing the computationally expensive spatial weighting.
    try:
        subset = ds[var_name].sel(
            lat_band=lat_band, 
            lon_band=lon_band, 
            time=slice(start_year, end_year)
        )
    except KeyError as e:
        raise KeyError(f"Band selection failed. Check if '{lat_band}' or '{lon_band}' exist.") from e

    # Check for empty data early
    if subset.size == 0:
        return np.nan

    # 2. Area Weighting
    # We weight by cosine of latitude to account for grid cell area distortion near poles.
    # Note: 'lat' must be a coordinate available in the subset.
    weights = np.cos(np.deg2rad(subset.lat))
    
    # 3. Aggregation
    # Compute spatial mean (weighted) -> then temporal mean
    # standardizing on skipping NaNs to handle the sparse arctic dataset structure
    weighted_mean = subset.weighted(weights).mean(dim=['lat', 'lon'], skipna=True)
    
    # Compute temporal mean over the remaining time series
    baseline_value = weighted_mean.mean(dim='time', skipna=True)

    return baseline_value.item()


def calculate_temperature_anomalies(
    ds: xr.Dataset, 
    ds_baseline: Optional[xr.Dataset] = None, 
    start_year: Union[str, int] = 1900, 
    end_year: Union[str, int] = 2098,
    baseline_start: str = "1850",
    baseline_end: str = "1900"
) -> xr.Dataset:
    """
    Calculates temperature anomalies for Arctic bands relative to a pre-industrial baseline.

    This function performs area-weighted averaging across 'lat' and 'lon' dimensions 
    while preserving 'lat_band' and 'lon_band' structures. It operates in a fully 
    vectorized manner, avoiding loops for performance.

    Args:
        ds: Input dataset containing the target variable and band coordinates 
            (must contain 'lat_band' and 'lon_band').
        ds_baseline: Reference dataset for calculating the climatological baseline. 
            If None, the function returns absolute weighted means instead of anomalies. 
            Defaults to None.
        start_year: Start year for the output anomaly time series.
        end_year: End year for the output anomaly time series.
        baseline_start: Start year for the baseline period (only used if ds_baseline is provided).
        baseline_end: End year for the baseline period (only used if ds_baseline is provided).

    Returns:
        xr.Dataset: Dataset containing anomalies (or absolute means) with dimensions 
                    ordered (lon_band, lat_band, ...).
    
    Raises:
        KeyError: If the specified variable name is not found in the datasets.
    """
    
    # 1. Calculate Spatial Means for Target Dataset (Vectorized)
    # ---------------------------------------------------------
    # Create weights based on latitude to account for grid cell area
    weights = np.cos(np.deg2rad(ds.lat))
    
    # Slice time first for efficiency, then apply weights and mean
    # Result is a DataArray with dimensions preserving input order (e.g., time, member...)
    ds_subset = ds.sel(time=slice(str(start_year), str(end_year)))
    da_weighted_mean = ds_subset.weighted(weights).mean(dim=['lat', 'lon'])

    # 2. Conditional Baseline Subtraction
    # ---------------------------------------------------------
    if ds_baseline is None:
        result_da = da_weighted_mean
    else:
        # Calculate baseline mean (Vectorized)
        baseline_subset = ds_baseline.sel(time=slice(baseline_start, baseline_end))
        baseline_weights = np.cos(np.deg2rad(ds_baseline.lat))
        
        # Shape: (lon_band, lat_band) - averaged over time
        baseline_mean = baseline_subset.weighted(baseline_weights).mean(dim=['lat', 'lon']).mean(dim='time')
        
        # Subtract baseline from target means
        # Xarray automatically broadcasts the shapes correctly
        result_da = da_weighted_mean - baseline_mean
    
    return result_da


@lru_cache(maxsize=1)
def load_icefrac_params(model_dir: Union[str, Path]) -> xr.DataArray:
    """
    Loads and caches logistic interpolation parameters from the model directory.

    It assumes the parameters are stored in 'icefrac_arctic/interpolator.nc' 
    and strictly validates that the DataArray is named 'logistic_params'.

    Args:
        model_dir: Base directory containing the model subdirectories.

    Returns:
        xr.DataArray: The loaded parameters array.

    Raises:
        FileNotFoundError: If the interpolator file does not exist.
        ValueError: If the DataArray in the file is not named 'logistic_params'.
    """
    # Normalize input to Path object
    path = Path(model_dir) / "icefrac_arctic" / "interpolator.nc"

    if not path.exists():
        raise FileNotFoundError(f"Icefrac interpolator not found at: {path}")

    # Use a context manager to ensure file handles are released immediately
    with xr.open_dataarray(path) as da:
        # Strict validation of the internal variable name
        if da.name != 'logistic_params':
            raise ValueError(
                f"Invalid parameter file. Expected DataArray named 'logistic_params', "
                f"but found '{da.name}' in {path}."
            )
            
        return da.load()


def get_parameter_values(
    params: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Extracts logistic parameters (L, x0, k, b) from the parameter array.

    This function uses label-based selection for safety. It assumes the input 
    contains a coordinate named 'features' with values 'L', 'x0', 'k', and 'b'.

    Args:
        params: DataArray containing logistic parameters.

    Returns:
        Tuple[DataArray, ...]: The separated parameter arrays (L, x0, k, b).

    Raises:
        KeyError: If the 'features' dimension or specific parameter names are missing.
    """
    if 'features' not in params.coords:
        raise KeyError("Input parameters must have a 'features' coordinate.")

    try:
        # Select by label (safer than index 0, 1, 2...)
        # We use .sel() to preserve xarray metadata (coordinates), which is 
        # crucial for alignment in later broadcasting steps.
        L  = params.sel(features='L')
        x0 = params.sel(features='x0')
        k  = params.sel(features='k')
        b  = params.sel(features='b')
    except KeyError as e:
        raise KeyError(f"Missing required parameter in 'features' coordinate: {e}")

    # If you strictly need numpy arrays (stripping metadata), you can call .values here.
    # However, keeping them as DataArrays is usually better for subsequent xarray math.
    return L, x0, k, b


def predict_icefrac(
    temp_ds: xr.Dataset, 
    model_dir: Union[str, Path], 
    start_lat: float = 60.0, 
    end_lat: float = 90.0
) -> xr.DataArray:
    """
    Emulates ICEFRAC response to temperature anomalies using a logistic model.

    This function automatically detects if the input time coordinate represents 
    datetimes or integer years and handles both cases safely.

    Args:
        temp_ds: Input temperature dataset (Global).
        model_dir: Directory containing the 'icefrac_arctic/interpolator.nc' file.
        start_lat: Southernmost latitude of the Arctic region. Defaults to 60.0.
        end_lat: Northernmost latitude of the Arctic region. Defaults to 90.0.

    Returns:
        xr.DataArray: Global ICEFRAC predictions (time, lat, lon). 
                      Non-arctic regions are filled with 0.
    """
    # 1. Robust Year Extraction
    # ------------------------------------------------------------------
    # Check if we have datetime objects (which need .dt accessor) 
    # or simple numbers (which don't).
    try:
        # Try accessing .dt.year (works for datetime64 and cftime)
        start_year = int(temp_ds['time'].dt.year.min())
        end_year = int(temp_ds['time'].dt.year.max())
    except AttributeError:
        # Fallback: Assume time is already numeric (e.g. integer years 2020, 2021)
        start_year = int(temp_ds['time'].min())
        end_year = int(temp_ds['time'].max())

    # 2. Load Parameters
    # ------------------------------------------------------------------
    # params shape: (features, lon_band, lat_band, lat, lon)
    params_da = load_icefrac_params(model_dir)
    params_subset = params_da.sel(lat=slice(start_lat, end_lat))
    L, x0, k, b = get_parameter_values(params_subset)

    # 3. Calculate Anomalies
    # ------------------------------------------------------------------
    arctic_ds = create_arctic_dataset(temp_ds)
    
    anomalies_da = calculate_temperature_anomalies(
        arctic_ds, 
        ds_baseline=None,
        start_year=start_year, 
        end_year=end_year
    )

    # 4. Apply Logistic Model (Vectorized Broadcasting)
    # ------------------------------------------------------------------
    # Xarray broadcasts dims automatically: (time, bands) vs (lat, lon, bands)
    prediction_bands = logistic(anomalies_da, L, x0, k, b)

    # 5. Aggregation
    # ------------------------------------------------------------------
    # Sum over bands (handling NaNs as 0) to flatten the sparse arctic structure
    prediction_arctic = prediction_bands.sum(dim=['lat_band', 'lon_band'], min_count=0)

    # 6. Global Reconstruction
    # ------------------------------------------------------------------
    # Reindex back to global grid, filling non-arctic areas with 0
    prediction_global = prediction_arctic.reindex(
        lat=temp_ds.lat, 
        lon=temp_ds.lon, 
        fill_value=0.0
    )

    return prediction_global.transpose('time', 'lat', 'lon')
