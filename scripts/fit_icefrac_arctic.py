import fire
import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.sigmoid import LogisticFitter
from backend.icefrac import calculate_temperature_anomalies, create_arctic_dataset
from backend.utils import open_xarray_datasets, save_processed_data


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Define experiment scenarios to ensure consistent processing and avoid code duplication.
# Keys represent internal scenario names, values contain file names and time ranges.
SCENARIOS = {
    "baseline": {"file": "output_gauss-baseline.nc", "start": 1900, "end": 2098},
    "sai_0_5":  {"file": "output_gauss-0.5.nc",      "start": 2050, "end": 2069},
    "sai_1_0":  {"file": "output_gauss-1.0.nc",      "start": 2050, "end": 2069},
    "sai_1_5":  {"file": "output_gauss-1.5.nc",      "start": 2050, "end": 2069},
}

VAR_TEMP = "tas"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def reshape_icefrac_data(
    dataset: xr.Dataset, 
    var_name: str = 'icefrac', 
    start_year: str = '2050', 
    end_year: str = '2069'
) -> np.ndarray:
    """
    Extracts a time slice of ice fraction data and flattens the spatial dimensions.
    
    Args:
        dataset: xarray Dataset (expected to be pre-processed/arctic-adjusted).
        var_name: The name of the data variable to access (default: 'icefrac').
        start_year: Start year for time slice. 
        end_year: End year for time slice.
    
    Returns:
        np.ndarray: Array with shape (time, spatial_points).
        
    Raises:
        KeyError: If var_name is not in the dataset.
        ValueError: If the time slice returns empty data.
    """
    # 1. Validation: Ensure variable exists
    if var_name not in dataset:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    # 2. Slicing: Perform slicing lazily on the xarray object
    # Note: We assume the input dataset is already processed (e.g. create_arctic_dataset applied beforehand)
    # to maintain separation of concerns.
    subset = dataset[var_name].sel(time=slice(start_year, end_year))
    # Remove 2015 from the time selection if it's present
    if 'time' in subset.dims:
        subset = subset.sel(time=subset.time[subset.time != 2015])
    
    # print(f"Subset shape: {subset.shape}")
    # print(f"Subset time range: {subset.time.values}")

    if subset.size == 0:
        raise ValueError(f"No data found for time range {start_year}-{end_year}")

    # 3. Reshaping: Use xarray stack to safely combine lat/lon (or last two dims)
    # This automatically handles dimension ordering better than raw reshape
    # Assuming the last two dimensions are the spatial ones we want to flatten
    spatial_dims = subset.dims[-2:] 
    stacked = subset.stack(spatial=spatial_dims)
    
    # 4. Conversion: Transpose to ensure (time, spatial) order, then load to numpy
    # '...' handles any leading dimensions (like ensemble members) automatically
    return stacked.transpose(..., "spatial").values.squeeze()


def filter_icefrac_data(
    data: np.ndarray, 
    min_valid_obs: int = 10, 
    low_value_threshold: float = 0.01, 
    intensity_threshold: float = 0.15
):
    """
    Filters ice fraction data based on observational quality and intensity thresholds.
    
    This function performs three filtering steps:
    1. Hard Threshold: Values <= `low_value_threshold` are set to NaN.
    2. Insufficient Data: Grid cells with fewer than `min_valid_obs` non-NaN values 
       along the time dimension are entirely masked (set to NaN).
    3. Low Intensity: Grid cells where all remaining valid values are below 
       `intensity_threshold` are entirely masked (set to NaN).

    Note: This is a vectorized implementation. It avoids loops for performance and 
    fixes a logic issue in older versions where "Insufficient Data" cells were 
    incorrectly double-counted as "Low Intensity".

    Args:
        data (np.ndarray): Input array with shape (lon, lat, time, grid_cells).
        min_valid_obs (int, optional): Minimum required non-NaN observations per time series. 
            Defaults to 10.
        low_value_threshold (float, optional): Values less than or equal to this are 
            considered noise and set to NaN. Defaults to 0.01.
        intensity_threshold (float, optional): If the maximum value in a time series 
            is below this, the series is considered low intensity. Defaults to 0.15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - filtered_data: The processed data array (same shape as input).
            - count_insufficient: Array (lon, lat) counting cells dropped due to insufficient data.
            - count_low_intensity: Array (lon, lat) counting cells dropped due to low intensity.
    """
    # Work on a copy to ensure the input array is not mutated (no side effects)
    Y = data.copy()

    # ---------------------------------------------------------
    # Step 1: Filter noise (Vectorized)
    # ---------------------------------------------------------
    # Set all individual values below threshold to NaN instantly
    Y[Y <= low_value_threshold] = np.nan

    # ---------------------------------------------------------
    # Step 2: Analyze Time Series (Axis 2)
    # ---------------------------------------------------------
    # Count how many non-NaN values exist for every grid cell along the time axis
    valid_counts = np.count_nonzero(~np.isnan(Y), axis=2)
    
    # Calculate the max value per time series to check against intensity threshold.
    # We suppress RuntimeWarnings because some cells are already all-NaN, 
    # and np.nanmax raises a warning for empty slices (which is expected here).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_vals = np.nanmax(Y, axis=2)

    # ---------------------------------------------------------
    # Step 3: Create Boolean Masks (Shape: lon, lat, grid_cells)
    # ---------------------------------------------------------
    
    # Mask A: "Insufficient Data"
    # Cells that don't have enough data points.
    mask_too_few = valid_counts < min_valid_obs
    
    # Mask B: "Low Intensity"
    # Cells where the max value never exceeds the intensity threshold.
    # CRITICAL LOGIC: We use `& (~mask_too_few)` to ensure mutual exclusivity.
    # If a cell has too few obs, it is ONLY counted in mask A, not mask B.
    mask_low_intensity = (max_vals < intensity_threshold) & (~mask_too_few)

    # ---------------------------------------------------------
    # Step 4: Apply Masks and Aggregate Counts
    # ---------------------------------------------------------
    
    # Expand masks to match the 4D shape (lon, lat, time, grid_cells)
    # This broadcasts the 3D mask across the time dimension (axis 2)
    # Shape becomes: (lon, lat, 1, grid_cells) -> broadcasted to (lon, lat, time, grid_cells)
    
    # Apply Mask A
    mask_A_broadcast = np.expand_dims(mask_too_few, axis=2)
    Y = np.where(mask_A_broadcast, np.nan, Y)

    # Apply Mask B
    mask_B_broadcast = np.expand_dims(mask_low_intensity, axis=2)
    Y = np.where(mask_B_broadcast, np.nan, Y)

    # Sum the masks over the grid_cells dimension (axis 3 implies the last axis of the mask)
    # The masks are shape (lon, lat, grid_cells), so we sum over the last axis.
    count_insufficient_data = np.sum(mask_too_few, axis=-1)
    count_low_intensity = np.sum(mask_low_intensity, axis=-1)

    return Y, count_insufficient_data, count_low_intensity


def _load_and_process_temp(
    data_dir: Path, 
    filename: str, 
    start_year: int, 
    end_year: int, 
    baseline_ref: xr.Dataset
) -> np.ndarray:
    """
    Loads temperature data, subsets the Arctic region, and calculates anomalies.

    Args:
        data_dir: Base directory containing data.
        filename: Name of the NetCDF file to load.
        start_year: Start year for time slicing.
        end_year: End year for time slicing.
        baseline_ref: The reference dataset used to calculate anomalies.

    Returns:
        np.ndarray: Flattened temperature anomalies for the specified time range.
    """
    # Load raw dataset
    ds = open_xarray_datasets(data_dir / VAR_TEMP / filename)
    
    # Subset to Arctic region
    arctic_ds = create_arctic_dataset(ds)
    
    # Calculate anomalies relative to the provided baseline reference
    anomalies = calculate_temperature_anomalies(
        arctic_ds[VAR_TEMP], 
        baseline_ref, 
        start_year=start_year, 
        end_year=end_year
    )

    # print(f"Anomalies shape: {anomalies[VAR_TEMP].shape}")
    # print(f"Anomalies time range: {anomalies[VAR_TEMP].time.values}")
    return anomalies[VAR_TEMP].values.squeeze()


def _load_and_process_ice(
    data_dir: Path, 
    var_name: str, 
    filename: str, 
    start_year: int, 
    end_year: int
) -> np.ndarray:
    """
    Loads ice fraction data, subsets the Arctic region, and reshapes for training.

    Args:
        data_dir: Base directory containing data.
        var_name: Variable name (e.g., 'icefrac').
        filename: Name of the NetCDF file to load.
        start_year: Start year for time slicing.
        end_year: End year for time slicing.

    Returns:
        np.ndarray: Reshaped ice fraction data suitable for model fitting.
    """
    ds = open_xarray_datasets(data_dir / var_name / filename)
    
    # Subset to Arctic region and reshape spatial dims
    reshaped_data = reshape_icefrac_data(
        create_arctic_dataset(ds), 
        var_name=var_name, 
        start_year=str(start_year), 
        end_year=str(end_year)
    )
    
    return reshaped_data


def _save_params_to_netcdf(
    params: np.ndarray, 
    template_ds: xr.Dataset, 
    arctic_ref: xr.Dataset, 
    output_path: Path,
) -> None:
    """
    Expands fitted parameters back to the full global grid and saves to NetCDF.

    This handles the complexity of padding the non-Arctic regions with NaNs so the 
    output matches the dimensions of the original global input files.

    Args:
        params: The fitted logistic parameters (features, lon_band, lat_band, spatial_flat).
        template_ds: A global dataset used to extract full lat/lon coordinates.
        arctic_ref: The arctic-subset dataset used to extract band coordinates.
        output_path: Path where the NetCDF file will be saved.
    """
    # 1. Reshape params from flat spatial dim back to (lat, lon) for the arctic region
    # Expected shape: (features, lon_band, lat_band, arctic_lat, lon)
    params_reshaped = params.reshape(params.shape[:-1] + arctic_ref.tas.shape[-2:])
    
    # 2. Create a container for the full global grid filled with NaNs
    # Structure: (features, lon_band, lat_band, global_lat, global_lon)
    expanded_shape = list(params_reshaped.shape[:-2] + template_ds.tas.shape[-2:])
    expanded_params = np.full(expanded_shape, np.nan)
    
    # 3. Fill the Arctic portion of the grid
    # We assume the Arctic data corresponds to the highest latitudes (end of the array)
    lat_cutoff = params_reshaped.shape[-2]
    expanded_params[..., -lat_cutoff:, :] = params_reshaped

    # 4. Construct DataArray
    params_xr = xr.DataArray(
        expanded_params, 
        dims=['features', 'lon_band', 'lat_band', 'lat', 'lon'],
        coords={
            'features': ['L', 'x0', 'k', 'b'],
            'lon_band': arctic_ref.lon_band,
            'lat_band': arctic_ref.lat_band,
            'lat': template_ds.lat,
            'lon': template_ds.lon,
        },
        name='logistic_params'
    )

    save_processed_data(params_xr, output_path, overwrite=True)


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def fit_arctic_logistic(
    var: str, 
    data_dir: str, 
    output_dir: str, 
    ignore_existing: bool = False
) -> None:
    """
    Orchestrates the loading, transforming, and fitting of logistic models 
    relating Arctic temperature anomalies to ice fraction.

    The workflow follows these steps:
    1. Checks if output exists (skips if so).
    2. Prepares a baseline reference for temperature anomalies.
    3. Iterates through configured scenarios (Baseline, SAI 0.5, 1.0, 1.5) to 
       load and process both Temperature (X) and Ice Fraction (Y) data.
    4. Concatenates all scenarios into a single training set.
    5. Filters invalid or low-quality ice data.
    6. Fits logistic models (in parallel).
    7. Saves the resulting parameters to a NetCDF file.

    Args:
        var: The variable name to fit (e.g., 'icefrac').
        data_dir: Path to the directory containing processed input data.
        output_dir: Path to the directory where model outputs will be saved.
        ignore_existing: If True, overwrites existing output files. Defaults to False.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir) / f"{var}_arctic"
    out_path.mkdir(exist_ok=True, parents=True)
    
    interpolator_path = out_path / "interpolator.nc"
    if interpolator_path.exists() and not ignore_existing:
        return

    # --- 1. Prepare Baseline Reference ---
    # We load the raw baseline temperature first to serve as the climatology reference
    # for anomaly calculations across all scenarios.
    baseline_raw = open_xarray_datasets(data_path / VAR_TEMP / SCENARIOS["baseline"]["file"])
    baseline_arctic_ref = create_arctic_dataset(baseline_raw)

    # --- 2. Load & Process Data Scenarios ---
    X_arrays = []
    Y_arrays = []
    
    for config in SCENARIOS.values():
        # Process Temperature (X)
        x_data = _load_and_process_temp(
            data_path, 
            config["file"], 
            config["start"], 
            config["end"], 
            baseline_arctic_ref
        )
        X_arrays.append(x_data)
        
        # Process Ice Fraction (Y)
        y_data = _load_and_process_ice(
            data_path, 
            var, 
            config["file"], 
            config["start"], 
            config["end"]
        )
        Y_arrays.append(y_data)

    # --- 3. Merge & Filter ---
    # Concatenate all scenarios along the time axis (axis 2)
    X = np.concatenate(X_arrays, axis=2)
    Y = np.concatenate(Y_arrays, axis=2)
    
    # Filter Y data (removes noise and insufficient data points)
    # Note: X is not filtered here; the fitter likely handles alignment or masking
    Y, _, _ = filter_icefrac_data(Y)
    # print(f"Y shape: {Y.shape}")
    # print(f"X shape: {X.shape}")

    # --- 4. Model Fitting ---
    fitter = LogisticFitter(X, Y)
    
    # Fit models using all available cores (n_jobs=-1)
    params, _, _, _ = fitter.fit_all(n_jobs=-1)

    # --- 5. Save Output ---
    _save_params_to_netcdf(params, baseline_raw, baseline_arctic_ref, interpolator_path)


if __name__ == "__main__":
    fire.Fire(fit_arctic_logistic)