import fire
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
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

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Define scenarios constant to avoid magic strings in the function body
# (Name, Filename, Start Year, End Year)
SCENARIOS = [
    ("baseline", "output_gauss-baseline-all-members.nc", 1900, 2098),
    ("sai_1_5",  "output_gauss-1.5-all-members.nc",      2050, 2069),
    ("sai_1_0",  "output_gauss-1.0-all-members.nc",      2050, 2069),
    ("sai_0_5",  "output_gauss-0.5-all-members.nc",      2050, 2069),
]

REFERENCE_FILE = "output_gauss-baseline.nc"
VAR_TEMP = "tas"
VAR_ICE = "icefrac"

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

def matching_common_time_ranges(data_dir: Path, var1: str, var2: str) -> np.ndarray:
    """
    Loads two variables and returns the common time ranges.
    """
    common_time_ranges = []

    for _, filename, start_yr, end_yr in SCENARIOS:
        file_path1 = data_dir / var1 / filename
        file_path2 = data_dir / var2 / filename
        ds1 = open_xarray_datasets(file_path1)
        ds2 = open_xarray_datasets(file_path2)
        common_time_range = xr.align(ds1, ds2, join="inner", copy=False)
        common_time_ranges.append(common_time_range[0].time.values)
    
    return common_time_ranges


def load_temperature_training_data(data_dir: Path, common_time_ranges: List[np.ndarray]) -> np.ndarray:
    """
    Loads temperature data, calculates anomalies, and flattens for training.

    This function processes multiple scenarios, calculates anomalies relative 
    to a fixed control baseline, and reshapes the result into a 3D feature matrix.

    Args:
        data_dir: Path to the root data directory.

    Returns:
        np.ndarray: Training data with shape (lon_band, lat_band, total_samples).
                    'total_samples' is the flattened combination of members and time.
    """
    data_path = Path(data_dir) / VAR_TEMP

    # 1. Load the fixed reference baseline (Control Run) for anomaly calculation
    ref_path = data_path / REFERENCE_FILE
    ds_ref_raw = open_xarray_datasets(ref_path)
    ds_ref_arctic = create_arctic_dataset(ds_ref_raw)

    processed_arrays = []

    # 2. Iterate through scenarios
    for i, (_, filename, start_yr, end_yr) in enumerate(SCENARIOS):
        
        file_path = data_path / filename
        
        # Load raw data and subset Arctic
        ds_raw = open_xarray_datasets(file_path)
        ds_raw = ds_raw.sel(time=common_time_ranges[i])
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


def load_icefrac_training_data(data_dir: Path, common_time_ranges: List[np.ndarray]) -> np.ndarray:
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
    ref_ds = open_xarray_datasets(data_path / REFERENCE_FILE)
    global_lat = ref_ds.lat
    global_lon = ref_ds.lon

    # Store unflattened DataArrays here
    scenario_arrays = []

    for i, (_, filename, start_yr, end_yr) in enumerate(SCENARIOS):
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


def calculate_variability(var_name: str, data_dir: Union[str, Path], model_dir: Union[str, Path]):
    """
    Calculates natural variability (residuals) of the model against training data.

    Currently supports: 'icefrac'.

    Args:
        var_name: Variable to analyze (e.g., 'icefrac').
        data_dir: Path to processed data directory.
        model_dir: Path to directory containing model parameters.
    """
    data_path = Path(data_dir)
    model_path = Path(model_dir)

    if var_name == "icefrac":
        _calculate_icefrac_variability(data_path, model_path)
    else:
        raise NotImplementedError(f"Variability calculation for '{var_name}' is not implemented.")


def _calculate_icefrac_variability(data_dir: Path, model_dir: Path):
    """Internal handler for Ice Fraction variability."""
    
    # Load Data & Parameters
    # ------------------------------------------------------------------
    area_da = get_area(data_dir)  # Returns DataArray (lat, lon)
    params_da = load_icefrac_params(model_dir) # Returns DataArray (features, lon_band, lat_band, lat, lon)
    
    # Matching Common Time Ranges
    common_time_ranges = matching_common_time_ranges(data_dir, VAR_TEMP, VAR_ICE)

    # Load processed training data
    # X: Temperature anomalies (lon_band, lat_band, samples)
    temp_train = load_temperature_training_data(data_dir, common_time_ranges)
    print(f"Temperature training data shape: {temp_train.shape}")
    
    # Y: simulated Ice Global (samples, global_lat, global_lon)
    simulated_ice_global = load_icefrac_training_data(data_dir, common_time_ranges)
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

    # Save NetCDF (Gridcell Map)
    xr.DataArray(
        std_gridcell,
        dims=['lat', 'lon'],
        coords={'lat': area_da.lat, 'lon': area_da.lon},
        name='icefrac_std',
        attrs={
            'description': 'Arctic ice fraction residuals standard deviation',
            'unit': 'million km^2',
        }
    ).to_netcdf(output_dir / "grid_level_model_internal_variability.nc")

    # Save Scalar (Numpy)
    np.save(output_dir / "model_internal_variability.npy", global_std.item())


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
    weights: xr.Dataset
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
        avg_col_name = f"variability_{var}"

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
    fire.Fire(calculate_variability)
