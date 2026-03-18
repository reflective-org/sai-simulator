import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import Union
from pathlib import Path


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_interpolator(interpolator_path: Path) -> xr.Dataset:
    with xr.open_dataset(interpolator_path) as ds:
        return ds.load()  # materialize -> file closed

def create_mask(xarray_dataset, gdf):
    # The geometries are -180 to 180 but the xarray dataset is 0 to 360, so convert this
    xarray_dataset = xarray_dataset.copy()
    original_lon = xarray_dataset.lon
    original_lat = xarray_dataset.lat
    xarray_dataset.coords['lon'] = xr.where(xarray_dataset.lon > 180, xarray_dataset.lon - 360, xarray_dataset.lon)
    xarray_dataset = xarray_dataset.sortby('lon')
    xarray_dataset = xarray_dataset.rio.write_crs("EPSG:4326")
    xarray_dataset = xarray_dataset.rio.set_spatial_dims("lon", "lat")
    mask = xarray_dataset.rio.clip(gdf.geometry, invert=True, drop=False)
    # Convert the mask back to 0 to 360
    mask.coords['lon'] = xr.where(mask.lon < 0, mask.lon + 360, mask.lon)
    mask = mask.sortby(['lon', 'lat'])
    mask.coords['lon'] = original_lon
    mask.coords['lat'] = original_lat

    return mask.isnull()


def regional_aggregation(xarray_dataset, weights, op):
    xarray_dataset = xarray_dataset.weighted(weights)
    if op == "mean":
        return xarray_dataset.mean(dim=('lat', 'lon'))
    elif op == "max":
        return xarray_dataset.quantile(1, dim=('lat', 'lon'))
    elif op == "min":
        return xarray_dataset.quantile(0, dim=('lat', 'lon'))
    elif op == "sum":
        return xarray_dataset.sum(dim=('lat', 'lon'))
    else:
        raise ValueError(f"Unknown operation: {op}")


def apply_constraints(time_series, smooth_type, filter_width=10, filter_order=3):

    if smooth_type == "min_norm":
        # Calculate the long-term mean of the time series
        long_term_mean = np.mean(time_series)

        # Minimum norm: pad with the long-term mean
        pad_min_norm = np.pad(time_series, (filter_width, filter_width), 'constant', constant_values=(long_term_mean, long_term_mean))

    elif smooth_type == "min_slope":
        # Minimum slope: reflect the series about the boundary
        pad_min_slope = np.pad(time_series, (filter_width, filter_width), 'reflect')
    
    elif smooth_type == "min_roughness":
        # Minimum roughness: reflect about the time boundary and vertically about the y-axis
        pad_min_roughness = np.pad(time_series, (filter_width, filter_width), 'symmetric')

    else:
        raise ValueError(f"Unknown smooth type: {smooth_type}")

    # Butterworth low-pass filter setup
    b, a = butter(filter_order, 0.1)  # 0.1 is a normalized frequency; adjust as needed

    # Apply filter to the padded series
    if smooth_type == "min_norm":
        smoothed_min_norm = filtfilt(b, a, pad_min_norm)
        smoothed =  smoothed_min_norm[filter_width:-filter_width]
    elif smooth_type == "min_slope":
        smoothed_min_slope = filtfilt(b, a, pad_min_slope)
        smoothed = smoothed_min_slope[filter_width:-filter_width]
    elif smooth_type == "min_roughness":
        smoothed_min_roughness = filtfilt(b, a, pad_min_roughness)
        smoothed = smoothed_min_roughness[filter_width:-filter_width]

    # Convert to xarray DataArray
    smoothed = xr.DataArray(smoothed, dims=('time'), coords={'time': time_series.time})

    return smoothed


def clip_to_land(data_dir, regional_map):
    """
    Mask regional_map to land areas only (continental regions included in IPCC-WGII definition).
    """
    geojson_dir = data_dir / "geojsons"
    continental_gdf = gpd.read_file(geojson_dir / "IPCC-WGII-continental-regions.geojson")

    # Take union of all geometries
    continental_gdf = gpd.GeoDataFrame(geometry=[continental_gdf.unary_union], crs=continental_gdf.crs)
    land_mask = create_mask(regional_map, continental_gdf)
    regional_map = regional_map.where(land_mask, np.nan)
    return regional_map

def clip_to_ocean(data_dir, regional_map):
    """
    Mask regional_map to ocean areas only (everything NOT included in the IPCC-WGII continental regions).
    """
    geojson_dir = data_dir / "geojsons"
    continental_gdf = gpd.read_file(geojson_dir / "IPCC-WGII-continental-regions.geojson")

    # Take union of all geometries
    continental_gdf = gpd.GeoDataFrame(geometry=[continental_gdf.unary_union], crs=continental_gdf.crs)
    # Create land mask as before
    land_mask = create_mask(regional_map, continental_gdf)
    # Invert the mask to get ocean (True where NOT land)
    ocean_mask = ~land_mask
    regional_map = regional_map.where(ocean_mask, np.nan)
    return regional_map


def load_spatial_aggregation_gdfs(data_dir):
    data_dir = Path(data_dir)
    geojsons_dir = data_dir / "geojsons"
    
    if not geojsons_dir.exists():
        raise FileNotFoundError(f"Geojsons directory not found: {geojsons_dir}")
    
    spatial_agg_gdfs = {}
    for geojson_file in geojsons_dir.glob("*.geojson"):
        try:
            gdf = gpd.read_file(geojson_file)
            # Normalize column naming to match downstream expectations
            if "Name" in gdf.columns and "name" not in gdf.columns:
                gdf = gdf.rename(columns={"Name": "name"})
            spatial_agg_gdfs[geojson_file.stem] = gdf
        except Exception as e:
            print(f"Warning: Could not load {geojson_file}: {e}")
    
    return spatial_agg_gdfs


def open_xarray_datasets(
    dataset_path: Union[str, Path], 
    decode_times: bool = True
    ) -> xr.Dataset:
    dataset_path = Path(dataset_path)
    """
    Opens a NetCDF file, loads it into memory, and safely closes the file handle.

    This function ensures that the underlying file handler is closed immediately 
    after loading the data into memory (.load()), preventing 'Too many open files' 
    errors in workflows processing many files.

    Args:
        dataset_path (Union[str, Path]): Path to the NetCDF dataset file.
        decode_times (bool, optional): If True, forces the use of cftime for 
            date decoding (useful for non-standard calendars). If False, time 
            coordinates remain as numerical values. Defaults to True.

    Returns:
        xr.Dataset: The fully loaded xarray Dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is corrupt or cannot be read by xarray.
    """
    dataset_path = Path(dataset_path)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

    if not dataset_path.exists():
        # ANSI Red for error visibility
        error_msg = f"\033[91mDataset file not found: {dataset_path}\033[0m"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    # Configure arguments: Force cftime if decoding, otherwise disable decoding entirely
    open_kwargs = {"decode_times": time_coder} if decode_times else {"decode_times": False}

    try:
        # Use a context manager to ensure the file is closed automatically
        with xr.open_dataset(dataset_path, **open_kwargs) as ds:
            ds.load()  # Persist data in RAM
            return ds  # Exiting the block closes the file handle
            
    except Exception as e:
        error_msg = f"\033[91mFailed to open dataset: {dataset_path}\nError: {e}\033[0m"
        print(error_msg)
        raise ValueError(error_msg) from e


def save_processed_data(
    data: Union[xr.DataArray, xr.Dataset], 
    path: Union[str, Path], 
    overwrite: bool = False,
    complevel: int = 5
) -> None:
    """
    Save an xarray DataArray or Dataset to a NetCDF file using atomic writes and compression.

    Args:
        data: The data to save.
        path: The target file path.
        overwrite: If True, replaces existing files. If False, skips saving.
        complevel: Compression level (1-9). 5 is a good balance of speed/size.

    Returns:
        None
    """
    
    target_path = Path(path)

    # Check if the file exists
    if target_path.exists():
        if not overwrite:
            print(f"\033[93mWARNING: {target_path} already exists. Skipping save.\033[0m")
            return
        else:
            print(f"\033[96mINFO: File {target_path} exists. Overwriting enabled.\033[0m")

    # Prepare Compression Encoding
    # We apply compression to all data variables, but NOT to coordinate variables (lat, lon, time)
    # as compressing coordinates can slow down slicing operations.
    comp_args = {'zlib': True, 'complevel': complevel}
    encoding = {}

    if isinstance(data, xr.Dataset):
        for var_name in data.data_vars:
            encoding[var_name] = comp_args
    elif isinstance(data, xr.DataArray):
        if data.name:
            encoding[data.name] = comp_args
    
    # Define a temporary path (atomic write strategy)
    temp_path = target_path.with_suffix(target_path.suffix + '.tmp')

    try:
        # Write to TEMPORARY file
        # Passing encoding ensures the file is compressed on disk
        data.to_netcdf(temp_path, encoding=encoding, compute=True)
        
        # Atomic Rename
        temp_path.replace(target_path)
        
    except Exception as e:
        print(f"\033[91mERROR: Failed to save {target_path}: {e}\033[0m")
        # Cleanup garbage
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass 
        raise
