"""Generic helpers shared by the SAI simulator backend.

Hosts cached I/O loaders (:func:`get_interpolator`, :func:`open_xarray_datasets`), the
land/ocean clipping primitives that wrap the IPCC-WGII continental geometries
(:func:`create_mask`, :func:`clip_to_land`, :func:`clip_to_ocean`), the smoothing and
regional-aggregation utilities used by the dashboard pipeline (:func:`apply_constraints`,
:func:`regional_aggregation`), the GeoJSON loader for spatial-aggregation polygons
(:func:`load_spatial_aggregation_gdfs`), the atomic NetCDF writer
(:func:`save_processed_data`), and :func:`apply_ramp_up_factor` for SAI onset smoothing.
"""
import uuid
import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache
from scipy.signal import butter, filtfilt
from pathlib import Path
from typing import Optional, Union

from fair_wrap.fair_utils import SIM_END_YEAR


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_interpolator(interpolator_path: Path) -> xr.Dataset:
    """Open and fully load an ``interpolator.nc`` file, returning the in-memory Dataset.

    Uses :func:`xr.open_dataset` inside a context manager so the file handle is released
    immediately after ``.load()``; the result is cached on ``interpolator_path`` so
    re-opening the same file in another simulation iteration is free.
    """
    with xr.open_dataset(interpolator_path) as ds:
        return ds.load()  # materialize -> file closed

def create_mask(
    xarray_dataset: xr.DataArray,
    gdf: gpd.GeoDataFrame,
) -> xr.DataArray:
    """Build a boolean mask that is ``True`` for grid cells outside the ``gdf`` polygons.

    Used by :func:`clip_to_land` / :func:`clip_to_ocean` and by the regional aggregation
    code in :mod:`get_outputs`. The polygons live in a ``[-180, 180)`` longitude frame
    while the simulator's xarrays use ``[0, 360)``, so the dataset is rotated, clipped via
    :meth:`xr.Dataset.rio.clip` (``invert=True, drop=False`` to preserve shape), and then
    rotated back. Returns ``mask.isnull()`` so callers can plug it straight into
    :meth:`xr.DataArray.where`.
    """
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


def regional_aggregation(
    xarray_dataset: xr.DataArray,
    weights: xr.DataArray,
    op: str,
) -> xr.DataArray:
    """Apply ``weights`` to ``xarray_dataset`` and reduce over the ``lat``/``lon`` dimensions.

    ``op`` is one of ``"mean"`` (cos-lat-weighted mean), ``"sum"`` (used for exposure /
    icefrac totals), ``"max"`` / ``"min"`` (mapped to the 1.0 / 0.0 spatial quantile so they
    still respect the weighting). Anything else raises :class:`ValueError`.
    """
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


def apply_constraints(
    time_series: xr.DataArray,
    smooth_type: str,
    filter_width: int = 10,
    filter_order: int = 3,
) -> xr.DataArray:
    """Smooth ``time_series`` with a Butterworth low-pass filter under one of three boundary policies.

    ``smooth_type`` selects how the series is extended before filtering:

    * ``"min_norm"`` — pads with the long-term mean (minimum-norm boundary).
    * ``"min_slope"`` — reflects about the boundary (preserves slope).
    * ``"min_roughness"`` — reflects + flips vertically (preserves second derivative).

    A Butterworth filter of order ``filter_order`` with normalized cutoff ``0.1`` is then
    applied via :func:`scipy.signal.filtfilt` (zero-phase), and the ``filter_width``
    padding is trimmed off before returning the result as an :class:`xr.DataArray` aligned
    with the input's ``time`` coordinate.
    """
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


@lru_cache(maxsize=8)
def _land_mask_values(data_dir: Path, lat_key: tuple, lon_key: tuple) -> np.ndarray:
    """Rasterized IPCC-WGII continental-regions mask for one lat/lon grid, as a bool array.

    Reading the geojson, unioning its geometries and rasterizing them costs about a second
    and used to happen on every clip (six times per p-value call); the grid is always the
    same, so the result is cached on the grid's coordinates (rounded to 1e-6 degrees).
    """
    continental_gdf = gpd.read_file(Path(data_dir) / "geojsons" / "IPCC-WGII-continental-regions.geojson")
    continental_gdf = gpd.GeoDataFrame(geometry=[continental_gdf.unary_union], crs=continental_gdf.crs)
    template = xr.DataArray(
        np.zeros((len(lat_key), len(lon_key))), dims=("lat", "lon"),
        coords={"lat": np.array(lat_key), "lon": np.array(lon_key)},
    )
    return create_mask(template, continental_gdf).values


def _land_mask(data_dir: Path, regional_map: xr.DataArray) -> xr.DataArray:
    values = _land_mask_values(
        Path(data_dir),
        tuple(np.round(regional_map.lat.values.astype(float), 6)),
        tuple(np.round(regional_map.lon.values.astype(float), 6)),
    )
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": regional_map.lat, "lon": regional_map.lon})


def clip_to_land(data_dir: Path, regional_map: xr.DataArray) -> xr.DataArray:
    """
    Mask regional_map to land areas only (continental regions included in IPCC-WGII definition).
    """
    return regional_map.where(_land_mask(data_dir, regional_map), np.nan)


def clip_to_ocean(data_dir: Path, regional_map: xr.DataArray) -> xr.DataArray:
    """
    Mask regional_map to ocean areas only (everything NOT included in the IPCC-WGII continental regions).
    """
    return regional_map.where(~_land_mask(data_dir, regional_map), np.nan)


def load_spatial_aggregation_gdfs(data_dir: Union[str, Path]) -> dict[str, gpd.GeoDataFrame]:
    """Load every ``*.geojson`` under ``data_dir / "geojsons"`` into a dict keyed by file stem.

    Used by the dashboard to populate the spatial-aggregation dropdowns (continents, regions,
    countries, …) without hardcoding the available layers. Normalizes the ``Name`` column to
    lowercase ``name`` so downstream code can rely on a single attribute, and skips
    individual GeoJSONs that fail to parse (printing a warning) instead of aborting the
    whole load.
    """
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
    decode_times: bool = True,
    *,
    group: Optional[str] = None,
    consolidated: Optional[bool] = None,
    storage_options: Optional[dict] = None,
) -> xr.Dataset:
    """Open a NetCDF file or Zarr store and eagerly load it into memory.

    Dispatches on ``dataset_path``: Zarr when the path ends in ``.zarr`` (with
    or without a trailing slash), contains ``.zarr/``, or starts with a remote
    URL scheme (``s3://``, ``gs://``, ``http://``, ``https://``); NetCDF
    otherwise. Both branches call :meth:`xarray.Dataset.load` so the returned
    Dataset owns no open file handles or HTTP connections — important for
    workflows that open many sources in sequence.

    Args:
        dataset_path: Path to a local NetCDF file, or a path / URL to a Zarr
            store.
        decode_times: If True, force cftime decoding (handles non-standard
            calendars like ``noleap``). If False, leave time coords as raw
            numerical values.
        group: Zarr only — subgroup path inside the store (e.g.
            ``"Mon/atmos_2d"`` for the CESM2-WACCM SAI stores whose root group
            holds no data variables). Ignored for NetCDF.
        consolidated: Zarr only — whether to read consolidated metadata.
            ``None`` (the default) lets xarray auto-detect and fall back if
            absent. Ignored for NetCDF.
        storage_options: Zarr only — fsspec kwargs (e.g. R2 ``endpoint_url``,
            ``anon``). Ignored for NetCDF.

    Returns:
        The fully loaded xarray Dataset.

    Raises:
        FileNotFoundError: NetCDF only — raised when the file is not on disk.
        ValueError: If the file or store is corrupt or unreadable.
    """
    path_str = str(dataset_path)
    is_zarr = (
        path_str.endswith(".zarr")
        or path_str.endswith(".zarr/")
        or ".zarr/" in path_str
        or path_str.startswith(("s3://", "gs://", "http://", "https://"))
    )

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    decode_kwargs = {"decode_times": time_coder} if decode_times else {"decode_times": False}

    if is_zarr:
        open_kwargs = {**decode_kwargs}
        if group is not None:
            open_kwargs["group"] = group
        if consolidated is not None:
            open_kwargs["consolidated"] = consolidated
        if storage_options is not None:
            open_kwargs["storage_options"] = storage_options
        try:
            with xr.open_zarr(path_str, **open_kwargs) as ds:
                ds.load()
                return ds
        except Exception as e:
            error_msg = f"\033[91mFailed to open zarr store: {path_str}\nError: {e}\033[0m"
            print(error_msg)
            raise ValueError(error_msg) from e

    # NetCDF path — existing behavior unchanged.
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        error_msg = f"\033[91mDataset file not found: {dataset_path}\033[0m"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with xr.open_dataset(dataset_path, **decode_kwargs) as ds:
            ds.load()
            return ds
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
    
    # Unique temp name so concurrent writers (e.g. Celery workers) never clobber
    # the same ".tmp.nc" while building different copies of the same cache key.
    temp_path = target_path.with_name(
        f"{target_path.stem}.{uuid.uuid4().hex}.tmp{target_path.suffix}"
    )

    try:
        # Clear inherited encodings from source NetCDFs before writing derived outputs.
        # Stale coordinate encodings (especially on "time") can cause netCDF4 write
        # failures like "Invalid argument: (variable 'time', group '/')".
        clean_data = data.copy(deep=False)
        clean_data.encoding = {}

        if isinstance(clean_data, xr.Dataset):
            for var_name in clean_data.data_vars:
                clean_data[var_name].encoding = {}
            for coord_name in clean_data.coords:
                clean_data[coord_name].encoding = {}
        elif isinstance(clean_data, xr.DataArray):
            for coord_name in clean_data.coords:
                clean_data[coord_name].encoding = {}

        # Write to TEMPORARY file
        # Passing encoding ensures the file is compressed on disk
        clean_data.to_netcdf(temp_path, encoding=encoding, compute=True)
        
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

def apply_ramp_up_factor(
    data: xr.DataArray,
    temp_diff: xr.DataArray,
    ramp_up: int,
) -> xr.DataArray:
    """Linearly ramp ``data`` from ``1/ramp_up`` up to ``1`` over the first ``ramp_up`` SAI years.

    Identifies the first year where SAI is actually active (``temp_diff <= 0``) and scales
    ``data`` in-place from ``1/ramp_up_eff`` at that year up to ``1`` over the next
    ``ramp_up_eff`` years, where ``ramp_up_eff = min(ramp_up, SIM_END_YEAR - start_year)`` so
    the ramp never exceeds the remaining simulation horizon. ``ramp_up <= 0`` or a temp_diff
    that is always positive (i.e., SAI is never triggered) is a no-op.
    """
    no_sai_indices = temp_diff > 0
    if ramp_up > 0 and not no_sai_indices.all():
        sai_mask = ~np.asarray(no_sai_indices, dtype=bool)
        actual_sai_start_year = int(temp_diff.time.values[sai_mask][0])
        ramp_up_eff = min(ramp_up, SIM_END_YEAR - actual_sai_start_year)
        if ramp_up_eff > 0:
            ramp_up_indices = np.arange(
                actual_sai_start_year, actual_sai_start_year + ramp_up_eff
            )
            ramp_up_factor = np.linspace(1 / ramp_up_eff, 1, ramp_up_eff)
            data.loc[dict(time=ramp_up_indices)] *= ramp_up_factor
    return data
