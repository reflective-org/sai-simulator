import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache
from scipy.signal import butter, filtfilt


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_interpolator(interpolator_path):
    interpolator = xr.open_dataset(interpolator_path)
    return interpolator


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
    geojson_dir = data_dir / "geojsons"
    continental_gdf = gpd.read_file(geojson_dir / "IPCC-WGII-continental-regions.geojson")

    # Take union of all geometries
    continental_gdf = gpd.GeoDataFrame(geometry=[continental_gdf.unary_union], crs=continental_gdf.crs)
    land_mask = create_mask(regional_map, continental_gdf)
    regional_map = regional_map.where(land_mask, np.nan)
    return regional_map
