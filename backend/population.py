import numpy as np
import xarray as xr
import geopandas as gpd
from functools import lru_cache


@lru_cache(maxsize=1)  # Caches the last call
def get_population_data(data_dir):
    # Went with the pregridded data from Dan instead of our processed data
    processed_population_path = data_dir / "Regridded_pop_num.npy"
    population_np = np.load(processed_population_path)

    return population_np


def aggregate_xarray_by_geopandas(xr_data, gdf, fill_value=0):
    # Ensure CRS alignment
    if not gdf.crs:
        raise ValueError("GeoDataFrame has no CRS. Define a CRS for the GeoDataFrame.")
    if not xr_data.rio.crs:
        raise ValueError("xarray data has no CRS. Define a CRS for the xarray data.")

    gdf = gdf.to_crs(xr_data.rio.crs.to_string())

    # # Revert longitude shift (0 to 360) to (-180 to 180)
    # [180, 360] -> [-180, 0]
    # [0, 180] -> [0, 180]
    xr_data = xr_data.assign_coords(lon=(((xr_data.lon + 180) % 360) - 180)).sortby('lon')

    # For each lat/lon in xr_data, find its correspondig country and create a mask
    # where each value of the mask corresponds to the country ID
    lon, lat = np.meshgrid(xr_data.lon.values, xr_data.lat.values)
    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon.flatten(), lat.flatten()), 
                                  crs=xr_data.rio.crs.to_string())
    gdf.index = gdf.index + 1
    joined_gdf = gpd.sjoin(points_gdf, gdf, how="left", predicate='intersects')
    joined_gdf = joined_gdf.fillna(0)
    rasterized_countries = joined_gdf['index_right'].values.reshape(xr_data.lat.size, xr_data.lon.size)

    # Determine the number of time steps and the unique group IDs
    if xr_data.dims != ('time', 'lat', 'lon'):
        # Array is in the wrong order (lat, lon, time), so transpose it
        xr_data = xr_data.transpose('time', 'lat', 'lon')
    time_series_values = xr_data.values
    num_time_steps = time_series_values.shape[0]
    unique_groups = np.unique(rasterized_countries)

    # Prepare an output array with the same shape
    output_values = np.zeros_like(time_series_values)
    
    # Iterate over each time step
    for t in range(num_time_steps):
        # For each group, sum values over the group members
        for group in unique_groups:
            if group == 0:
                continue
            mask = (rasterized_countries == group)
            # Sum values where the mask is True and assign to the same locations in the output array
            try:
                group_sum = np.nansum(time_series_values[t][mask])
            except:
                breakpoint()
            output_values[t][mask] = group_sum

    # Convert the output array to an xarray DataArray
    xr_data.values = output_values

    # Revert the longitude shift back to 0 to 360
    # [-180, 0] -> [180, 360]
    # [0, 180] -> [0, 180]
    xr_data = xr_data.assign_coords(lon=((xr_data.lon + 360) % 360)).sortby('lon')

    return xr_data


@lru_cache(maxsize=1)  # Caches the last call
def get_country_boundaries(data_dir):
    # Source: world administrative boundaries from OpenDataSoft v2.1
    country_boundaries = gpd.read_file(data_dir / "world-administrative-boundaries.geojson")
    
    return country_boundaries


def get_population_exposure(data_dir, ds, country_level=False):

    population_np = get_population_data(data_dir)
    population_xr = xr.DataArray(population_np, dims=('lon', 'lat'), coords={'lat': ds.lat, 'lon': ds.lon})
    # Shift lon values by 180
    population_xr = population_xr.assign_coords(lon=(((population_xr.lon + 180) % 360))).sortby('lon').transpose()

    exposure = ds * population_xr

    exposure /= 1e6  # Convert to millions

    if country_level:
        # Aggregate to country level by summing over lat and lon
        country_boundaries = get_country_boundaries(data_dir)

        exposure = exposure.rio.write_crs("EPSG:4326")
        exposure = exposure.rio.set_spatial_dims("lon", "lat")
        exposure = aggregate_xarray_by_geopandas(exposure, country_boundaries)

    return exposure
