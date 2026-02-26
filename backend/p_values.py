import numpy as np
import xarray as xr
from scipy.stats import t

from .constants import *
from .utils import clip_to_land, clip_to_ocean
from .population import get_population_xr


def compute_ttest(mean1, mean2, std, n):
    """
    Perform per-cell two-sided t-tests on the given data.
    Parameters:
        mean1 (numpy.ndarray): Mean values for group 1, shape (W, H).
        mean2 (numpy.ndarray): Mean values for group 2, shape (W, H).
        std (numpy.ndarray): Shared standard deviation, shape (W, H).
        n (int): Number of samples per group.
    Returns:
        numpy.ndarray: p-values for each cell, shape (W, H).
    """
    # Compute the t-statistic
    mean_diff = mean1 - mean2
    # Avoid division by zero by masking where std is 0
    denominator = std * np.sqrt(2 / n)
    # Set denominator to 1 where it's 0 to avoid division warning, result will be overwritten
    safe_denominator = np.where(denominator == 0, 1, denominator)
    t_stat = mean_diff / safe_denominator
    # Override t_stat where denominator was 0
    t_stat = np.where(denominator == 0, 0, t_stat)

    # Degrees of freedom
    df = 2 * n - 2

    # Compute two-sided p-values from the t-distribution
    p_values = 2 * t.cdf(-np.abs(t_stat), df)

    # Assign a p-value of 1 to cells where the standard deviation is 0 and the means are equal
    p_values[(std == 0) & (mean_diff == 0)] = 1

    # Assign a p-value of 0 to cells where the standard deviation is 0 and the means are different
    p_values[(std == 0) & (mean_diff != 0)] = 0

    return p_values


def get_grid_level_p_values(var, data_dir, reference, comparison):
    
    # Compute two-sided t-test p-values between reference and comparison
    is_exposure = "exposure" in var
    if is_exposure:
        var = exposurevar2var[var]
        # Apply global population weighting
        population_xr = get_population_xr(data_dir)

    with xr.open_dataarray(data_dir / var / f"grid_level_model_internal_variability.nc") as da:
        std_np = da.load().values
        if is_exposure:
            std_np = std_np * population_xr
            std_np = std_np.values / 1.0e6

    # For every decade, compute the p-values, then concatenate across the time dimension
    decadal_grid_level_p_values = []
    decades = list(zip(range(2041, 2092, 10), range(2050, 2101, 10)))
    for _decade_start_year, _decade_end_year in decades:
        reference_np = reference.values # this is historical rebase
        comparison_np = comparison.sel(time=slice(_decade_start_year, _decade_end_year)).mean(dim='time').values

        grid_level_p_values = compute_ttest(reference_np, comparison_np, std_np, n=(2039-2025+1) * 3) # 3 members, 2025-2039
        # Convert regional_p_values to xarray DataArray
        grid_level_p_values = xr.DataArray(grid_level_p_values, dims=('lat', 'lon'), coords={'lat': reference.lat, 'lon': reference.lon})
        # Use the end of the decade as the time coordinate
        grid_level_p_values = grid_level_p_values.expand_dims(time=[_decade_end_year])
        decadal_grid_level_p_values.append(grid_level_p_values)
    decadal_grid_level_p_values = xr.concat(decadal_grid_level_p_values, dim='time').sortby('time')

    # Always clip p values to land
    if "icefrac" in var:
        grid_level_p_values = clip_to_ocean(data_dir, decadal_grid_level_p_values)
        # Set p-values to NaN for latitudes below 60
        grid_level_p_values = grid_level_p_values.where(grid_level_p_values.lat >= 60, np.nan)
    else:    
        grid_level_p_values = clip_to_land(data_dir, decadal_grid_level_p_values)

    return grid_level_p_values
