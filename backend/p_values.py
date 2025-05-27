import numpy as np
import xarray as xr
from scipy.stats import t

from .constants import *
from .utils import clip_to_land


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
    t_stat = mean_diff / (std * np.sqrt(2 / n))

    # Degrees of freedom
    df = 2 * n - 2

    # Compute two-sided p-values from the t-distribution
    p_values = 2 * t.cdf(-np.abs(t_stat), df)

    # Assign a p-value of 1 to cells where the standard deviation is 0 and the means are equal
    p_values[(std == 0) & (mean_diff == 0)] = 1

    # Assign a p-value of 0 to cells where the standard deviation is 0 and the means are different
    p_values[(std == 0) & (mean_diff != 0)] = 0

    return p_values


def get_regional_p_values(var, data_dir, reference, comparison):
    # Compute two-sided t-test p-values between reference and comparison
    is_exposure = "exposure" in var
    if is_exposure:
        var = exposurevar2var[var]

    std_np = xr.open_dataarray(data_dir / var / f"regional_natural_variability.nc", autoclose=True).values

    # For every decade, compute the p-values, then concatenate across the time dimension
    decadal_regional_p_values = []
    decades = list(zip(range(2041, 2092, 10), range(2050, 2101, 10)))
    for _decade_start_year, _decade_end_year in decades:
        reference_np = reference.values
        comparison_np = comparison.sel(time=slice(_decade_start_year, _decade_end_year)).mean(dim='time').values

        regional_p_values = compute_ttest(reference_np, comparison_np, std_np, n=(2030-2010+1) * 3) # 3 members, 2010-2030
        # Convert regional_p_values to xarray DataArray
        regional_p_values = xr.DataArray(regional_p_values, dims=('lat', 'lon'), coords={'lat': reference.lat, 'lon': reference.lon})
        # Use the end of the decade as the time coordinate
        regional_p_values = regional_p_values.expand_dims(time=[_decade_end_year])
        decadal_regional_p_values.append(regional_p_values)
    regional_p_values = xr.concat(decadal_regional_p_values, dim='time').sortby('time')

    # Always clip p values to land
    regional_p_values = clip_to_land(data_dir, regional_p_values)

    return regional_p_values
