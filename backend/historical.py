import pandas as pd
import xarray as xr
from functools import lru_cache

from .constants import exposurevar2var
from .utils import clip_to_land
from .population import get_population_exposure


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_historical_model(var, data_dir, cache_dir):
    """Gets the unrebased historical model data for the given variable."""
    historical_cache_path = cache_dir / f"historical_{var}.nc"

    if historical_cache_path.exists():
        print(f"Found {historical_cache_path}")
        historical_model_data = xr.open_dataarray(historical_cache_path, autoclose=True)
    else:
        is_above_below = "above" in var or "below" in var
        is_exposure = "exposure" in var
        if var in ["pr", "tas", "p-e"]:
            historical_model_data = xr.open_dataarray(data_dir / var / "output_gauss-baseline.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        elif var in ["tasmin", "tasmax", "tas_above_40", "tas_above_35", "tas_below_0", "pr_above_10", "pr_above_20"]:
            historical_model_data = xr.open_dataarray(data_dir / var / "output_gauss-cmip_historical.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        elif is_exposure:
            daily_var = exposurevar2var[var]
            historical_model_data = xr.open_dataarray(data_dir / daily_var / "output_gauss-cmip_historical.nc", autoclose=True)
            historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
            historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
        else:
            historical_model_data = None

        if historical_model_data is not None:
            if is_above_below:
                historical_model_data = historical_model_data.where(historical_model_data > 0, 0)

            if is_above_below or var in ["pr", "p-e"]:
                historical_model_data = clip_to_land(data_dir, historical_model_data)
                if is_exposure:
                    historical_model_data = get_population_exposure(data_dir, historical_model_data)
            historical_model_data = historical_model_data.sel(time=slice(None, 2014))

            historical_model_data.to_netcdf(historical_cache_path)

    return historical_model_data


def get_historical_obs_global_mean_temp(data_dir):
    # Source: https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt
    historical_obs_data = pd.read_csv(data_dir / "Land_and_Ocean_summary.txt", skiprows=56, sep="\s+")
    historical_obs_data = historical_obs_data.rename(columns={"%": "Year", "Year,": "Temperature"})

    # Convert to xarray
    historical_obs_data = xr.DataArray(historical_obs_data["Temperature"].values, dims=('time'), coords={'time': historical_obs_data["Year"].values})

    # Rebase by datasets 1850-1900 mean
    historical_obs_data = historical_obs_data - historical_obs_data.sel(time=slice(1850, 1900)).mean()

    return historical_obs_data