import sys
import fire
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.get_outputs import get_temp_diff, get_so2_by_latitude
from backend.baseline import get_global_temp, get_regional_models, get_regional_map_from_global_temp
from backend.delta import get_regional_delta
from backend.historical import get_historical_model
from backend.p_values import get_regional_p_values
from backend.constants import *
from backend.utils import get_interpolator
from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, CANVAS_START_YEAR, SIM_END_YEAR


def cache(data_dir, model_dir, output_dir):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for ssp_scenario in REVERSE_FANCY_SSP_TITLES:
        print(ssp_scenario)
        simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
        num_targets = int((MAX_TARGET - MIN_TARGET) / STEP_TARGET) + 1
        # First get FaIR global temperature
        global_temp = get_global_temp(ssp_scenario)

        for var in tqdm(VAR2INFO):
            no_sai_cache_paths = [output_dir / f"{simple_ssp}_regional_{var}.nc",
                                  output_dir / f"{simple_ssp}_regional_no_sai_p_values_{var}.nc"]
            if all(cache_path.exists() for cache_path in no_sai_cache_paths):
                print("Found no sai cache for", var)
                for cache_path in no_sai_cache_paths:
                    print(cache_path)
                regional_map = xr.open_dataarray(no_sai_cache_paths[0], autoclose=True)
                regional_p_values = xr.open_dataarray(no_sai_cache_paths[1], autoclose=True)
            else:
                is_exposure = "exposure" in var

                if not is_exposure:
                    # Get regional models for tas
                    variable_dir = model_dir / var
                else:
                    daily_var = exposurevar2var[var]
                    variable_dir = model_dir / daily_var
                fair2smip = get_regional_models(variable_dir)

                # Cache regional map and historical data (the latter is saved under the hood)
                regional_map, regional_p_values = get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, output_dir)

                regional_map.to_netcdf(no_sai_cache_paths[0])
                regional_p_values.to_netcdf(no_sai_cache_paths[1])

            for temp_target in np.linspace(MIN_TARGET, MAX_TARGET, num_targets):
                temp_diff = get_temp_diff(data_dir, model_dir, output_dir, ssp_scenario, temp_target)
                so2_cache_path = output_dir / f"{simple_ssp}_{temp_target:.1f}_so2_by_latitude.nc"
                if not so2_cache_path.exists():
                    so2_by_latitudes = []
                    start_years = list(range(2035, 2091))
                    for start_year in start_years:
                        _temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
                        interpolator = get_interpolator(model_dir / "so2" / "interpolator.nc")
                        # Assumes ramp-up is always 10 years, and caches so2 based on this.
                        # This would need to be changed if we want to support other ramp-up periods.
                        so2_by_latitude = get_so2_by_latitude(interpolator, _temp_diff, ramp_up=10)["so2"]
                        so2_by_latitude = so2_by_latitude.drop_vars('model')

                        so2_by_latitudes.append(so2_by_latitude)

                    so2_by_latitude = xr.concat(so2_by_latitudes, dim='start_year')
                    so2_by_latitude['start_year'] = start_years
                    so2_by_latitude.to_netcdf(so2_cache_path)

                sai_cache_paths = [output_dir / f"{simple_ssp}_{temp_target:.1f}_regional_delta_{var}.nc",
                                   output_dir / f"{simple_ssp}_{temp_target:.1f}_regional_sai_p_values_{var}.nc"]

                if all(cache_path.exists() for cache_path in sai_cache_paths) and so2_cache_path.exists():
                    print("Found sai cache for", var)
                    for cache_path in sai_cache_paths + [so2_cache_path]:
                        print(cache_path)
                    continue

                # Caches the regional delta (saved under the hood)
                # Assumes 0 ramp up, so that means ttest is unaffected by a ramp up in the middle of the selected decade.
                # This simplifies the implementation drastically.
                regional_delta = get_regional_delta(var, data_dir, model_dir, output_dir, ssp_scenario, temp_target, 0, temp_diff, CANVAS_START_YEAR)

                # Save the p_values
                historical_model = get_historical_model(var, data_dir, output_dir)
                if historical_model is None:
                    continue
                historical_rebase = historical_model.sel(time=slice(1850, 1900)).mean('time')
                regional_sai_p_values = get_regional_p_values(var, data_dir, historical_rebase, regional_map + regional_delta + historical_rebase)
                regional_sai_p_values.to_netcdf(sai_cache_paths[1])


if __name__ == "__main__":
    fire.Fire(cache)
