import sys
import fire
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend import (
    get_global_temp, get_regional_models, get_temp_diff,
    get_regional_map_from_global_temp,
    get_regional_delta_without_rampup,
    REVERSE_FANCY_SSP_TITLES, VAR2INFO,
    MIN_TARGET, MAX_TARGET, STEP_TARGET,
    exposurevar2var
)


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

        for temp_target in tqdm(np.linspace(MIN_TARGET, MAX_TARGET, num_targets)):
            temp_diff = get_temp_diff(data_dir, model_dir, output_dir, ssp_scenario, temp_target)
            var2cache_path = {
                var: output_dir / f"{simple_ssp}_{temp_target:.1f}_regional_{var}.nc"
                for var in VAR2INFO
            }

            for var in var2cache_path:
                cache_path = var2cache_path[var]
                if cache_path.exists():
                    print(f"Found {cache_path}")
                    continue

                is_exposure = "exposure" in var

                if not is_exposure:
                    # Get regional models for tas
                    variable_dir = model_dir / var
                else:
                    daily_var = exposurevar2var[var]
                    variable_dir = model_dir / daily_var
                fair2smip = get_regional_models(variable_dir)

                # Cache regional map and historical data (saved under the hood)
                regional_map = get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, output_dir)

                # Save the regional map
                regional_map.to_netcdf(cache_path)

                # Caches the regional delta
                get_regional_delta_without_rampup(var, data_dir, model_dir, output_dir, ssp_scenario, temp_target, temp_diff)


if __name__ == "__main__":
    fire.Fire(cache)
