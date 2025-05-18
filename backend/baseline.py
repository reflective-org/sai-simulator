import joblib
import numpy as np
import xarray as xr
from functools import lru_cache
from sklearn.linear_model import LinearRegression

from fair_wrap.fair_utils import *
from .constants import *
from .historical import get_historical_model
from .utils import clip_to_land
from .population import get_population_exposure
from .p_values import get_regional_p_values


def get_global_temp(ssp_scenario, initial_dir=None):
    # Get global temperature from FaIR
    df_emis, df_configs, df_solar, df_volcanic = get_dataframes()

    # Optionally load initial conditions from a directory
    if initial_dir is not None and initial_dir.exists():
        initial_conditions = {
            path.stem: xr.load_dataarray(path)
            for path in initial_dir.iterdir()
            if path.suffix == ".nc"
        }
        # Replace scenario in the initial conditions with the ssp scenario
        for key in list(initial_conditions.keys()):
            initial_conditions[key].coords["scenario"] = [REVERSE_FANCY_SSP_TITLES[ssp_scenario]]
        preindustrial_temp = np.load(initial_dir / "preindustrial_temperature.npy")
    else:
        initial_conditions = None

    runner = get_runner(
        df_emis, df_configs,
        df_solar, df_volcanic,
        co2_input=None, ssp_scenario=ssp_scenario,
        initial_conditions=initial_conditions,
        calibrate_to="cesm"
    )

    runner.run(progress=False)
    preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(1850, 1900)).mean('timebounds')

    if initial_dir is not None and not initial_dir.exists():
        preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(1850, 1900)).mean('timebounds')

    global_temp = (runner.temperature.sel(layer=0) - preindustrial_temp).mean('config')

    return global_temp



@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_models(model_dir):

    fair2smip_coefs, fair2smip_intercepts = [], []

    for model in REGIONAL_MODEL_NAMES:
        for i in range(NUM_EMULATORS):
            fair2smip = joblib.load(model_dir / f"fair_to_smip_{model}_{i}.joblib")
            fair2smip_coefs.append(fair2smip.coef_)
            fair2smip_intercepts.append(fair2smip.intercept_)

    fair2smip_coefs = np.concatenate(fair2smip_coefs)
    fair2smip_intercepts = np.concatenate(fair2smip_intercepts)

    fair2smip = LinearRegression(n_jobs=-1)
    fair2smip.coef_ = fair2smip_coefs
    fair2smip.intercept_ = fair2smip_intercepts

    return fair2smip


def get_smip(global_temp, fair2smip):
    # Get regional projections for each model
    smip = fair2smip.predict(global_temp.sel(timebounds=slice(2015, None)))

    smip = smip.reshape((smip.shape[0], len(REGIONAL_MODEL_NAMES), NUM_EMULATORS, NUM_LAT, NUM_LON))

    return smip


def get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, cache_dir):
    # Get regional map from global temperature
    smip = get_smip(global_temp, fair2smip)
    regional_map = xr.DataArray(smip, dims=('time', 'model', 'emulator', 'lat', 'lon'))
    regional_map.coords['time'] = np.arange(2015, SIM_END_YEAR+1)
    regional_map.coords['model'] = REGIONAL_MODEL_NAMES
    regional_map.coords['emulator'] = np.arange(NUM_EMULATORS)
    correct_lat = np.load(data_dir / "correct_lat.npy")
    regional_map.coords['lat'] = correct_lat
    regional_map.coords['lon'] = np.linspace(0, 360, NUM_LON, endpoint=False)
    regional_map = xr.Dataset({var: regional_map})
    regional_map = regional_map.sel(model="CESM2-WACCM").mean(dim='emulator')[var]

    is_above_below = "above" in var or "below" in var
    is_exposure = "exposure" in var

    # Floor the values to 0 if the variable counts number of days
    if is_above_below:
        regional_map = regional_map.where(regional_map > 0, 0)

    # Clip pr and days to land only
    if is_above_below or var in ["pr", "p-e"]:
        regional_map = clip_to_land(data_dir, regional_map)
        if is_exposure:
            regional_map = get_population_exposure(data_dir, regional_map)

    ### Get historical temperature data ###
    historical_map = get_historical_model(var, data_dir, cache_dir)
    # Get regional p values
    if historical_map is not None:
        historical_rebase = historical_map.sel(time=slice(1850, 1900)).mean('time')
        regional_p_values = get_regional_p_values(var, data_dir, historical_rebase, regional_map)
    else:
        regional_p_values = None

    ### Rebase the regional map ###
    if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
        regional_map = regional_map - historical_rebase

    return regional_map, regional_p_values


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_map(var, data_dir, model_dir, cache_dir, ssp_scenario):
    """Returns the regional map for the given variable, SSP scenario, and temperature target."""
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_cache_path = cache_dir / f"{simple_ssp}_regional_{var}.nc"
    regional_p_cache_path = cache_dir / f"{simple_ssp}_regional_no_sai_p_values_{var}.nc"

    if regional_cache_path.exists() and regional_p_cache_path.exists():
        print(f"Found {regional_cache_path} and {regional_p_cache_path}")
        regional_map = xr.open_dataarray(regional_cache_path, autoclose=True)
        regional_p_values = xr.open_dataarray(regional_p_cache_path, autoclose=True)
    else:
        # First get FaIR global temperature
        global_temp = get_global_temp(ssp_scenario)

        is_exposure = "exposure" in var

        if is_exposure:
            daily_var = exposurevar2var[var]
            variable_dir = model_dir / daily_var
        else:
            variable_dir = model_dir / var
        fair2smip = get_regional_models(variable_dir)

        # Get regional map
        regional_map, regional_p_values = get_regional_map_from_global_temp(global_temp, fair2smip, var, data_dir, cache_dir)

    return regional_map, regional_p_values
