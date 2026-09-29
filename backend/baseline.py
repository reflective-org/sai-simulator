"""Baseline (no-SAI) regional projections derived from FaIR global temperature.

Loads the FaIR global-mean temperature time series for a given SSP scenario, runs the
fitted ``FaIR → SMIP`` linear emulator to produce regional projections for each variable,
clips, rebases, and computes the no-SAI grid-level p-values. Outputs are cached on disk
under ``cache_dir`` so subsequent calls for the same ``(ssp, var)`` reuse the same maps.
"""
import json
import sys
import joblib
import numpy as np
import xarray as xr
from functools import lru_cache
from pathlib import Path
from sklearn.linear_model import LinearRegression

from fair_wrap.fair_utils import *
from .constants import *
from .historical import get_historical_model
from .utils import clip_to_land
from .population import get_population_exposure
from .p_values import get_grid_level_p_values

sys.path.append(str(Path(__file__).resolve().parents[1]))
from version import SIMULATOR_VERSION  # noqa: E402  (needs the repo-root path above)


# Climate-response calibration handed to FaIR (see fair_wrap.fair_utils.CLIMATE_CALIBRATIONS).
# "constrained" runs the calibrated-constrained ensemble and averages across its members, so
# the global magnitude reflects the ensemble-mean response rather than one model's
# sensitivity (the member spread itself is collapsed by that mean and is not propagated).
# Set to "cesm" to restore the pre-2026-08 FaIR driver — note that only reverts the global
# warming magnitude; the multi-model regional emulators and multi-model historical rebase
# remain in effect either way. Deployment-fixed: change requires a process restart (the
# lru caches below key on this only via the cache-dir fingerprint).
FAIR_CALIBRATION = DEFAULT_CALIBRATION

# Every FaIR-dependent artifact in a cache dir is only valid for the (simulator version,
# FaIR climate calibration) pair that produced it, and neither is recoverable from the
# files themselves. The pair is therefore recorded in this file when a cache dir is first
# populated and checked before any cached file is trusted (see validate_cache_dir).
CACHE_META_FILENAME = "cache_meta.json"


def stamp_cache_dir(cache_dir: Path) -> None:
    """Record this process's (simulator version, FaIR calibration) fingerprint in ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {"simulator_version": SIMULATOR_VERSION, "fair_calibration": FAIR_CALIBRATION}
    # Write-then-rename so a concurrent reader never sees a truncated stamp
    # (parallel cache.py --scenario runs stamp the same dir with identical bytes).
    tmp_path = cache_dir / (CACHE_META_FILENAME + ".tmp")
    tmp_path.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_path.replace(cache_dir / CACHE_META_FILENAME)


def validate_cache_dir(cache_dir: Path) -> None:
    """Refuse a cache dir whose artifacts were produced under a different version or calibration.

    A matching ``cache_meta.json`` passes; an empty (or artifact-free) dir is stamped for the
    current process and passes; anything else raises. In particular a populated dir *without*
    a stamp predates the fingerprint mechanism — and therefore the AR6-constrained FaIR
    default — so serving its files would silently mix climate calibrations.
    """
    expected = {"simulator_version": SIMULATOR_VERSION, "fair_calibration": FAIR_CALIBRATION}
    meta_path = cache_dir / CACHE_META_FILENAME
    if meta_path.exists():
        try:
            found = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Unreadable {CACHE_META_FILENAME} in {cache_dir} ({e}). Delete the file "
                "and regenerate the cache with scripts/cache.py."
            ) from e
        if found != expected:
            raise ValueError(
                f"Cache dir {cache_dir} was generated under {found}, but this process "
                f"expects {expected}. Regenerate it with scripts/cache.py or point at a "
                "matching cache dir."
            )
        return
    if any(cache_dir.glob("*.nc")):
        raise ValueError(
            f"Cache dir {cache_dir} contains .nc artifacts but no {CACHE_META_FILENAME} — "
            "it predates the cache fingerprint (pre-AR6-constrained FaIR default), so its "
            "files may not match the current calibration. Regenerate it with "
            "scripts/cache.py or delete the stale files."
        )
    stamp_cache_dir(cache_dir)


def get_global_temp(
    ssp_scenario: str,
    initial_dir: Path | None = None,
) -> xr.DataArray:
    """Run FaIR for ``ssp_scenario`` and return global temperature anomaly vs the 1850-1900 mean.

    Loads the standard FaIR input dataframes and, optionally, NetCDF initial-conditions plus a
    precomputed pre-industrial temperature from ``initial_dir`` (each ``initial_conditions``
    DataArray has its ``scenario`` coord re-pointed at ``ssp_scenario`` so cached conditions
    work across scenarios). The FaIR ``runner`` is calibrated per :data:`FAIR_CALIBRATION`
    (default: the AR6 calibrated-constrained ensemble's own per-member configs), run without
    a progress bar, and the layer-0 temperature is rebased to the 1850-1900 mean and
    averaged across configs before being returned.
    """
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
        calibrate_to=FAIR_CALIBRATION,
    )

    runner.run(progress=False)
    preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(1850, 1900)).mean('timebounds')

    if initial_dir is not None and not initial_dir.exists():
        preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(1850, 1900)).mean('timebounds')

    global_temp = (runner.temperature.sel(layer=0) - preindustrial_temp).mean('config')

    return global_temp

@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_models(variable_dir: Path) -> LinearRegression:
    """Load and concatenate the per-model, per-emulator ``FaIR → SMIP`` linear regressors.

    For every ``(model, emulator)`` pair in ``REGIONAL_MODEL_NAMES × range(NUM_EMULATORS)``
    loads ``fair_to_smip_{model}_{i}.joblib`` from ``variable_dir`` and concatenates the
    coefficients and intercepts into a single :class:`sklearn.linear_model.LinearRegression`
    instance whose ``.predict`` returns stacked outputs for all emulators at once. Cached on
    ``variable_dir`` via :func:`functools.lru_cache` (32 entries) since the load is I/O-heavy.
    """
    var = variable_dir.name
    
    fair2smip_coefs, fair2smip_intercepts = [], []

    for model in REGIONAL_MODEL_NAMES:
        for i in range(NUM_EMULATORS):
            fair2smip = joblib.load(variable_dir / f"fair_to_smip_{model}_{i}.joblib")
            fair2smip_coefs.append(fair2smip.coef_)
            fair2smip_intercepts.append(fair2smip.intercept_)

    fair2smip_coefs = np.concatenate(fair2smip_coefs)
    fair2smip_intercepts = np.concatenate(fair2smip_intercepts)

    fair2smip = LinearRegression(n_jobs=-1)
    fair2smip.coef_ = fair2smip_coefs
    fair2smip.intercept_ = fair2smip_intercepts

    return fair2smip

def get_smip(global_temp: xr.DataArray, fair2smip: LinearRegression) -> np.ndarray:
    """Run the concatenated ``FaIR → SMIP`` regressor and reshape its output into a 5-D array.

    Slices ``global_temp`` to ``[2015, …]`` (the emulator's training range), predicts the
    stacked per-emulator regional fields, then reshapes the flat output into
    ``(time, model, emulator, lat, lon)`` using ``NUM_LAT`` / ``NUM_LON`` / ``NUM_EMULATORS``
    and the regional model list.
    """
    # Get regional projections for each model
    smip = fair2smip.predict(global_temp.sel(timebounds=slice(2015, None)))

    smip = smip.reshape((smip.shape[0], len(REGIONAL_MODEL_NAMES), NUM_EMULATORS, NUM_LAT, NUM_LON))

    return smip

def get_regional_map_from_global_temp(
    global_temp: xr.DataArray,
    fair2smip: LinearRegression,
    var: str,
    data_dir: Path,
    cache_dir: Path,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Convert FaIR global temperature into a labelled regional map plus its no-SAI p-values.

    Runs :func:`get_smip` to get the raw emulator output, wraps it as an :class:`xr.DataArray`
    with the canonical ``time / model / emulator / lat / lon`` coords (loading ``correct_lat.npy``
    for the latitude axis), takes the equal-weight mean over models and emulators, then applies the
    variable-specific post-processing: floors at zero for ``"above"``/``"below"`` count variables,
    clips to land for precipitation/water-availability variables, and converts to population
    exposure for exposure variables.

    Finally rebases temperature/precipitation variables to the 1850-1900 historical mean and
    returns ``(regional_map, regional_p_values)``. The p-values are ``None`` when no historical
    model exists for this variable (see :func:`get_historical_model`).
    """
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
    # Equal-weight multi-model mean: every model has NUM_EMULATORS members, so
    # averaging over both dims gives each model identical weight.
    regional_map = regional_map.mean(dim=['model', 'emulator'])[var]

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
        regional_p_values = get_grid_level_p_values(var, data_dir, historical_rebase, regional_map)
    else:
        regional_p_values = None

    ### Rebase the regional map ###
    if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
        regional_map = regional_map - historical_rebase

    return regional_map, regional_p_values


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_map(
    var: str,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    ssp_scenario: str,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Returns the regional map for the given variable, SSP scenario, and temperature target."""
    validate_cache_dir(cache_dir)
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_cache_path = cache_dir / f"{simple_ssp}_regional_{var}.nc"
    regional_p_cache_path = cache_dir / f"{simple_ssp}_regional_no_sai_p_values_{var}.nc"

    if regional_cache_path.exists() and regional_p_cache_path.exists():
        print(f"Found {regional_cache_path} and {regional_p_cache_path}")
        with xr.open_dataarray(regional_cache_path) as da:
            regional_map = da.load()
        with xr.open_dataarray(regional_p_cache_path) as da:
            regional_p_values = da.load()
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
