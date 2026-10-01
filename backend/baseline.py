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
from .historical import get_historical_model, get_historical_model_per_model
from .utils import clip_to_land
from .population import get_population_exposure
from .p_values import get_grid_level_p_values, no_sai_p_values_cache_path

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


def _cache_fingerprint() -> dict:
    """Everything a cache artifact's numbers depend on besides the input data.

    The model roster and the per-variable exclusions are part of it: adding a
    model changes every multi-model mean and the pooled SAI delta, so a cache
    produced under a different roster must be refused, not served.
    """
    return {
        "simulator_version": SIMULATOR_VERSION,
        "fair_calibration": FAIR_CALIBRATION,
        "regional_models": list(REGIONAL_MODEL_NAMES),
        "var_exclusions": {k: list(v) for k, v in sorted(REGIONAL_MODEL_VAR_EXCLUSIONS.items())},
    }


def stamp_cache_dir(cache_dir: Path) -> None:
    """Record this process's (simulator version, FaIR calibration) fingerprint in ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = _cache_fingerprint()
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
    expected = _cache_fingerprint()
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
    preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(PREINDUSTRIAL_START, PREINDUSTRIAL_END)).mean('timebounds')

    if initial_dir is not None and not initial_dir.exists():
        preindustrial_temp = runner.temperature.sel(layer=0, timebounds=slice(PREINDUSTRIAL_START, PREINDUSTRIAL_END)).mean('timebounds')

    global_temp = (runner.temperature.sel(layer=0) - preindustrial_temp).mean('config')

    return global_temp

@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_models(variable_dir: Path) -> LinearRegression:
    """Load the per-model ``FaIR → SMIP`` regressors, one emulator-averaged regression per model.

    Each model has ``NUM_EMULATORS`` bootstrap regressors ``fair_to_smip_{model}_{i}.joblib``.
    They are linear, so the mean of their predictions IS the prediction of their mean
    coefficients and intercepts: ``mean_i(X·a_i + b_i) = X·mean_i(a_i) + mean_i(b_i)``. The
    emulators are therefore averaged here, per model, and the returned
    :class:`sklearn.linear_model.LinearRegression` stacks one regression per model in
    ``regional_models_for_var(var)`` order. Predicting every emulator separately and averaging
    afterwards gives the same maps but needs ``NUM_EMULATORS`` times the memory — about
    23 GB per scenario with three models, which is what froze parallel cache runs. Cached on
    ``variable_dir`` via :func:`functools.lru_cache` (32 entries) since the load is I/O-heavy.
    """
    var = variable_dir.name

    fair2smip_coefs, fair2smip_intercepts = [], []

    for model in regional_models_for_var(var):
        coefs, intercepts = [], []
        for i in range(NUM_EMULATORS):
            fair2smip = joblib.load(variable_dir / f"fair_to_smip_{model}_{i}.joblib")
            coefs.append(fair2smip.coef_)
            intercepts.append(fair2smip.intercept_)
        fair2smip_coefs.append(np.mean(coefs, axis=0))
        fair2smip_intercepts.append(np.mean(intercepts, axis=0))

    fair2smip_coefs = np.concatenate(fair2smip_coefs)
    fair2smip_intercepts = np.concatenate(fair2smip_intercepts)

    fair2smip = LinearRegression(n_jobs=-1)
    fair2smip.coef_ = fair2smip_coefs
    fair2smip.intercept_ = fair2smip_intercepts

    return fair2smip

def get_smip(
    global_temp: xr.DataArray, fair2smip: LinearRegression, models: list[str]
) -> np.ndarray:
    """Run the stacked ``FaIR → SMIP`` regressor and reshape its output into a 4-D array.

    Slices ``global_temp`` to ``[2015, …]`` (the emulator's training range), predicts the
    stacked per-model (emulator-averaged, see :func:`get_regional_models`) regional fields,
    then reshapes the flat output into ``(time, model, lat, lon)`` using ``NUM_LAT`` /
    ``NUM_LON`` and ``models`` — the variable's participating model list
    (:func:`backend.constants.regional_models_for_var`), which must match the list the
    regressor was concatenated over.
    """
    # Get regional projections for each model
    smip = fair2smip.predict(global_temp.sel(timebounds=slice(2015, None)))

    smip = smip.reshape((smip.shape[0], len(models), NUM_LAT, NUM_LON))

    return smip


def get_regional_map_from_global_temp(
    global_temp: xr.DataArray,
    fair2smip: LinearRegression,
    var: str,
    data_dir: Path,
    cache_dir: Path,
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray]:
    """Convert FaIR global temperature into a labelled regional map, its no-SAI p-values, and the per-model maps.

    Runs :func:`get_smip` to get each model's emulator-mean output, wraps it as an
    :class:`xr.DataArray` with the canonical ``time / model / lat / lon`` coords (loading
    ``correct_lat.npy`` for the latitude axis), then keeps two views of it:

    * the **combined** map — equal-weight mean over models (each already the mean of its
      emulators), rebased against the multi-model historical mean; this is what the dashboard
      draws;
    * the **per-model** maps ``(time, model, lat, lon)`` — each model's emulator mean,
      rebased against **its own** model's historical mean. These are the end-to-end
      per-model outcomes behind the combined uncertainty band (:mod:`backend.uncertainty`).

    Both get the same variable-specific post-processing (floor at zero for ``above``/``below``
    counts, land clip for precipitation/water availability, population exposure). Returns
    ``(regional_map, regional_p_values, regional_map_per_model)``; the p-values are ``None``
    when no historical model exists for this variable (see :func:`get_historical_model`).
    """
    # Get regional map from global temperature
    models = regional_models_for_var(var)
    smip = get_smip(global_temp, fair2smip, models)
    regional_map = xr.DataArray(smip, dims=('time', 'model', 'lat', 'lon'))
    regional_map.coords['time'] = np.arange(2015, SIM_END_YEAR+1)
    regional_map.coords['model'] = models
    correct_lat = np.load(data_dir / "correct_lat.npy")
    regional_map.coords['lat'] = correct_lat
    regional_map.coords['lon'] = np.linspace(0, 360, NUM_LON, endpoint=False)
    regional_map = xr.Dataset({var: regional_map})
    # Per-model maps: each model's emulator mean (averaged in get_regional_models).
    # Equal-weight multi-model mean on top: every model has NUM_EMULATORS members, so
    # averaging the per-model means gives each model identical weight.
    regional_map_per_model = regional_map[var]
    regional_map = regional_map_per_model.mean(dim='model')

    regional_map = _postprocess_map(var, data_dir, regional_map)
    regional_map_per_model = _postprocess_map(var, data_dir, regional_map_per_model)

    ### Get historical temperature data ###
    historical_map = get_historical_model(var, data_dir, cache_dir)
    # Get regional p values
    if historical_map is not None:
        historical_rebase = historical_map.sel(time=slice(PREINDUSTRIAL_START, PREINDUSTRIAL_END)).mean('time')
        # Significance is judged against the current-climate baseline (1995-2014), so the map's
        # hatching means "significantly different from today's climate" rather than from 1850-1900.
        significance_reference = historical_map.sel(time=slice(DISPLAY_BASELINE_START, DISPLAY_BASELINE_END)).mean('time')
        regional_p_values = get_grid_level_p_values(var, data_dir, significance_reference, regional_map)
    else:
        regional_p_values = None

    ### Rebase the regional maps: the combined map against the multi-model historical mean,
    ### each per-model map against its own model's historical record. ###
    # The projection stays in the pre-industrial frame here: the SAI temp_target, the icefrac
    # model, and get_temp_diff all assume anomalies vs 1850-1900. The dashboard re-expresses the
    # map against the current-climate baseline at display time (see get_outputs._load_variable_fields).
    if var in REBASED_VARS:
        regional_map = regional_map - historical_rebase
        historical_per_model = get_historical_model_per_model(var, data_dir, cache_dir)
        rebase_per_model = historical_per_model.sel(time=slice(1850, 1900)).mean('time')
        regional_map_per_model = regional_map_per_model - rebase_per_model

    return regional_map, regional_p_values, regional_map_per_model


# Variables whose maps are shown as anomalies from the 1850-1900 historical mean.
REBASED_VARS = ("tas", "tasmin", "tasmax", "pr", "p-e")


def _postprocess_map(var: str, data_dir: Path, regional_map: xr.DataArray) -> xr.DataArray:
    """Floor / land-clip / population-scale a map; broadcasts over any leading dims (e.g. ``model``)."""
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
    return regional_map


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_regional_map(
    var: str,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    ssp_scenario: str,
) -> tuple[xr.DataArray, xr.DataArray | None, xr.DataArray]:
    """Returns ``(regional_map, no-SAI p-values, per-model maps)`` for the variable and scenario.

    Served from ``cache_dir`` when all three cache files exist (``scripts/cache.py`` writes
    them); otherwise computed via :func:`get_regional_map_from_global_temp`.
    """
    validate_cache_dir(cache_dir)
    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_cache_path = cache_dir / f"{simple_ssp}_regional_{var}.nc"
    regional_p_cache_path = no_sai_p_values_cache_path(cache_dir, simple_ssp, var)
    regional_per_model_cache_path = cache_dir / f"{simple_ssp}_regional_per_model_{var}.nc"

    if all(p.exists() for p in (regional_cache_path, regional_p_cache_path, regional_per_model_cache_path)):
        print(f"Found {regional_cache_path}, {regional_p_cache_path} and {regional_per_model_cache_path}")
        with xr.open_dataarray(regional_cache_path) as da:
            regional_map = da.load()
        with xr.open_dataarray(regional_p_cache_path) as da:
            regional_p_values = da.load()
        with xr.open_dataarray(regional_per_model_cache_path) as da:
            regional_map_per_model = da.load()
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
        regional_map, regional_p_values, regional_map_per_model = get_regional_map_from_global_temp(
            global_temp, fair2smip, var, data_dir, cache_dir
        )

    return regional_map, regional_p_values, regional_map_per_model
