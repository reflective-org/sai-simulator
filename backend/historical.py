"""Historical model and observation loaders for the SAI dashboard.

Provides cached access to the per-variable historical model output — the equal-weight
multi-model mean of each model's projection-baseline run (see
:func:`backend.constants.resolve_projection_baseline_file`) — plus two observation series —
Berkeley Earth global mean temperature and NSIDC September Arctic sea-ice extent — used to
overlay observations on the ``tas`` and ``icefrac`` dashboard charts.
"""
import pandas as pd
import xarray as xr
from functools import lru_cache
from pathlib import Path

from .constants import (
    exposurevar2var,
    regional_models_for_var,
    resolve_projection_baseline_file,
)
from .utils import clip_to_land
from .utils import save_processed_data
from .population import get_population_exposure


def _load_baselines(variable_dir: Path, models: list[str]) -> list[xr.DataArray]:
    """Each model's no-SAI projection baseline for one variable, as ``(time, lat, lon)``.

    Resolves each model's file via :func:`backend.constants.resolve_projection_baseline_file`
    — the same file ``scripts/fit_map.py`` trained that model's emulators on — and selects
    ``ssp245``.

    Every model in ``models`` must carry the stitched historical run in its baseline
    file: the multi-model mean inner-joins on the years the models SHARE, so a model whose
    file starts in 2015 would silently erase everyone's pre-2015 series. That is what
    ``REGIONAL_MODEL_VAR_EXCLUSIONS`` guards — callers pass ``regional_models_for_var(var)``,
    never the raw roster. For a
    model that does carry it, the ``time <= 2014`` slice is unaffected by which no-SAI run backs the
    projection years.
    """
    per_model = []
    for model in models:
        path = resolve_projection_baseline_file(variable_dir, model)
        if path is None:
            raise FileNotFoundError(
                f"No processed projection baseline for {model} in {variable_dir}. "
                "Run scripts/process_monthly_gauss.py first."
            )
        with xr.open_dataarray(path) as da:
            da = da.load()
        per_model.append(da.sel(model=model, ssp="ssp245").drop_vars(["ssp", "model"]))
    return per_model


def _load_multi_model_baseline(variable_dir: Path, models: list[str]) -> xr.DataArray:
    """Equal-weight mean of the models' baselines over their overlapping years.

    Keeps the historical reference consistent with the multi-model projection mean taken
    in :mod:`backend.baseline`. Note the inner join: CESM's annual series starts in 1851
    and MIROC's in 1850, so the combined 1850-1900 reference covers 1851-1900 (see the
    review doc, X5).
    """
    per_model = _load_baselines(variable_dir, models)
    if len(per_model) == 1:
        return per_model[0]
    per_model = xr.align(*per_model, join="inner")
    return xr.concat(per_model, dim="model").mean(dim="model")


def _load_per_model_baseline(variable_dir: Path, models: list[str]) -> xr.DataArray:
    """The models' baselines stacked on a ``model`` dim, ``(model, time, lat, lon)``.

    Outer-joined on time (a year one model lacks is NaN for it), so each model's own
    1850-1900 reference uses every year that model has. Feeds the end-to-end per-model
    series behind the combined uncertainty band (:mod:`backend.uncertainty`).
    """
    per_model = _load_baselines(variable_dir, models)
    return xr.concat(
        [da.expand_dims(model=[model]) for da, model in zip(per_model, models)],
        dim="model",
        join="outer",
    )


def _postprocess_historical(var: str, data_dir: Path, historical_model_data: xr.DataArray) -> xr.DataArray:
    """Floor / land-clip / population-scale exactly as the projection map is, then cut at 2014.

    Masks and population weights are ``(lat, lon)`` and broadcast over any leading dims, so
    this serves both the combined ``(time, lat, lon)`` and per-model ``(model, time, lat, lon)``
    arrays.
    """
    is_above_below = "above" in var or "below" in var
    is_exposure = "exposure" in var
    if is_above_below:
        historical_model_data = historical_model_data.where(historical_model_data > 0, 0)
    if is_above_below or var in ["pr", "p-e"]:
        historical_model_data = clip_to_land(data_dir, historical_model_data)
        if is_exposure:
            historical_model_data = get_population_exposure(data_dir, historical_model_data)
    return historical_model_data.sel(time=slice(None, 2014))


_HISTORICAL_VARS = [
    "tas", "tasmin", "tasmax", "tas_above_40", "tas_above_35", "tas_below_0",
    "pr", "p-e", "pr_above_10", "pr_above_20",
    "icefrac",
]


def _historical_source_dir(var: str, data_dir: Path) -> Path | None:
    """Directory holding the processed baselines for ``var`` (its daily variable for exposure)."""
    if var in _HISTORICAL_VARS:
        return data_dir / var
    if "exposure" in var:
        return data_dir / exposurevar2var[var]
    return None


@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_historical_model(
    var: str,
    data_dir: Path,
    cache_dir: Path,
) -> xr.DataArray | None:
    """Gets the unrebased historical model data (multi-model mean) for the given variable."""
    # Local import: baseline imports this module, so a top-level import would cycle.
    # Historical files are FaIR-independent, but the fingerprint is dir-level — an
    # unstamped or mismatched cache dir must be refused here too, not just at the
    # FaIR-dependent readers, so direct callers can't sidestep the check.
    from .baseline import validate_cache_dir
    validate_cache_dir(cache_dir)
    historical_cache_path = cache_dir / f"historical_{var}.nc"

    if historical_cache_path.exists():
        print(f"Found {historical_cache_path}")
        try:
            with xr.open_dataarray(historical_cache_path) as da:
                historical_model_data = da.load()
            return historical_model_data
        except (OSError, ValueError) as e:
            print(f"Invalid historical cache at {historical_cache_path}; rebuilding. Error: {e}")
            historical_cache_path.unlink(missing_ok=True)

    source_dir = _historical_source_dir(var, data_dir)
    if source_dir is None:
        return None

    # The models taking part in this variable, icefrac included: its historical reference
    # was CESM-only while the sea-ice sigmoid was CESM-only, and every model in
    # scripts/fit_icefrac_arctic.MODEL_SCENARIOS has its own sigmoid now.
    historical_model_data = _load_multi_model_baseline(source_dir, regional_models_for_var(var))
    historical_model_data = _postprocess_historical(var, data_dir, historical_model_data)
    save_processed_data(historical_model_data, historical_cache_path, overwrite=True)
    return historical_model_data


@lru_cache(maxsize=32)
def get_historical_model_per_model(
    var: str,
    data_dir: Path,
    cache_dir: Path,
) -> xr.DataArray | None:
    """Unrebased historical model data per model, ``(model, time, lat, lon)``.

    Same sources and post-processing as :func:`get_historical_model`, but stacked on a
    ``model`` dim (outer-joined on time) instead of averaged, so each model's end-to-end
    series can be rebased against its own 1850-1900 record. Cached as
    ``historical_per_model_<var>.nc``; ``None`` for variables without a historical run.
    """
    from .baseline import validate_cache_dir
    validate_cache_dir(cache_dir)
    cache_path = cache_dir / f"historical_per_model_{var}.nc"

    if cache_path.exists():
        try:
            with xr.open_dataarray(cache_path) as da:
                return da.load()
        except (OSError, ValueError) as e:
            print(f"Invalid per-model historical cache at {cache_path}; rebuilding. Error: {e}")
            cache_path.unlink(missing_ok=True)

    source_dir = _historical_source_dir(var, data_dir)
    if source_dir is None:
        return None

    per_model = _load_per_model_baseline(source_dir, regional_models_for_var(var))
    per_model = _postprocess_historical(var, data_dir, per_model)
    save_processed_data(per_model, cache_path, overwrite=True)
    return per_model


def get_historical_obs_global_mean_temp(data_dir: Path) -> xr.DataArray:
    """Load Berkeley Earth historical global-mean temperatures, rebased to the 1850-1900 mean.

    Reads ``Land_and_Ocean_summary.txt`` (Berkeley Earth Land+Ocean summary; the first 56
    rows are header / metadata), renames the columns into ``Year`` / ``Temperature``, wraps
    the result as an :class:`xr.DataArray` with a ``time`` dimension, and subtracts the
    1850-1900 mean so it aligns with the rebased simulator outputs on the dashboard.
    """
    # Source: https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_summary.txt
    historical_obs_data = pd.read_csv(data_dir / "Land_and_Ocean_summary.txt", skiprows=56, sep=r"\s+")
    historical_obs_data = historical_obs_data.rename(columns={"%": "Year", "Year,": "Temperature"})

    # Convert to xarray
    historical_obs_data = xr.DataArray(historical_obs_data["Temperature"].values, dims=('time'), coords={'time': historical_obs_data["Year"].values})

    # Rebase by datasets 1850-1900 mean
    historical_obs_data = historical_obs_data - historical_obs_data.sel(time=slice(1850, 1900)).mean()

    return historical_obs_data


def get_historical_obs_sea_ice_extent(data_dir: Path) -> xr.DataArray:
    """Load the NSIDC September Arctic sea-ice extent observations (1979-2025) as a ``time``-indexed DataArray.

    Reads ``september-sea-ice-extent-observation.csv`` (an NSIDC summary with ``year`` and
    ``extent`` columns) and returns it as an :class:`xr.DataArray` with the year as the
    ``time`` coordinate; the values are absolute extents (no rebasing) since the dashboard
    plots sea ice in physical units.
    """
    # Source: NSIDC September Arctic sea ice extent (1979-2025)
    historical_obs_data = pd.read_csv(
        data_dir / "september-sea-ice-extent-observation.csv",
        skipinitialspace=True,
    )

    # Convert to xarray using year and extent columns
    historical_obs_data = xr.DataArray(
        historical_obs_data["extent"].values,
        dims=('time'),
        coords={'time': historical_obs_data["year"].values},
    )

    return historical_obs_data
