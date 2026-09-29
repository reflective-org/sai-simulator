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
    REGIONAL_MODEL_NAMES,
    resolve_projection_baseline_file,
)
from .utils import clip_to_land
from .utils import save_processed_data
from .population import get_population_exposure


def _load_multi_model_baseline(variable_dir: Path, models: list[str]) -> xr.DataArray:
    """Equal-weight mean of each model's no-SAI projection baseline for one variable.

    Resolves each model's baseline file via
    :func:`backend.constants.resolve_projection_baseline_file` — the same file
    ``scripts/fit_map.py`` trained that model's emulators on — selects ``ssp245``, and
    averages the models over their overlapping years so the historical reference stays
    consistent with the multi-model projection mean taken in :mod:`backend.baseline`.

    Every model's baseline file carries the same stitched historical run, so the
    ``time <= 2014`` slice the caller takes is unaffected by which no-SAI run backs the
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
    if len(per_model) == 1:
        return per_model[0]
    per_model = xr.align(*per_model, join="inner")
    return xr.concat(per_model, dim="model").mean(dim="model")


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

    is_above_below = "above" in var or "below" in var
    is_exposure = "exposure" in var
    if var in [
        "tas", "tasmin", "tasmax", "tas_above_40", "tas_above_35", "tas_below_0",
        "pr", "p-e", "pr_above_10", "pr_above_20",
        "icefrac"
    ]:
        # icefrac projections come from the CESM-trained interpolator rather than the
        # REGIONAL_MODEL_NAMES emulators, so its historical reference stays CESM-only.
        models = ["CESM2-WACCM"] if var == "icefrac" else REGIONAL_MODEL_NAMES
        historical_model_data = _load_multi_model_baseline(data_dir / var, models)
    # elif var in []:
    #     with xr.open_dataarray(data_dir / var / "output_gauss-cmip_historical.nc") as da:
    #         historical_model_data = da.load()
    #     historical_model_data = historical_model_data.sel(model="CESM2-WACCM", ssp="ssp245")
    #     historical_model_data = historical_model_data.drop_vars(["ssp", "model"])
    elif is_exposure:
        daily_var = exposurevar2var[var]
        historical_model_data = _load_multi_model_baseline(data_dir / daily_var, REGIONAL_MODEL_NAMES)
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

        save_processed_data(historical_model_data, historical_cache_path, overwrite=True)

    return historical_model_data


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
