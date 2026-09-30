"""Top-level SAI dashboard pipeline: :func:`get_outputs` plus its per-step helpers.

Orchestrates the full per-request workflow used by the simulator front-end. For each variable
in the request, the pipeline loads the no-SAI regional projection, computes the SAI ``delta``,
runs grid-level t-tests, aggregates spatial means / smoothed time series / PDF histograms,
optionally writes GeoTIFFs, PNG/SVG plots, and NetCDF exports, and finally tacks on the
aggregate SO2 injection time series. Most of the public surface is intentionally narrow —
:func:`get_outputs` is the entry point, with the per-output writers (:func:`create_geotiff_files`,
:func:`create_vectorized_files`, :func:`generate_sai_netcdf`) callable directly when needed.

Private helpers (``_load_variable_fields``, ``_compute_regional_delta``,
``_compute_sai_p_values``, ``_compute_mean_over_time``, ``_compute_distribution``,
``_compute_internal_variability``) decompose :func:`get_outputs` into per-stage units; their
typed ``_VariableFields`` / ``_DeltaResult`` / ``_MeanOverTimeResult`` dataclasses are the
glue between stages and intentionally not part of the public API.
"""
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
import hashlib
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import struct
import logging
from dataclasses import dataclass
from datetime import datetime

from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR

from .constants import *
from .historical import get_historical_model, get_historical_obs_global_mean_temp, get_historical_obs_sea_ice_extent
from .baseline import get_regional_map, validate_cache_dir
from .delta import get_regional_delta
from .p_values import get_grid_level_p_values, sai_p_values_cache_path
from .population import get_population_data
from .variable import get_variable_regional_delta
from .utils import get_interpolator, create_mask, regional_aggregation, apply_constraints, clip_to_land, apply_ramp_up_factor
from .icefrac import get_area, predict_icefrac


logger = logging.getLogger(__name__)

@lru_cache(maxsize=32)  # Caches the last 32 unique calls
def get_temp_diff(
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    ssp_scenario: str,
    temp_target: float,
) -> xr.DataArray:
    """Compute the global-mean temperature shortfall (target − projected) used to drive SAI.

    Loads the SSP regional temperature projection for ``tas`` via :func:`get_regional_map`,
    averages it with cosine-of-latitude area weights to a single global-mean time series,
    and returns ``temp_target - global_mean``. Positive values mean SAI is needed to close
    the gap; non-positive values are zeroed out by :func:`get_so2_by_latitude` downstream.

    Cached on the full argument tuple via :func:`functools.lru_cache` (32 entries).
    """
    regional_temp, _ = get_regional_map("tas", data_dir, model_dir, cache_dir, ssp_scenario)

    global_mean = regional_temp.weighted(np.cos(np.deg2rad(regional_temp.lat))).mean(('lat', 'lon'))
    temp_diff = temp_target - global_mean

    return temp_diff


def get_so2_by_latitude(
    interpolator: xr.DataArray,
    temp_diff: xr.DataArray,
    ramp_up: int,
) -> xr.DataArray:
    """Compute the SO2 injection-by-latitude needed to close ``temp_diff`` at each year.

    Multiplies the ``slope`` feature of ``interpolator`` (loaded from the SO2 model) by the
    temperature shortfall to get a per-latitude SO2 amount, zeros out years where SAI is
    not required (``temp_diff <= 0``), then routes through :func:`apply_ramp_up_factor` to
    soften the onset over the ``ramp_up`` years following the SAI start.
    """
    so2_by_latitude = interpolator.sel(features='slope') * temp_diff

    no_sai_indices = temp_diff > 0
    so2_by_latitude = so2_by_latitude.where(~no_sai_indices, 0)
    so2_by_latitude = apply_ramp_up_factor(so2_by_latitude, temp_diff, ramp_up)

    return so2_by_latitude


@dataclass
class _VariableFields:
    """Per-variable inputs loaded once per :func:`get_outputs` iteration."""
    regional_map: xr.DataArray
    regional_no_sai_p_values: xr.DataArray | None
    historical_model: xr.DataArray | None
    historical_rebase: xr.DataArray | None  # 1850-1900 mean (pre-industrial frame; used to undo the rebase)
    significance_reference: xr.DataArray | None  # current-climate mean used as the t-test reference
    display_offset: xr.DataArray | None  # current-climate mean expressed vs 1850-1900; shifts displayed anomalies
    area: xr.DataArray | None  # Cell areas, only populated when var == "icefrac"
    # Per-climate-model sigmoid predictions (model, time, lat, lon), only
    # populated when var == "icefrac"; regional_map is their across-model mean.
    icefrac_per_model: xr.DataArray | None = None


@dataclass
class _DeltaResult:
    """Output of :func:`_compute_regional_delta`."""
    regional_delta: xr.DataArray
    icefrac_with_sai: xr.DataArray | None  # Only populated when var == "icefrac"
    regional_no_sai_p_values: xr.DataArray | None  # Recomputed for icefrac, otherwise unchanged
    # Per-climate-model with-SAI predictions; icefrac_with_sai is their mean.
    icefrac_with_sai_per_model: xr.DataArray | None = None


@dataclass
class _MeanOverTimeResult:
    """Output of :func:`_compute_mean_over_time`."""
    mean_no_sai: xr.DataArray
    mean_with_sai: xr.DataArray
    historical_model_global_mean: xr.DataArray
    # Masked views needed by the PDF helper in regional mode (None in global mode):
    masked_regional_map: xr.DataArray | None
    masked_regional_delta: xr.DataArray | None
    masked_historical_model: xr.DataArray | None
    # icefrac only, global mode only: ±2σ across-climate-model half-widths of
    # the aggregated extent series, aligned to mean_no_sai / mean_with_sai:
    model_spread_2sigma_no_sai: xr.DataArray | None = None
    model_spread_2sigma_with_sai: xr.DataArray | None = None


def _load_variable_fields(
    var: str,
    *,
    ssp_scenario: str,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    regional_temp: xr.DataArray,
    regional_temp_no_sai_p_values: xr.DataArray | None,
) -> _VariableFields:
    """Load projection map, no-SAI p-values, historical model, rebase, and (for icefrac) cell area.

    For ``"tas"`` the already-loaded ``regional_temp`` is reused to avoid a redundant disk read.
    For ``"icefrac"`` the regional map is predicted from temperature; no-SAI p-values are left
    ``None`` here and computed later in :func:`_compute_regional_delta`. For all other variables
    :func:`get_regional_map` is used.

    The historical model is rebased to the 1850–1900 mean for ``tas/tasmin/tasmax/pr/p-e`` so
    downstream code can reason in anomalies; for other variables it is returned as-is.

    Two extra references are derived from the (un-rebased) historical model when present:
    ``significance_reference`` (the current-climate ``DISPLAY_BASELINE`` mean, used as the
    t-test reference so hatching means "different from today's climate") and ``display_offset``
    (that same current-climate mean expressed vs 1850-1900, i.e. ``significance_reference -
    historical_rebase``), which the caller subtracts from the pre-industrial-framed anomalies to
    present the map against the current-climate baseline. Only the map moves; the time-series and
    distribution outputs stay in the pre-industrial frame.
    """
    is_icefrac = var == "icefrac"
    area = get_area(data_dir) if is_icefrac else None
    icefrac_per_model = None

    if var == "tas":
        regional_map = regional_temp
        regional_no_sai_p_values = regional_temp_no_sai_p_values
    elif is_icefrac:
        # One sigmoid prediction per climate model; everything downstream uses
        # their mean, and the per-model fields feed the uncertainty band.
        icefrac_per_model = predict_icefrac(regional_temp, model_dir, per_model=True)
        regional_map = icefrac_per_model.mean(dim='model')
        regional_no_sai_p_values = None
    else:
        regional_map, regional_no_sai_p_values = get_regional_map(
            var, data_dir, model_dir, cache_dir, ssp_scenario
        )

    historical_model = get_historical_model(var, data_dir, cache_dir)
    historical_rebase = None
    significance_reference = None
    display_offset = None
    if historical_model is not None:
        historical_rebase = historical_model.sel(
            time=slice(PREINDUSTRIAL_START, PREINDUSTRIAL_END)
        ).mean('time')
        significance_reference = historical_model.sel(
            time=slice(DISPLAY_BASELINE_START, DISPLAY_BASELINE_END)
        ).mean('time')
        display_offset = significance_reference - historical_rebase
        if var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
            historical_model = historical_model - historical_rebase

    return _VariableFields(
        regional_map=regional_map,
        regional_no_sai_p_values=regional_no_sai_p_values,
        historical_model=historical_model,
        historical_rebase=historical_rebase,
        significance_reference=significance_reference,
        display_offset=display_offset,
        area=area,
        icefrac_per_model=icefrac_per_model,
    )


def _compute_regional_delta(
    var: str,
    *,
    fields: _VariableFields,
    regional_temp: xr.DataArray,
    ssp_scenario: str,
    temp_target: float | None,
    ramp_up: int | None,
    start_year: int | None,
    temp_diff: xr.DataArray | None,
    variable_injection: np.ndarray | None,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
) -> _DeltaResult:
    """Compute the SAI ``regional_delta`` for one variable.

    For ``"icefrac"`` the delta is the predicted-with-SAI fraction minus the no-SAI fraction,
    derived by routing the temperature delta through :func:`predict_icefrac`; this branch also
    computes the no-SAI p-values that ``_load_variable_fields`` could not provide.
    For ``"p-e"`` the delta is built from separate ``"pr"`` and ``"e"`` injection deltas.
    Otherwise the standard analytic (``variable_injection is None``) or injection-array path
    is used. ``icefrac_with_sai`` is returned so :func:`_compute_sai_p_values` can reuse it.
    """
    is_icefrac = var == "icefrac"
    icefrac_with_sai = None
    icefrac_with_sai_per_model = None
    regional_no_sai_p_values = fields.regional_no_sai_p_values

    if is_icefrac:
        if variable_injection is None:
            temp_regional_delta = get_regional_delta(
                "tas", data_dir, model_dir, cache_dir, ssp_scenario,
                temp_target, ramp_up, temp_diff, start_year,
            )
        else:
            temp_regional_delta = get_variable_regional_delta(
                "tas", data_dir, cache_dir, variable_injection
            )
        temp_with_sai = regional_temp + temp_regional_delta
        icefrac_with_sai_per_model = predict_icefrac(
            temp_with_sai, model_dir, per_model=True
        )
        icefrac_with_sai = icefrac_with_sai_per_model.mean(dim='model')
        regional_delta = icefrac_with_sai - fields.regional_map
        regional_no_sai_p_values = get_grid_level_p_values(
            var, data_dir, fields.significance_reference, fields.regional_map
        )
    else:
        if variable_injection is None:
            regional_delta = get_regional_delta(
                var, data_dir, model_dir, cache_dir, ssp_scenario,
                temp_target, ramp_up, temp_diff, start_year,
            )
        elif var != "p-e":
            regional_delta = get_variable_regional_delta(
                var, data_dir, cache_dir, variable_injection
            )
        else:
            regional_delta_p = get_variable_regional_delta(
                "pr", data_dir, cache_dir, variable_injection
            )
            regional_delta_e = get_variable_regional_delta(
                "e", data_dir, cache_dir, variable_injection
            )
            regional_delta = regional_delta_p - regional_delta_e

    return _DeltaResult(
        regional_delta=regional_delta,
        icefrac_with_sai=icefrac_with_sai,
        regional_no_sai_p_values=regional_no_sai_p_values,
        icefrac_with_sai_per_model=icefrac_with_sai_per_model,
    )


def _compute_sai_p_values(
    var: str,
    *,
    fields: _VariableFields,
    regional_delta: xr.DataArray,
    icefrac_with_sai: xr.DataArray | None,
    simple_ssp: str,
    temp_target: float | None,
    variable_injection: np.ndarray | None,
    data_dir: Path,
    cache_dir: Path,
    decade_end_year: int,
) -> xr.DataArray | None:
    """Compute (or load from cache) the with-SAI grid-level p-values, sliced to ``decade_end_year``.

    Returns ``None`` when the variable has no historical model (in which case a t-test is not
    meaningful). When ``variable_injection`` is ``None`` we look for a per-(scenario, target)
    cache file under ``cache_dir`` first; otherwise the values are recomputed.

    For temperature/precipitation variables the regional map is rebased before comparison
    (``map + delta + rebase``); for icefrac the absolute ``icefrac_with_sai`` is used directly;
    other variables compare ``map + delta`` against the reference. In every case the t-test
    reference is the current-climate ``significance_reference`` (1995-2014 mean), so significance
    is judged against today's climate rather than pre-industrial.
    """
    if fields.historical_model is None:
        return None

    if variable_injection is None:
        validate_cache_dir(cache_dir)
        regional_p_cache_path = sai_p_values_cache_path(cache_dir, simple_ssp, temp_target, var)
        regional_p_cache_path_exists = regional_p_cache_path.exists()
    else:
        regional_p_cache_path = None
        regional_p_cache_path_exists = False

    if regional_p_cache_path_exists:
        logger.info("Found %s", regional_p_cache_path)
        with xr.open_dataarray(regional_p_cache_path) as da:
            regional_sai_p_values = da.load()
    elif var in ["tas", "tasmin", "tasmax", "pr", "p-e"]:
        # The regional map is already rebased; add the historical rebase back to recover the
        # absolute field, then compare against the current-climate reference.
        regional_sai_p_values = get_grid_level_p_values(
            var, data_dir, fields.significance_reference,
            fields.regional_map + regional_delta + fields.historical_rebase,
        )
    elif var == "icefrac":
        regional_sai_p_values = get_grid_level_p_values(
            var, data_dir, fields.significance_reference, icefrac_with_sai
        )
    else:
        regional_sai_p_values = get_grid_level_p_values(
            var, data_dir, fields.significance_reference,
            fields.regional_map + regional_delta,
        )

    return regional_sai_p_values.sel(time=decade_end_year)


def _compute_mean_over_time(
    var: str,
    *,
    regional_map: xr.DataArray,
    regional_delta: xr.DataArray,
    recent_regional_map: xr.DataArray,
    historical_model: xr.DataArray | None,
    area: xr.DataArray | None,
    spatial_gdf,  # GeoDataFrame | None — untyped to avoid a module-level geopandas import
    spatial_item: str | None,
    start_year: int,
    variable_injection: np.ndarray | None,
    data_dir: Path,
    icefrac_per_model: xr.DataArray | None = None,
    icefrac_with_sai_per_model: xr.DataArray | None = None,
) -> _MeanOverTimeResult:
    """Aggregate regional fields into the historical / no-SAI / with-SAI time series shown on the dashboard.

    Operates in two modes that share the same return shape:

    * **Global** (``spatial_gdf`` or ``spatial_item`` is ``None``): area-weighted aggregation
      over the full lat/lon grid. ``masked_*`` outputs are ``None``.
    * **Regional**: the same aggregation but restricted to the region described by
      ``spatial_gdf[spatial_gdf.name == spatial_item]``. The masked DataArrays are
      returned so :func:`_compute_distribution` can reuse them without re-masking.

    Aggregation op is ``"sum"`` for exposure / icefrac variables and ``"mean"`` otherwise;
    the lat-weight array is uniform for sum-op variables and ``cos(lat)`` otherwise. After
    aggregation the historical/recent/no-SAI series is smoothed via :func:`apply_constraints`
    to give a continuous historical→projection boundary, and ``mean_with_sai`` is anchored
    to ``mean_no_sai`` for years before SAI (or before the first nonzero ``variable_injection``
    year) so the curves split exactly at the SAI start.

    For icefrac in global mode, ``icefrac_per_model`` / ``icefrac_with_sai_per_model``
    (per-climate-model sigmoid predictions) additionally yield the ±2σ across-model
    half-width series of the aggregated extent — the uncertainty band around the
    multi-model-mean curves.
    """
    is_exposure = "exposure" in var
    is_above_below = "above" in var or "below" in var
    is_icefrac = var == "icefrac"

    op = "sum" if (is_exposure or is_icefrac) else "mean"
    if is_exposure or is_icefrac:
        weights = xr.DataArray(
            np.ones_like(regional_map.lat),
            dims=('lat'),
            coords={'lat': regional_map.lat},
        )
    else:
        weights = np.cos(np.deg2rad(regional_map.lat))

    masked_regional_map = None
    masked_regional_delta = None
    masked_historical_model = None

    if spatial_gdf is None or spatial_item is None:
        if historical_model is not None:
            if is_icefrac:
                historical_mean = regional_aggregation(
                    historical_model * area.broadcast_like(historical_model), weights, op
                )
            else:
                historical_mean = regional_aggregation(historical_model, weights, op)

        if is_icefrac:
            recent_global_mean = regional_aggregation(
                recent_regional_map * area.broadcast_like(recent_regional_map), weights, op
            )
            mean_no_sai = regional_aggregation(
                regional_map * area.broadcast_like(regional_map), weights, op
            )
        else:
            recent_global_mean = regional_aggregation(recent_regional_map, weights, op)
            mean_no_sai = regional_aggregation(regional_map, weights, op)

        regional_map_with_sai = regional_map + regional_delta
        if is_above_below:
            regional_map_with_sai = regional_map_with_sai.where(regional_map_with_sai > 0, 0)
            # Re-clip because the line above sets NaN values to 0.
            regional_map_with_sai = clip_to_land(data_dir, regional_map_with_sai)
        if is_icefrac:
            mean_with_sai = regional_aggregation(
                regional_map_with_sai * area.broadcast_like(regional_map_with_sai), weights, op
            )
        else:
            mean_with_sai = regional_aggregation(regional_map_with_sai, weights, op)
    else:
        region = spatial_gdf[spatial_gdf.name == spatial_item]
        if historical_model is not None:
            historical_mask = create_mask(historical_model, region)
            masked_historical_model = historical_model.where(historical_mask, np.nan)
            historical_mean = regional_aggregation(masked_historical_model, weights, op)

        recent_mask = create_mask(recent_regional_map, region)
        masked_recent_regional_map = recent_regional_map.where(recent_mask, np.nan)
        recent_global_mean = regional_aggregation(masked_recent_regional_map, weights, op)

        mask = create_mask(regional_map, region)
        masked_regional_map = regional_map.where(mask, np.nan)
        masked_regional_delta = regional_delta.where(mask, np.nan)
        mean_no_sai = regional_aggregation(masked_regional_map, weights, op)

        masked_regional_map_with_sai = masked_regional_map + masked_regional_delta
        if is_above_below:
            masked_regional_map_with_sai = masked_regional_map_with_sai.where(
                masked_regional_map_with_sai > 0, 0
            )
        mean_with_sai = regional_aggregation(masked_regional_map_with_sai, weights, op)

    if historical_model is not None:
        historical_model_global_mean = xr.concat(
            [historical_mean, recent_global_mean], dim='time'
        )
    else:
        historical_model_global_mean = recent_global_mean

    # Smooth across the historical→projection boundary.
    concatenated_global_mean = xr.concat(
        [historical_model_global_mean, mean_no_sai], dim='time'
    )
    concatenated_global_mean = apply_constraints(concatenated_global_mean, "min_roughness")
    historical_model_global_mean = concatenated_global_mean.sel(
        time=slice(None, historical_model_global_mean.time[-1] + 1)
    )
    mean_no_sai = concatenated_global_mean.sel(
        time=slice(historical_model_global_mean.time[-1], None)
    )

    if variable_injection is not None:
        global_injection_amounts = variable_injection.sum(axis=0)
        if global_injection_amounts.max() > 0:
            # Anchor mean_with_sai to mean_no_sai before the first nonzero injection year so
            # the historical→projection boundary remains smooth.
            first_nonzero_index = np.argmax(global_injection_amounts > 0)
            first_nonzero_year = list(range(2035, 2101))[first_nonzero_index]
            mean_with_sai.loc[dict(time=slice(None, first_nonzero_year - 1))] = \
                mean_no_sai.loc[dict(time=slice(None, first_nonzero_year - 1))]
        else:
            # Zero injection → identical to no-SAI (also matches the smoothing applied above).
            mean_with_sai = mean_no_sai
    else:
        # Anchor mean_with_sai to mean_no_sai before SAI start year for a smooth split.
        mean_with_sai.loc[dict(time=slice(None, start_year - 1))] = \
            mean_no_sai.loc[dict(time=slice(None, start_year - 1))]

    # Across-climate-model uncertainty band for icefrac (global mode): 2σ of
    # the per-model aggregated extent series, i.e. "2 sigmas of the mean" the
    # dashboard shades around the multi-model-mean curves.
    model_spread_2sigma_no_sai = None
    model_spread_2sigma_with_sai = None
    if (
        is_icefrac
        and (spatial_gdf is None or spatial_item is None)
        and icefrac_per_model is not None
        and icefrac_per_model.sizes.get('model', 1) >= 2
    ):
        no_sai_series = regional_aggregation(
            icefrac_per_model * area.broadcast_like(icefrac_per_model), weights, op
        )
        model_spread_2sigma_no_sai = (
            2 * no_sai_series.std(dim='model', ddof=1)
        ).sel(time=mean_no_sai.time)
        if icefrac_with_sai_per_model is not None:
            with_sai_series = regional_aggregation(
                icefrac_with_sai_per_model
                * area.broadcast_like(icefrac_with_sai_per_model),
                weights,
                op,
            )
            # The with-SAI curve is anchored to no-SAI before the SAI split, so
            # anchor its band the same way where the with-SAI series has no data.
            model_spread_2sigma_with_sai = (
                2 * with_sai_series.std(dim='model', ddof=1)
            ).reindex(time=mean_with_sai.time).fillna(model_spread_2sigma_no_sai)

    return _MeanOverTimeResult(
        mean_no_sai=mean_no_sai,
        mean_with_sai=mean_with_sai,
        historical_model_global_mean=historical_model_global_mean,
        masked_regional_map=masked_regional_map,
        masked_regional_delta=masked_regional_delta,
        masked_historical_model=masked_historical_model,
        model_spread_2sigma_no_sai=model_spread_2sigma_no_sai,
        model_spread_2sigma_with_sai=model_spread_2sigma_with_sai,
    )


def _compute_distribution(
    var: str,
    *,
    regional_map: xr.DataArray,
    regional_delta: xr.DataArray,
    historical_model: xr.DataArray | None,
    masked_regional_map: xr.DataArray | None,
    masked_regional_delta: xr.DataArray | None,
    masked_historical_model: xr.DataArray | None,
    decade_start_year: int,
    decade_end_year: int,
    spatial_gdf,  # GeoDataFrame | None — untyped to avoid a module-level geopandas import
    spatial_item: str | None,
) -> dict:
    """Build the no-SAI / with-SAI / historical PDF dict shown as the histogram on the dashboard.

    Bins are variable-dependent: ``[1, 5, 10, 20, 50, ∞]`` for exposure (``above``/``below``)
    variables (with non-positive values dropped); ``[0, 0.1, …, 1.0]`` for icefrac;
    otherwise data-driven 0.1-wide bins spanning the union of the three series. The historical
    series is taken from the 1950–1959 slice of ``historical_model`` (or its masked counterpart
    in regional mode), and for icefrac all three series are restricted to grid cells where
    historical icefrac was nonzero so the histograms compare like with like.
    """
    is_above_below = "above" in var or "below" in var
    is_icefrac = var == "icefrac"

    if spatial_gdf is None or spatial_item is None:
        reg_map = regional_map
        reg_delta = regional_delta
    else:
        reg_map = masked_regional_map
        reg_delta = masked_regional_delta

    reg_map = reg_map.sel(time=slice(decade_start_year, decade_end_year))
    reg_delta = reg_delta.sel(time=slice(decade_start_year, decade_end_year))

    pdf_no_sai = reg_map.values.flatten()
    pdf_with_sai = (reg_map + reg_delta).values.flatten()
    pdf_historical = None

    if historical_model is not None:
        if spatial_gdf is None or spatial_item is None:
            hist_model = historical_model
        else:
            hist_model = masked_historical_model
        pdf_historical = hist_model.sel(time=slice(1950, 1959)).values.flatten()
        if is_icefrac:
            pdf_with_sai = pdf_with_sai[pdf_historical > 0]
            pdf_no_sai = pdf_no_sai[pdf_historical > 0]
            pdf_historical = pdf_historical[pdf_historical > 0]

    pdf_no_sai = pdf_no_sai[~np.isnan(pdf_no_sai)]
    pdf_with_sai = pdf_with_sai[~np.isnan(pdf_with_sai)]

    if is_above_below:
        bins = [1, 5, 10, 20, 50, np.inf]
        pdf_no_sai = pdf_no_sai[pdf_no_sai > 0]
        pdf_with_sai = pdf_with_sai[pdf_with_sai > 0]
    elif is_icefrac:
        bins = np.arange(0, 1.01, 0.1)
    else:
        max_v = np.nanmax(np.concatenate([pdf_no_sai, pdf_with_sai, pdf_historical]))
        min_v = np.nanmin(np.concatenate([pdf_no_sai, pdf_with_sai, pdf_historical]))
        bins = np.arange(min_v, max_v + 0.1, 0.1)

    no_sai_counts, no_sai_bins = np.histogram(pdf_no_sai, bins=bins, density=False)
    with_sai_counts, with_sai_bins = np.histogram(pdf_with_sai, bins=bins, density=False)
    no_sai_counts = no_sai_counts / no_sai_counts.sum()
    with_sai_counts = with_sai_counts / with_sai_counts.sum()

    distribution = {
        "no_sai": {
            "counts": no_sai_counts.tolist(),
            "bin_edges": no_sai_bins.tolist(),
        },
        "with_sai": {
            "counts": with_sai_counts.tolist(),
            "bin_edges": with_sai_bins.tolist(),
        },
    }

    if historical_model is not None:
        pdf_historical = pdf_historical[~np.isnan(pdf_historical)]
        historical_counts, historical_bins = np.histogram(
            pdf_historical, bins=bins, density=False
        )
        historical_counts = historical_counts / historical_counts.sum()
        distribution["historical"] = {
            "counts": historical_counts.tolist(),
            "bin_edges": historical_bins.tolist(),
        }

    return distribution


def _compute_internal_variability(
    var: str,
    *,
    regional_map: xr.DataArray,
    spatial_gdf,  # GeoDataFrame | None — untyped to avoid a module-level geopandas import
    spatial_item: str | None,
    data_dir: Path,
) -> float | int | np.ndarray:
    """Look up the model's internal-variability number used to size the dashboard error band.

    Global mode (``spatial_gdf`` / ``spatial_item`` is ``None``) loads ``model_internal_variability.npy``
    from disk and, for exposure variables, multiplies by total population in millions.
    Regional mode reads a per-region ``variability_<var>`` column from the GeoJSON and, for
    exposure variables, scales by the population inside the selected geometry. NaN/None values
    fall back to ``0``. The return type is a Python scalar (``float`` or numpy ``ndarray.item()``)
    so it serializes cleanly into the output JSON.
    """
    is_exposure = "exposure" in var

    if spatial_gdf is None or spatial_item is None:
        if is_exposure:
            model_internal_variability = np.load(
                data_dir / exposurevar2var[var] / "model_internal_variability.npy"
            )
            population_np = get_population_data(data_dir)
            model_internal_variability = (
                model_internal_variability * population_np.sum() / 1e6
            )
            return model_internal_variability.item()
        return np.load(data_dir / var / "model_internal_variability.npy")

    # Regional mode.
    population_xr = None
    region_mask = None
    if is_exposure:
        var_for_variability = exposurevar2var[var]
        population_np = get_population_data(data_dir)
        population_xr = xr.DataArray(
            population_np,
            dims=('lon', 'lat'),
            coords={'lat': regional_map.lat, 'lon': regional_map.lon},
        )
        # Shift lon values by 180 to align with regional_map.
        population_xr = (
            population_xr.assign_coords(lon=(((population_xr.lon + 180) % 360)))
            .sortby('lon')
            .transpose()
        )
        region_mask = create_mask(
            population_xr, spatial_gdf[spatial_gdf.name == spatial_item]
        )
    else:
        var_for_variability = var

    variability_col = f"variability_{var_for_variability}"
    region_row = spatial_gdf[spatial_gdf.name == spatial_item]
    if variability_col not in region_row.columns:
        raise ValueError(
            f"Column {variability_col} not found in spatial_gdf. "
            f"Available columns: {list(region_row.columns)}"
        )

    model_internal_variability = region_row[variability_col].iloc[0]
    if is_exposure:
        model_internal_variability = (
            model_internal_variability * population_xr.where(region_mask).sum() / 1e6
        )
        model_internal_variability = model_internal_variability.item()

    if model_internal_variability is None or pd.isna(model_internal_variability):
        model_internal_variability = 0
    return model_internal_variability


def get_outputs(
    ssp_scenario: str,
    temp_target: float | None,
    spatial_gdf,  # GeoDataFrame | None — untyped to avoid a module-level geopandas import
    spatial_item: str | None,
    decade_start_year: int,
    decade_end_year: int,
    start_year: int | None,
    ramp_up: int,
    data_dir: str | Path,
    model_dir: str | Path,
    cache_dir: str | Path,
    var: str | list[str] | None = None,
    variable_injection: np.ndarray | None = None,
    output_config: dict = {},
) -> dict:
    """Run the full SAI dashboard pipeline for one (scenario, target, region, decade) request.

    For each variable in ``var`` (or all of ``VAR2INFO`` if ``None``) the function loads the
    projection map, computes the SAI delta, and assembles regional means, p-values, mean-over-time
    series with model-internal-variability bands, and PDF histograms. Optional outputs (controlled
    by ``output_config``) include GeoTIFFs, PNG/SVG plots, and NetCDF exports. The aggregate SO2
    injection time series is appended once after the per-variable loop.

    Parameters
    ----------
    ssp_scenario, temp_target, start_year, ramp_up
        Standard SAI knobs. ``temp_target`` and ``start_year`` may be ``None`` when
        ``variable_injection`` is provided, since the analytic SAI parameters are then ignored.
    spatial_gdf, spatial_item
        Optional GeoDataFrame + region name to restrict aggregation to a single region. Pass
        ``None`` for both to run in global mode. ``spatial_gdf`` is intentionally untyped here
        because GeoPandas is not imported at module scope; pass anything with a ``name`` column.
    decade_start_year, decade_end_year
        Inclusive year range (e.g., 2091, 2100) used for the regional/decadal-mean fields.
    data_dir, model_dir, cache_dir
        Filesystem locations for processed data, fitted models, and the precomputed cache.
        Strings are accepted and converted to :class:`pathlib.Path`.
    var
        A single variable name, a list of names, or ``None`` to process all of ``VAR2INFO``.
    variable_injection
        Optional ``(lat, time)`` SO2 injection array. When provided, switches every helper from
        the analytic ``temp_target``/``start_year`` path to a custom-injection path.
    output_config
        Dict of optional toggles: ``create_geotiffs``, ``create_pngs``, ``create_svgs``,
        ``create_netcdf``, ``output_dir``, ``image_format``, ``dpi``. The keys
        ``verification_minimal`` / ``validation_minimal`` short-circuit the per-variable loop
        after the GeoTIFF/PNG step (used by the verification harness).

    Returns
    -------
    dict
        Per-variable nested dict (``regional_map``, ``regional_mean``, ``mean_over_time``,
        ``distribution``, …) plus top-level ``temp_diff``, ``so2_by_latitude``, ``global_so2``,
        and any ``geotiff_paths`` / ``image_paths`` produced by the optional output stages.
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    cache_dir = Path(cache_dir)

    if var is None:
        variables_to_process = VAR2INFO.keys()
    elif isinstance(var, str):
        variables_to_process = [var]
    else:
        variables_to_process = var

    simple_ssp = REVERSE_FANCY_SSP_TITLES[ssp_scenario]

    regional_temp, regional_temp_no_sai_p_values = get_regional_map(
        "tas", data_dir, model_dir, cache_dir, ssp_scenario
    )
    temp_diff = (
        get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, temp_target)
        if variable_injection is None else None
    )
    historical_obs_data = get_historical_obs_global_mean_temp(data_dir)
    historical_obs_sea_ice = get_historical_obs_sea_ice_extent(data_dir)

    output_data = defaultdict(dict)

    for var in variables_to_process:
        fields = _load_variable_fields(
            var,
            ssp_scenario=ssp_scenario,
            data_dir=data_dir,
            model_dir=model_dir,
            cache_dir=cache_dir,
            regional_temp=regional_temp,
            regional_temp_no_sai_p_values=regional_temp_no_sai_p_values,
        )
        output_data[var]["regional_map"] = fields.regional_map

        delta_result = _compute_regional_delta(
            var,
            fields=fields,
            regional_temp=regional_temp,
            ssp_scenario=ssp_scenario,
            temp_target=temp_target,
            ramp_up=ramp_up,
            start_year=start_year,
            temp_diff=temp_diff,
            variable_injection=variable_injection,
            data_dir=data_dir,
            model_dir=model_dir,
            cache_dir=cache_dir,
        )
        regional_delta = delta_result.regional_delta
        regional_no_sai_p_values = delta_result.regional_no_sai_p_values
        if regional_no_sai_p_values is not None:
            regional_no_sai_p_values = regional_no_sai_p_values.sel(time=decade_end_year)

        regional_sai_p_values = _compute_sai_p_values(
            var,
            fields=fields,
            regional_delta=regional_delta,
            icefrac_with_sai=delta_result.icefrac_with_sai,
            simple_ssp=simple_ssp,
            temp_target=temp_target,
            variable_injection=variable_injection,
            data_dir=data_dir,
            cache_dir=cache_dir,
            decade_end_year=decade_end_year,
        )

        # tas/tasmin/tasmax/pr/p-e maps are pre-industrial anomalies; re-express them against the
        # current-climate baseline for display by subtracting display_offset (recent mean vs
        # 1850-1900). Only the map moves: the time-series and distribution outputs below stay
        # relative to pre-industrial. The SAI delta is a change, so it is left untouched and both the no-SAI map
        # (regional_mean) and the with-SAI map (regional_mean + regional_delta_mean) shift together.
        is_display_anomaly = var in ["tas", "tasmin", "tasmax", "pr", "p-e"]
        display_offset = fields.display_offset if is_display_anomaly else None

        regional_mean = fields.regional_map.sel(
            time=slice(decade_start_year, decade_end_year)
        ).mean('time')
        if display_offset is not None:
            regional_mean = regional_mean - display_offset
        regional_delta_mean = regional_delta.sel(
            time=slice(decade_start_year, decade_end_year)
        ).mean('time')

        if spatial_gdf is not None and spatial_item is not None:
            mask = create_mask(regional_mean, spatial_gdf[spatial_gdf.name == spatial_item])
            output_data[var]["mask"] = mask

        output_data[var]["regional_mean"] = regional_mean
        output_data[var]["regional_delta_mean"] = regional_delta_mean
        output_data[var]["regional_no_sai_p_values"] = regional_no_sai_p_values
        output_data[var]["regional_sai_p_values"] = regional_sai_p_values

        if output_config.get("create_geotiffs", False):
            output_data["geotiff_paths"] = create_geotiff_files(
                var=var,
                regional_mean=regional_mean,
                regional_delta_mean=regional_delta_mean,
                regional_no_sai_p_values=regional_no_sai_p_values,
                regional_sai_p_values=regional_sai_p_values,
                ssp_scenario=ssp_scenario,
                temp_target=temp_target,
                decade_start_year=decade_start_year,
                decade_end_year=decade_end_year,
                start_year=start_year,
                ramp_up=ramp_up,
                variable_injection=variable_injection,
                output_config=output_config,
            )
        if output_config.get("create_pngs", False) or output_config.get("create_svgs", False):
            output_data["image_paths"] = create_vectorized_files(
                var=var,
                regional_mean=regional_mean,
                regional_delta_mean=regional_delta_mean,
                regional_no_sai_p_values=regional_no_sai_p_values,
                regional_sai_p_values=regional_sai_p_values,
                ssp_scenario=ssp_scenario,
                temp_target=temp_target,
                decade_start_year=decade_start_year,
                decade_end_year=decade_end_year,
                start_year=start_year,
                ramp_up=ramp_up,
                variable_injection=variable_injection,
                output_config=output_config,
            )

        # Validation-only: GeoTIFFs (and core regional fields above) are enough; skip
        # mean-over-time, PDFs, NetCDF export, and per-variable plotting work.
        if output_config.get("verification_minimal") or output_config.get("validation_minimal"):
            continue

        if start_year is None:
            start_year = MIN_SAI_START
        recent_regional_map = fields.regional_map.sel(time=slice(2015, start_year - 1))
        regional_map_post = fields.regional_map.sel(time=slice(start_year, SIM_END_YEAR))

        # In the regional + above/below case we clip negatives out of historical_model so the
        # same clipped array feeds both the mean-over-time aggregation and the PDF histogram.
        # In the original code this was an in-place rebind inside the mean-over-time block;
        # hoisting it here makes the data flow visible to the orchestrator.
        is_above_below = "above" in var or "below" in var
        historical_model = fields.historical_model
        if (
            spatial_gdf is not None
            and spatial_item is not None
            and is_above_below
            and historical_model is not None
        ):
            historical_model = historical_model.where(historical_model > 0, 0)

        mot_result = _compute_mean_over_time(
            var,
            regional_map=regional_map_post,
            regional_delta=regional_delta,
            recent_regional_map=recent_regional_map,
            historical_model=historical_model,
            area=fields.area,
            spatial_gdf=spatial_gdf,
            spatial_item=spatial_item,
            start_year=start_year,
            variable_injection=variable_injection,
            data_dir=data_dir,
            icefrac_per_model=fields.icefrac_per_model,
            icefrac_with_sai_per_model=delta_result.icefrac_with_sai_per_model,
        )

        output_data[var]["mean_over_time"] = {
            "no_sai": mot_result.mean_no_sai,
            "with_sai": mot_result.mean_with_sai,
            "historical_model": mot_result.historical_model_global_mean,
        }
        if var == 'tas':
            output_data[var]["mean_over_time"]["historical_obs"] = historical_obs_data
        elif var == 'icefrac':
            output_data[var]["mean_over_time"]["historical_obs"] = historical_obs_sea_ice
        if mot_result.model_spread_2sigma_no_sai is not None:
            # icefrac, global mode: ±2σ across-model band half-widths, one value
            # per year of the corresponding mean-over-time series.
            output_data[var]["mean_over_time"]["model_spread_2sigma_no_sai"] = (
                mot_result.model_spread_2sigma_no_sai
            )
            output_data[var]["mean_over_time"]["model_spread_2sigma_with_sai"] = (
                mot_result.model_spread_2sigma_with_sai
            )

        output_data[var]["mean_over_time"]["model_internal_variability"] = (
            _compute_internal_variability(
                var,
                regional_map=regional_map_post,
                spatial_gdf=spatial_gdf,
                spatial_item=spatial_item,
                data_dir=data_dir,
            )
        )

        output_data[var]["distribution"] = _compute_distribution(
            var,
            regional_map=regional_map_post,
            regional_delta=regional_delta,
            historical_model=historical_model,
            masked_regional_map=mot_result.masked_regional_map,
            masked_regional_delta=mot_result.masked_regional_delta,
            masked_historical_model=mot_result.masked_historical_model,
            decade_start_year=decade_start_year,
            decade_end_year=decade_end_year,
            spatial_gdf=spatial_gdf,
            spatial_item=spatial_item,
        )

        if output_config.get("create_netcdf", False):
            is_exposure = "exposure" in var
            if is_exposure:
                var_internal_variability = np.load(
                    data_dir / exposurevar2var[var] / "model_internal_variability.npy"
                )
            else:
                var_internal_variability = np.load(
                    data_dir / var / "model_internal_variability.npy"
                )
            generate_sai_netcdf(
                var=var,
                recent_regional_map=recent_regional_map,
                regional_map=regional_map_post,
                regional_delta=regional_delta,
                historical_model=historical_model,
                historical_rebase=fields.historical_rebase,
                model_internal_variability=var_internal_variability,
                ssp_scenario=ssp_scenario,
                temp_target=temp_target,
                decade_start_year=decade_start_year,
                decade_end_year=decade_end_year,
                start_year=start_year,
                ramp_up=ramp_up,
                variable_injection=variable_injection,
                output_config=output_config,
            )

    if output_config.get("verification_minimal") or output_config.get("validation_minimal"):
        return output_data

    ### Latitude vs. Tg SO2 plot ###
    if variable_injection is None:
        temp_diff = temp_diff.sel(time=slice(start_year, SIM_END_YEAR))
        # Note: ``var`` here is the last loop-variable, intentionally preserved from the
        # original implementation — the SO2 cache filename is keyed by that incidental name.
        validate_cache_dir(cache_dir)
        so2_cache_path = cache_dir / f"{simple_ssp}_{temp_target:.1f}_so2_{var}.nc"
        if so2_cache_path.exists():
            with xr.open_dataarray(so2_cache_path) as da:
                so2_by_latitude = da.load()
            so2_by_latitude = so2_by_latitude.sel(start_year=start_year)
        else:
            interpolator = get_interpolator(model_dir / "so2" / "interpolator.nc")
            so2_by_latitude = get_so2_by_latitude(interpolator, temp_diff, ramp_up)["so2"]
            # The multi-model-mean temp_diff no longer carries a leftover
            # 'model' coord (the old single-model .sel did) — drop if present.
            so2_by_latitude = so2_by_latitude.drop_vars('model', errors='ignore')
        global_so2 = so2_by_latitude.sum(('lat'))
    else:
        temp_diff = None
        so2_by_latitude = xr.DataArray(
            variable_injection,
            dims=('lat', 'time'),
            coords={
                'lat': [-60, -30, -15, 0, 15, 30, 60],
                'time': np.arange(2035, 2101),
            },
        )
        global_so2 = so2_by_latitude.sum(('lat'))

    output_data["temp_diff"] = temp_diff
    output_data["so2_by_latitude"] = so2_by_latitude
    output_data["global_so2"] = global_so2

    return output_data


def _reproject_to_arctic_stereo(
    data_array: np.ndarray,
    significance_mask: np.ndarray,
    src_transform: "rasterio.transform.Affine",
) -> tuple[np.ndarray, np.ndarray, "rasterio.transform.Affine", "rasterio.crs.CRS"]:
    """Reproject WGS84 arrays to North Polar Stereographic for polar visualization.

    Used by :func:`create_geotiff_files` for ``var == 'icefrac'`` so the GeoTIFF renders as a
    polar-cap view rather than the warped equirectangular projection. The source raster is
    first clipped to the northern hemisphere to avoid GDAL "Invalid latitude" warnings, then
    resampled into a 512×512 destination raster spanning ±6,000,000 m in stereographic meters
    (≈40°N at the edges). Data uses bilinear resampling; the significance mask uses nearest.

    Returns ``(dst_data, dst_sig, dst_transform, dst_crs)`` — both arrays in stereographic
    coordinates, plus the destination affine and CRS for the GeoTIFF writer.
    """
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds, array_bounds

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_proj4(
        '+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    )

    # Clip source to northern hemisphere to avoid "Invalid latitude" warnings
    src_height, src_width = data_array.shape
    src_bounds = array_bounds(src_height, src_width, src_transform)
    if src_bounds[1] < 0:
        pixel_height = abs(src_transform.e)
        rows_to_skip = int(max(0, (0 - src_bounds[1]) / pixel_height))
        clip_row = src_height - rows_to_skip
        data_array = data_array[:clip_row, :]
        significance_mask = significance_mask[:clip_row, :]
        src_transform = from_bounds(
            src_bounds[0], 0, src_bounds[2], src_bounds[3],
            src_width, clip_row
        )

    # Extent in stereographic meters. 6 000 000 m ≈ edges at ~40°N.
    extent = 6_000_000
    dst_size = 512

    dst_transform = from_bounds(-extent, -extent, extent, extent, dst_size, dst_size)

    dst_data = np.full((dst_size, dst_size), np.nan, dtype=np.float32)
    dst_sig = np.full((dst_size, dst_size), np.nan, dtype=np.float32)

    reproject(
        source=data_array,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    reproject(
        source=significance_mask,
        destination=dst_sig,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return dst_data, dst_sig, dst_transform, dst_crs


def _geotiff_coord_step_degrees(coord: xr.DataArray, *, axis_name: str) -> float:
    """Pixel size along one axis; avoids empty reductions on degenerate coordinates."""
    vals = np.asarray(coord.values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"GeoTIFF build: {axis_name} has no finite coordinate values")
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    span = hi - lo
    n = vals.size
    if n == 1:
        return 1.0
    return span / (n - 1)


def create_geotiff_files(
    var: str,
    regional_mean: xr.DataArray,
    regional_delta_mean: xr.DataArray,
    regional_no_sai_p_values: xr.DataArray | None,
    regional_sai_p_values: xr.DataArray | None,
    ssp_scenario: str,
    temp_target: float | None = None,
    decade_start_year: int | None = None,
    decade_end_year: int | None = None,
    start_year: int | None = None,
    ramp_up: int | None = None,
    variable_injection: np.ndarray | None = None,
    output_config: dict = {},
) -> dict[str, str]:
    """Write no-SAI and with-SAI GeoTIFFs for one variable, returning ``{var}_{suffix} → path``.

    Each output is a 2-band GeoTIFF: band 1 holds the data values (``regional_mean`` for
    ``no_sai``, ``regional_mean + regional_delta_mean`` for ``with_sai``), band 2 holds a
    significance mask built by :func:`_significance_mask`. Filenames embed a 10-char hash
    from :func:`_compute_params_hash` so cached files are reused on identical input. For
    ``var == 'icefrac'`` the rasters are reprojected via :func:`_reproject_to_arctic_stereo`
    so they render as a polar cap; all other variables use plain WGS84 longlat.

    The output directory is ``{output_config['output_dir']}/geotiffs/`` (default ``output/``).
    """
    geotiff_paths = {}

    output_dir = Path(output_config.get("output_dir", "output")) / "geotiffs"
    output_dir.mkdir(parents=True, exist_ok=True)

    geotiff_params_hash = _compute_params_hash(
        ssp_scenario, temp_target, decade_start_year, decade_end_year,
        start_year, ramp_up, variable_injection,
    )

    pvals_by_suffix = {
        "no_sai": regional_no_sai_p_values,
        "with_sai": regional_sai_p_values,
    }

    for data, suffix in [(regional_mean, "no_sai"), (regional_mean + regional_delta_mean, "with_sai")]:
        output_filename = f"{var}_{suffix}_{geotiff_params_hash}.tif"
        output_path = output_dir / output_filename

        # Skip if file already exists
        if output_path.exists():
            geotiff_paths[f"{var}_{suffix}"] = str(output_path)
            continue

        # Shift lon to [-180, 180), sort, and flip latitude (north-up)
        data_adjusted = _adjust_polar_lonlat(data)

        # Prepare the data for GeoTIFF
        data_array = data_adjusted.values

        # Get the appropriate significance mask
        significance_mask = _significance_mask(pvals_by_suffix[suffix])
        if significance_mask is None:
            significance_mask = np.full_like(data_array, np.nan, dtype=np.float32)

        # Convert to float32 if it's not already
        if data_array.dtype != np.float32:
            data_array = data_array.astype(np.float32)
        
        # Calculate the correct transform (robust to single-point axes / bad caches)
        lon_res = _geotiff_coord_step_degrees(data_adjusted.lon, axis_name="lon")
        lat_res = _geotiff_coord_step_degrees(data_adjusted.lat, axis_name="lat")
        lon_vals = np.asarray(data_adjusted.lon.values, dtype=np.float64).ravel()
        lon_vals = lon_vals[np.isfinite(lon_vals)]
        lat_vals = np.asarray(data_adjusted.lat.values, dtype=np.float64).ravel()
        lat_vals = lat_vals[np.isfinite(lat_vals)]
        transform = rasterio.transform.from_origin(
            float(np.min(lon_vals)) - lon_res / 2,
            float(np.max(lat_vals)) + lat_res / 2,
            lon_res, lat_res
        )

        # For icefrac, reproject to Arctic LAEA for polar visualization
        geotiff_crs = '+proj=longlat +datum=WGS84 +no_defs'
        if var == 'icefrac':
            data_array, significance_mask, transform, geotiff_crs = \
                _reproject_to_arctic_stereo(data_array, significance_mask, transform)

        # Write to memory first, then save to disk atomically
        with rasterio.MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=data_array.shape[0],
                width=data_array.shape[1],
                count=2,  # Now using 2 bands
                dtype=data_array.dtype,
                crs=geotiff_crs,
                transform=transform,
                compress='deflate',
                zlevel=9,
                predictor=3,
                tiled=False,
                blockxsize=data_array.shape[1],
                blockysize=1,
            ) as dst:
                dst.write(data_array, 1)  # Data values in band 1
                dst.write(significance_mask, 2)  # Significance mask in band 2
            
            # Only write to disk if file doesn't exist
            if not output_path.exists():
                with open(output_path, 'wb') as f:
                    f.write(memfile.read())

        geotiff_paths[f"{var}_{suffix}"] = str(output_path)
    
    return geotiff_paths

@lru_cache(maxsize=32)
def _borders_50m() -> cfeature.NaturalEarthFeature:
    """Cached 50m Natural Earth country-borders feature for cartopy plots.

    Building this feature triggers a one-time download/parse, so memoizing it via
    :func:`functools.lru_cache` keeps repeated :func:`create_vectorized_files` calls fast.
    """
    return cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_0_boundary_lines_land",
        scale="50m",
        facecolor="none",
    )

def _adjust_polar_lonlat(da: xr.DataArray) -> xr.DataArray:
    """Shift lon to [-180, 180), sort, and flip latitude (north-up)."""
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    # flip latitude order if increasing south->north
    # (your original code always flipped; this keeps it robust)
    if da.lat[0] < da.lat[-1]:
        da = da.isel(lat=slice(None, None, -1))
    return da

def _hash_variable_injection(variable_injection: np.ndarray) -> str:
    """Stable 8-char MD5 hash of a 2-D injection array, including its shape.

    The shape is folded into the digest (via a struct-packed prefix) so two arrays with
    identical bytes but different shapes hash differently. Used by :func:`_compute_params_hash`
    when ``variable_injection`` replaces the analytic SAI parameters in the cache key.
    """
    arr = np.asarray(variable_injection, dtype=np.float64)
    rows, cols = arr.shape
    payload = struct.pack('<II', rows, cols) + arr.astype('<f8', copy=False).tobytes()
    return hashlib.md5(payload).hexdigest()[:8]


def _compute_params_hash(
    ssp_scenario: str,
    temp_target: float | None,
    decade_start_year: int | None,
    decade_end_year: int | None,
    start_year: int | None,
    ramp_up: int | None,
    variable_injection: np.ndarray | None,
) -> str:
    """Stable 10-char MD5 cache-key shared by GeoTIFF / vectorized / NetCDF outputs.

    When ``variable_injection`` is ``None`` the key encodes the analytic SAI parameters
    (``ssp_scenario``, ``temp_target``, decade range, ``start_year``, ``ramp_up``).
    Otherwise the injection array is hashed via :func:`_hash_variable_injection` and
    used in place of the analytic params, so two cases that differ only in injection
    produce different keys.

    The display baseline period is folded into the key for both paths so cached map tiles are
    not reused across deployments that present anomalies against a different baseline.
    """
    display_baseline = f"base_{DISPLAY_BASELINE_START}_{DISPLAY_BASELINE_END}"
    if variable_injection is None:
        params = [
            ssp_scenario,
            temp_target,
            decade_start_year,
            decade_end_year,
            start_year,
            ramp_up,
            display_baseline,
        ]
    else:
        variable_hash = _hash_variable_injection(variable_injection)
        params = [
            ssp_scenario,
            decade_start_year,
            decade_end_year,
            f"vinj_{variable_hash}",
            display_baseline,
        ]
    params_str = "_".join(map(str, params))
    return hashlib.md5(params_str.encode()).hexdigest()[:10]


def _significance_mask(p_values_da: xr.DataArray | None) -> np.ndarray | None:
    """Build a float32 mask from a p-values DataArray for hatching non-significant regions.

    The input is reprojected with :func:`_adjust_polar_lonlat` so the mask aligns with
    polar-stereographic data fields. Returns ``1.0`` where ``p >= 0.05`` (not significant)
    and ``NaN`` elsewhere, or ``None`` when ``p_values_da`` is ``None``.
    """
    if p_values_da is None:
        return None
    pv = _adjust_polar_lonlat(p_values_da).values
    return np.where(
        (~np.isnan(pv)) & (pv >= 0.05),
        1.0,
        np.nan,
    ).astype(np.float32)

def create_vectorized_files(
    var: str,
    regional_mean: xr.DataArray,
    regional_delta_mean: xr.DataArray,
    regional_no_sai_p_values: xr.DataArray | None,
    regional_sai_p_values: xr.DataArray | None,
    ssp_scenario: str,
    temp_target: float | None = None,
    decade_start_year: int | None = None,
    decade_end_year: int | None = None,
    start_year: int | None = None,
    ramp_up: int | None = None,
    variable_injection: np.ndarray | None = None,
    output_config: dict | None = None,
) -> dict[str, str]:
    """
    Create image files (PNG or SVG) for visualizing variables over arctic (e.g. sea-ice extent) data.

    This function generates polar stereographic projection plots of climate variables,
    with options for statistical significance hatching and different output formats.

    Args:
        var (str): Name of the climate variable being plotted
        regional_mean (xarray.DataArray): Regional mean values without SAI
        regional_delta_mean (xarray.DataArray): Change in regional mean values due to SAI
        regional_no_sai_p_values (xarray.DataArray, optional): P-values for no-SAI scenario
        regional_sai_p_values (xarray.DataArray, optional): P-values for SAI scenario
        ssp_scenario (str): SSP scenario identifier
        temp_target (float, optional): Temperature target in degrees Celsius
        decade_start_year (int, optional): Start year of the decade being analyzed
        decade_end_year (int, optional): End year of the decade being analyzed
        start_year (int, optional): Year SAI intervention begins
        ramp_up (int, optional): Duration of SAI ramp-up period in years
        variable_injection (numpy.ndarray, optional): Custom injection profile if used
        output_config (dict, optional): Configuration options including:
            - image_format (str): Output format, either 'png' or 'svg' (default: 'png')
            - dpi (int): Resolution in dots per inch (default: 300)
            - output_dir (str): Directory for output files (default: 'output')
        **kwargs: Additional keyword arguments passed through

    Returns:
        dict: Dictionary mapping scenario keys to file paths, with keys in format:
            "{var}_no_sai" and "{var}_with_sai"

    Features:
        - Generates both no-SAI and with-SAI visualizations
        - Uses polar stereographic projection focused on 60°N-90°N
        - Applies hatching to regions without statistical significance
        - Includes coastlines and land features
        - Caches results using hash-based filenames
        - Supports both PNG and SVG output formats

    Production-optimized: less memory, cached features, robust hashing, leak-safe figure handling.
    """
    image_paths = {}

    # ---- config (no mutable defaults) ----
    output_config = output_config or {}
    image_format = output_config.get("image_format", "png").lower()
    dpi = int(output_config.get("dpi", 50))
    output_dir = Path(output_config.get("output_dir", "output")) / f"{image_format}s"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache key (stable + cheap) ----
    plot_params_hash = _compute_params_hash(
        ssp_scenario, temp_target, decade_start_year, decade_end_year,
        start_year, ramp_up, variable_injection,
    )

    # ---- prepare base fields once (avoid extra copies) ----
    # keep in float32 to halve memory bandwidth
    base = _adjust_polar_lonlat(regional_mean.astype(np.float32))
    with_sai = _adjust_polar_lonlat((regional_mean + regional_delta_mean).astype(np.float32))

    lon = base.lon.values
    lat = base.lat.values

    no_sai_mask = _significance_mask(regional_no_sai_p_values)
    with_sai_mask = _significance_mask(regional_sai_p_values)

    # ---- color normalization (reuse) ----
    cmap_blues = plt.cm.Blues_r
    bounds_blues = np.linspace(0, 1, 11)
    norm_blues = colors.BoundaryNorm(boundaries=bounds_blues, ncolors=cmap_blues.N, clip=True)

    for data, suffix, sigmask in [
        (base, "no_sai", no_sai_mask),
        (with_sai, "with_sai", with_sai_mask),
    ]:
        output_path = output_dir / f"{var}_{suffix}_{plot_params_hash}.{image_format}"
        if output_path.exists():
            image_paths[f"{var}_{suffix}"] = str(output_path)
            continue

        # materialize the data just-in-time (keeps xarray advantages until here)
        z = data.values  # float32

        fig = None
        try:
            # Create figure/axes
            fig = plt.figure(figsize=(10, 10), dpi=dpi)
            fig.patch.set_visible(False)
            ax = plt.subplot(111, projection=ccrs.NorthPolarStereo())
            ax.set_frame_on(False)

            # Map features (borders cached)
            # To increase the resolution of both land and borders, use 'scale="50m"' for higher-res:
            highres_land = cfeature.NaturalEarthFeature(
                category='physical',
                name='land',
                scale='50m',    # Higher resolution
                facecolor="gray"
            )
            ax.add_feature(highres_land, zorder=2)
            ax.add_feature(_borders_50m(), edgecolor="black", linewidth=1, zorder=2)
            ax.coastlines(resolution="50m")

            # Use 1D lon/lat (avoids making a meshgrid copy)
            mesh = ax.pcolormesh(
                lon,
                lat,
                z,
                transform=ccrs.PlateCarree(),
                cmap=cmap_blues,
                norm=norm_blues,
                rasterized=True,  # shrinks vector-output size; harmless for PNG
            )

            # Hatching for non-significant regions (if provided)
            if sigmask is not None and not np.all(np.isnan(sigmask)):
                ax.contourf(
                    lon,
                    lat,
                    sigmask,
                    levels=[-0.5, 0.5, 1.5],
                    hatches=["",".."], #["", "///"],
                    colors="none",
                    transform=ccrs.PlateCarree(),
                )

            # Light gridlines (no labels -> faster)
            ax.gridlines(draw_labels=False)
            ax.set_extent([-180, 180, 40, 90], ccrs.PlateCarree())

            plt.tight_layout()
            fig.savefig(output_path, bbox_inches="tight", dpi=dpi, format=image_format)
            image_paths[f"{var}_{suffix}"] = str(output_path)

        finally:
            # Leak-safe: always close, even if save or plotting raises
            if fig is not None:
                plt.close(fig)

    return image_paths

def generate_sai_netcdf(
    var: str,
    recent_regional_map: xr.DataArray,
    regional_map: xr.DataArray,
    regional_delta: xr.DataArray,
    historical_model: xr.DataArray | None,
    historical_rebase: xr.DataArray | None,
    model_internal_variability: np.ndarray | float | int,
    ssp_scenario: str,
    temp_target: float | None,
    decade_start_year: int,
    decade_end_year: int,
    start_year: int | None,
    ramp_up: int | None,
    variable_injection: np.ndarray | None,
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str | Path | None = None,
    output_config: dict | None = None,
) -> xr.Dataset:
    """Write a per-variable NetCDF bundling historical, no-SAI, and with-SAI fields plus metadata.

    The output dataset contains three time-aligned data variables — ``{var}_historical``,
    ``{var}_nosai``, ``{var}_sai`` — and a scalar ``model_std_dev`` for the dashboard error band.
    For temperature/precipitation variables the historical-rebase value is added back so all
    three series are in absolute units; icefrac and exposure variables stay in their natural
    absolute units (fraction, person-days). When ``variable_injection`` is provided, a
    ``so2_injection_amount`` (lat-bin × time) variable is appended.

    The file is gzipped (zlib level 5, float32) and written to
    ``{output_config['output_dir']}/netcdfs/`` (default ``output/``). If ``output_path`` is
    ``None`` the filename is built from :func:`_compute_params_hash` so identical inputs land
    in the same file. Returns the xarray ``Dataset`` that was written, which callers can
    inspect or persist further.
    """

    if output_config is None:
        output_config = {}

    # Determine output directory from output_config
    output_dir = Path(output_config.get("output_dir", "output")) / "netcdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compose output file name if not provided
    if output_path is None:
        netcdf_params_hash = _compute_params_hash(
            ssp_scenario, temp_target, decade_start_year, decade_end_year,
            start_year, ramp_up, variable_injection,
        )
        output_filename = f"{var}_{netcdf_params_hash}.nc"
        output_path = output_dir / output_filename
    else:
        # Ensure Path object for uniformity
        output_path = Path(output_path)

        # If output_path is not absolute and not under the intended output_dir, 
        # resolve it under output_dir
        if not output_path.is_absolute() and not str(output_path).startswith(str(output_dir)):
            output_path = output_dir / output_path

    # 1. Define Metadata Mapping
    var_meta = {
        'tas': {'long_name': 'Near-Surface Air Temperature', 'units': 'K'},
        'p-e': {'long_name': 'Water Availability (Precipitation-Evaporation)', 'units': 'mm/day'},
        'tasmin': {'long_name': 'Daily Minimum Near-Surface Air Temperature', 'units': 'K'},
        'tasmax': {'long_name': 'Daily Maximum Near-Surface Air Temperature', 'units': 'K'},
        'exposure_above_40': {'long_name': 'Person-Days Above 40°C', 'units': 'Millions'},
        'exposure_above_35': {'long_name': 'Person-Days Above 35°C', 'units': 'Millions'},
        'exposure_below_0': {'long_name': 'Person-Days Below 0°C', 'units': 'Millions'},
        'exposure_above_10': {'long_name': 'Person-Days Above 10mm/day', 'units': 'Millions'},
        'exposure_above_20': {'long_name': 'Person-Days Above 20mm/day', 'units': 'Millions'},
        'icefrac': {'long_name': 'Arctic Sea Ice Fraction', 'units': 'fraction'}
    }
    
    meta = var_meta.get(var, {'long_name': var, 'units': 'unknown'})

    # 2. Perform Calculations
    # No-SAI: projection + historical rebase (1850-1900 avg)
    # For icefrac and exposure vars, don't add historical_rebase as they're already absolute values
    is_exposure = "exposure" in var
    is_icefrac = var == "icefrac"

    # Concatenate recent_regional_map and regional_map along time dimension
    concatenated_regional_map = xr.concat([recent_regional_map, regional_map], dim="time")

    if is_icefrac or is_exposure or historical_rebase is None:
        ds_nosai = concatenated_regional_map.rename(f"{var}_nosai")
        ds_sai = (regional_map + regional_delta).rename(f"{var}_sai")
    else:
        ds_nosai = (concatenated_regional_map + historical_rebase).rename(f"{var}_nosai")
        ds_sai = (regional_map + regional_delta + historical_rebase).rename(f"{var}_sai")

    # Historical
    if historical_model is not None:
        # historical_model was converted to anomalies (original - historical_rebase)
        # For consistency with nosai/sai, add historical_rebase back for absolute values
        if is_icefrac or is_exposure or historical_rebase is None:
            ds_hist = historical_model.rename(f"{var}_historical")
        else:
            ds_hist = (historical_model + historical_rebase).rename(f"{var}_historical")
    else:
        # Create a dummy historical with NaN values or skip it
        ds_hist = xr.DataArray(
            np.nan,
            coords=regional_map.coords,
            dims=regional_map.dims,
            name=f"{var}_historical"
        ).sel(time=slice(1850, 2014))  # Historical period

    # 3. Create the Dataset
    # Convert model_internal_variability scalar to DataArray
    if isinstance(model_internal_variability, (int, float, np.ndarray)):
        model_std_dev = xr.DataArray(
            model_internal_variability,
            name="model_std_dev"
        )
    else:
        model_std_dev = model_internal_variability.rename("model_std_dev")

    ds = xr.merge([ds_hist, ds_nosai, ds_sai, model_std_dev])

    # 4. Handle Optional SO2 Injection
    if variable_injection is not None:
        # variable_injection is (lat, time) with time from 2035-2101
        # latitude_bin coordinates are [-60, -30, -15, 0, 15, 30, 60]
        lat_bins = [-60, -30, -15, 0, 15, 30, 60]
        injection_time = np.arange(2035, 2101)
        inj_da = xr.DataArray(
            variable_injection,
            coords={'latitude_bin': lat_bins, 'time': injection_time},
            dims=['latitude_bin', 'time'],
            name="so2_injection_amount",
            attrs={'units': 'Tg SO2', 'long_name': 'Strategic SO2 Injection Amount'}
        )
        ds = xr.merge([ds, inj_da])

    # 5. Set Variable Attributes
    for v in [f"{var}_nosai", f"{var}_sai", f"{var}_historical"]:
        ds[v].attrs['units'] = meta['units']
        ds[v].attrs['long_name'] = meta['long_name']
        if is_icefrac or is_exposure:
            ds[v].attrs['description'] = f"Absolute values of {var}"
        elif historical_rebase is None:
            ds[v].attrs['description'] = f"Projected values of {var} without historical rebase."
        else:
            ds[v].attrs['description'] = "Values calculated as delta from PI (1850-1900) rebased to historical mean."

    ds['model_std_dev'].attrs.update({
        'units': meta['units'],
        'long_name': f'Internal Variability Standard Deviation for {var}'
    })

    # 6. Global Attributes
    ds.attrs = {
        'title': f'Emulated Climate Projections for {meta["long_name"]}',
        'source': f'Emulated output from Reflective simulator based on {model_name}',
        'datetime_start': '1850',
        'datetime_end': '2100',
        'temporal_resolution': 'yearly',
        'spatial_resolution_lon': '1.25°',
        'spatial_resolution_lat': '0.9375°',
        'cooling_target': f'{temp_target}°C above PI' if temp_target is not None else 'N/A (custom variable injection used)',
        'scenario': ssp_scenario,
        'sai_start_year': start_year if start_year is not None else 'N/A (custom variable injection used)',
        'sai_ramp_up_years': ramp_up if ramp_up is not None else 'N/A (custom variable injection used)',
        'decade_start_year': decade_start_year,
        'decade_end_year': decade_end_year,
        'injection_strategy': 'ARISE' if variable_injection is not None else 'Standard',
        'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'paper_ref': 'Beall, Charlotte and Irvin, Jeremy A. and Dexheimer, Jake and Gruener, Dakota and Ng, Andrew Y. and Watson-Parris, Duncan and MacMartin, Douglas G. and Visioni, Daniele and Administrator, Sneak Peek, The Stratospheric Aerosol Injection (SAI) Simulator: An Open-Source Web Tool for Exploring Climate and SAI Deployment Scenarios. ONE-EARTH-D-25-00518, Available at SSRN: https://ssrn.com/abstract=5200736 or http://dx.doi.org/10.2139/ssrn.5200736',
        'data_ref': 'https://doi.org/10.5281/zenodo.15531372',
        'github_ref': 'https://github.com/reflective-org/sai-simulator',
    }

    # 7. Save to NetCDF with compression
    # Set up encoding for compression
    encoding = {}
    for var_name in ds.data_vars:
        encoding[var_name] = {
            'zlib': True,
            'complevel': 5,
            'dtype': 'float32'
        }

    ds.to_netcdf(str(output_path), engine='netcdf4', encoding=encoding)
    logger.info("File saved successfully to %s", output_path)
    return ds
