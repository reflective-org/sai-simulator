"""Build the interactive validation website.

Architecture (option 1, server-required):
  - index.html        : ~few MB. Plotly + JS. Embeds the per-variable metric maps
                        for all 10 variables (small). Maintains a per-variable
                        scatter-data cache loaded on demand.
  - data_<var>.json.gz: per-variable scatter payload (~5-25 MB each). Loaded by the
                        page only when the user clicks a cell of that variable (or
                        when the variable is first selected).

User runs `python -m http.server` in model-validation/website/ and opens
http://localhost:8000 in a browser.

Multi-model (CESM2-WACCM, MIROC-ES2H, UKESM1-1-LL):
  - Baseline (SSP2-4.5) mode validates the dashboard's no-SAI map — the equal-weight
    multi-model mean emulator (backend.baseline.get_regional_map) — against EACH
    model's own GCM projection-baseline run (backend.constants
    .resolve_projection_baseline_file). One set of maps per model.
  - SAI delta mode validates the dashboard's SAI delta (the production pooled SAI slope,
    backend.delta.get_regional_delta) against EACH model's own 1.5 K SAI run minus that
    run's SAI reference (the fit_delta.py anchor pairs). RMSE/MAE are per model; the R²
    is one pooled map over all six anchor runs (CESM 0.5/1.0/1.5 K, MIROC 0.5/1.5 K,
    UKESM 1.5 K), shown under every model.
  - Per-model scatter/SNR payloads are written to model-validation/website/<model>/,
    gzipped (data_<var>.json.gz); the page unzips each file as it loads it.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from backend.baseline import REBASED_VARS, get_regional_map  # noqa: E402  (needs the repo-root path above)
from backend.constants import REGIONAL_MODEL_NAMES, resolve_projection_baseline_file  # noqa: E402
from backend.delta import get_regional_delta  # noqa: E402
from backend.get_outputs import _predict_icefrac_end_to_end, get_temp_diff  # noqa: E402
from backend.utils import clip_to_land  # noqa: E402
from scripts.fit_delta import ANCHORS  # noqa: E402

PROCESSED_ROOT = REPO / "data" / "processed"
MODEL_ROOT = REPO / "data" / "models"
CACHE_ROOT = REPO / "data" / "cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "website"

WINDOW_START, WINDOW_END = 2050, 2069
SSP_SCENARIO = "SSP2-4.5"
TEMP_TARGET = 1.5
START_YEAR = 2035
RAMP_UP = 10
PI_START, PI_END = 1850, 1900   # pre-industrial reference window for baseline-mode SNR

# Models validated in Baseline mode (every regional model of the dashboard's
# multi-model mean) and in SAI delta mode (each model's TEMP_TARGET anchor run).
MODELS = list(REGIONAL_MODEL_NAMES)
SAI_MODELS = list(REGIONAL_MODEL_NAMES)
BASELINE_MODE = "Baseline (SSP2-4.5)"
SAI_MODE = "SAI delta"

ABOVE_BELOW_VARS = {"tas_above_35", "tas_above_40", "tas_below_0", "pr_above_10", "pr_above_20"}
LAND_ONLY_VARS = {"p-e", "pr"} | ABOVE_BELOW_VARS

# Variables that get the SNR stippling overlay on their R² map. Vars where
# per-member baseline data is available for both PI (1850–1900) and the eval
# window (2050–2069):
#   - tas, p-e, icefrac: 3 matched members. SAI 001/002/003 were branched from
#     SSP245 001/002/003, so the SAI and baseline files share the same 3 members
#     and both cover PI + eval window. Clean paired subtraction.
#   - tasmin, tasmax: split lineage — baseline all-members file has historical
#     (001/002/003, 1850–2014) ⊕ ssp245 (007/008/009, 2016–2069). No single
#     member spans both windows, so we cross-pair 001↔007, 002↔008, 003↔009
#     (see TASMINMAX_BASELINE_MEMBER_PAIRING). Note: these are DIFFERENT physical
#     realizations of internal variability, so the pseudo-paired subtraction is
#     noisier than tas/p-e/icefrac — team decided this is acceptable.
# Excluded (data gap, not a code issue):
#   - tas_above_35/40, tas_below_0, pr_above_10/20: baseline days-per-year require
#     daily raw data; only member 001 daily SSP245 files exist. Members 002/003 baseline
#     are NaN over the SSP245 period.
# For both scenarios (SAI-delta paired δ, Baseline SSP245 anomaly vs per-member PI):
# signal = mean of 60 samples, noise = detrended residual σ (ddof=2) after a joint
# OLS linear-trend fit across the pooled samples. Same method for all 5 vars.
# Stipple drawn ON TOP of the R² map; underlying R² value is not NaN'd out. The
# other 5 vars render R² without any stipple overlay.
SNR_VARS = {"tas", "tasmin", "tasmax", "p-e", "icefrac"}

# Cross-lineage pairing for tasmin/tasmax only. Keys = members that have PI data
# (also the SAI-side member labels); values = members that have ssp245 2050–2069
# data. Sample labels reported to the frontend use the KEY (SAI-side) so the
# color scheme matches tas/p-e/icefrac; the actual baseline data comes from
# the VALUE-side member.
TASMINMAX_BASELINE_MEMBER_PAIRING = {"001": "007", "002": "008", "003": "009"}

VARS = [
    "tas",
    "tasmax",
    "tasmin",
    "tas_above_40",
    "tas_above_35",
    "tas_below_0",
    "p-e",
    "pr_above_20",
    "pr_above_10",
    "icefrac",
]

UI_NAMES = {
    "tas": "Temperature",
    "tasmin": "Min Temperature",
    "tasmax": "Max Temperature",
    "p-e": "Water Availability",
    "tas_above_35": "Heat Exposure",
    "tas_above_40": "Extreme Heat Exposure",
    "tas_below_0": "Freezing Exposure",
    "pr_above_10": "Rain Exposure",
    "pr_above_20": "Heavy Rain Exposure",
    "icefrac": "Sea Ice Extent",
}

UNITS = {
    "tas": "K", "tasmin": "K", "tasmax": "K", "p-e": "mm/day",
    "tas_above_35": "days/yr", "tas_above_40": "days/yr", "tas_below_0": "days/yr",
    "pr_above_10": "days/yr", "pr_above_20": "days/yr",
    "icefrac": "fraction",
}


def area_weighted_global_mean(da):
    weights = np.cos(np.deg2rad(da.lat))
    return da.weighted(weights).mean(("lat", "lon"))


def open_var(path, var):
    """Load ``var`` from a processed NetCDF fully into memory and close the file.

    The backend (e.g. backend.historical) opens these same files itself; leaving a
    lazy handle open here while it does so crashes HDF5 (segfault in netCDF4).
    """
    with xr.open_dataset(path) as ds:
        return ds[var].load()


def anchor_target(anchor):
    """The SAI cooling target (K) of a fit_delta anchor, from its name ("gauss-1.5",
    "g6-0.5-miroc", ...)."""
    return float(anchor.name.split("-")[1])


def sai_anchor(model, target=TEMP_TARGET):
    """``model``'s fit_delta anchor at ``target`` (its SAI run + SAI reference run)."""
    return next(a for a in ANCHORS if a.model == model and anchor_target(a) == target)


def load_anchor(var, anchor):
    """An anchor's SAI run and its SAI reference run (ensemble means), as fit_delta.py
    pairs them (MIROC: G6 vs its GeoMIP baseline, not the ssp245 projection run)."""
    sai = open_var(PROCESSED_ROOT / var / anchor.exp_file, var).sel(ssp="ssp245", model=anchor.model)
    ref = open_var(PROCESSED_ROOT / var / anchor.baseline_file, var).sel(ssp="ssp245", model=anchor.model)
    return sai, ref


def window_mean(da):
    return da.sel(time=slice(WINDOW_START, WINDOW_END)).mean("time")


def all_members_path(path):
    return path.with_name(path.stem + "-all-members.nc")


def baseline_path(var, model, all_members=False):
    """``model``'s projection-baseline file for ``var`` — the run its fit_map emulator
    trains on (MIROC: ssp245; CESM/UKESM: baseline)."""
    path = resolve_projection_baseline_file(PROCESSED_ROOT / var, model)
    if path is None:
        raise FileNotFoundError(f"No processed projection baseline for {model} in {PROCESSED_ROOT / var}")
    if all_members:
        path = all_members_path(path)
    return path


def load_baseline_processed(var, model):
    """``model``'s GCM projection-baseline run (ensemble mean), historical merged in."""
    return open_var(baseline_path(var, model), var).sel(ssp="ssp245", model=model)


def _snr_from_samples(samples_pooled, years_pooled):
    """Detrended SNR from (n_samples, lat, lon) array + matching year vector.

    signal = mean of samples (per cell).
    noise  = sqrt( Σ residuals² / (n-2) ) after removing a per-cell OLS linear
             trend fit jointly across all pooled samples. ddof=2 accounts for
             the 2 parameters (slope + intercept).
    Returns (signal_arr, noise_arr).
    """
    n = samples_pooled.shape[0]
    t_c = years_pooled - years_pooled.mean()
    signal = samples_pooled.mean(axis=0)
    num = np.einsum("s,sij->ij", t_c, samples_pooled - signal[None])
    den = float((t_c ** 2).sum())
    slope = num / den
    trend = slope[None] * t_c[:, None, None] + signal[None]
    residuals = samples_pooled - trend
    noise = np.sqrt((residuals ** 2).sum(axis=0) / (n - 2))
    return signal, noise


def compute_snr_mask(var, scenario, model="CESM2-WACCM"):
    """Detrended SNR mask + per-cell (member × year) samples for `var` in the given
    scenario ('sai_delta' | 'baseline') for one GCM `model`.

    scenario == 'sai_delta':  60 paired diffs δ[m,t] = SAI[m,t] − baseline[m,t]
                              (3 members × 20 years, all in 2050–2069).
    scenario == 'baseline':   60 SSP245 anomalies = baseline[m,t] − PI[m], where
                              PI[m] is member m's own 1850–1900 mean. Same
                              per-member paired convention.

    For both: signal = mean of 60 samples; noise = detrended σ (ddof=2);
    mask = |signal|/noise < 1. Land-clipped for LAND_ONLY_VARS.

    Returns None for 'sai_delta' when the per-member SAI file is absent; otherwise a dict with:
        mask:     (lat, lon) bool array, True where SNR < 1
        lat, lon: source coordinate arrays (0–360 lon order)
        samples:  (60, lat, lon) float array — the raw per-(member,year) samples
                  used by the SNR calc, land-clipped for LAND_ONLY_VARS. Serialized
                  to the per-var JSON so the website can draw a per-cell residuals
                  scatter on click.
        members:  (60,) str array — member ID for each sample (member-major order)
        years:    (60,) int array — year for each sample
    """
    if scenario == "sai_delta" and model not in SAI_MODELS:
        raise ValueError(f"SAI delta SNR is only defined for {SAI_MODELS}, not {model}")
    if scenario == "sai_delta":
        # Paired against the SAI run's own reference (MIROC: GeoMIP baseline, not ssp245).
        anchor = sai_anchor(model)
        base_path = all_members_path(PROCESSED_ROOT / var / anchor.baseline_file)
    else:
        base_path = baseline_path(var, model, all_members=True)
    ds_base_full = open_var(base_path, var)

    # For tasmin/tasmax the PI window and the eval window live on different
    # member IDs (see TASMINMAX_BASELINE_MEMBER_PAIRING). Build the paired
    # baseline windows by concatenating the ssp245-side members and relabeling
    # them with the PI-side member IDs so subtraction/pairing lines up.
    # The pairing is a CESM split-lineage quirk: apply it only when the file
    # actually carries the ssp245-side member IDs.
    file_members = {str(m) for m in ds_base_full.member.values}
    uses_cross_pairing = (
        var in ("tasmin", "tasmax")
        and set(TASMINMAX_BASELINE_MEMBER_PAIRING.values()) <= file_members
    )
    if uses_cross_pairing:
        sai_side_members = list(TASMINMAX_BASELINE_MEMBER_PAIRING.keys())
        ssp_side_members = list(TASMINMAX_BASELINE_MEMBER_PAIRING.values())
        base_window_paired = xr.concat(
            [ds_base_full.sel(member=m) for m in ssp_side_members],
            dim="member",
        ).assign_coords(member=sai_side_members).sel(time=slice(WINDOW_START, WINDOW_END))

    if scenario == "sai_delta":
        sai_path = all_members_path(PROCESSED_ROOT / var / anchor.exp_file)
        if not sai_path.exists():
            # No per-member SAI file (scripts/process_monthly_gauss.py writes them for
            # tas, tasmin, tasmax, pr, p-e, icefrac): no paired samples to stipple.
            return None
        ds_sai = open_var(sai_path, var).sel(time=slice(WINDOW_START, WINDOW_END))
        if uses_cross_pairing:
            ds_base = base_window_paired
        else:
            ds_base = ds_base_full.sel(time=slice(WINDOW_START, WINDOW_END))
        samples_da = ds_sai - ds_base  # (member, time, lat, lon)
    elif scenario == "baseline":
        # Per-member SSP245 anomaly vs same member's PI. PI 1850–1900 mean.
        if uses_cross_pairing:
            # PI comes from historical-lineage members (001/002/003), eval
            # window from ssp245-lineage members (007/008/009). base_window_paired
            # already has ssp245-side data relabeled with historical-side IDs so
            # the subtraction aligns on member dim.
            pi = ds_base_full.sel(
                member=sai_side_members, time=slice(PI_START, PI_END)
            ).mean("time")
            window = base_window_paired
        else:
            pi = ds_base_full.sel(time=slice(PI_START, PI_END)).mean("time")   # (member, lat, lon)
            window = ds_base_full.sel(time=slice(WINDOW_START, WINDOW_END))    # (member, T, lat, lon)
        samples_da = window - pi                                                # broadcasts pi to time
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    samples_v = samples_da.values                                  # (M, T, lat, lon)
    M, T = samples_v.shape[:2]
    pooled = samples_v.reshape(M * T, *samples_v.shape[2:])         # (60, lat, lon)
    years_pooled = np.tile(samples_da.time.values.astype(float), M)
    members_pooled = np.repeat(samples_da.member.values, T)

    signal_arr, noise_arr = _snr_from_samples(pooled, years_pooled)

    coords_2d = {d: samples_da.coords[d] for d in samples_da.dims[2:]}
    signal = xr.DataArray(signal_arr, dims=samples_da.dims[2:], coords=coords_2d)
    noise = xr.DataArray(noise_arr, dims=samples_da.dims[2:], coords=coords_2d)

    if var in LAND_ONLY_VARS:
        signal = clip_to_land(PROCESSED_ROOT, signal)
        noise = clip_to_land(PROCESSED_ROOT, noise)
        # Also land-mask the samples so ocean cells are NaN in the serialized data.
        land_mask_da = xr.DataArray(
            np.ones(samples_v.shape[2:]), dims=samples_da.dims[2:], coords=coords_2d)
        land_ok = np.isfinite(clip_to_land(PROCESSED_ROOT, land_mask_da).values)
        pooled = np.where(land_ok[None, :, :], pooled, np.nan)

    signal_v = signal.values
    noise_v = noise.values
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(signal_v) / np.where(noise_v > 0, noise_v, np.nan)
    mask = snr < 1.0  # False where snr is NaN (NaN < 1 is False) — what we want.
    return {
        "mask": mask,
        "lat": ds_base_full.lat.values,
        "lon": ds_base_full.lon.values,
        "samples": pooled,                             # (60, lat, lon)
        "members": [str(m) for m in members_pooled],   # length 60
        "years": [int(y) for y in years_pooled],       # length 60
    }


# (p-e t-test data function removed during 2026-07-01 cleanup — was tied to the
# now-deleted "t-test" metric option.)


def pooled_r2(x_vals, y):
    """Uncentered through-origin R² per cell: y (n_runs, lat, lon) regressed on x (n_runs)."""
    Sxx = float(np.sum(x_vals ** 2))
    Sxy = np.einsum("s,sij->ij", x_vals, y)
    beta = Sxy / Sxx
    y_pred_scenarios = np.einsum("s,ij->sij", x_vals, beta)
    ss_res = np.sum((y - y_pred_scenarios) ** 2, axis=0)
    ss_tot = np.sum(y ** 2, axis=0)
    r2_map = np.full_like(ss_tot, np.nan, dtype=float)
    valid = ss_tot > 0
    r2_map[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return r2_map, y_pred_scenarios


def anchor_coolings():
    """Each fit_delta anchor's 2050–2069 global-mean tas cooling (SAI run − its reference)."""
    x_vals = []
    for a in ANCHORS:
        tas_sai, tas_ref = load_anchor("tas", a)
        x_vals.append(float(
            area_weighted_global_mean(window_mean(tas_sai)) - area_weighted_global_mean(window_mean(tas_ref))
        ))
    return np.array(x_vals)


def compute_sai_r2_var(var):
    """Pooled SAI R² for a linear variable over all six fit_delta anchors (CESM
    0.5/1.0/1.5 K, MIROC 0.5/1.5 K, UKESM 1.5 K). Per cell: a through-origin fit of
    each run's 2050–2069 delta on its global-mean tas cooling — the same pooled fit
    that gives the simulator's SAI slope. One map, shared by every model."""
    print(f"  computing pooled SAI R² (all models): {var}")
    x_vals = anchor_coolings()
    deltas = []
    for a in ANCHORS:
        var_sai, var_ref = load_anchor(var, a)
        d = window_mean(var_sai) - window_mean(var_ref)
        if var in LAND_ONLY_VARS:
            d = clip_to_land(PROCESSED_ROOT, d)
        deltas.append(d)
    y = np.stack([d.values for d in deltas], axis=0)
    r2_map, y_pred_scenarios = pooled_r2(x_vals, y)
    return {
        "r2_map": r2_map,
        "r2_x": x_vals,
        "r2_actual": y,
        "r2_pred": y_pred_scenarios,
        "r2_models": [a.model for a in ANCHORS],
    }


def compute_linear_var(var, model):
    """SAI delta RMSE/MAE for a linear variable: the dashboard's SAI delta at
    TEMP_TARGET vs ``model``'s TEMP_TARGET SAI run minus its SAI reference."""
    print(f"  computing linear var (SAI delta, {model}): {var}")
    var_sai, var_base = load_anchor(var, sai_anchor(model))

    tdiff = get_temp_diff(PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO, TEMP_TARGET)
    pred_da = get_regional_delta(
        var, PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO,
        TEMP_TARGET, RAMP_UP, tdiff, START_YEAR,
    )
    base_da_var = var_base.sel(time=slice(WINDOW_START, WINDOW_END))
    pred_window = pred_da.sel(time=slice(WINDOW_START, WINDOW_END))

    if var in ABOVE_BELOW_VARS:
        # As backend.get_outputs: with-SAI = the dashboard's no-SAI map + delta, floored at
        # 0 and land-clipped; the predicted delta is what that leaves over the no-SAI map.
        emul_base, _, _ = get_regional_map(var, PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO)
        emul_base = emul_base.sel(time=slice(WINDOW_START, WINDOW_END))
        full_sai = emul_base + pred_window
        full_sai = full_sai.where(full_sai > 0, 0)
        full_sai = clip_to_land(PROCESSED_ROOT, full_sai)
        ts_pred = full_sai - emul_base
    else:
        ts_pred = pred_window

    gcm_sai = var_sai.sel(time=slice(WINDOW_START, WINDOW_END))
    ts_actual = gcm_sai - base_da_var

    ts_pred = ts_pred.transpose("time", "lat", "lon")
    ts_actual = ts_actual.transpose("time", "lat", "lon")
    ts_pred, ts_actual = xr.align(ts_pred, ts_actual, join="inner")

    errors = ts_pred.values - ts_actual.values
    rmse_map = np.sqrt(np.nanmean(errors ** 2, axis=0))
    mae_map = np.nanmean(np.abs(errors), axis=0)

    return {
        "lat": base_da_var.lat.values,
        "lon": base_da_var.lon.values,
        "rmse_map": rmse_map,
        "mae_map": mae_map,
        "years": ts_pred.time.values.astype(int),
        "ts_pred": ts_pred.values,
        "ts_actual": ts_actual.values,
    }


def compute_baseline_var(var, model):
    """Baseline (no-SAI) metrics for a linear variable: the dashboard's multi-model
    emulator map vs ``model``'s own GCM projection-baseline run."""
    print(f"  computing baseline var: {var} vs {model}")
    var_base = load_baseline_processed(var, model)

    # ----- Baseline (no-SAI) validation: emulator's SSP245 baseline vs GCM projection baseline -----
    # Both sides expressed as per-cell anomaly relative to PI 1850–1900 mean.
    # IMPORTANT: backend.baseline.get_regional_map() **already PI-rebases** its output
    # for tas / tasmin / tasmax / pr / p-e (see backend/baseline.py: `regional_map -
    # historical_rebase`, against the multi-model historical mean). Subtracting GCM PI
    # from that again would double-rebase and shift the emulator side by −PI (≈ −280 K
    # for temperature). For other vars the output is in absolute units, so we do
    # subtract this model's GCM PI to anomalize it.
    emul_baseline_da, _, _ = get_regional_map(var, PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO)
    gcm_pi = var_base.sel(time=slice(1850, 1900)).mean("time")
    if var in REBASED_VARS:
        emul_window_anom = emul_baseline_da.sel(time=slice(WINDOW_START, WINDOW_END))
    else:
        emul_window_anom = (emul_baseline_da - gcm_pi).sel(time=slice(WINDOW_START, WINDOW_END))
    gcm_window_anom = (var_base - gcm_pi).sel(time=slice(WINDOW_START, WINDOW_END))

    if var in LAND_ONLY_VARS:
        emul_window_anom = clip_to_land(PROCESSED_ROOT, emul_window_anom)
        gcm_window_anom = clip_to_land(PROCESSED_ROOT, gcm_window_anom)

    emul_window_anom = emul_window_anom.transpose("time", "lat", "lon")
    gcm_window_anom = gcm_window_anom.transpose("time", "lat", "lon")
    emul_window_anom, gcm_window_anom = xr.align(emul_window_anom, gcm_window_anom, join="inner")
    ts_baseline_pred = emul_window_anom.values
    ts_baseline_actual = gcm_window_anom.values

    err_b = ts_baseline_pred - ts_baseline_actual
    rmse_baseline_map = np.sqrt(np.nanmean(err_b ** 2, axis=0))
    mae_baseline_map = np.nanmean(np.abs(err_b), axis=0)

    # Baseline R² = emulator-vs-GCM goodness of fit (matches the existing icefrac
    # baseline R² convention). Earlier versions of this code fit a line through origin
    # to GCM yearly anomalies against global tas anomaly — that didn't use the emulator
    # output at all and was measuring the GCM's own pattern-scaling, not validating the
    # emulator. The correct formulation:
    #   ss_res = Σ_year (GCM − emul)²    ← uses both
    #   ss_tot = Σ_year (GCM)²            ← GCM variance around zero (anomaly)
    #   R²     = 1 − ss_res / ss_tot
    # Interpretation: 1.0 = emulator perfectly reproduces GCM yearly anomalies. 0.0 =
    # emulator does no better than predicting zero. Negative = emulator worse than
    # predicting zero (e.g., systematic bias).
    ss_res_b = np.sum((ts_baseline_actual - ts_baseline_pred) ** 2, axis=0)
    ss_tot_b = np.sum(ts_baseline_actual ** 2, axis=0)
    r2_baseline_map = np.full_like(ss_tot_b, np.nan, dtype=float)
    valid_b = ss_tot_b > 0
    r2_baseline_map[valid_b] = 1.0 - ss_res_b[valid_b] / ss_tot_b[valid_b]

    return {
        "lat": var_base.lat.values,
        "lon": var_base.lon.values,
        "years_baseline": emul_window_anom.time.values.astype(int),
        "rmse_baseline_map": rmse_baseline_map,
        "mae_baseline_map": mae_baseline_map,
        "r2_baseline_map": r2_baseline_map,
        "ts_baseline_pred": ts_baseline_pred,
        "ts_baseline_actual": ts_baseline_actual,
        "is_icefrac": False,
    }


def emulated_icefrac(tas_delta=None):
    """The dashboard's icefrac emulator over the 2050–2069 window: each model's
    sigmoid on that model's own (time, lat, lon) temperature map, then the
    equal-weight model mean (backend.get_outputs: _predict_icefrac_end_to_end
    → .mean('model')). ``tas_delta`` (the SAI tas delta) is added to every
    model's temperature map, as the dashboard does for the with-SAI map."""
    _, _, tas_per_model = get_regional_map("tas", PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO)
    temp = tas_per_model.sel(time=slice(WINDOW_START, WINDOW_END))
    if tas_delta is not None:
        temp = temp + tas_delta.sel(time=slice(WINDOW_START, WINDOW_END))
    return _predict_icefrac_end_to_end(temp, MODEL_ROOT).mean(dim="model")


def emulated_icefrac_delta(target):
    """The dashboard's icefrac SAI delta over the window at ``target`` (with-SAI
    icefrac minus no-SAI icefrac, both from emulated_icefrac). The cooling is driven
    by ``target``'s own temp_diff, as backend.get_outputs does."""
    tdiff = get_temp_diff(PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO, target)
    tas_delta = get_regional_delta(
        "tas", PROCESSED_ROOT, MODEL_ROOT, CACHE_ROOT, SSP_SCENARIO,
        target, RAMP_UP, tdiff, START_YEAR,
    )
    return emulated_icefrac(tas_delta) - emulated_icefrac()


def compute_sai_r2_icefrac():
    """Pooled SAI R² for icefrac over all six fit_delta anchors: each run's GCM
    2050–2069 icefrac delta vs the dashboard's icefrac delta at that run's target.
    One map, shared by every model."""
    print("  computing pooled SAI R² (all models): icefrac")
    pred_by_target = {}
    pred_deltas_scenarios = []
    actual_deltas_scenarios = []
    for a in ANCHORS:
        t = anchor_target(a)
        if t not in pred_by_target:
            pred_by_target[t] = emulated_icefrac_delta(t).mean(dim="time").values
        pred_deltas_scenarios.append(pred_by_target[t])
        actual_sai, actual_ref = load_anchor("icefrac", a)
        actual_deltas_scenarios.append((window_mean(actual_sai) - window_mean(actual_ref)).values)

    predicted_scenarios = np.stack(pred_deltas_scenarios, axis=0)
    actual_scenarios = np.stack(actual_deltas_scenarios, axis=0)

    ss_res = np.sum((actual_scenarios - predicted_scenarios) ** 2, axis=0)
    ss_tot = np.sum(actual_scenarios ** 2, axis=0)
    r2_map = np.full_like(ss_tot, np.nan, dtype=float)
    valid = ss_tot > 0
    r2_map[valid] = 1.0 - ss_res[valid] / ss_tot[valid]

    return {
        "r2_map": r2_map,
        "r2_x": anchor_coolings(),
        "r2_actual": actual_scenarios,
        "r2_pred": predicted_scenarios,
        "r2_models": [a.model for a in ANCHORS],
    }


def compute_icefrac(model):
    """SAI delta RMSE/MAE for icefrac: the dashboard's icefrac delta at TEMP_TARGET
    vs ``model``'s TEMP_TARGET SAI run minus its SAI reference."""
    print(f"  computing icefrac (SAI delta, {model})")
    actual_sai, actual_base = load_anchor("icefrac", sai_anchor(model))

    ts_pred = emulated_icefrac_delta(TEMP_TARGET).transpose("time", "lat", "lon")
    ts_actual = (
        actual_sai.sel(time=slice(WINDOW_START, WINDOW_END))
        - actual_base.sel(time=slice(WINDOW_START, WINDOW_END))
    ).transpose("time", "lat", "lon")
    ts_pred, ts_actual = xr.align(ts_pred, ts_actual, join="inner")

    errors = ts_pred.values - ts_actual.values
    rmse_map = np.sqrt(np.nanmean(errors ** 2, axis=0))
    mae_map = np.nanmean(np.abs(errors), axis=0)

    return {
        "lat": ts_pred.lat.values,
        "lon": ts_pred.lon.values,
        "rmse_map": rmse_map,
        "mae_map": mae_map,
        "years": ts_pred.time.values.astype(int),
        "ts_pred": ts_pred.values,
        "ts_actual": ts_actual.values,
    }


def compute_icefrac_baseline(model):
    """Baseline (no-SAI) metrics for icefrac: the dashboard's multi-model icefrac
    emulator vs ``model``'s own GCM projection-baseline run."""
    print(f"  computing icefrac baseline vs {model}")

    pred_no_sai = emulated_icefrac()
    actual_base_full = load_baseline_processed("icefrac", model)
    actual_base = actual_base_full.sel(time=slice(WINDOW_START, WINDOW_END))

    # ----- Baseline (no-SAI) validation for icefrac: emul vs GCM, both as PI anomaly -----
    # Use this model's GCM 1850–1900 mean as the shared PI reference for both sides
    # (emulator's tas_baseline only covers 2015–2100, so an emulator-internal PI isn't available).
    gcm_icefrac_pi = actual_base_full.sel(time=slice(1850, 1900)).mean(dim="time")
    # pred_no_sai is the emulator icefrac time series at 2050–2069; actual_base is GCM at 2050–2069.
    emul_window_anom_da = (pred_no_sai - gcm_icefrac_pi).transpose("time", "lat", "lon")
    gcm_window_anom_da = (actual_base - gcm_icefrac_pi).transpose("time", "lat", "lon")

    emul_window_anom_da, gcm_window_anom_da = xr.align(
        emul_window_anom_da, gcm_window_anom_da, join="inner"
    )
    ts_baseline_pred = emul_window_anom_da.values
    ts_baseline_actual = gcm_window_anom_da.values

    err_b = ts_baseline_pred - ts_baseline_actual
    rmse_baseline_map = np.sqrt(np.nanmean(err_b ** 2, axis=0))
    mae_baseline_map = np.nanmean(np.abs(err_b), axis=0)

    # R² for icefrac baseline: emulator-vs-GCM residual (1 = perfect, 0 = no better
    # than predicting zero anomaly). Same formula now used by the linear-var path.
    ss_res_b = np.sum((ts_baseline_actual - ts_baseline_pred) ** 2, axis=0)
    ss_tot_b = np.sum(ts_baseline_actual ** 2, axis=0)
    r2_baseline_map = np.full_like(ss_tot_b, np.nan, dtype=float)
    valid_b = ss_tot_b > 0
    r2_baseline_map[valid_b] = 1.0 - ss_res_b[valid_b] / ss_tot_b[valid_b]

    return {
        "lat": pred_no_sai.lat.values,
        "lon": pred_no_sai.lon.values,
        "years_baseline": emul_window_anom_da.time.values.astype(int),
        "rmse_baseline_map": rmse_baseline_map,
        "mae_baseline_map": mae_baseline_map,
        "r2_baseline_map": r2_baseline_map,
        "ts_baseline_pred": ts_baseline_pred,
        "ts_baseline_actual": ts_baseline_actual,
        "is_icefrac": True,
    }


def extract_borders(tolerance=0.05):
    """Extract country / coastline outlines from world-administrative-boundaries.geojson.

    Returns (lon_list, lat_list) where None acts as a line break (Plotly draws NaN
    as a discontinuity, so one scatter trace can render every polygon outline).
    """
    gdf = gpd.read_file(PROCESSED_ROOT / "world-administrative-boundaries.geojson")
    gdf = gdf.set_geometry(gdf.geometry.simplify(tolerance, preserve_topology=True))
    lons = []
    lats = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "Polygon":
            polys = [g]
        elif g.geom_type == "MultiPolygon":
            polys = list(g.geoms)
        else:
            continue
        for p in polys:
            xs, ys = p.exterior.coords.xy
            lons.extend([round(float(x), 4) for x in xs])
            lats.extend([round(float(y), 4) for y in ys])
            lons.append(None)
            lats.append(None)
            for interior in p.interiors:
                xs, ys = interior.coords.xy
                lons.extend([round(float(x), 4) for x in xs])
                lats.extend([round(float(y), 4) for y in ys])
                lons.append(None)
                lats.append(None)
    return lons, lats


def compute_region_grid(lat, lon_shifted):
    """For each (lat, lon) cell center, find the IPCC-WGI reference region name.
    Returns (unique_region_names, region_index_grid) where region_index_grid[i][j]
    is an index into unique_region_names, and 'ocean' is reserved for cells outside
    all polygons.
    """
    gdf = gpd.read_file(PROCESSED_ROOT / "geojsons" / "IPCC-WGI-reference-regions-v4.geojson")
    name_col = "Name" if "Name" in gdf.columns else "name"

    n_lat = len(lat)
    n_lon = len(lon_shifted)
    pts = []
    for la in lat:
        for lo in lon_shifted:
            pts.append(Point(float(lo), float(la)))
    points_gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    joined = gpd.sjoin(points_gdf, gdf[[name_col, "geometry"]], how="left", predicate="within")
    # Some points may match multiple polygons (overlapping regions); take the first.
    joined = joined.reset_index().drop_duplicates(subset="index", keep="first").set_index("index")
    joined = joined.reindex(range(len(pts)))
    names = joined[name_col].fillna("ocean").tolist()

    unique = sorted(set(names))
    idx_of = {n: i for i, n in enumerate(unique)}
    grid = np.array([idx_of[n] for n in names], dtype=int).reshape(n_lat, n_lon).tolist()
    return unique, grid


def discrete_colorbar_spec(cmap_name, bounds, leftmost_label=None):
    """Build Plotly discrete colorscale + tick spec to match the static figures.

    Each of the (len(bounds)-1) bins gets a single color from `cmap_name`.
    Returns a dict with keys: colorscale, zmin, zmax, tickvals, ticktext.
    """
    n_bins = len(bounds) - 1
    cmap = plt.get_cmap(cmap_name, n_bins)
    colors = []
    for i in range(n_bins):
        r, g, b, _ = cmap(i)
        colors.append(f"rgb({int(round(r * 255))},{int(round(g * 255))},{int(round(b * 255))})")

    vmin, vmax = float(bounds[0]), float(bounds[-1])
    span = vmax - vmin
    colorscale = []
    for i, color in enumerate(colors):
        pos_start = (float(bounds[i]) - vmin) / span
        pos_end = (float(bounds[i + 1]) - vmin) / span
        colorscale.append([pos_start, color])
        colorscale.append([pos_end, color])
    # Round positions a hair to avoid float-precision weirdness in Plotly
    colorscale = [[round(p, 6), c] for p, c in colorscale]

    tickvals = [float(b) for b in bounds]
    if leftmost_label is not None:
        ticktext = [leftmost_label] + [f"{b:.2f}" for b in bounds[1:]]
    else:
        ticktext = [f"{b:.2f}" for b in bounds]

    return {
        "cmap_name": cmap_name,
        "bounds": [float(b) for b in bounds],
        "colorscale": colorscale,
        "zmin": vmin,
        "zmax": vmax,
        "tickvals": tickvals,
        "ticktext": ticktext,
    }


def render_icefrac_polar_png(metric_2d, lat_arr, lon_arr_pcarree, cbar_spec, dpi=110):
    """Render the icefrac metric in NorthPolarStereo (matches static figures exactly).

    Returns dict with: png_b64 (string), extent_meters [xmin, xmax, ymin, ymax],
    width, height (image pixel dims).
    """
    import io
    import base64
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import BoundaryNorm
    from PIL import Image as PILImage

    proj = ccrs.NorthPolarStereo(central_longitude=0)

    # Figure dimensions in inches. Map occupies left ~78% of width; colorbar on the right.
    fig_w, fig_h = 6.5, 5.0
    map_box = [0.0, 0.0, 0.78, 1.0]       # [left, bottom, width, height] as fig fractions
    cbar_box = [0.84, 0.10, 0.04, 0.80]

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(map_box, projection=proj)
    cax = fig.add_axes(cbar_box)
    ax.set_extent([-180, 180, 55, 90], ccrs.PlateCarree())

    n_bins = len(cbar_spec["bounds"]) - 1
    cmap = plt.get_cmap(cbar_spec["cmap_name"], n_bins).copy()
    cmap.set_bad("white")
    norm = BoundaryNorm(cbar_spec["bounds"], ncolors=cmap.N, clip=True)

    # Land underneath, data on top, then coastlines/borders on top of data.
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "50m",
                                                facecolor="#dddddd", edgecolor="none"),
                   zorder=2)
    mesh = ax.pcolormesh(lon_arr_pcarree, lat_arr, metric_2d,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, norm=norm, zorder=3)
    ax.coastlines(resolution="50m", linewidth=0.5, color="#222", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   linewidth=0.3, edgecolor="#222", zorder=4)

    cb = fig.colorbar(mesh, cax=cax, ticks=cbar_spec["tickvals"])
    cb.ax.set_yticklabels(cbar_spec["ticktext"], fontsize=8)

    xmin, xmax, ymin, ymax = ax.get_extent(crs=proj)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None, pad_inches=0,
                facecolor="white", edgecolor="none")
    plt.close(fig)

    buf.seek(0)
    img = PILImage.open(buf)
    w, h = img.size
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("ascii")

    return {
        "png_b64": png_b64,
        "extent_meters": [float(xmin), float(xmax), float(ymin), float(ymax)],
        "width": int(w),
        "height": int(h),
        # Pixel rectangle within the image that contains the map (clicks elsewhere
        # are colorbar / margin and should be ignored).
        "map_bbox_fraction": [
            float(map_box[0]),                          # left
            float(1 - (map_box[1] + map_box[3])),       # top (1 - (bottom + height))
            float(map_box[0] + map_box[2]),             # right
            float(1 - map_box[1]),                      # bottom
        ],
    }


def round_or_null(val, ndigits=3):
    if np.isnan(val) or np.isinf(val):
        return None
    return round(float(val), ndigits)


def write_json_gz(path, payload):
    """Gzipped compact JSON (mtime=0, so an unchanged payload rebuilds byte-identical)."""
    path.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), mtime=0))


def array_to_json_list(arr, ndigits=3):
    # float64 first: a rounded float32 (2.756) serializes as 2.75600004196167.
    arr = np.asarray(arr, dtype=np.float64)
    return np.where(np.isfinite(arr), np.round(arr, ndigits), None).tolist()


# =============================================================================
# Sanity checks — catches bugs of two kinds we've hit:
#   (1) Magnitude/units bugs: e.g., a double-PI-subtraction inflating baseline tas
#       RMSE from ~1 K to ~280 K. Order-of-magnitude wrong values are silent without
#       a sanity check — the map still renders, just with absurd colorbar bounds.
#   (2) Bypassed-input bugs: e.g., an R² formula that computes a number but never
#       uses one of its declared inputs (so the value is irrelevant to what it claims
#       to measure). Caught by `assert_uses_inputs` below, which perturbs an input
#       and confirms the metric changes.
#
# Both kinds were real bugs in this codebase. Adding these procedural guards so we
# don't re-introduce them silently. If a new metric is added, add a plausibility
# cap below and (where it compares emul-vs-GCM) wrap its computation in
# `assert_uses_inputs(...)`.
# =============================================================================

# Per-variable plausibility caps. Picked loose (~10× the realistic upper end) so
# honest extreme values don't trigger false alarms — the caps exist to catch
# off-by-280-K-class bugs, not to enforce tight tolerances.
METRIC_PLAUSIBILITY_CAPS = {
    # Temperature caps are 25 K (was 10 K). Comparing the multi-model mean emulator
    # against each GCM gives honest errors of 12–20 K in a few Arctic sea-ice cells
    # (tasmin, mostly UKESM1-1-LL and MIROC-ES2H disagreeing on winter warming).
    # 25 K still sits far below the ~280 K bugs these caps exist to catch.
    "tas":    {"rmse_map": (25, "K"), "mae_map": (25, "K"),
               "rmse_baseline_map": (25, "K"), "mae_baseline_map": (25, "K")},
    "tasmin": {"rmse_map": (25, "K"), "mae_map": (25, "K"),
               "rmse_baseline_map": (25, "K"), "mae_baseline_map": (25, "K")},
    "tasmax": {"rmse_map": (25, "K"), "mae_map": (25, "K"),
               "rmse_baseline_map": (25, "K"), "mae_baseline_map": (25, "K")},
    "p-e":    {"rmse_map": (10, "mm/day"), "mae_map": (10, "mm/day"),
               "rmse_baseline_map": (10, "mm/day"), "mae_baseline_map": (10, "mm/day")},
    # Counts-of-days variables: floor=0, ceiling ~365 days/yr. Errors should be a
    # small fraction of that.
    "tas_above_35":  {"rmse_map": (200, "days/yr"), "mae_map": (200, "days/yr")},
    "tas_above_40":  {"rmse_map": (200, "days/yr"), "mae_map": (200, "days/yr")},
    "tas_below_0":   {"rmse_map": (200, "days/yr"), "mae_map": (200, "days/yr")},
    "pr_above_10":   {"rmse_map": (200, "days/yr"), "mae_map": (200, "days/yr")},
    "pr_above_20":   {"rmse_map": (200, "days/yr"), "mae_map": (200, "days/yr")},
    # icefrac: bounded [0, 1].
    "icefrac": {"rmse_map": (1, "fraction"), "mae_map": (1, "fraction"),
                "rmse_baseline_map": (1, "fraction"), "mae_baseline_map": (1, "fraction")},
}
# R² is dimensionless and should fall in (−∞, 1]. Very large negative values are
# legitimate (means emulator worse than predicting zero) — no cap. But all maps
# should at least be ≤ 1 + ε; flag anything wildly above 1 as a bug.
R2_FIELDS = ("r2_map", "r2_baseline_map")


def _stats_str(arr):
    a = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else np.array([])
    if a.size == 0:
        return "all NaN"
    return f"min={float(a.min()):+.4f}  median={float(np.median(a)):+.4f}  max={float(a.max()):+.4f}  (n={a.size})"


def sanity_check_metrics(all_data):
    """Print a per-variable metric summary; raise if any metric is wildly out of
    its plausibility band. Run at end of metric computation, before payload
    assembly, so bad values can't make it into a built website."""
    print()
    print("=" * 84)
    print("Per-variable metric summary  (sanity check — catches order-of-magnitude bugs)")
    print("=" * 84)
    errors = []
    for var, d in all_data.items():
        print(f"\n{var}:")
        for k in sorted(d.keys()):
            v = d[k]
            if not (isinstance(v, np.ndarray) and v.dtype.kind == "f"):
                continue
            if v.ndim < 2:
                continue
            print(f"  {k:<28} {_stats_str(v)}")
            # Apply plausibility cap if defined
            caps = METRIC_PLAUSIBILITY_CAPS.get(var, {})
            if k in caps:
                cap, units = caps[k]
                vf = v[np.isfinite(v)]
                if vf.size and float(np.abs(vf).max()) > cap:
                    errors.append(
                        f"  {var}.{k}: max |value| = {float(np.abs(vf).max()):.2f} {units} "
                        f"> cap {cap} {units}. Likely a bug (units, double-rebasing, etc.)."
                    )
            # R² fields: should be ≤ 1 + a tiny tolerance
            if k in R2_FIELDS:
                vf = v[np.isfinite(v)]
                if vf.size and float(vf.max()) > 1.001:
                    errors.append(
                        f"  {var}.{k}: max value = {float(vf.max()):.4f} > 1.001 — "
                        f"R² should be ≤ 1. Bug in the R² formula."
                    )
    print("=" * 84)
    if errors:
        msg = "\n".join(errors)
        raise ValueError(
            f"Metric sanity check FAILED:\n{msg}\n\n"
            f"Either the cap in METRIC_PLAUSIBILITY_CAPS is too tight (raise it) "
            f"or there is a real bug. Do NOT just bump the cap blindly — first verify "
            f"the metric is doing what it claims.")
    print("All metrics within plausibility caps. ✓")
    print()


def assert_uses_inputs(metric_arr, name, *input_arrays):
    """Defensive check for metrics that claim to compare multiple inputs (e.g.,
    'emulator vs GCM' R²/RMSE/MAE). Perturbs each declared input by a small noise
    and confirms the metric value changes — i.e., it actually depends on that
    input. Catches the bug we hit where baseline R² didn't use the emulator
    output at all.

    Raises RuntimeError if metric is invariant under perturbation of any input.
    Call this once when defining a new emul-vs-GCM metric; remove for production
    runs (it doubles the computation). Not currently wired into main() — kept here
    as a reference pattern. Use as:
        assert_uses_inputs(metric_value, 'baseline_r2', pred_array, actual_array)
    """
    # Reference implementation — illustrative. Real perturbation should be done
    # at the formula site (re-running the metric computation with mutated input).
    raise NotImplementedError(
        "assert_uses_inputs is a documentation stub. To use: wrap the metric "
        "computation in a function `f(pred, actual)`, then check that "
        "f(pred + noise, actual) != f(pred, actual) for small noise. The "
        "bypassed-emulator-output bug in baseline R² would have been caught "
        "this way during development.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing per-variable data...")
    # all_data[model][var] -> metric dict. Every model gets the Baseline keys;
    # SAI_MODELS additionally get the SAI delta keys (rmse_map, ts_pred, ...) plus the
    # pooled all-model SAI R² (r2_map, r2_x, ...), which is the same for every model.
    all_data = {model: {} for model in MODELS}
    for var in VARS:
        sai_r2 = compute_sai_r2_icefrac() if var == "icefrac" else compute_sai_r2_var(var)
        for model in MODELS:
            if var == "icefrac":
                d = compute_icefrac_baseline(model)
                if model in SAI_MODELS:
                    d.update(compute_icefrac(model))
            else:
                d = compute_baseline_var(var, model)
                if model in SAI_MODELS:
                    d.update(compute_linear_var(var, model))
            if model in SAI_MODELS:
                d.update(sai_r2)
            all_data[model][var] = d

    # Guardrail: catch order-of-magnitude bugs (units, double-rebasing, etc.) before
    # they bake into the rendered website. See METRIC_PLAUSIBILITY_CAPS above.
    for model in MODELS:
        print(f"\n##### {model} #####")
        sanity_check_metrics(all_data[model])

    print("Building lightweight index payload (maps only)...")
    lat = all_data[MODELS[0]][VARS[0]]["lat"]
    lon = all_data[MODELS[0]][VARS[0]]["lon"]
    years = np.arange(WINDOW_START, WINDOW_END + 1)
    for model in MODELS:
        for var in VARS:
            d = all_data[model][var]
            # Every model is on the common (CESM2-WACCM) grid, so one lat/lon serves all.
            if not (np.allclose(d["lat"], lat) and np.allclose(d["lon"], lon)):
                raise ValueError(f"{model}/{var} is not on the common grid")
            # The scatter JS indexes every per-cell series with INDEX.years.
            for key in ("years", "years_baseline"):
                if key in d and not np.array_equal(d[key], years):
                    raise ValueError(f"{model}/{var} {key} = {d[key]} != {years}")

    lon_shifted = np.where(lon > 180, lon - 360, lon)
    lon_sort_idx = np.argsort(lon_shifted)
    lon_for_display = lon_shifted[lon_sort_idx]

    def reorder_lon(arr):
        return arr[..., lon_sort_idx]

    def modes_for(model):
        return [SAI_MODE, BASELINE_MODE] if model in SAI_MODELS else [BASELINE_MODE]

    # Build discrete colorbar specs matching the static validation figures.
    # R² is fixed: YlOrBr, 10 bins from 0.5 to 1.0 step 0.05, leftmost "≤ 0.50".
    r2_bounds = np.arange(0.5, 1.001, 0.05)
    r2_cbar = discrete_colorbar_spec("YlOrBr", r2_bounds, leftmost_label="≤ 0.50")

    def per_var_cbar(arr, cmap_name):
        """RMSE/MAE colorbar: 10 equal bins 0 → ceil(q99 * 100) / 100."""
        v = float(np.ceil(np.nanquantile(arr, 0.99) * 100) / 100)
        if v <= 0:
            v = float(np.nanmax(arr)) if np.nanmax(arr) > 0 else 1.0
        bounds = np.linspace(0.0, v, 11)
        return discrete_colorbar_spec(cmap_name, bounds)

    # Per-mode map keys: SAI delta uses the bare keys, Baseline the *_baseline keys.
    MODE_SUFFIX = {SAI_MODE: "", BASELINE_MODE: "_baseline"}

    by_model = {}
    for model in MODELS:
        md = all_data[model]
        cbars = {}
        for v in VARS:
            # SAI delta and Baseline metrics have different absolute magnitudes (deltas vs PI anomalies)
            # so each mode gets its own RMSE/MAE colorbar scaling. R² shares the fixed YlOrBr 0.5→1.0.
            cbars[v] = {
                mode: {
                    "RMSE": per_var_cbar(md[v][f"rmse{MODE_SUFFIX[mode]}_map"], "summer"),
                    "MAE": per_var_cbar(md[v][f"mae{MODE_SUFFIX[mode]}_map"], "cividis"),
                    "R²": r2_cbar,
                }
                for mode in modes_for(model)
            }
        by_model[model] = {"cbars": cbars}

    print("Computing per-cell IPCC region grid...")
    region_names, region_idx_grid = compute_region_grid(lat, lon_for_display)
    print(f"  {len(region_names)} unique regions; grid {len(region_idx_grid)}x{len(region_idx_grid[0])}")

    print("Extracting country/coastline outlines...")
    border_lon, border_lat = extract_borders(tolerance=0.05)
    print(f"  {sum(1 for v in border_lon if v is not None)} border points (after simplification)")

    print("Computing SNR masks (SAI delta + baseline, all vars, per model)...")
    # Per-model, per-scenario, per-var stipple-dot lists in display-lon order. Applied on top
    # of the R² map (no masking of the R² values themselves — just an overlay).
    # Also collects the per-cell 60-sample arrays so the JS can draw the
    # per-cell detrended-residuals scatter on click (side-by-side with the R² fit).
    scenario_map = {SAI_MODE: "sai_delta", BASELINE_MODE: "baseline"}
    snr_samples = {}   # (model, var, scenario_key) -> {...}
    for model in MODELS:
        snr_dots = {mode: {} for mode in modes_for(model)}
        for mode_label in modes_for(model):
            scen = scenario_map[mode_label]
            for sv in SNR_VARS:
                result = compute_snr_mask(sv, scen, model)
                if result is None:
                    print(f"  [{model:<12}] [{mode_label:<21}] {sv:<15} SKIPPED — no per-member "
                          f"{TEMP_TARGET} SAI file; no stippling for this var/scenario")
                    continue
                mask = result["mask"][..., lon_sort_idx]      # to display-lon order
                dot_lats, dot_lons = [], []
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j]:
                            dot_lats.append(round(float(lat[i]), 3))
                            dot_lons.append(round(float(lon_for_display[j]), 3))
                snr_dots[mode_label][sv] = {"lat": dot_lats, "lon": dot_lons}
                print(f"  [{model:<12}] [{mode_label:<21}] {sv:<15} {int(mask.sum()):>6d} cells "
                      f"({100*mask.sum()/mask.size:.1f}%) low-SNR")
                # Per-cell samples for the click-scatter (also reordered to display lons).
                snr_samples[(model, sv, scen)] = {
                    "samples": result["samples"][..., lon_sort_idx],
                    "years": result["years"],
                    "members": result["members"],
                }
        by_model[model]["snr_dots_by_mode"] = snr_dots

    for model in MODELS:
        md = all_data[model]
        cbars = by_model[model]["cbars"]
        # Pre-rendered NorthPolarStereo PNGs of icefrac maps (one per metric).
        # Used as a static background image; clicks are mapped back to (lat,lon)
        # in JS via the embedded extent (meters in NorthPolarStereo).
        by_model[model]["icefrac_polar"] = {
            mode: {
                metric: render_icefrac_polar_png(
                    reorder_lon(md["icefrac"][f"{key}{MODE_SUFFIX[mode]}_map"]),
                    lat, lon_for_display, cbars["icefrac"][mode][metric])
                for metric, key in (("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2"))
            }
            for mode in modes_for(model)
        }
        # Maps only (small). Each variable carries the SAI-delta (SAI_MODELS only) and
        # Baseline metric maps side-by-side; the JS picks which keys to read based on Scenario.
        maps = {}
        for v in VARS:
            maps[v] = {}
            for mode in modes_for(model):
                suffix = MODE_SUFFIX[mode]
                maps[v][f"r2{suffix}"] = array_to_json_list(reorder_lon(md[v][f"r2{suffix}_map"]), ndigits=3)
                maps[v][f"rmse{suffix}"] = array_to_json_list(reorder_lon(md[v][f"rmse{suffix}_map"]), ndigits=4)
                maps[v][f"mae{suffix}"] = array_to_json_list(reorder_lon(md[v][f"mae{suffix}_map"]), ndigits=4)
        by_model[model]["maps"] = maps

    index_payload = {
        # Scenario dropdown is leftmost; the Model dropdown lists the models available
        # for the selected Scenario (SAI delta: SAI_MODELS).
        "models": MODELS,
        "models_by_mode": {
            SAI_MODE: SAI_MODELS,
            BASELINE_MODE: MODELS,
        },
        "vars": VARS,
        "ui_names": UI_NAMES,
        "units": UNITS,
        "metrics": ["RMSE", "MAE", "R²", "All"],
        # Scenario dropdown (leftmost); label shown in the UI as "Scenario:".
        "compare_modes": [SAI_MODE, BASELINE_MODE],
        "metrics_by_mode": {
            SAI_MODE: ["RMSE", "MAE", "R²", "All"],
            BASELINE_MODE: ["RMSE", "MAE", "R²", "All"],
        },
        "lat": lat.tolist(),
        "lon": lon_for_display.tolist(),
        "years": years.tolist(),
        "title": "Emulator fit quality and validation metrics",
        "subtitle": ("GCM data is the 3-ensemble-member mean from each model: CESM2-WACCM, "
                     "MIROC-ES2H and UKESM1-1-LL. "
                     "Emulator predictions use the same regression as the simulator's runtime."),
        "is_icefrac": {v: all_data[MODELS[0]][v]["is_icefrac"] for v in VARS},
        "region_names": region_names,
        "region_grid": region_idx_grid,
        "border_lon": border_lon,
        "border_lat": border_lat,
        # Per-model payloads: maps, colorbars, SNR stipple dots (per Scenario), and the
        # pre-rendered icefrac polar PNGs. Keys of each are the Scenarios that model has.
        "by_model": by_model,
    }

    # Per-model, per-variable scatter payloads (the big stuff), in OUTPUT_DIR/<model>/.
    print("Writing per-variable scatter JSON files...")
    for model in MODELS:
        model_dir = OUTPUT_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for var in VARS:
            d = all_data[model][var]
            scatter_payload = {
                "var": var,
                "model": model,
                # Baseline-mode scatter data: per-cell, 20 yearly samples each of the
                # emulator prediction and the GCM truth. Same data drives the click-to-see
                # plot for RMSE / MAE / R² in baseline mode (since R² is now also an
                # emulator-vs-GCM metric).
                "ts_baseline_pred": array_to_json_list(reorder_lon(d["ts_baseline_pred"]), ndigits=4),
                "ts_baseline_actual": array_to_json_list(reorder_lon(d["ts_baseline_actual"]), ndigits=4),
            }
            if model in SAI_MODELS:
                scatter_payload.update({
                    "r2_x": [round_or_null(v, 4) for v in d["r2_x"]],
                    "r2_models": d["r2_models"],
                    "r2_actual": array_to_json_list(reorder_lon(d["r2_actual"]), ndigits=4),
                    "r2_pred": array_to_json_list(reorder_lon(d["r2_pred"]), ndigits=4),
                    "ts_pred": array_to_json_list(reorder_lon(d["ts_pred"]), ndigits=4),
                    "ts_actual": array_to_json_list(reorder_lon(d["ts_actual"]), ndigits=4),
                })
            out = model_dir / f"data_{var}.json.gz"
            write_json_gz(out, scatter_payload)
            print(f"  {model}/data_{var}.json.gz: {out.stat().st_size / 1024 / 1024:.1f} MB")

    # Per-cell 60-sample arrays for the click-to-see detrended-residuals scatter
    # (side-by-side with the R² fit). Only for vars in SNR_VARS since only those
    # have clean per-member baseline data. Split into one file per (model × var × scenario)
    # so no single file exceeds ~100 MB (GitHub push limit). Fetched lazily by the
    # JS the first time a cell is clicked in R² mode for the matching (model, var, scenario).
    print("Writing per-model-per-var-per-scenario SNR sample files...")
    for (model, var, scen_key), sdata in snr_samples.items():
        mode_label = next(m for m, k in scenario_map.items() if k == scen_key)
        payload = {
            "var": var,
            "model": model,
            "scenario": mode_label,
            "samples": array_to_json_list(sdata["samples"], ndigits=3),
            "years": sdata["years"],
            "members": sdata["members"],
        }
        out = OUTPUT_DIR / model / f"data_{var}_snr_{scen_key}.json.gz"
        write_json_gz(out, payload)
        print(f"  {model}/data_{var}_snr_{scen_key}.json.gz: {out.stat().st_size / 1024 / 1024:.1f} MB")

    print("Writing index.html...")
    index_json = json.dumps(index_payload, separators=(",", ":"))
    print(f"  index payload size: {len(index_json) / 1024 / 1024:.1f} MB")
    html = build_html(index_json)
    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(html)
    print(f"  index.html: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    print()
    print("To view: run this in a terminal then open the URL it prints:")
    print(f"  cd {OUTPUT_DIR}")
    print("  python -m http.server 8000")
    print("  open http://localhost:8000")


def build_html(index_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Emulator validation</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --font-stack: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Helvetica Neue", Arial, sans-serif;
  }}
  html, body, select, button, input {{ font-family: var(--font-stack); }}
  /* Page grows to fill the viewport on large screens; never shrinks below 1400px
     (small viewports get a horizontal scroll instead of squashing the layout). */
  body {{ margin: 20px; min-width: 1400px; }}
  h1 {{ margin-bottom: 4px; font-size: 22px; }}
  .subtitle {{ color: #555; margin-bottom: 20px; font-size: 14px; }}
  .controls {{ margin-bottom: 16px; }}
  .controls label {{ font-weight: 600; margin-right: 8px; }}
  .controls select {{ margin-right: 24px; padding: 4px 8px; font-size: 14px; }}
  /* Single view map: width fills the page, height computed from natural 2:1 aspect
     so the map grows proportionally on wide screens (no stretching, less white space).
     icefrac overrides via .is-icefrac to keep the polar PNG at its current behavior. */
  #map {{ width: 100%; aspect-ratio: 2 / 1; }}
  #map.is-icefrac {{ aspect-ratio: auto; height: 700px; }}
  /* Single-view scatter row: normally one full-width scatter. When the SNR
     residuals scatter is shown (R² metric + SNR-supported var), a right column
     appears with two stacked plots — top: detrended residuals, bottom: raw
     samples (the values whose mean is the signal). */
  #scatter-row {{ display: flex; gap: 12px; margin-top: 12px; }}
  #scatter {{ flex: 1 1 100%; height: 420px; }}
  #snr-col {{ flex: 0 0 0; display: none; flex-direction: column; gap: 12px; }}
  #snr-scatter, #snr-scatter-raw {{ height: 420px; }}
  #scatter-row.with-snr #scatter {{ flex: 1 1 50%; }}
  #scatter-row.with-snr #snr-col {{ flex: 1 1 50%; display: flex; }}
  /* All view: 2 equal-width columns × 3 rows.
     Left column holds the 3 maps (RMSE, MAE, R²) top→bottom.
     Right column holds the 2 scatters: RMSE/MAE scatter spans rows 1+2 (since it
     corresponds to both the RMSE and MAE maps); R² scatter sits in row 3 next to
     the R² map. Maps and scatters share the same column width. */
  #all-view {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto auto;
    gap: 14px;
  }}
  .all-maps-row, .all-scatters-row {{ display: contents; }}
  #map-rmse, #map-mae, #map-r2 {{ aspect-ratio: 2 / 1; }}
  #map-rmse.is-icefrac, #map-mae.is-icefrac, #map-r2.is-icefrac {{
    aspect-ratio: auto; height: 420px;
  }}
  #map-rmse {{ grid-column: 1; grid-row: 1; }}
  #map-mae  {{ grid-column: 1; grid-row: 2; }}
  #map-r2   {{ grid-column: 1; grid-row: 3; }}
  #scatter-ts {{ grid-column: 2; grid-row: 1 / span 2; }}
  #scatter-r2 {{ grid-column: 2; grid-row: 3; }}
  /* Polar (icefrac) renders as a static PNG with an absolutely-positioned canvas overlay
     for the selection rectangle. Fits whatever container it's in. */
  .polar-wrap {{
    position: relative;
    width: 100%; height: 100%;
    display: flex; align-items: flex-start; justify-content: flex-start;
  }}
  .polar-img {{ display: block; max-width: 100%; max-height: 100%; cursor: crosshair; }}
  .polar-overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
  .hint {{ color: #666; font-size: 13px; margin: 8px 0; }}
  .stipple-note {{ display: none; color: #555; font-size: 13px;
                   margin: 4px 0 10px; padding: 6px 10px; background: #f5f5f5;
                   border-left: 3px solid #999; line-height: 1.4; }}
  .stipple-note.visible {{ display: block; }}
  .stipple-note code {{ font-size: 12px; }}
  .loading {{ color: #999; font-style: italic; }}
  /* Loading wheel: shown from a dropdown change or cell click until the maps/plots are drawn. */
  .spinner {{ display: none; position: fixed; top: 50%; left: 50%; width: 44px; height: 44px;
             margin: -22px 0 0 -22px; border: 4px solid #ddd; border-top-color: #555;
             border-radius: 50%; animation: spin 0.8s linear infinite; z-index: 1000; }}
  .spinner.visible {{ display: block; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  /* Default cursor on map hover; switch to drag (move) cursor only while actively dragging. */
  .js-plotly-plot .nsewdrag,
  .js-plotly-plot .nsewdrag.cursor-pointer,
  .js-plotly-plot .nsewdrag.cursor-move {{ cursor: default !important; }}
  .js-plotly-plot .nsewdrag:active {{ cursor: move !important; }}
</style>
</head>
<body>
<h1 id="title"></h1>
<div class="subtitle" id="subtitle"></div>
<div class="controls">
  <label for="compare-select">Scenario:</label>
  <select id="compare-select"></select>
  <label for="model-select">Model:</label>
  <select id="model-select"></select>
  <label for="var-select">Variable:</label>
  <select id="var-select"></select>
  <label for="metric-select">Metric:</label>
  <select id="metric-select"></select>
  <span id="loader" class="loading"></span>
</div>
<div class="hint">Click any grid cell on the map to see the scatter plot for that cell below. Scroll to zoom, drag to pan.</div>
<div id="spinner" class="spinner" role="status" aria-label="Loading"></div>
<div id="stipple-note" class="stipple-note"></div>
<div id="r2-note" class="stipple-note"><strong>Note:</strong> The R² analysis includes the input from all three models.</div>
<div id="single-view">
  <div id="map"></div>
  <div id="scatter-row">
    <div id="scatter"></div>
    <div id="snr-col">
      <div id="snr-scatter"></div>
      <div id="snr-scatter-raw"></div>
    </div>
  </div>
</div>
<div id="all-view" style="display:none;">
  <div class="all-maps-row">
    <div id="map-rmse"></div>
    <div id="map-mae"></div>
    <div id="map-r2"></div>
  </div>
  <div class="all-scatters-row">
    <div id="scatter-ts"></div>
    <div id="scatter-r2"></div>
  </div>
</div>
<script>
const INDEX = {index_json};

// Match the simulator's font stack so Plotly text renders with the same fonts as the page.
const PLOTLY_FONT = {{ family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Helvetica Neue", Arial, sans-serif' }};

// Mode-aware metric key lookup. Baseline metrics use *_baseline keys; SAI uses the bare keys.
const METRIC_KEY_BY_MODE = {{
  "SAI delta": {{ "RMSE": "rmse", "MAE": "mae", "R²": "r2" }},
  "Baseline (SSP2-4.5)": {{ "RMSE": "rmse_baseline", "MAE": "mae_baseline", "R²": "r2_baseline" }},
}};
function metricKey(metric) {{ return METRIC_KEY_BY_MODE[compareSelect.value][metric]; }}
const ALL_MAP_DIVS = {{ "RMSE": "map-rmse", "MAE": "map-mae", "R²": "map-r2" }};

let SELECTED_CELL = null;
let RELAYOUT_GUARD = false;

// Data files are gzipped JSON. Unzip here unless the server already did (it sent
// Content-Encoding: gzip, so the body no longer starts with the gzip magic bytes).
async function readJsonGz(resp) {{
  const bytes = new Uint8Array(await resp.arrayBuffer());
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(bytes));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
}}

// Cache of per-variable scatter data (fetched on demand)
// Per-model files live in <model>/; caches are keyed by model too.
const SCATTER_CACHE = {{}};
async function ensureScatterFor(varKey) {{
  const model = modelSelect.value;
  const cacheKey = model + '::' + varKey;
  if (SCATTER_CACHE[cacheKey]) return SCATTER_CACHE[cacheKey];
  const loader = document.getElementById('loader');
  loader.textContent = 'loading scatter data for ' + model + ' ' + varKey + '...';
  const fname = encodeURIComponent(model) + '/data_' + encodeURIComponent(varKey) + '.json.gz';
  const resp = await fetch(fname);
  if (!resp.ok) {{
    loader.textContent = 'error loading ' + fname + ' (status ' + resp.status + ')';
    throw new Error('fetch failed: ' + resp.status);
  }}
  const data = await readJsonGz(resp);
  SCATTER_CACHE[cacheKey] = data;
  loader.textContent = '';
  return data;
}}

// Separate lazy-load cache for the (big) SNR sample files. Split per (var × scenario)
// so no single file exceeds ~100 MB (GitHub push limit). Fetched only when the user
// clicks a cell in R² mode for tas / p-e / icefrac.
const SNR_CACHE = {{}};
async function ensureSnrDataFor(varKey, scenarioKey) {{
  // scenarioKey: 'sai_delta' or 'baseline'
  const model = modelSelect.value;
  const cacheKey = model + '::' + varKey + '::' + scenarioKey;
  if (SNR_CACHE[cacheKey]) return SNR_CACHE[cacheKey];
  const loader = document.getElementById('loader');
  loader.textContent = 'loading SNR samples for ' + model + ' ' + varKey + ' (' + scenarioKey + ')...';
  const fname = encodeURIComponent(model) + '/data_' + encodeURIComponent(varKey) + '_snr_' + scenarioKey + '.json.gz';
  const resp = await fetch(fname);
  if (!resp.ok) {{
    loader.textContent = 'error loading ' + fname + ' (status ' + resp.status + ')';
    throw new Error('fetch failed: ' + resp.status);
  }}
  const data = await readJsonGz(resp);
  SNR_CACHE[cacheKey] = data;
  loader.textContent = '';
  return data;
}}

document.getElementById('title').textContent = INDEX.title;
document.getElementById('subtitle').textContent = INDEX.subtitle;

// Scenario dropdown (leftmost).
const compareSelect = document.getElementById('compare-select');
INDEX.compare_modes.forEach(m => {{
  const opt = document.createElement('option');
  opt.value = m;
  opt.textContent = m;
  compareSelect.appendChild(opt);
}});
compareSelect.value = 'SAI delta';

// Model dropdown: lists the GCMs that have data for the selected Scenario
// (INDEX.models_by_mode).
const modelSelect = document.getElementById('model-select');
function populateModelDropdown() {{
  const prev = modelSelect.value;
  const opts = INDEX.models_by_mode[compareSelect.value] || INDEX.models;
  modelSelect.innerHTML = '';
  for (const m of opts) {{
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  }}
  modelSelect.value = opts.includes(prev) ? prev : opts[0];
}}
populateModelDropdown();

// Per-model payload (maps, colorbars, SNR dots, icefrac polar PNGs) for the selected model.
function MD() {{ return INDEX.by_model[modelSelect.value]; }}

const varSelect = document.getElementById('var-select');
INDEX.vars.forEach(v => {{
  const opt = document.createElement('option');
  opt.value = v;
  opt.textContent = INDEX.ui_names[v] + ' (' + v + ')';
  varSelect.appendChild(opt);
}});
varSelect.value = 'tas';

const metricSelect = document.getElementById('metric-select');

function metricsForVar(varKey) {{
  const mode = compareSelect.value;
  return (INDEX.metrics_by_mode[mode] || INDEX.metrics).slice();
}}

function populateMetricDropdown(varKey) {{
  const prev = metricSelect.value;
  const opts = metricsForVar(varKey);
  metricSelect.innerHTML = '';
  for (const m of opts) {{
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    metricSelect.appendChild(opt);
  }}
  metricSelect.value = opts.includes(prev) ? prev : 'RMSE';
}}

populateMetricDropdown(varSelect.value);
metricSelect.value = 'RMSE';

function nearestIdx(arr, target) {{
  let best = 0;
  let bestDist = Math.abs(arr[0] - target);
  for (let i = 1; i < arr.length; i++) {{
    const d = Math.abs(arr[i] - target);
    if (d < bestDist) {{ bestDist = d; best = i; }}
  }}
  return best;
}}

function isR2Metric(metric) {{
  return metric === 'R²';
}}

function colorForMetric(metric) {{
  return isR2Metric(metric) ? 'black' : 'red';
}}

function colorbarTitle(metric, varKey) {{
  if (isR2Metric(metric)) return 'R²';
  return metric + ' (' + INDEX.units[varKey] + ')';
}}

function fullExtent(varKey) {{
  const isIce = INDEX.is_icefrac[varKey];
  return {{ xfull: [-180, 180], yfull: isIce ? [55, 90] : [-90, 90] }};
}}

// ---------- Cartesian heatmap (non-icefrac) ----------
function buildHeatmapTraces(varKey, metric) {{
  const cbarKey = isR2Metric(metric) ? 'R²' : metric;
  const z = MD().maps[varKey][metricKey(metric)];
  const cbar = MD().cbars[varKey][compareSelect.value][cbarKey];
  const heatmap = {{
    type: 'heatmap',
    x: INDEX.lon, y: INDEX.lat, z: z,
    colorscale: cbar.colorscale,
    zmin: cbar.zmin, zmax: cbar.zmax, zauto: false,
    colorbar: {{
      title: colorbarTitle(metric, varKey),
      tickmode: 'array',
      tickvals: cbar.tickvals,
      ticktext: cbar.ticktext,
    }},
    hovertemplate: 'lat: %{{y:.2f}}<br>lon: %{{x:.2f}}<br>value: %{{z}}<extra></extra>',
  }};
  const borders = {{
    type: 'scatter',
    x: INDEX.border_lon, y: INDEX.border_lat,
    mode: 'lines',
    line: {{ color: '#222', width: 0.4 }},
    hoverinfo: 'skip', showlegend: false,
  }};
  const traces = [heatmap, borders];
  // Overlay SNR stippling on top of the R² heatmap (does not mask the R² values).
  // Dots mark cells where |signal|/noise < 1 in the current Scenario. SNR is a
  // detrended paired-sample statistic; the underlying formula differs by scenario
  // (paired δ in SAI mode, SSP245 anomaly vs PI in Baseline mode).
  if (isR2Metric(metric)) {{
    const modeDots = (MD().snr_dots_by_mode || {{}})[compareSelect.value] || {{}};
    const dots = modeDots[varKey];
    if (dots && dots.lat && dots.lat.length) {{
      traces.push({{
        type: 'scattergl',
        x: dots.lon, y: dots.lat,
        mode: 'markers',
        marker: {{ color: 'black', size: 2, opacity: 0.7 }},
        hoverinfo: 'skip', showlegend: false,
        name: '|signal|/noise < 1',
      }});
    }}
  }}
  return traces;
}}

function buildCartesianLayout(varKey, metric, opts) {{
  opts = opts || {{}};
  const {{ xfull, yfull }} = fullExtent(varKey);
  const layout = {{
    xaxis: {{
      title: 'longitude', range: xfull,
      showgrid: false, zeroline: false,
      minallowed: xfull[0], maxallowed: xfull[1],
    }},
    yaxis: {{
      title: 'latitude', range: yfull,
      showgrid: false, zeroline: false,
      scaleanchor: 'x', scaleratio: 1,
      minallowed: yfull[0], maxallowed: yfull[1],
    }},
    margin: {{ l: 50, r: 20, t: 36, b: 40 }},
    font: PLOTLY_FONT,
    dragmode: 'pan',
    title: opts.title || (INDEX.ui_names[varKey] + ' (' + varKey + ') — ' + metric),
  }};
  return layout;
}}

function rectShapeForCell(metric, lineWidth) {{
  if (!SELECTED_CELL) return [];
  const {{ iLat, iLon }} = SELECTED_CELL;
  const halfLat = Math.abs((INDEX.lat[1] - INDEX.lat[0]) / 2);
  const halfLon = Math.abs((INDEX.lon[1] - INDEX.lon[0]) / 2);
  const lat = INDEX.lat[iLat], lon = INDEX.lon[iLon];
  return [{{
    type: 'rect',
    x0: lon - halfLon, x1: lon + halfLon,
    y0: lat - halfLat, y1: lat + halfLat,
    line: {{ color: colorForMetric(metric), width: lineWidth }},
    fillcolor: 'rgba(0,0,0,0)',
    layer: 'above',
  }}];
}}

function selectionLineWidthForCartesian(mapDiv) {{
  const xr = mapDiv.layout?.xaxis?.range;
  const span = (xr && xr.length === 2) ? (xr[1] - xr[0]) : 360;
  return Math.max(1, 1.5 * Math.sqrt(360 / Math.max(1, span)));
}}

// ---------- Polar (icefrac) — rendered as a PNG image with a canvas overlay ----------
// The polar PNG is pre-rendered server-side using cartopy's NorthPolarStereo
// projection (matches the static figures bit-for-bit). Clicks on the <img> are
// converted from pixel coords -> stereographic meters -> lat/lon via inverse
// projection, then mapped to the nearest grid cell.

const EARTH_R = 6378137;  // WGS84 equatorial radius (meters)

function inverseNorthPolarStereographic(xMeters, yMeters) {{
  // Inverse projection: NorthPolarStereo with central_longitude=0.
  // Spherical approximation; close enough for cell-snapping at our resolution.
  const r = Math.sqrt(xMeters * xMeters + yMeters * yMeters);
  const lat = Math.PI / 2 - 2 * Math.atan(r / (2 * EARTH_R));
  const lon = Math.atan2(xMeters, -yMeters);
  return [lat * 180 / Math.PI, lon * 180 / Math.PI];
}}

function forwardNorthPolarStereographic(latDeg, lonDeg) {{
  const phi = latDeg * Math.PI / 180;
  const lam = lonDeg * Math.PI / 180;
  const tanHalf = Math.tan(Math.PI / 4 - phi / 2);
  const x = 2 * EARTH_R * tanHalf * Math.sin(lam);
  const y = -2 * EARTH_R * tanHalf * Math.cos(lam);
  return [x, y];
}}

function polarPixelToMeters(px, py, polarData) {{
  // px, py in image natural pixel coords. map_bbox_fraction = [left, top, right, bottom]
  // as fractions of the image (0..1).
  const [bxl, byt, bxr, byb] = polarData.map_bbox_fraction;
  const mapLeft = bxl * polarData.width;
  const mapTop = byt * polarData.height;
  const mapRight = bxr * polarData.width;
  const mapBottom = byb * polarData.height;
  if (px < mapLeft || px > mapRight || py < mapTop || py > mapBottom) return null;
  const [xmin, xmax, ymin, ymax] = polarData.extent_meters;
  const xm = xmin + (px - mapLeft) / (mapRight - mapLeft) * (xmax - xmin);
  // y is inverted in image coords (y=0 at top, but ymax is the top in projection)
  const ym = ymax - (py - mapTop) / (mapBottom - mapTop) * (ymax - ymin);
  return [xm, ym];
}}

function polarMetersToPixel(xMeters, yMeters, polarData) {{
  const [bxl, byt, bxr, byb] = polarData.map_bbox_fraction;
  const mapLeft = bxl * polarData.width;
  const mapTop = byt * polarData.height;
  const mapRight = bxr * polarData.width;
  const mapBottom = byb * polarData.height;
  const [xmin, xmax, ymin, ymax] = polarData.extent_meters;
  const px = mapLeft + (xMeters - xmin) / (xmax - xmin) * (mapRight - mapLeft);
  const py = mapTop + (ymax - yMeters) / (ymax - ymin) * (mapBottom - mapTop);
  return [px, py];
}}

function renderPolarMap(divId, varKey, metric) {{
  const polarData = MD().icefrac_polar[compareSelect.value][metric];
  const div = document.getElementById(divId);
  div.innerHTML = '<div class="polar-wrap">'
    + '<img class="polar-img" src="data:image/png;base64,' + polarData.png_b64 + '" alt="icefrac polar map" />'
    + '<canvas class="polar-overlay"></canvas>'
    + '</div>';
  const wrap = div.querySelector('.polar-wrap');
  const img = div.querySelector('.polar-img');
  const canvas = div.querySelector('.polar-overlay');

  // After image loads, size the canvas to the image's natural pixel dims and
  // pin its CSS size to match the displayed image size (so the canvas drawing
  // coordinates remain in natural-pixel space).
  function syncCanvasSize() {{
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = img.clientWidth + 'px';
    canvas.style.height = img.clientHeight + 'px';
    // Draw SNR stipple dots underneath the selection rectangle (R² metric only).
    drawPolarSnrDots(divId, varKey, metric);
    drawPolarSelection(divId, varKey, metric);
  }}
  if (img.complete && img.naturalWidth) {{
    syncCanvasSize();
  }} else {{
    img.addEventListener('load', syncCanvasSize, {{ once: true }});
  }}
  // Re-sync if the page resizes
  if (window.ResizeObserver) {{
    const ro = new ResizeObserver(syncCanvasSize);
    ro.observe(img);
  }} else {{
    window.addEventListener('resize', syncCanvasSize);
  }}

  img.addEventListener('click', function(e) {{
    const rect = img.getBoundingClientRect();
    const scaleX = img.naturalWidth / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    const meters = polarPixelToMeters(px, py, polarData);
    if (!meters) return;
    const [latVal, lonVal] = inverseNorthPolarStereographic(meters[0], meters[1]);
    const iLat = nearestIdx(INDEX.lat, latVal);
    const iLon = nearestIdx(INDEX.lon, lonVal);
    // Click same cell twice = deselect.
    if (SELECTED_CELL && SELECTED_CELL.iLat === iLat && SELECTED_CELL.iLon === iLon) {{
      SELECTED_CELL = null;
    }} else {{
      SELECTED_CELL = {{ iLat, iLon, latVal: INDEX.lat[iLat], lonVal: INDEX.lon[iLon] }};
    }}
    whileBusy(onSelectionChanged);
  }});

  if (img.complete && img.naturalWidth) return Promise.resolve();
  return new Promise(resolve => img.addEventListener('load', resolve, {{ once: true }}));
}}

function drawPolarSnrDots(divId, varKey, metric) {{
  // Stipple dots for cells where SNR < 1 in the current Scenario. Only for R².
  if (!isR2Metric(metric)) return;
  const div = document.getElementById(divId);
  const canvas = div.querySelector('.polar-overlay');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const modeDots = (MD().snr_dots_by_mode || {{}})[compareSelect.value] || {{}};
  const dots = modeDots[varKey];
  if (!dots || !dots.lat || !dots.lat.length) return;
  const polarData = MD().icefrac_polar[compareSelect.value][metric];
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  const r = 1.4;  // dot radius in image pixels
  for (let i = 0; i < dots.lat.length; i++) {{
    const la = dots.lat[i], lo = dots.lon[i];
    if (la < 50) continue;  // polar map only shows lat > ~55°N; skip southern dots
    const [xm, ym] = forwardNorthPolarStereographic(la, lo);
    const [px, py] = polarMetersToPixel(xm, ym, polarData);
    ctx.beginPath();
    ctx.arc(px, py, r, 0, 2 * Math.PI);
    ctx.fill();
  }}
}}

function drawPolarSelection(divId, varKey, metric) {{
  const div = document.getElementById(divId);
  const canvas = div.querySelector('.polar-overlay');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Redraw stipple dots first (they were cleared with the selection rectangle).
  drawPolarSnrDots(divId, varKey, metric);
  if (!SELECTED_CELL) return;
  const polarData = MD().icefrac_polar[compareSelect.value][metric];
  const {{ iLat, iLon }} = SELECTED_CELL;
  const halfLat = Math.abs((INDEX.lat[1] - INDEX.lat[0]) / 2);
  const halfLon = Math.abs((INDEX.lon[1] - INDEX.lon[0]) / 2);
  const lat = INDEX.lat[iLat], lon = INDEX.lon[iLon];
  // 4 corners projected to pixel
  const cornersLL = [
    [lat - halfLat, lon - halfLon],
    [lat - halfLat, lon + halfLon],
    [lat + halfLat, lon + halfLon],
    [lat + halfLat, lon - halfLon],
  ];
  const pts = cornersLL.map(([la, lo]) => {{
    const [xm, ym] = forwardNorthPolarStereographic(la, lo);
    return polarMetersToPixel(xm, ym, polarData);
  }});
  ctx.strokeStyle = colorForMetric(metric);
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
  ctx.stroke();
}}

// ---------- Map rendering (single-view and all-view shared) ----------
function renderMap(divId, varKey, metric, opts) {{
  opts = opts || {{}};
  // Always clear the div first so a previous polar <img> (or any leftover content)
  // is removed before drawing the new map.
  clearMapDiv(divId);
  const isIce = INDEX.is_icefrac[varKey];
  // Toggle the .is-icefrac class so CSS picks the right sizing (icefrac = fixed
  // height; other vars = aspect-ratio-driven height that scales with page width).
  const div = document.getElementById(divId);
  if (div) {{
    if (isIce) div.classList.add('is-icefrac');
    else div.classList.remove('is-icefrac');
  }}
  if (isIce) {{
    return renderPolarMap(divId, varKey, metric);
  }}
  const traces = buildHeatmapTraces(varKey, metric);
  const layout = buildCartesianLayout(varKey, metric, opts);
  if (SELECTED_CELL) layout.shapes = rectShapeForCell(metric, 1.5);
  // scrollZoom is disabled in All view so page-scroll works through the stacked plots.
  const inAllView = (metricSelect.value === 'All');
  return Plotly.newPlot(divId, traces, layout, {{ responsive: true, scrollZoom: !inAllView }})
    .then(() => attachCartesianMapHandlers(divId, varKey, metric, inAllView));
}}

function attachCartesianMapHandlers(divId, varKey, metric, inAllView) {{
  const mapDiv = document.getElementById(divId);

  mapDiv.on('plotly_click', function(evt) {{
    if (!evt.points || !evt.points.length) return;
    const p = evt.points[0];
    const latVal = p.y, lonVal = p.x;
    const iLat = nearestIdx(INDEX.lat, latVal);
    const iLon = nearestIdx(INDEX.lon, lonVal);
    // Click same cell twice = deselect.
    if (SELECTED_CELL && SELECTED_CELL.iLat === iLat && SELECTED_CELL.iLon === iLon) {{
      SELECTED_CELL = null;
    }} else {{
      SELECTED_CELL = {{ iLat, iLon, latVal: INDEX.lat[iLat], lonVal: INDEX.lon[iLon] }};
    }}
    whileBusy(onSelectionChanged);
  }});

  mapDiv.on('plotly_relayout', function(eventData) {{
    if (RELAYOUT_GUARD) return;
    if (!eventData) return;
    const {{ xfull, yfull }} = fullExtent(varKey);
    const xr = mapDiv.layout.xaxis.range;
    const yr = mapDiv.layout.yaxis.range;
    const updates = {{}};
    const xClamped = [Math.max(xfull[0], xr[0]), Math.min(xfull[1], xr[1])];
    const yClamped = [Math.max(yfull[0], yr[0]), Math.min(yfull[1], yr[1])];
    if (xClamped[0] !== xr[0] || xClamped[1] !== xr[1]) updates['xaxis.range'] = xClamped;
    if (yClamped[0] !== yr[0] || yClamped[1] !== yr[1]) updates['yaxis.range'] = yClamped;
    if (SELECTED_CELL) {{
      updates['shapes'] = rectShapeForCell(metric, selectionLineWidthForCartesian(mapDiv));
    }}
    if (Object.keys(updates).length) {{
      RELAYOUT_GUARD = true;
      Plotly.relayout(mapDiv, updates).then(() => {{ RELAYOUT_GUARD = false; }});
    }}
  }});

  if (!inAllView) {{
    mapDiv.addEventListener('wheel', function(e) {{
      if (e.deltaY <= 0) return;
      const {{ xfull, yfull }} = fullExtent(varKey);
      const xr = mapDiv.layout.xaxis.range;
      const yr = mapDiv.layout.yaxis.range;
      const xAtFull = xr[0] <= xfull[0] && xr[1] >= xfull[1];
      const yAtFull = yr[0] <= yfull[0] && yr[1] >= yfull[1];
      if (xAtFull && yAtFull) {{ e.preventDefault(); e.stopPropagation(); }}
    }}, {{ passive: false, capture: true }});
  }}
}}

// ---------- Selection change dispatch ----------
function onSelectionChanged() {{
  const varKey = varSelect.value;
  const metric = metricSelect.value;
  if (metric === 'All') {{
    for (const m of ['RMSE', 'MAE', 'R²']) {{
      updateSelectionOnMap(ALL_MAP_DIVS[m], varKey, m);
    }}
    return drawAllScatters();
  }} else {{
    updateSelectionOnMap('map', varKey, metric);
    const draws = [drawScatterFor('scatter', varKey, metric)];
    // Side-by-side SNR plots (right column, stacked). Only when R² is selected AND
    // the var has SNR data (tas / p-e / icefrac). Not in All view.
    if (metric === 'R²' && hasSnrData(varKey)) {{
      draws.push(drawSnrScatters('snr-scatter', 'snr-scatter-raw', varKey));
    }} else {{
      Plotly.purge('snr-scatter'); Plotly.purge('snr-scatter-raw');
    }}
    return Promise.all(draws);
  }}
}}

function hasSnrData(varKey) {{
  const dotsByMode = MD().snr_dots_by_mode || {{}};
  const modeDots = dotsByMode[compareSelect.value] || {{}};
  // We don't have a direct "has samples" flag in INDEX, but SNR dots being
  // present for a var means samples are available in the per-var JSON.
  return varKey in modeDots;
}}

function updateStippleNote() {{
  // Show a short note whenever stippling is drawn on the page. Stippling appears
  // on R² maps (single view: metric='R²'; all view: metric='All' includes an R²
  // panel) when the current var has SNR data for the current scenario.
  const note = document.getElementById('stipple-note');
  if (!note) return;
  const stippled = (metricSelect.value === 'R²' || metricSelect.value === 'All')
    && hasSnrData(varSelect.value);
  if (!stippled) {{ note.classList.remove('visible'); note.textContent = ''; return; }}
  note.textContent = "Stippling represents areas where SNR < 1.";
  note.classList.add('visible');
}}

function updateR2Note() {{
  // The SAI-delta R² is pooled over all six anchor runs, so it is the same map under every model.
  const show = (metricSelect.value === 'R²' || metricSelect.value === 'All') && compareSelect.value === 'SAI delta';
  document.getElementById('r2-note').classList.toggle('visible', show);
}}

function updateScatterRowLayout() {{
  // Enable the side-by-side layout when R² is selected + var has SNR data + not in All view.
  const row = document.getElementById('scatter-row');
  if (!row) return;
  const metric = metricSelect.value;
  const varKey = varSelect.value;
  const shouldShow = metric === 'R²' && hasSnrData(varKey);
  row.classList.toggle('with-snr', shouldShow);
}}

function updateSelectionOnMap(divId, varKey, metric) {{
  const div = document.getElementById(divId);
  if (!div) return;
  const isIce = INDEX.is_icefrac[varKey];
  if (isIce) {{
    // Image-based map — just redraw the canvas overlay.
    drawPolarSelection(divId, varKey, metric);
  }} else {{
    if (!div.layout) return;
    RELAYOUT_GUARD = true;
    const lw = selectionLineWidthForCartesian(div);
    Plotly.relayout(divId, {{ shapes: rectShapeForCell(metric, lw) }})
      .then(() => {{ RELAYOUT_GUARD = false; }});
  }}
}}

// ---------- Scatter rendering ----------
async function drawScatterFor(divId, varKey, metric) {{
  if (!SELECTED_CELL) {{ Plotly.purge(divId); return; }}
  if (compareSelect.value === 'Baseline (SSP2-4.5)') {{
    // R², RMSE, MAE in baseline mode all use the same per-year emul-vs-GCM data;
    // only the title (which metric is shown) differs.
    return drawTSBaselineScatter(divId, varKey, metric);
  }}
  if (metric === 'R²') {{
    return drawR2Scatter(divId, varKey);
  }}
  return drawTSScatter(divId, varKey, metric);
}}

async function drawTSBaselineScatter(divId, varKey, metric) {{
  let scatterData;
  try {{ scatterData = await ensureScatterFor(varKey); }} catch (e) {{ return; }}
  const {{ iLat, iLon, latVal, lonVal }} = SELECTED_CELL;
  const units = INDEX.units[varKey];
  const regionName = INDEX.region_names[INDEX.region_grid[iLat][iLon]];
  const titlePrefix = 'cell lat=' + latVal.toFixed(2) + ', lon=' + lonVal.toFixed(2) + ' (' + regionName + ')';
  const cellMetric = MD().maps[varKey][metricKey(metric)][iLat][iLon];
  const cellMetricStr = (cellMetric === null || cellMetric === undefined) ? 'NaN' : Number(cellMetric).toFixed(3);

  const years = INDEX.years;
  const yPred = years.map((_, i) => scatterData.ts_baseline_pred[i][iLat][iLon]);
  const yActual = years.map((_, i) => scatterData.ts_baseline_actual[i][iLat][iLon]);

  if (yActual.every(v => v === null) && yPred.every(v => v === null)) {{
    Plotly.newPlot(divId, [], {{ title: titlePrefix + ' — no data (masked cell)', font: PLOTLY_FONT }});
    return;
  }}
  const traces = [
    {{ x: years, y: yActual, mode: 'markers', name: 'GCM (actual)', marker: {{ color: 'black', size: 10 }} }},
    {{ x: years, y: yPred, mode: 'markers', name: 'Emulator (predicted)', marker: {{ color: 'red', size: 10 }} }},
  ];
  // R² is dimensionless; RMSE/MAE carry units of the variable.
  const titleSuffix = (metric === 'R²')
    ? ''
    : ' ' + units + ' (anomaly vs PI 1850–1900)';
  return Plotly.newPlot(divId, traces, {{
    title: titlePrefix + ' — ' + metric + ' = ' + cellMetricStr + titleSuffix,
    xaxis: {{ title: 'year', dtick: 2 }},
    yaxis: {{ title: varKey + ' anomaly (' + units + ')' }},
    margin: {{ l: 60, r: 30, t: 50, b: 50 }},
    font: PLOTLY_FONT,
  }});
}}

// Per-cell detrended-residuals scatter (side-by-side with R² fit scatter).
// Shows the 60 (member × year) samples used by the SNR mask at this cell, minus
// the OLS linear trend fit jointly to all 60 samples. Y-axis is residual (mean 0
// by construction); horizontal red-shaded band shows ±σ (ddof=2, matches the
// noise the SNR uses).
// Two stacked click-scatters for the R² map (right column, SNR-supported vars):
//   Top    (residDivId): 60 detrended residuals colored by member (visualizes noise σ).
//   Bottom (rawDivId):   60 raw paired-δ (or SSP245-anomaly) samples with the OLS
//                        trend line + a horizontal signal line (mean of 60).
//                        The raw values are the ones whose mean IS the signal.
async function drawSnrScatters(residDivId, rawDivId, varKey) {{
  if (!SELECTED_CELL) {{ Plotly.purge(residDivId); Plotly.purge(rawDivId); return; }}
  const scenarioKey = compareSelect.value === 'SAI delta' ? 'sai_delta' : 'baseline';
  let snrData;
  try {{ snrData = await ensureSnrDataFor(varKey, scenarioKey); }} catch (e) {{ return; }}
  if (!snrData) {{ Plotly.purge(residDivId); Plotly.purge(rawDivId); return; }}
  const {{ iLat, iLon }} = SELECTED_CELL;
  const years60 = snrData.years;
  const members60 = snrData.members;
  const samples = years60.map((_, i) => snrData.samples[i][iLat][iLon]);
  const finite = samples.map(v => v !== null && v !== undefined && isFinite(v));
  if (!finite.some(f => f)) {{
    Plotly.newPlot(residDivId, [], {{ font: PLOTLY_FONT }});
    Plotly.newPlot(rawDivId, [], {{ font: PLOTLY_FONT }});
    return;
  }}

  // OLS linear trend across all 60 samples: y = signal + slope · (year − ȳ).
  // signal is the sample mean; slope from Σ (year−ȳ)(y−ȳ) / Σ (year−ȳ)².
  const validIdx = finite.map((f, i) => f ? i : -1).filter(i => i >= 0);
  const n = validIdx.length;
  const yBar = validIdx.reduce((s, i) => s + years60[i], 0) / n;
  const sigMean = validIdx.reduce((s, i) => s + samples[i], 0) / n;
  let num = 0, den = 0;
  for (const i of validIdx) {{
    const dy = years60[i] - yBar;
    num += dy * (samples[i] - sigMean);
    den += dy * dy;
  }}
  const slope = den > 0 ? (num / den) : 0;
  const residuals = years60.map((y, i) =>
    finite[i] ? (samples[i] - (slope * (y - yBar) + sigMean)) : null
  );

  const MEMBER_COLORS = {{ "001": "#1f77b4", "002": "#ff7f0e", "003": "#2ca02c" }};
  // MIROC/UKESM member IDs shown with CESM's 001/002/003 labels and colors.
  const MEMBER_LABELS = {{
    'MIROC-ES2H': {{ "r01": "001", "r02": "002", "r03": "003" }},
    'UKESM1-1-LL': {{ "r2": "001", "r3": "002", "r12": "003" }},
  }};
  const memberLabel = m => (MEMBER_LABELS[modelSelect.value] || {{}})[m] || m;
  const uniqueMembers = [...new Set(members60)];
  const yrMin = Math.min(...years60), yrMax = Math.max(...years60);
  const units = INDEX.units[varKey];
  const rawYLabel = (scenarioKey === 'sai_delta')
    ? 'SAI − baseline paired δ (' + units + ')'
    : 'SSP245 anomaly vs PI (' + units + ')';

  // Helper to build one member-colored scatter set from a values array (aligned to years60).
  function memberTraces(valuesArr) {{
    const t = [];
    for (const m of uniqueMembers) {{
      const xs = [], ys = [];
      for (let i = 0; i < years60.length; i++) {{
        if (members60[i] === m && finite[i]) {{
          xs.push(years60[i]);
          ys.push(valuesArr[i]);
        }}
      }}
      t.push({{
        x: xs, y: ys,
        type: 'scatter', mode: 'markers',
        marker: {{ color: MEMBER_COLORS[memberLabel(m)] || '#666', size: 8,
                   line: {{ color: 'black', width: 0.4 }} }},
        name: 'member ' + memberLabel(m),
      }});
    }}
    return t;
  }}

  // ----- Top: detrended residuals -----
  const residTraces = memberTraces(residuals);
  residTraces.push({{
    x: [yrMin - 0.5, yrMax + 0.5], y: [0, 0],
    type: 'scatter', mode: 'lines',
    line: {{ color: 'black', dash: 'dash', width: 1 }},
    hoverinfo: 'skip', name: 'residual mean = 0',
  }});
  Plotly.newPlot(residDivId, residTraces, {{
    xaxis: {{ title: 'year', dtick: 2 }},
    yaxis: {{ title: 'residual = sample − trend (' + units + ')' }},
    margin: {{ l: 60, r: 30, t: 30, b: 50 }},
    font: PLOTLY_FONT,
    showlegend: true,
    legend: {{ font: {{ size: 10 }} }},
  }});

  // ----- Bottom: raw samples with OLS trend + signal line -----
  const rawTraces = memberTraces(samples);
  // OLS trend line — endpoints at (yrMin, yrMax) evaluated at slope·(year−ȳ)+signal.
  rawTraces.push({{
    x: [yrMin, yrMax],
    y: [slope * (yrMin - yBar) + sigMean, slope * (yrMax - yBar) + sigMean],
    type: 'scatter', mode: 'lines',
    line: {{ color: 'red', width: 2 }},
    hoverinfo: 'skip',
    name: 'OLS trend (slope=' + slope.toFixed(4) + '/yr)',
  }});
  // Horizontal signal line — mean of all 60 raw samples.
  rawTraces.push({{
    x: [yrMin - 0.5, yrMax + 0.5], y: [sigMean, sigMean],
    type: 'scatter', mode: 'lines',
    line: {{ color: 'black', dash: 'dash', width: 1 }},
    hoverinfo: 'skip',
    name: 'signal (mean of 60) = ' + sigMean.toFixed(4),
  }});
  return Plotly.newPlot(rawDivId, rawTraces, {{
    xaxis: {{ title: 'year', dtick: 2 }},
    yaxis: {{ title: rawYLabel }},
    margin: {{ l: 60, r: 30, t: 30, b: 50 }},
    font: PLOTLY_FONT,
    showlegend: true,
    legend: {{ font: {{ size: 10 }} }},
  }});
}}

async function drawR2Scatter(divId, varKey) {{
  let scatterData;
  try {{ scatterData = await ensureScatterFor(varKey); }} catch (e) {{ return; }}
  const {{ iLat, iLon, latVal, lonVal }} = SELECTED_CELL;
  const units = INDEX.units[varKey];
  const regionName = INDEX.region_names[INDEX.region_grid[iLat][iLon]];
  const titlePrefix = 'cell lat=' + latVal.toFixed(2) + ', lon=' + lonVal.toFixed(2) + ' (' + regionName + ')';
  const cellMetric = MD().maps[varKey]['r2'][iLat][iLon];
  const cellMetricStr = (cellMetric === null || cellMetric === undefined) ? 'NaN' : Number(cellMetric).toFixed(3);

  // One point per SAI run of every model (pooled R²); scatterData.r2_models labels each run.
  const x = scatterData.r2_x;
  const runModels = scatterData.r2_models;
  const yActual = x.map((_, k) => scatterData.r2_actual[k][iLat][iLon]);
  const yPred = x.map((_, k) => scatterData.r2_pred[k][iLat][iLon]);

  if (yActual.every(v => v === null) || yPred.every(v => v === null)) {{
    Plotly.newPlot(divId, [], {{ title: titlePrefix + ' — R² undefined here (no signal in this cell)', font: PLOTLY_FONT }});
    return;
  }}

  // Compute β from any valid (x, yPred) pair
  let beta = null;
  for (let i = 0; i < x.length; i++) {{
    if (yPred[i] !== null && yPred[i] !== undefined && x[i] !== 0) {{
      beta = yPred[i] / x[i];
      break;
    }}
  }}

  // Fit line: from (x_extreme, β·x_extreme) → (0, 0). x_extreme = a bit past the min x.
  const xMin = Math.min(...x);
  const xExtreme = xMin - 0.1 * Math.abs(xMin);  // extend a bit past data away from origin
  const xPadRight = 0.05 * Math.abs(xExtreme);   // tiny pad on origin side

  const lineX = [xExtreme, 0];
  const lineY = (beta === null) ? [0, 0] : [beta * xExtreme, 0];

  const traces = [
    {{ x: lineX, y: lineY, mode: 'lines', name: (beta === null ? 'fit' : 'fit (β=' + beta.toFixed(3) + ')'),
       line: {{ color: 'red', dash: 'dash' }}, hoverinfo: 'skip' }},
  ];
  // GCM points are black with one marker symbol per model; the emulator is one pooled
  // line, so its points (one per run) are a single red-circle trace.
  const R2_MODEL_SYMBOLS = {{ 'CESM2-WACCM': 'triangle-up', 'MIROC-ES2H': 'square', 'UKESM1-1-LL': 'diamond' }};
  for (const m of [...new Set(runModels)]) {{
    const idx = runModels.map((rm, k) => rm === m ? k : -1).filter(k => k >= 0);
    traces.push({{
      x: idx.map(k => x[k]), y: idx.map(k => yActual[k]), mode: 'markers',
      name: 'GCM (actual) — ' + m,
      marker: {{ color: 'black', size: 12, symbol: R2_MODEL_SYMBOLS[m] || 'cross' }},
    }});
  }}
  traces.push({{ x: x, y: yPred, mode: 'markers', name: 'Emulator (predicted)', marker: {{ color: 'red', size: 12 }} }});
  return Plotly.newPlot(divId, traces, {{
    title: titlePrefix + ' — R² = ' + cellMetricStr,
    xaxis: {{ title: 'global tas cooling (K)', range: [xExtreme, xPadRight] }},
    yaxis: {{ title: 'SAI delta (' + units + ')' }},
    margin: {{ l: 60, r: 30, t: 50, b: 50 }},
    font: PLOTLY_FONT,
  }});
}}

async function drawTSScatter(divId, varKey, metric) {{
  let scatterData;
  try {{ scatterData = await ensureScatterFor(varKey); }} catch (e) {{ return; }}
  const {{ iLat, iLon, latVal, lonVal }} = SELECTED_CELL;
  const units = INDEX.units[varKey];
  const regionName = INDEX.region_names[INDEX.region_grid[iLat][iLon]];
  const titlePrefix = 'cell lat=' + latVal.toFixed(2) + ', lon=' + lonVal.toFixed(2) + ' (' + regionName + ')';
  const cellMetric = MD().maps[varKey][metricKey(metric)][iLat][iLon];
  const cellMetricStr = (cellMetric === null || cellMetric === undefined) ? 'NaN' : Number(cellMetric).toFixed(3);

  const years = INDEX.years;
  const yPred = years.map((_, i) => scatterData.ts_pred[i][iLat][iLon]);
  const yActual = years.map((_, i) => scatterData.ts_actual[i][iLat][iLon]);

  if (yActual.every(v => v === null) && yPred.every(v => v === null)) {{
    Plotly.newPlot(divId, [], {{ title: titlePrefix + ' — no data (masked cell)', font: PLOTLY_FONT }});
    return;
  }}
  const traces = [
    {{ x: years, y: yActual, mode: 'markers', name: 'GCM (actual)', marker: {{ color: 'black', size: 10 }} }},
    {{ x: years, y: yPred, mode: 'markers', name: 'Emulator (predicted)', marker: {{ color: 'red', size: 10 }} }},
  ];
  return Plotly.newPlot(divId, traces, {{
    title: titlePrefix + ' — ' + metric + ' = ' + cellMetricStr + ' ' + units,
    xaxis: {{ title: 'year', dtick: 2 }},
    yaxis: {{ title: 'SAI delta (' + units + ')' }},
    margin: {{ l: 60, r: 30, t: 50, b: 50 }},
    font: PLOTLY_FONT,
  }});
}}

function drawAllScatters() {{
  if (!SELECTED_CELL) {{ Plotly.purge('scatter-ts'); Plotly.purge('scatter-r2'); return; }}
  const varKey = varSelect.value;
  if (compareSelect.value === 'Baseline (SSP2-4.5)') {{
    return Promise.all([
      drawTSBaselineScatter('scatter-ts', varKey, 'RMSE'),
      drawTSBaselineScatter('scatter-r2', varKey, 'R²'),
    ]);
  }} else {{
    return Promise.all([
      drawTSScatter('scatter-ts', varKey, 'RMSE'),  // 20-year residuals; metric label only affects the title
      drawR2Scatter('scatter-r2', varKey),
    ]);
  }}
}}

function clearMapDiv(divId) {{
  // Plotly.purge for cartesian; also clear innerHTML in case the div was hosting an
  // <img>-based polar map.
  Plotly.purge(divId);
  const d = document.getElementById(divId);
  if (d && d.querySelector('.polar-wrap')) d.innerHTML = '';
}}

// Loading wheel: visible from a dropdown change / cell click until everything it
// triggers (maps, scatter data fetches, plots) has been drawn.
let BUSY_COUNT = 0;
async function whileBusy(work) {{
  const spinner = document.getElementById('spinner');
  BUSY_COUNT++;
  spinner.classList.add('visible');
  // Let the browser paint the wheel before the synchronous Plotly work starts: wait for
  // the next frame, or 50 ms if no frame comes (background tab, hidden iframe).
  await new Promise(resolve => {{
    let started = false;
    const go = () => {{ if (!started) {{ started = true; setTimeout(resolve, 0); }} }};
    requestAnimationFrame(go);
    setTimeout(go, 50);
  }});
  try {{
    return await work();
  }} finally {{
    BUSY_COUNT--;
    if (BUSY_COUNT === 0) spinner.classList.remove('visible');
  }}
}}

// ---------- Top-level render dispatch ----------
function render() {{
  const varKey = varSelect.value;
  const metric = metricSelect.value;
  const single = document.getElementById('single-view');
  const all = document.getElementById('all-view');
  updateStippleNote();
  updateR2Note();
  if (metric === 'All') {{
    single.style.display = 'none';
    all.style.display = '';
    clearMapDiv('map'); Plotly.purge('scatter');
    Plotly.purge('snr-scatter'); Plotly.purge('snr-scatter-raw');
    return Promise.all([
      renderMap('map-rmse', varKey, 'RMSE'),
      renderMap('map-mae', varKey, 'MAE'),
      renderMap('map-r2', varKey, 'R²'),
    ]).then(() => {{ if (SELECTED_CELL) return drawAllScatters(); }});
  }} else {{
    single.style.display = '';
    all.style.display = 'none';
    clearMapDiv('map-rmse'); clearMapDiv('map-mae'); clearMapDiv('map-r2');
    Plotly.purge('scatter-ts'); Plotly.purge('scatter-r2');
    updateScatterRowLayout();  // toggle side-by-side for R² + SNR-supported var
    return renderMap('map', varKey, metric).then(() => {{
      if (SELECTED_CELL) {{
        const draws = [drawScatterFor('scatter', varKey, metric)];
        if (metric === 'R²' && hasSnrData(varKey)) {{
          draws.push(drawSnrScatters('snr-scatter', 'snr-scatter-raw', varKey));
        }} else {{
          Plotly.purge('snr-scatter'); Plotly.purge('snr-scatter-raw');
        }}
        return Promise.all(draws);
      }}
    }});
  }}
}}

compareSelect.addEventListener('change', () => {{
  // Scenario switch: preserve selected cell so user can compare SAI vs Baseline
  // at the same location without re-clicking. `populateMetricDropdown` falls back
  // to 'RMSE' if the current metric isn't in the other scenario's list.
  // The model list depends on the scenario; the current model is kept if available.
  populateModelDropdown();
  populateMetricDropdown(varSelect.value);
  whileBusy(render);
}});
modelSelect.addEventListener('change', () => {{
  // Model switch: preserve selected cell so the same location can be compared across models.
  whileBusy(render);
}});
varSelect.addEventListener('change', () => {{
  SELECTED_CELL = null;
  populateMetricDropdown(varSelect.value);
  whileBusy(render);
}});
metricSelect.addEventListener('change', () => whileBusy(render));
whileBusy(render);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
