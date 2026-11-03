"""Raw model output per climate model, for model-comparison/index.html.

Reads each model's processed no-SAI run (historical stitched to SSP2-4.5, every member, on
the shared 192x288 grid; written by scripts/process_monthly_gauss.py) and writes JSON for
the page. Nothing here goes through the emulator.

* series.json        Global series per variable: each member, each model's ensemble mean
                     and member spread (sigma), the multi-model mean, and observations.
* maps_<var>.json    Decade-mean maps of each model's ensemble mean and the multi-model mean.

Each global series is aggregated as the simulator aggregates its lines: cos-lat mean (p-e
over land only), population-weighted land sum for exposure, sea-ice fraction x cell area for
icefrac. tas, tasmin, tasmax and p-e are anomalies against the model's own ensemble-mean
1850-1900 mean; on the maps, against its own 1995-2014 mean.

Years a member lacks are left out of that year's ensemble mean. Sigma is the standard
deviation across a model's members (ddof=1), only where at least two have data. The
multi-model mean, and its band 2*sqrt(sum sigma_m^2)/M, exist only in years every model has.

Needs data/processed (see the main README), from this repo or from ``--data-root``.

Run from the repo root with the ``sai-simulator`` conda env:
    python model-comparison/pipeline/preview_data.py --out model-comparison/data [--vars tas p-e]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(BACKEND))

import xarray as xr  # noqa: E402

from backend.constants import (  # noqa: E402
    DISPLAY_BASELINE_END,
    DISPLAY_BASELINE_START,
    PREINDUSTRIAL_END,
    PREINDUSTRIAL_START,
    VAR2INFO,
    exposurevar2var,
    regional_models_for_var,
)
from backend.historical import (  # noqa: E402
    get_historical_obs_global_mean_temp,
    get_historical_obs_sea_ice_extent,
)
from backend.icefrac import get_area  # noqa: E402
from backend.population import get_population_exposure  # noqa: E402
from backend.uncertainty import Z, internal_sigma_of_mean  # noqa: E402
from backend.utils import clip_to_land, regional_aggregation  # noqa: E402
from scripts.variability import resolve_all_members_file  # noqa: E402

YEARS = np.arange(1850, 2100)
DECADE = (2090, 2099)
# Maps use the last decade every model has: MIROC's p-e stops in 2084 (its ssp245 evaporation
# is published for 2020-2084 only), CESM's tasmin/tasmax in 2069 (its baseline store's
# tasminmax group covers 2015-2069).
VAR_DECADE = {"p-e": (2075, 2084), "tasmin": (2060, 2069), "tasmax": (2060, 2069)}
ANOMALY_VARS = ("tas", "tasmin", "tasmax", "p-e")


def _r(x, nd=4):
    """Round; NaN/inf -> None for JSON."""
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


def _vals(da, nd=4):
    return [_r(v, nd) for v in da.values]


def _grid(da, nd=3):
    arr = np.asarray(da.transpose("lat", "lon").values, dtype=float)
    return [[_r(v, nd) for v in row] for row in arr]


def members_field(var: str, model: str, data_dir: Path) -> xr.DataArray:
    """The model's members for ``var`` as (member, time, lat, lon), processed as the simulator's
    lines are: day counts floored at 0, land-clipped and population-weighted for exposure, p-e
    land-clipped. Member-years with no data stay NaN."""
    src = exposurevar2var.get(var, var)
    with xr.open_dataset(resolve_all_members_file(data_dir / src, model)) as ds:
        da = ds[src].load()
    for dim in ("model", "ssp"):
        if dim in da.dims:
            da = da.squeeze(dim, drop=True)
    has_data = da.notnull().any(("lat", "lon"))
    da = da.sel(member=has_data.any("time"))
    if "above" in src or "below" in src:
        da = da.clip(min=0)  # NaN-preserving floor
    if "exposure" in var or var == "p-e":
        da = clip_to_land(data_dir, da)
    if "exposure" in var:
        da = get_population_exposure(data_dir, da)
    da = da.where(has_data.sel(member=da.member))
    return da.reindex(time=YEARS)


def global_series(var: str, field: xr.DataArray, data_dir: Path) -> xr.DataArray:
    """(member, time) global series, aggregated as get_outputs aggregates its lines."""
    has_data = field.notnull().any(("lat", "lon"))
    if var == "icefrac":
        ones = xr.ones_like(field.lat)
        out = regional_aggregation(field * get_area(data_dir).broadcast_like(field), ones, "sum")
    elif "exposure" in var:
        out = regional_aggregation(field, xr.ones_like(field.lat), "sum")
    else:
        out = regional_aggregation(field, np.cos(np.deg2rad(field.lat)), "mean")
    return out.where(has_data)  # an all-NaN year would otherwise sum to 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars", nargs="*", default=list(VAR2INFO))
    ap.add_argument("--data-root", default=str(BACKEND),
                    help="checkout whose data/processed to read (default: this repo)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_root) / "data/processed"
    obs = {"tas": get_historical_obs_global_mean_temp(data_dir),
           "icefrac": get_historical_obs_sea_ice_extent(data_dir)}

    series_out = {"time": [int(y) for y in YEARS], "vars": {}}
    for var in args.vars:
        decade = VAR_DECADE.get(var, DECADE)
        models = regional_models_for_var(var)
        entry = {"label": VAR2INFO[var][1], "decade": decade, "models": {}}
        maps = {"var": var, "label": VAR2INFO[var][1], "decade": decade, "models": {}}
        means, sigmas, model_maps = [], [], []
        for m in models:
            field = members_field(var, m, data_dir)
            series = global_series(var, field, data_dir)
            if var in ANOMALY_VARS:
                series = series - series.mean("member").sel(
                    time=slice(PREINDUSTRIAL_START, PREINDUSTRIAL_END)).mean("time")
            n = series.notnull().sum("member")
            mean = series.mean("member")
            sigma = series.std("member", ddof=1).where(n >= 2)
            means.append(mean)
            sigmas.append(sigma)
            entry["models"][m] = {
                "mean": _vals(mean), "sigma": _vals(sigma), "n_members": [int(k) for k in n.values],
                "members": {str(k): _vals(series.sel(member=k)) for k in series.member.values},
            }

            ens = field.mean("member")
            dec = ens.sel(time=slice(*decade)).mean("time")
            if var in ANOMALY_VARS:
                dec = dec - ens.sel(time=slice(DISPLAY_BASELINE_START, DISPLAY_BASELINE_END)).mean("time")
            model_maps.append(dec)
            maps["models"][m] = _grid(dec)
            print(f"{var} {m}: members {[str(k) for k in series.member.values]}", flush=True)

        # Multi-model mean: only where every model has a value (no NaN-skipping).
        mmm = xr.concat(means, "model").mean("model", skipna=False)
        sig = xr.concat(sigmas, "model")
        half = xr.apply_ufunc(
            lambda s: Z * internal_sigma_of_mean(list(s)) if np.isfinite(s).all() else np.nan,
            sig, input_core_dims=[["model"]], vectorize=True,
        )
        entry["mmm"] = {"mean": _vals(mmm), "half_band": _vals(half)}
        if var in obs:
            o = obs[var]
            entry["obs"] = {"time": [int(t) for t in o.time.values], "values": _vals(o)}
        series_out["vars"][var] = entry

        maps["lat"] = [_r(v, 3) for v in model_maps[0].lat.values]
        maps["lon"] = [_r(v, 3) for v in model_maps[0].lon.values]
        maps["mmm"] = _grid(xr.concat(model_maps, "model").mean("model", skipna=False))
        (out / f"maps_{var}.json").write_text(json.dumps(maps, separators=(",", ":")))

    (out / "series.json").write_text(json.dumps(series_out, separators=(",", ":")))


if __name__ == "__main__":
    main()
