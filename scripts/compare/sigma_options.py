"""What-if maps for the p-value test σ: internal, between-model, and the resulting stipple.

For one variable and the harness's combined intermediates, draws (last decade, with SAI):

* ``σ_pooled``  = sqrt(mean_m σ²_int,m) — the year-to-year noise one realization shows;
* ``σ_model``   = std over models (ddof=0) of each model's decadal-mean with-SAI field —
  where the models disagree about the outcome;
* their ratio; today's stipple (p<0.05 from the baseline dump); and the stipple under three
  candidate test σ's, all with the "detectability of a decadal mean" statistic
  ``z = Δ / SE``, ``SE_int = σ · sqrt(((1+ρ)/(1−ρ)) / k)`` (k = 10 years, per-cell lag-1 ρ):

  - **A** σ = σ_pooled
  - **B** σ = sqrt(Σσ²_m)/M (noise of the ensemble mean)
  - **C** SE = sqrt(SE_int(σ_pooled)² + σ²_model) — model spread added, not averaged down
    over years (it is not year-to-year noise).

Δ is the decadal with-SAI mean minus the 1850-1900 reference. Read-only; nothing in
``backend/`` changes. Outputs ``<out>/<var>_sigma_options.png`` and a JSON of the percentages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import norm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.constants import exposurevar2var  # noqa: E402
from backend.population import get_population_xr  # noqa: E402
from backend.utils import clip_to_land, clip_to_ocean  # noqa: E402
from scripts.variability import _detrended_residuals, resolve_all_members_file  # noqa: E402

REBASED_VARS = ("tas", "tasmin", "tasmax", "pr", "p-e")
K_YEARS = 10
P_THRESHOLD = 0.05


def _lag1_per_cell(var: str, models: list[str], data_dir: Path) -> xr.DataArray:
    """Per-cell lag-1 autocorrelation of detrended annual residuals, averaged over members and models."""
    daily_var = exposurevar2var[var] if "exposure" in var else var
    rhos = []
    for model in models:
        path = resolve_all_members_file(data_dir / daily_var, model)
        if path is None:
            continue
        da = xr.open_dataset(path)[daily_var]
        if daily_var == "icefrac":
            da = da.fillna(0.0)
        res = _detrended_residuals(da)
        x0, x1 = res.isel(time=slice(None, -1)), res.isel(time=slice(1, None))
        x1 = x1.assign_coords(time=x0.time)
        num = (x0 * x1).sum("time")
        den = (res**2).sum("time")
        rhos.append((num / den.where(den > 0)).mean("member"))
    rho = xr.concat(rhos, dim="model", join="override").mean("model").fillna(0.0)
    return rho.clip(-0.9, 0.9)


def _on_grid(field: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Pin ``field`` to ``ref``'s lat/lon coordinates (same shape, float-digit differences).

    Artifacts written from ``area.nc`` (icefrac σ) and from ``correct_lat.npy`` (maps) carry
    latitudes that differ in the last digits; xarray arithmetic between them would silently
    inner-join to the ~125 latitudes that match exactly (review doc, X6).
    """
    assert field.shape[-2:] == ref.shape[-2:], (field.shape, ref.shape)
    return field.assign_coords(lat=ref.lat.values, lon=ref.lon.values)


def _grid_sigma(var: str, model: str, data_dir: Path, ref: xr.DataArray) -> xr.DataArray:
    daily_var = exposurevar2var[var] if "exposure" in var else var
    with xr.open_dataarray(data_dir / daily_var / f"grid_level_model_internal_variability_{model}.nc") as da:
        sigma = _on_grid(da.load(), ref)
    if "exposure" in var:
        # Multiply by values, as backend.p_values does: the population grid's float lat
        # coords differ from the sigma file's in the last digits, and an xarray product
        # would silently inner-join to the ~125 latitudes that match exactly.
        pop = get_population_xr(data_dir)
        pop_values = pop.transpose("lat", "lon").values if set(pop.dims) == {"lat", "lon"} else pop.values
        assert pop_values.shape == sigma.shape, (pop_values.shape, sigma.shape)
        sigma = sigma * pop_values / 1.0e6
    return sigma


def _mask_like_backend(var: str, data_dir: Path, field: xr.DataArray) -> xr.DataArray:
    if var == "icefrac":
        return clip_to_ocean(data_dir, field).where(field.lat >= 60)
    return clip_to_land(data_dir, field)


def sigma_options(
    var: str,
    inter_dir: str | Path,
    baseline_qdir: str | Path,
    out_dir: str | Path,
    data_dir: str | Path = "data/processed",
    decade: tuple[int, int] = (2091, 2100),
) -> dict:
    inter, base, out, data_dir = Path(inter_dir), Path(baseline_qdir), Path(out_dir), Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    def load(name, required=True):
        f = inter / f"{name}.nc"
        if not f.exists():
            if required:
                raise FileNotFoundError(f)
            return None
        with xr.open_dataarray(f) as da:
            return da.load()

    regional_map = load("regional_map")
    regional_delta = load("regional_delta")
    rebase = load("historical_rebase")
    if var == "icefrac":
        per_model_with_sai = load("icefrac_with_sai_per_model")
    else:
        per_model_map = load("regional_map_per_model")
        per_model_with_sai = per_model_map + regional_delta
    models = [str(m) for m in per_model_with_sai.model.values]
    M = len(models)

    W_m = per_model_with_sai.sel(time=slice(*decade)).mean("time")  # (model, lat, lon)
    W = W_m.mean("model")
    if var in REBASED_VARS:
        delta = W  # per-model maps are anomalies already
    else:
        delta = W - rebase

    sig_m = xr.concat([_grid_sigma(var, m, data_dir, delta) for m in models], dim="model").assign_coords(model=models)
    sigma_pooled = np.sqrt((sig_m**2).mean("model"))
    sigma_ens = np.sqrt((sig_m**2).sum("model")) / M
    sigma_model = W_m.std("model", ddof=0)
    rho = _on_grid(_lag1_per_cell(var, models, data_dir), delta)
    infl = np.sqrt(((1 + rho) / (1 - rho)) / K_YEARS)

    def pvals(se):
        z = delta / se.where(se > 0)
        return 2 * (1 - norm.cdf(np.abs(z)))

    options = {
        "A: σ_pooled": pvals(sigma_pooled * infl),
        "B: σ of ensemble mean": pvals(sigma_ens * infl),
        "C: sqrt(SE_int² + σ_model²)": pvals(np.sqrt((sigma_pooled * infl) ** 2 + sigma_model**2)),
    }
    with xr.open_dataarray(base / "p_sai.nc") as da:
        p_today = da.load().sel(time=decade[1])

    masked = {k: _mask_like_backend(var, data_dir, xr.DataArray(v, coords=delta.coords, dims=delta.dims)) for k, v in options.items()}
    valid = p_today.notnull()
    pct = {"today (two-sample t, CESM σ)": float((p_today < P_THRESHOLD).sum() / valid.sum())}
    for k, p in masked.items():
        pct[k] = float(((p < P_THRESHOLD) & valid).sum() / valid.sum())

    ratio = (sigma_model / sigma_pooled.where(sigma_pooled > 0))
    ratio_masked = _mask_like_backend(var, data_dir, ratio)
    fig, axes = plt.subplots(2, 4, figsize=(22, 7.5))
    axes = axes.ravel()
    lon, lat = delta.lon, delta.lat

    def show(ax, field, title, **kw):
        im = ax.pcolormesh(lon, lat, field, shading="auto", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    vmax = float(np.nanpercentile(_mask_like_backend(var, data_dir, sigma_pooled), 98))
    fig.colorbar(show(axes[0], _mask_like_backend(var, data_dir, sigma_pooled), f"{var}: σ_pooled (internal, per year)", vmin=0, vmax=vmax, cmap="viridis"), ax=axes[0], shrink=0.8)
    fig.colorbar(show(axes[1], _mask_like_backend(var, data_dir, sigma_model), "σ_model (between models, decadal mean)", vmin=0, vmax=max(vmax, float(np.nanpercentile(_mask_like_backend(var, data_dir, sigma_model), 98))), cmap="viridis"), ax=axes[1], shrink=0.8)
    fig.colorbar(show(axes[2], np.log10(ratio_masked.where(ratio_masked > 0)), "log10(σ_model / σ_pooled)  (>0: models disagree more than weather)", vmin=-1.5, vmax=1.5, cmap="RdBu_r"), ax=axes[2], shrink=0.8)
    fig.colorbar(show(axes[3], _mask_like_backend(var, data_dir, rho), "lag-1 ρ of residuals", vmin=-0.5, vmax=0.5, cmap="RdBu_r"), ax=axes[3], shrink=0.8)
    show(axes[4], (p_today < P_THRESHOLD).where(valid), f"today's stipple: {100*pct['today (two-sample t, CESM σ)']:.1f}%", vmin=0, vmax=1, cmap="Greys")
    for ax, (k, p) in zip(axes[5:], masked.items()):
        show(ax, (p < P_THRESHOLD).where(valid), f"{k}: {100*pct[k]:.1f}%", vmin=0, vmax=1, cmap="Greys")
    fig.suptitle(f"{var} — with SAI, {decade[0]}-{decade[1]}: what decides the hatching", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{var}_sigma_options.png", dpi=120)
    plt.close(fig)

    summary = {
        "var": var,
        "decade": list(decade),
        "models": models,
        "percent_significant": {k: 100 * v for k, v in pct.items()},
        "median_sigma_model_over_pooled": float(np.nanmedian(ratio_masked)),
        "median_rho": float(np.nanmedian(_mask_like_backend(var, data_dir, rho))),
    }
    (out / f"{var}_sigma_options.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
