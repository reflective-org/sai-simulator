"""Render before/after (or config-vs-config) figures from harness quantity dumps.

Inputs are N quantity directories written by :func:`scripts.compare.dump.compute_quantities`
(``<root>/<config>/<sha>/<params.key>/<var>``). The same figure set is drawn whether the
inputs differ by code revision (before/after a fix) or by model config (cesm-only vs
miroc-only vs combined):

* ``pvalues``  — fraction of valid cells with p < 0.05 per decade, raw and under a
  Benjamini–Hochberg FDR control at q = 0.05, for no-SAI and with-SAI; plus p < 0.05 masks
  for the last decade, one panel per input, and a "changed cells" panel (first vs last input).
* ``series``   — mean-over-time no-SAI / with-SAI / historical overlaid across inputs, with
  the uncertainty band each dump carries: the combined prediction interval (outer) and 95%
  CI of the ensemble mean (inner) from ``backend.uncertainty`` when present, otherwise the
  legacy icefrac-only inter-model ``model_spread_2sigma_*`` or ±2 × the internal-variability
  scalar. The band source is named in the legend so two different meanings of "uncertainty"
  never share a label silently.
* ``histogram`` — the no-SAI / with-SAI / historical distributions as step plots, overlaid.
* ``summary.json`` — the numbers behind the figures: significant counts per decade (raw and
  BH), max |Δ| between inputs for each series, L1 distance between histograms.

Plain matplotlib on a lat/lon grid (no cartopy) so it stays fast and dependency-light.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

P_THRESHOLD = 0.05
FDR_Q = 0.05
SERIES_KEYS = ("historical_model", "no_sai", "with_sai")
SERIES_COLORS = {"historical_model": "tab:red", "no_sai": "tab:red", "with_sai": "tab:blue"}
SERIES_STYLES = {"historical_model": "-", "no_sai": "-", "with_sai": "--"}
INPUT_ALPHAS = (1.0, 0.55, 0.35, 0.25)


def bh_significant_mask(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Benjamini–Hochberg: boolean mask of cells significant at FDR ``q``. NaNs are excluded."""
    flat = p.ravel()
    valid = np.isfinite(flat)
    out = np.zeros(flat.shape, dtype=bool)
    pv = flat[valid]
    m = pv.size
    if m == 0:
        return out.reshape(p.shape)
    order = np.argsort(pv)
    ranked = pv[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    passing = np.nonzero(ranked <= thresholds)[0]
    if passing.size:
        k = passing.max()
        sig_sorted = np.zeros(m, dtype=bool)
        sig_sorted[: k + 1] = True
        sig = np.empty(m, dtype=bool)
        sig[order] = sig_sorted
        out[valid] = sig
    return out.reshape(p.shape)


def _load_quantities(qdir: Path) -> dict:
    qdir = Path(qdir)
    d = {"dir": qdir, "meta": json.loads((qdir / "quantities.json").read_text())}
    for name in ("p_no_sai", "p_sai"):
        f = qdir / f"{name}.nc"
        d[name] = xr.open_dataarray(f).load() if f.exists() else None
    d["series"] = xr.open_dataset(qdir / "mean_over_time.nc").load()
    return d


def _label(d: dict, fallback: str) -> str:
    return fallback or f"{d['meta']['config']}@{d['meta']['sha']}"


def _sig_table(p: xr.DataArray | None) -> list[dict]:
    if p is None:
        return []
    rows = []
    for t in p.time.values:
        arr = p.sel(time=t).values
        valid = int(np.isfinite(arr).sum())
        raw = int((arr < P_THRESHOLD).sum())
        bh = int(bh_significant_mask(arr).sum())
        rows.append({"decade_end": int(t), "valid": valid, "raw": raw, "bh": bh,
                     "raw_frac": raw / valid if valid else float("nan"),
                     "bh_frac": bh / valid if valid else float("nan")})
    return rows


def render_pvalues(inputs: list[dict], labels: list[str], var: str, out: Path) -> dict:
    """Significant-fraction bars per decade (raw vs BH) and last-decade masks."""
    summary: dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, key in zip(axes, ("p_no_sai", "p_sai")):
        width = 0.8 / max(1, 2 * len(inputs))
        for i, (d, lab) in enumerate(zip(inputs, labels)):
            rows = _sig_table(d[key])
            summary.setdefault(key, {})[lab] = rows
            if not rows:
                continue
            x = np.arange(len(rows))
            ax.bar(x + (2 * i) * width, [r["raw_frac"] for r in rows], width,
                   label=f"{lab} — p<{P_THRESHOLD}", alpha=0.85)
            ax.bar(x + (2 * i + 1) * width, [r["bh_frac"] for r in rows], width,
                   label=f"{lab} — BH q={FDR_Q}", alpha=0.45, hatch="//")
            ax.set_xticks(x + width * (len(inputs) - 0.5))
            ax.set_xticklabels([str(r["decade_end"]) for r in rows])
        ax.set_title(f"{var}: {'no-SAI' if key == 'p_no_sai' else 'with-SAI'} — fraction of valid cells significant")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("decade end year")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("fraction significant")
    fig.tight_layout()
    fig.savefig(out / f"{var}_pvalues_fraction.png", dpi=130)
    plt.close(fig)

    # Last-decade masks + changed-cells panel, for with-SAI (the dashboard's stipple)
    key = "p_sai"
    usable = [(d, lab) for d, lab in zip(inputs, labels) if d[key] is not None]
    if usable:
        n = len(usable) + (1 if len(usable) > 1 else 0)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.2), squeeze=False)
        axes = axes[0]
        last = usable[0][0][key].time.values[-1]
        masks = []
        for ax, (d, lab) in zip(axes, usable):
            p = d[key].sel(time=last)
            mask = (p < P_THRESHOLD).where(p.notnull())
            masks.append(mask)
            ax.pcolormesh(p.lon, p.lat, mask, cmap="Greys", vmin=0, vmax=1, shading="auto")
            frac = float(mask.sum() / p.notnull().sum())
            ax.set_title(f"{lab}\nwith-SAI p<{P_THRESHOLD}, {int(last)}: {100 * frac:.1f}%", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        if len(usable) > 1:
            ax = axes[-1]
            a, b = masks[0], masks[-1]
            change = xr.where(a.notnull() & b.notnull(), b.fillna(0) - a.fillna(0), np.nan)
            ax.pcolormesh(a.lon, a.lat, change, cmap="RdBu", vmin=-1, vmax=1, shading="auto")
            gained = int((change == 1).sum())
            lost = int((change == -1).sum())
            ax.set_title(f"changed cells ({usable[-1][1]} vs {usable[0][1]})\n"
                         f"gained significance: {gained}, lost: {lost}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            summary["changed_cells_last_decade"] = {"gained": gained, "lost": lost,
                                                    "from": usable[0][1], "to": usable[-1][1]}
        fig.tight_layout()
        fig.savefig(out / f"{var}_pvalues_masks.png", dpi=130)
        plt.close(fig)
    return summary


_SEGMENT_OF = {"historical_model": "historical", "no_sai": "no_sai", "with_sai": "with_sai"}


def _bands_for(d: dict, series: xr.Dataset, key: str, time: xr.DataArray) -> list[tuple[np.ndarray, str, float]]:
    """Bands to shade around ``series[key]`` as ``(half-width aligned to ``time``, label, alpha)``.

    Prefers the band around the multi-model mean (``band_internal_<seg>`` from
    backend.uncertainty), then the earlier spread + internal band (``band_prediction_<seg>``,
    dumps made before 2026-09-28); falls back to the legacy icefrac-only
    ``model_spread_2sigma_*`` or the internal-variability scalar, so dumps from before the
    fix render with the band they actually had.
    """
    seg = _SEGMENT_OF[key]
    labels = d["meta"].get("bands", {}).get("band_labels", {})
    if f"band_internal_{seg}" in series:
        hw = series[f"band_internal_{seg}"].reindex(time=time).values
        return [(hw, labels.get("internal", "internal variability of the mean"), 0.2)]
    if f"band_prediction_{seg}" in series:
        pi = series[f"band_prediction_{seg}"].reindex(time=time).values
        return [(pi, labels.get("prediction", "prediction interval"), 0.2)]
    spread_key = f"model_spread_2sigma_{key}"
    if spread_key in series:
        hw = series[spread_key].reindex(time=time).values
        return [(hw, "legacy: inter-model 2σ (ddof=1)", 0.15)]
    iv = d["meta"].get("internal_variability")
    if iv is not None:
        return [(np.full(time.size, 2.0 * float(iv)), "legacy: internal variability ×2 (CESM-only scalar)", 0.15)]
    return []


def _band_numbers(d: dict) -> dict:
    """Last-time half-widths and spread components per segment, for summary.json."""
    series = d["series"]
    out: dict = {}
    for seg in ("historical", "no_sai", "with_sai"):
        key = next((f"band_{kind}_{seg}" for kind in ("internal", "prediction") if f"band_{kind}_{seg}" in series), None)
        if key is None:
            continue
        hw = series[key].dropna("time")
        sm = series[f"sigma_model_{seg}"].dropna("time")
        t = int(hw.time.values[-1])
        out[seg] = {
            "time": t,
            "band": key.rsplit(f"_{seg}", 1)[0],
            "band_half_width": float(hw.sel(time=t)),
            "sigma_model": float(sm.sel(time=t)),
            "band_half_width_max": float(hw.max()),
        }
    meta_b = d["meta"].get("bands", {})
    for k in ("band_models", "band_n_models", "band_ddof", "sigma_internal_per_model", "sigma_internal_mean"):
        if k in meta_b:
            out[k] = meta_b[k]
    return out


def render_series(inputs: list[dict], labels: list[str], var: str, out: Path) -> dict:
    """Mean-over-time overlays with each dump's own uncertainty band."""
    summary: dict = {"max_abs_delta_vs_first": {}, "bands": {}}
    fig, ax = plt.subplots(figsize=(11, 5))
    first_series = inputs[0]["series"]
    for i, (d, lab) in enumerate(zip(inputs, labels)):
        s = d["series"]
        alpha = INPUT_ALPHAS[min(i, len(INPUT_ALPHAS) - 1)]
        summary["bands"][lab] = _band_numbers(d)
        band_labelled: set[str] = set()
        for key in SERIES_KEYS:
            if key not in s:
                continue
            da = s[key].dropna("time")
            ax.plot(da.time, da, color=SERIES_COLORS[key], linestyle=SERIES_STYLES[key], alpha=alpha,
                    linewidth=1.6 if i == 0 else 1.2,
                    label=f"{key} — {lab}")
            for hw, source, band_alpha in _bands_for(d, s, key, da.time):
                lo, hi = da.values - hw, da.values + hw
                if var == "icefrac":
                    lo = np.maximum(lo, 0.0)
                label = None
                if source not in band_labelled:
                    label = f"{source} — {lab}"
                    band_labelled.add(source)
                ax.fill_between(da.time, lo, hi, color=SERIES_COLORS[key], alpha=band_alpha * alpha,
                                linewidth=0, label=label)
            if i > 0 and key in first_series:
                a, b = xr.align(first_series[key].dropna("time"), da, join="inner")
                summary["max_abs_delta_vs_first"].setdefault(key, {})[lab] = float(abs(b - a).max()) if a.size else None
    ax.set_title(f"{var}: mean over time (global) — {', '.join(labels)}")
    ax.set_xlabel("year")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"{var}_series.png", dpi=130)
    plt.close(fig)
    return summary


def _finite_edges(edges: np.ndarray) -> np.ndarray:
    """Replace a trailing +inf edge (exposure bins) with one more bin of the previous width."""
    edges = np.asarray(edges, dtype=float).copy()
    finite = np.isfinite(edges)
    if not finite.all():
        edges[~finite] = edges[finite][-1] + np.diff(edges[finite])[-1]
    return edges


def _hist_w1(a: dict, b: dict) -> float:
    """Wasserstein-1 distance between two binned distributions (mass at bin centres).

    Works across differing bin edges — the dump's data-driven bins span each input's own
    range, so a plain per-bin L1 is undefined between configs. Units are the variable's.
    """
    from scipy.stats import wasserstein_distance

    ea, eb = _finite_edges(a["bin_edges"]), _finite_edges(b["bin_edges"])
    ca, cb = np.asarray(a["counts"], dtype=float), np.asarray(b["counts"], dtype=float)
    if ca.sum() == 0 or cb.sum() == 0:
        return float("nan")
    return float(
        wasserstein_distance(
            (ea[:-1] + ea[1:]) / 2, (eb[:-1] + eb[1:]) / 2, u_weights=ca, v_weights=cb
        )
    )


def render_histogram(inputs: list[dict], labels: list[str], var: str, out: Path) -> dict:
    """Step-plot the dumped distributions; report Wasserstein-1 distances vs the first input."""
    summary: dict = {"w1_vs_first": {}}
    keys = [k for k in ("historical", "no_sai", "with_sai") if k in inputs[0]["meta"]["distribution"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8), squeeze=False)
    axes = axes[0]
    colors = {"historical": "black", "no_sai": "tab:red", "with_sai": "tab:blue"}
    first = inputs[0]["meta"]["distribution"]
    for ax, key in zip(axes, keys):
        for i, (d, lab) in enumerate(zip(inputs, labels)):
            dist = d["meta"]["distribution"].get(key)
            if dist is None:
                continue
            edges = _finite_edges(dist["bin_edges"])  # exposure bins end in +inf
            counts = np.asarray(dist["counts"], dtype=float)
            alpha = INPUT_ALPHAS[min(i, len(INPUT_ALPHAS) - 1)]
            ax.stairs(counts, edges, color=colors[key], alpha=alpha, linewidth=1.6 if i == 0 else 1.1,
                      label=lab, fill=(i == 0), baseline=0)
            if i > 0 and key in first:
                summary["w1_vs_first"].setdefault(key, {})[lab] = _hist_w1(first[key], dist)
        ax.set_title(f"{var}: {key} (fraction of cells per bin)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{var}_histogram.png", dpi=130)
    plt.close(fig)
    return summary


def render(qdirs: Sequence[str | Path], out_dir: str | Path, labels: Sequence[str] | None = None) -> Path:
    """Render the full figure set for N quantity dirs of the same variable into ``out_dir``."""
    inputs = [_load_quantities(Path(q)) for q in qdirs]
    vars_ = {d["meta"]["var"] for d in inputs}
    if len(vars_) != 1:
        raise ValueError(f"all inputs must be the same variable, got {sorted(vars_)}")
    var = vars_.pop()
    labels = list(labels) if labels else [_label(d, "") for d in inputs]
    if len(labels) != len(inputs):
        raise ValueError("labels must match inputs")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "var": var,
        "inputs": [{"label": lab, "dir": str(d["dir"]), "config": d["meta"]["config"], "sha": d["meta"]["sha"]}
                   for d, lab in zip(inputs, labels)],
        "pvalues": render_pvalues(inputs, labels, var, out),
        "series": render_series(inputs, labels, var, out),
        "histogram": render_histogram(inputs, labels, var, out),
        "internal_variability": {lab: d["meta"].get("internal_variability") for d, lab in zip(inputs, labels)},
        "band_keys": {lab: d["meta"].get("band_keys") for d, lab in zip(inputs, labels)},
    }
    (out / f"{var}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return out
