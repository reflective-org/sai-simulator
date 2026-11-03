"""Generate the before/after figures for the InVERT variability branch.

Run from the repo root:

    python analysis/add-invert-variability/generate_plots.py --var tas
    python analysis/add-invert-variability/generate_plots.py --var pr
    python analysis/add-invert-variability/generate_plots.py --var tasmin --line-only

Outputs land in ``analysis/add-invert-variability/plots/<var>/``. The script uses the same
data/model/cache directories that ``launch_gradio.py`` uses (``data/processed``,
``data/models``, ``data/cache``) by default; override via CLI flags if needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from backend.baseline import get_regional_map_with_variability  # noqa: E402
from backend.historical import get_historical_model  # noqa: E402
from backend.invert_variability import INVERT_VARIABLES  # noqa: E402

PLOT_ROOT = Path(__file__).resolve().parent / "plots"

SEAM_YEARS = [1901, 1951, 2001, 2051]

# pr is land-only (NaN over ocean) and its anomaly can have either sign, so it gets a
# diverging colormap centred on zero; tas anomalies are one-signed warming.
MEAN_CMAP = {"tas": "magma", "tasmin": "magma", "tasmax": "magma", "pr": "BrBG"}
AREA_LABEL = {"tas": "Global", "tasmin": "Global", "tasmax": "Global", "pr": "Global land"}


def _global_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def plot_global_annual_mean_line(mean: xr.DataArray, ensemble: xr.DataArray, var: str, units: str, ssp: str, plot_dir: Path):
    mean_gm = _global_mean(mean)
    ens_gm = _global_mean(ensemble)

    fig, ax = plt.subplots(figsize=(10, 5))
    envelope_low = ens_gm.min("ensemble")
    envelope_high = ens_gm.max("ensemble")
    ax.fill_between(
        ens_gm["time"], envelope_low, envelope_high,
        color="steelblue", alpha=0.4, label="InVERT full range of 20 members",
    )
    ax.plot(mean_gm["time"], mean_gm, color="black", lw=1.5, label="smooth mean")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{AREA_LABEL[var]}-mean {var} ({units} vs 1850-1900)")
    ax.set_title(f"{AREA_LABEL[var]} annual-mean {var}, {ssp.upper()} (mean + InVERT envelope)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    out = plot_dir / "01_global_annual_mean_line.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _plot_map(da_2d: xr.DataArray, title: str, out: Path, cmap: str, units: str, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    kwargs = {"cmap": cmap}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    im = ax.pcolormesh(da_2d["lon"], da_2d["lat"], da_2d.values, shading="auto", **kwargs)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=units)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_maps_2081_2100(mean: xr.DataArray, ensemble: xr.DataArray, var: str, units: str, ssp: str, plot_dir: Path):
    window = slice(2081, 2100)
    mean_map = mean.sel(time=window).mean("time")
    member0_map = ensemble.isel(ensemble=0).sel(time=window).mean("time")
    spread_map = ensemble.sel(time=window).std("ensemble").mean("time")
    diff_map = member0_map - mean_map

    vmin = float(min(mean_map.min(), member0_map.min()))
    vmax = float(max(mean_map.max(), member0_map.max()))
    if MEAN_CMAP[var] == "BrBG":
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax

    _plot_map(
        mean_map,
        f"BEFORE: mean {var} 2081-2100, {ssp.upper()}",
        plot_dir / "02_map_mean_2081_2100.png",
        cmap=MEAN_CMAP[var], units=units, vmin=vmin, vmax=vmax,
    )
    _plot_map(
        member0_map,
        f"AFTER: mean + InVERT member 0, {var} 2081-2100, {ssp.upper()}",
        plot_dir / "03_map_one_member_2081_2100.png",
        cmap=MEAN_CMAP[var], units=units, vmin=vmin, vmax=vmax,
    )
    _plot_map(
        spread_map,
        f"Ensemble std across 20 members, {var} 2081-2100, {ssp.upper()}",
        plot_dir / "04_map_ensemble_spread_2081_2100.png",
        cmap="viridis", units=units,
    )
    _plot_map(
        diff_map,
        f"Diff: member 0 - mean, {var} 2081-2100, {ssp.upper()}",
        plot_dir / "05_map_diff_member_minus_mean_2081_2100.png",
        cmap="RdBu_r", units=units,
        vmin=-float(np.abs(diff_map).max()), vmax=float(np.abs(diff_map).max()),
    )


def plot_global_annual_ensemble_std(ensemble: xr.DataArray, var: str, units: str, ssp: str, plot_dir: Path):
    gm = _global_mean(ensemble)
    yearly_std = gm.std("ensemble")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly_std["time"], yearly_std, color="darkorange", lw=1.5)
    for seam in SEAM_YEARS:
        ax.axvline(seam, color="grey", alpha=0.4, ls="--", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Std of {AREA_LABEL[var].lower()}-mean {var} across 20 members ({units})")
    ax.set_title(
        f"Per-year ensemble std, {ssp.upper()} (dashed lines = InVERT seam years)"
    )
    ax.grid(alpha=0.3)
    out = plot_dir / "06_global_annual_ensemble_std.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_negative_pr_fraction(ensemble: xr.DataArray, data_dir: Path, cache_dir: Path, plot_dir: Path):
    """pr only: where does mean + InVERT anomaly go below zero absolute precipitation?

    The ensemble is in the 1850-1900 anomaly frame, so add the pre-industrial climatology
    back to get absolute mm/day before testing the sign.
    """
    historical = get_historical_model("pr", data_dir, cache_dir)
    climatology = historical.sel(time=slice(1850, 1900)).mean("time")
    absolute = ensemble + climatology
    negative = absolute < 0
    frac_map = negative.mean(("ensemble", "time")).where(absolute.isel(ensemble=0, time=0).notnull())
    n_land = int(frac_map.notnull().sum())
    n_neg_cells = int((frac_map > 0).sum())
    total_frac = float(negative.sum() / absolute.notnull().sum())
    print(
        f"negative absolute pr: {100 * total_frac:.2f}% of all (member, year, land cell) values; "
        f"{n_neg_cells} of {n_land} land cells ever go negative; "
        f"worst cell {100 * float(frac_map.max()):.1f}% of member-years; "
        f"most negative value {float(absolute.min()):.2f} mm/day"
    )
    _plot_map(
        100 * frac_map,
        "pr: % of (member, year) values with mean + InVERT < 0 mm/day",
        plot_dir / "07_map_negative_pr_fraction.png",
        cmap="Reds", units="%", vmin=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", default="tas", choices=sorted(INVERT_VARIABLES))
    parser.add_argument(
        "--ssp",
        default="SSP2-4.5",
        help=(
            "SSP scenario in the FANCY_SSP_TITLES form, e.g. 'SSP2-4.5' for ssp245. "
            "InVERT is pre-detrended pure internal variability, so the scenario only "
            "affects the smooth mean plotted underneath -- the ensemble spread is "
            "scenario-agnostic."
        ),
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "data" / "models"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "cache"))
    parser.add_argument("--line-only", action="store_true",
                        help="Only write the global annual-mean line plot (skip the maps).")
    args = parser.parse_args()

    var = args.var
    units = INVERT_VARIABLES[var]["units"]
    plot_dir = PLOT_ROOT / var
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)

    print(f"Building {var} mean + InVERT ensemble for {args.ssp}...")
    mean, ensemble = get_regional_map_with_variability(
        var,
        data_dir=data_dir,
        model_dir=Path(args.model_dir),
        cache_dir=cache_dir,
        ssp_scenario=args.ssp,
    )
    print(f"mean dims={mean.dims} shape={mean.shape}")
    print(f"ensemble dims={ensemble.dims} shape={ensemble.shape}")
    variability = ensemble - mean
    print(
        f"InVERT {var} annual anomaly on the emulator grid: std {float(variability.std()):.3f} {units}, "
        f"global-mean std {float(_global_mean(variability).std()):.3f} {units}"
    )

    plot_global_annual_mean_line(mean, ensemble, var, units, args.ssp, plot_dir)
    if args.line_only:
        print(f"line plot written under {plot_dir}")
        return
    plot_maps_2081_2100(mean, ensemble, var, units, args.ssp, plot_dir)
    plot_global_annual_ensemble_std(ensemble, var, units, args.ssp, plot_dir)
    if var == "pr":
        plot_negative_pr_fraction(ensemble, data_dir, cache_dir, plot_dir)
    print(f"all plots written under {plot_dir}")


if __name__ == "__main__":
    main()
