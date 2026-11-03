"""Regional dashboard line plots with the InVERT envelope, and a check across all regions.

1. Runs the real ``get_outputs`` pipeline in regional mode for a few IPCC-WGI reference
   regions and draws each region's mean-over-time chart the way ``launch_gradio.py`` does:
   the InVERT envelope (lowest to highest member, over the selected region) shaded, with the
   legacy scalar +/-2 sigma band as dashed outlines for comparison.
2. For every IPCC-WGI reference region, compares the InVERT envelope half-width with the
   legacy 2 sigma half-width from the region GeoJSON, per variable.

Run from the repo root (the data dir needs ``invert/`` and the regional-model inputs):

    python analysis/add-invert-variability/plot_regional_envelopes.py \
        --data-dir <processed dir> --model-dir <models dir> --cache-dir <cache dir>

Outputs land in ``analysis/add-invert-variability/plots/regional/``.
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

from backend.get_outputs import get_outputs  # noqa: E402
from backend.invert_variability import INVERT_VARIABLES, _cache_path, get_regional_mean_variability  # noqa: E402
from backend.utils import clip_to_land, create_mask, load_spatial_aggregation_gdfs  # noqa: E402

PLOT_DIR = Path(__file__).resolve().parent / "plots" / "regional"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WGI = "IPCC-WGI-reference-regions-v4"
LINE_REGIONS = ["N.Europe", "West&Central-Europe", "C.North-America", "Sahara", "S.Asia", "N.Australia"]


def _series_and_colors(mot):
    return [(mot["no_sai"], "tab:red"), (mot["with_sai"], "tab:blue"), (mot["historical_model"], "tab:red")]


def plot_region_lines(var: str, mots: dict, units: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, (region, mot) in zip(axes.flat, mots.items()):
        low, high = mot["band"]["low"], mot["band"]["high"]
        sigma = float(mot["model_internal_variability"])
        for series, color in _series_and_colors(mot):
            ax.plot(series.time, series, color=color, lw=1)
            ax.fill_between(
                series.time, series + low.sel(time=series.time), series + high.sel(time=series.time),
                color=color, alpha=0.3, lw=0,
            )
            ax.plot(series.time, series - 2 * sigma, color=color, ls=":", lw=0.8)
            ax.plot(series.time, series + 2 * sigma, color=color, ls=":", lw=0.8)
        half = float((high - low).mean()) / 2
        centre = float((high + low).mean()) / 2
        ax.set_title(
            f"{region}\nInVERT half-width {half:.2f}, centre {centre:+.2f}, legacy 2σ {2 * sigma:.2f} {units}",
            fontsize=10,
        )
        ax.axvline(2035, color="black", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{var} ({units}, vs 1850-1900)")
    fig.suptitle(f"{title}\nshaded: InVERT full range of 20 members over the region; dotted: legacy ±2σ")
    out = PLOT_DIR / f"{var}_regional_lines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_all_regions(data_dir: Path, cache_dir: Path, gdf):
    fig, axes = plt.subplots(1, len(INVERT_VARIABLES), figsize=(5 * len(INVERT_VARIABLES), 5))
    for ax, var in zip(axes, sorted(INVERT_VARIABLES)):
        with xr.open_dataarray(_cache_path(cache_dir, var)) as ens:
            template = xr.zeros_like(ens.isel(ensemble=0, time=slice(0, 1), drop=False).drop_vars("ensemble"))
        if var == "pr":
            template = clip_to_land(data_dir, template)
        invert_half, legacy_half = [], []
        for _, row in gdf.iterrows():
            region = gdf[gdf.name == row["name"]]
            mask = create_mask(template.isel(time=0, drop=True), region)
            try:
                rm = get_regional_mean_variability(var, data_dir, cache_dir, template, mask)
            except ValueError:
                continue  # region has no cells with data (e.g. ocean-only for pr)
            invert_half.append(float((rm.max("ensemble") - rm.min("ensemble")).mean()) / 2)
            legacy_half.append(2 * float(row[f"variability_{var}"]))
        invert_half, legacy_half = np.array(invert_half), np.array(legacy_half)
        ax.scatter(legacy_half, invert_half, s=14)
        lim = max(legacy_half.max(), invert_half.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8, label="1:1")
        ratio = np.median(invert_half / legacy_half)
        ax.set_title(f"{var}: {len(invert_half)} regions, median ratio {ratio:.2f}")
        ax.set_xlabel(f"legacy 2σ half-width ({INVERT_VARIABLES[var]['units']})")
        ax.set_ylabel("InVERT full-range half-width")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend()
        print(f"{var}: {len(invert_half)} regions, InVERT/legacy half-width median {ratio:.2f}, "
              f"range {np.min(invert_half / legacy_half):.2f}-{np.max(invert_half / legacy_half):.2f}")
    fig.suptitle("IPCC-WGI reference regions: InVERT envelope vs legacy ±2σ band (time-mean half-widths)")
    out = PLOT_DIR / "all_regions_invert_vs_legacy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssp", default="SSP2-4.5")
    parser.add_argument("--temp-target", type=float, default=1.5)
    parser.add_argument("--start-year", type=int, default=2035)
    parser.add_argument("--ramp-up", type=int, default=10, help="SAI ramp-up years (launch_gradio default)")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "data" / "models"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "cache"))
    args = parser.parse_args()

    gdf = load_spatial_aggregation_gdfs(args.data_dir)[WGI]
    variables = sorted(INVERT_VARIABLES)
    mots = {var: {} for var in variables}
    for region in LINE_REGIONS:
        print(f"Running get_outputs (regional: {region}) for {variables}...")
        output = get_outputs(
            ssp_scenario=args.ssp,
            temp_target=args.temp_target,
            spatial_gdf=gdf,
            spatial_item=region,
            decade_start_year=2091,
            decade_end_year=2100,
            start_year=args.start_year,
            ramp_up=args.ramp_up,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            cache_dir=args.cache_dir,
            var=variables,
        )
        for var in variables:
            mots[var][region] = output[var]["mean_over_time"]
    title = f"{args.ssp}, target {args.temp_target} C, SAI from {args.start_year}"
    for var in variables:
        plot_region_lines(var, mots[var], INVERT_VARIABLES[var]["units"], title)
    plot_all_regions(Path(args.data_dir), Path(args.cache_dir), gdf)


if __name__ == "__main__":
    main()
