"""Before/after dashboard global-mean line plots with the InVERT envelope.

Runs the real ``get_outputs`` pipeline in global mode for the InVERT-enabled variables and
draws the mean-over-time chart the way ``launch_gradio.py`` does, twice per variable:
top panel with the legacy scalar +/-2 sigma band, bottom panel with the InVERT envelope
(lowest to highest of the 20 members) now returned under ``mean_over_time["band"]`` (source "invert").

Run from the repo root:

    python analysis/add-invert-variability/plot_dashboard_lines.py

Outputs land in ``analysis/add-invert-variability/plots/dashboard/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from backend.get_outputs import get_outputs  # noqa: E402
from backend.invert_variability import INVERT_VARIABLES  # noqa: E402

PLOT_DIR = Path(__file__).resolve().parent / "plots" / "dashboard"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _draw_series(ax, mot):
    ax.plot(mot["no_sai"].time, mot["no_sai"], label="No SAI", color="tab:red", ls="--")
    ax.plot(mot["with_sai"].time, mot["with_sai"], label="With SAI", color="tab:blue", ls="--")
    ax.plot(mot["historical_model"].time, mot["historical_model"], label="Historical CESM2-WACCM", color="tab:red")
    if "historical_obs" in mot:
        ax.plot(mot["historical_obs"].time, mot["historical_obs"], label="Historical Observations", color="black")


def _series_and_colors(mot):
    return [(mot["no_sai"], "tab:red"), (mot["with_sai"], "tab:blue"), (mot["historical_model"], "tab:red")]


def plot_variable(var: str, mot: dict, units: str, title: str):
    fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)

    _draw_series(ax_before, mot)
    sigma = mot["model_internal_variability"]
    for series, color in _series_and_colors(mot):
        ax_before.fill_between(series.time, series - 2 * sigma, series + 2 * sigma, color=color, alpha=0.3)
    ax_before.set_title(f"BEFORE: {var}, legacy scalar +/-2 sigma band (sigma = {float(sigma):.3f} {units})")

    _draw_series(ax_after, mot)
    low, high = mot["band"]["low"], mot["band"]["high"]
    for series, color in _series_and_colors(mot):
        ax_after.fill_between(
            series.time, series + low.sel(time=series.time), series + high.sel(time=series.time),
            color=color, alpha=0.3,
        )
    ax_after.set_title(f"AFTER: {var}, InVERT envelope, full range of 20 members (per year)")

    for ax in (ax_before, ax_after):
        ax.set_ylabel(f"{var} ({units}, vs 1850-1900)")
        ax.axvline(2035, color="black", ls="--", lw=0.8)
        ax.grid(alpha=0.3)
    ax_after.set_xlabel("Year")
    ax_before.legend(loc="upper left")
    fig.suptitle(title)
    out = PLOT_DIR / f"{var}_global_mean_line.png"
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

    variables = sorted(INVERT_VARIABLES)
    print(f"Running get_outputs (global) for {variables}...")
    output = get_outputs(
        ssp_scenario=args.ssp,
        temp_target=args.temp_target,
        spatial_gdf=None,
        spatial_item=None,
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
        mot = output[var]["mean_over_time"]
        low, high = mot["band"]["low"], mot["band"]["high"]
        units = INVERT_VARIABLES[var]["units"]
        print(
            f"{var}: envelope half-width mean {(float((high - low).mean()) / 2):.3f} {units} "
            f"vs legacy sigma {float(mot['model_internal_variability']):.3f} {units} "
            f"(legacy band half-width = 2 sigma = {2 * float(mot['model_internal_variability']):.3f})"
        )
        plot_variable(var, mot, units, f"{args.ssp}, target {args.temp_target} C, SAI from {args.start_year}")


if __name__ == "__main__":
    main()
