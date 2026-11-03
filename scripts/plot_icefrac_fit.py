"""Diagnostic figure for the Arctic sea-ice logistic fits (``img/icefrac_fit.png``).

``scripts/fit_icefrac_arctic.py`` fits one decreasing logistic per Arctic grid cell
per climate model, mapping that cell's regional ``tas`` anomaly to ice fraction, and
:func:`backend.icefrac.predict_icefrac` averages the models' sigmoids. This script
renders four panels so a reviewer can see what those fits do:

a. **Emulator vs model** — Arctic September sea-ice area against Arctic warming, for each
   model's own training data and its own fitted sigmoids.
b. **One region up close** — the actual sigmoid shape in a single band, with the
   training points that constrain it.
c. **Training coverage** — the range of regional anomalies each model supplies.
d. **Why UKESM1-1-LL is excluded** — the shipped two-model mean against the
   three-model mean that including UKESM would produce.

Panel (d) needs the rejected fit, so the script fits
:data:`scripts.fit_icefrac_arctic.EXCLUDED_MODEL_SCENARIOS` in memory. That costs a
minute or so and never touches ``data/models``; pass ``--skip_excluded`` to omit it.

Ice is **September** (Northern-Hemisphere) sea ice throughout: ``process_monthly_gauss``
masks ICEFRAC to NH September before the yearly aggregation, so that one month is
what the processed files and the sigmoids carry. September is the climatological
minimum, but no minimum is computed — nothing here is an annual mean.
Areas cover all cells 60-90°N, weighted by true cell area from ``area.nc``. A cell
whose per-cell fit failed predicts zero ice, matching
:func:`backend.icefrac.predict_icefrac`.

Reads only ``data/processed`` and ``data/models``; writes only the figure, which
is a generated artefact and is not committed.

CLI: ``python scripts/plot_icefrac_fit.py [--data_dir data/processed]
[--model_dir data/models] [--output img/icefrac_fit.png] [--skip_excluded]``
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.icefrac import create_arctic_dataset, get_area  # noqa: E402
from backend.sigmoid import logistic  # noqa: E402
from backend.utils import open_xarray_datasets  # noqa: E402
from scripts.fit_icefrac_arctic import (  # noqa: E402
    EXCLUDED_MODEL_SCENARIOS,
    MODEL_SCENARIOS,
    PRIMARY_MODEL,
    VAR_TEMP,
    _fit_one_model,
    _load_and_process_ice,
    _load_and_process_temp,
)

VAR_ICE = "icefrac"

# Colour per model, from the Wong colour-blind-safe palette. The excluded model
# takes the warm colour so it reads as the odd one out in every panel.
COLORS = {
    "CESM2-WACCM": "#0072B2",
    "MIROC-ES2H": "#009E73",
    "UKESM1-1-LL": "#D55E00",
}
SAI_MARKERS = ["^", "s", "D", "v"]

# The band panel (b) zooms into. Of the 24 regions this one has the largest DROP
# across the fitted curves over the cells every model fitted — about 0.89 at the
# cold end to 0.03 at +10 K — so both plateaus and the transition fit in one
# panel. A few high-Arctic bands sit marginally higher at the cold end but fall
# less far, which is why this is not simply the largest-amplitude band.
FOCUS_LON_BAND = "Pacific"
FOCUS_LAT_BAND = "75-80"


# ------------------------------------------------------------------------------
# Geometry and aggregation
# ------------------------------------------------------------------------------

def arctic_geometry(data_dir: Path, reference_file: str):
    """Region membership and per-cell areas on the flattened Arctic grid.

    ``create_arctic_dataset`` bins the Arctic into (lon_band x lat_band) regions by
    outer-joining slices, so every region carries the full Arctic (lat, lon) grid
    with NaN outside its own footprint. The fitted parameters inherit that layout,
    so a cell is "in" a region exactly where the temperature field is finite.

    Returns:
        (member, area, coords) where ``member`` is a boolean
        ``(lon_band, lat_band, n_cells)`` array, ``area`` is ``(n_cells,)`` in
        million km^2, and ``coords`` holds band labels plus the Arctic lat/lon axes.
    """
    ds = open_xarray_datasets(data_dir / VAR_TEMP / reference_file)
    arctic = create_arctic_dataset(ds)[VAR_TEMP].squeeze(drop=True)
    member = np.isfinite(arctic.isel(time=0).values)
    lat, lon = arctic.lat.values, arctic.lon.values

    # area.nc shares the processed grid but labels the pole rows +-90 where the
    # processed files carry the corrected cell centres (correct_lat.npy), so the
    # Arctic rows are selected by position rather than by label.
    area_da = get_area(Path(data_dir))
    full_lat = np.sort(ds.lat.values)
    if len(area_da.lat) != len(full_lat) or not np.allclose(area_da.lon.values, lon):
        raise ValueError("area.nc does not share the processed lat/lon grid")
    rows = np.searchsorted(full_lat, lat)
    if np.abs(area_da.lat.values[rows] - lat).max() > 0.3:
        raise ValueError("area.nc latitudes do not line up with the Arctic rows")
    area = area_da.isel(lat=rows).values.ravel()
    coords = {
        "lon_band": [str(v) for v in arctic.lon_band.values],
        "lat_band": [str(v) for v in arctic.lat_band.values],
        "lat": lat,
        "lon": lon,
    }
    return member.reshape(member.shape[0], member.shape[1], -1), area, coords


def region_index(coords: dict, lon_band: str, lat_band: str) -> tuple[int, int]:
    return coords["lon_band"].index(lon_band), coords["lat_band"].index(lat_band)


def _regions(member: np.ndarray):
    """Yield (i, j, cell_mask) for every non-empty region."""
    for i in range(member.shape[0]):
        for j in range(member.shape[1]):
            if member[i, j].any():
                yield i, j, member[i, j]


def ice_area_from_data(values: np.ndarray, member: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Total Arctic September sea-ice area per year from a per-cell field.

    ``values`` is ``(lon_band, lat_band, T, n_cells)``. Non-finite cells count as
    ice-free, the same convention the predictor uses for unfitted cells.
    """
    total = np.zeros(values.shape[2])
    for i, j, cells in _regions(member):
        total += np.nansum(values[i, j][:, cells] * area[cells], axis=1)
    return total


def arctic_mean_anomaly(x: np.ndarray, member: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Area-weighted Arctic mean of a per-region scalar series ``(lon_band, lat_band, T)``."""
    region_area = (member * area).sum(axis=-1)
    return (x * region_area[:, :, None]).sum(axis=(0, 1)) / region_area.sum()


# ------------------------------------------------------------------------------
# Fitted-sigmoid evaluation
# ------------------------------------------------------------------------------

def fitted_mask(params: np.ndarray) -> np.ndarray:
    """Cells whose logistic fit converged (``L`` is finite)."""
    return np.isfinite(params[0])


def evaluate_cells(params: np.ndarray, x, i: int, j: int, cells: np.ndarray) -> np.ndarray:
    """Evaluate one region's per-cell sigmoids at anomalies ``x`` → ``(len(x), n_cells)``.

    Cells whose fit failed are NaN in ``params``; :func:`backend.icefrac.predict_icefrac`
    treats those as zero ice, so this does too.
    """
    L, x0, k, b = (p[i, j, cells] for p in params)
    pred = logistic(np.atleast_1d(np.asarray(x, dtype=float))[:, None], L, x0, k, b)
    return np.where(np.isfinite(pred), pred, 0.0)


def region_response(params: np.ndarray, x, i: int, j: int, cells: np.ndarray,
                    area: np.ndarray) -> np.ndarray:
    """Area-weighted mean fitted ice fraction over ``cells`` at anomalies ``x``."""
    w = area[cells]
    return (evaluate_cells(params, x, i, j, cells) * w).sum(axis=1) / w.sum()


def uniform_response_area(params: np.ndarray, x, member: np.ndarray,
                          area: np.ndarray) -> np.ndarray:
    """Arctic September sea-ice area with every region held at the same anomaly ``x``.

    A sensitivity curve, not a trajectory: it answers "if the whole Arctic warmed by
    x, what would this model's sigmoids say?", which is the comparison panel (d) needs.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    total = np.zeros(len(x))
    for i, j, cells in _regions(member):
        total += (evaluate_cells(params, x, i, j, cells) * area[cells]).sum(axis=1)
    return total


def predicted_ice_area(params: np.ndarray, x: np.ndarray, member: np.ndarray,
                       area: np.ndarray) -> np.ndarray:
    """Arctic September sea-ice area at each training state ``(lon_band, lat_band, T)``.

    Unlike :func:`uniform_response_area` each region is evaluated at its OWN anomaly,
    so this is directly comparable to the model's actual sea-ice area.
    """
    total = np.zeros(x.shape[2])
    for i, j, cells in _regions(member):
        L, x0, k, b = (p[i, j, cells] for p in params)
        pred = logistic(x[i, j][:, None], L, x0, k, b)
        total += (np.where(np.isfinite(pred), pred, 0.0) * area[cells]).sum(axis=1)
    return total


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def load_training_data(data_dir: Path, scenarios: dict) -> dict:
    """Load each scenario's (anomaly, ice) training pair exactly as the fitter does."""
    baseline_ref = create_arctic_dataset(
        open_xarray_datasets(data_dir / VAR_TEMP / scenarios["baseline"]["file"])
    )
    out = {}
    for name, cfg in scenarios.items():
        if not (data_dir / VAR_TEMP / cfg["file"]).exists():
            continue
        if not (data_dir / VAR_ICE / cfg["file"]).exists():
            continue
        x = _load_and_process_temp(data_dir, cfg["file"], cfg["start"], cfg["end"], baseline_ref)
        y = _load_and_process_ice(data_dir, VAR_ICE, cfg["file"], cfg["start"], cfg["end"])
        out[name] = (x, y)
    return out


# Order the rest of this module unpacks parameter blocks in. Selected BY LABEL
# when the interpolator is read, so a file whose ``features`` axis is stored in
# another order still yields the same arrays.
FEATURES = ["L", "x0", "k", "b"]


def load_fitted_params(model_dir: Path, coords: dict) -> dict:
    """Read the shipped interpolator back onto the flattened Arctic grid, per model.

    Parameters come back stacked in :data:`FEATURES` order regardless of how the
    file stores them — everything downstream unpacks positionally, and a silently
    permuted axis produces a plausible but badly wrong figure rather than an error.
    """
    path = Path(model_dir) / f"{VAR_ICE}_arctic" / "interpolator.nc"
    with xr.open_dataarray(path) as da:
        da = da.load()
    if da.name != "logistic_params":
        raise ValueError(
            f"expected a DataArray named 'logistic_params', found {da.name!r} in {path}"
        )
    if "model" not in da.dims:
        da = da.expand_dims(model=["CESM2-WACCM"])
    missing = set(FEATURES) - {str(v) for v in da.features.values}
    if missing:
        raise ValueError(f"interpolator is missing logistic parameters: {sorted(missing)}")
    arctic = da.sel(lat=coords["lat"], features=FEATURES)
    if not np.allclose(arctic.lon.values, coords["lon"]):
        raise ValueError("interpolator lon axis does not match the processed Arctic grid")
    out = {}
    for name in [str(v) for v in arctic.model.values]:
        arr = (
            arctic.sel(model=name)
            .transpose("features", "lon_band", "lat_band", "lat", "lon")
            .values
        )
        out[name] = arr.reshape(arr.shape[:3] + (-1,))
    return out


# ------------------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------------------

def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _panel_emulator_vs_model(ax, models, member, area):
    """(a) Arctic sea-ice area vs Arctic warming: training points and fitted lines."""
    model_handles = []
    for name, entry in models.items():
        color = COLORS.get(name, "#666666")
        sai = 0
        for scenario, (x, y) in entry["training"].items():
            warming = arctic_mean_anomaly(x, member, area)
            ice = ice_area_from_data(y, member, area)
            if scenario == "baseline":
                ax.scatter(warming, ice, s=7, color=color, alpha=0.35, linewidths=0, zorder=2)
                continue
            marker = SAI_MARKERS[sai % len(SAI_MARKERS)]
            sai += 1
            ax.scatter(warming, ice, s=26, facecolors="none", edgecolors=color,
                       marker=marker, linewidths=0.9, alpha=0.85, zorder=3)
            if entry.get("params") is None:
                continue
            # The emulator is driven by each region's OWN anomaly, so its value at
            # an SAI year is not the value the no-SAI line shows above that year's
            # Arctic mean: under SAI the regional pattern differs at the same mean.
            # Plot the prediction at its own point, so the vertical gap to the open
            # marker is the emulator's actual error rather than a pattern artefact.
            sai_pred = predicted_ice_area(entry["params"], x, member, area)
            ax.scatter(warming, sai_pred, s=9, color=color, marker=marker,
                       linewidths=0, alpha=0.9, zorder=3)
        if entry.get("params") is None:
            model_handles.append(
                ax.scatter([], [], s=16, color=color, linewidths=0,
                           label=f"{name} — model only, not fitted")
            )
            continue
        x, _ = entry["training"]["baseline"]
        warming = arctic_mean_anomaly(x, member, area)
        pred = predicted_ice_area(entry["params"], x, member, area)
        order = np.argsort(warming)
        excluded = entry.get("excluded")
        line, = ax.plot(warming[order], pred[order], "--" if excluded else "-", color=color,
                        linewidth=2.0, zorder=4,
                        label=f"{name} — {'rejected fit' if excluded else 'emulator'}")
        model_handles.append(line)

    marker_handles = [
        ax.scatter([], [], s=12, color="#555555", alpha=0.5, linewidths=0,
                   label="no-SAI training years"),
        ax.scatter([], [], s=26, facecolors="none", edgecolors="#555555", marker="^",
                   linewidths=0.9, label="SAI training years (marker per scenario)"),
        ax.scatter([], [], s=9, color="#555555", marker="^", linewidths=0,
                   label="emulator at those SAI years"),
    ]
    ax.add_artist(ax.legend(handles=model_handles, loc="upper right", frameon=False, fontsize=8))
    ax.legend(handles=marker_handles, loc="lower left", frameon=False, fontsize=8)
    ax.set_xlabel("Arctic (60-90°N) mean $tas$ anomaly vs 1850-1900 (K)")
    ax.set_ylabel("Arctic September sea-ice area (10$^6$ km²)")
    ax.set_title("a  Emulator against its own model", loc="left", fontsize=10, fontweight="bold")
    _style(ax)


def _panel_focus_region(ax, models, member, area, coords, common_cells):
    """(b) One region's sigmoid, over the cells every model actually fitted."""
    i, j = region_index(coords, FOCUS_LON_BAND, FOCUS_LAT_BAND)
    cells = common_cells[i, j]
    if not cells.any():
        raise ValueError(
            f"no cells to plot for {FOCUS_LAT_BAND}N {FOCUS_LON_BAND}: the region has "
            f"{int(member[i, j].sum())} cells but none survive the fit masks"
        )
    w = area[cells]
    spans = {}
    lo, hi = np.inf, -np.inf
    for name, entry in models.items():
        color = COLORS.get(name, "#666666")
        model_lo, model_hi = np.inf, -np.inf
        for scenario, (x, y) in entry["training"].items():
            v = y[i, j][:, cells]
            ice = np.nansum(np.where(np.isfinite(v), v, 0.0) * w, axis=1) / w.sum()
            baseline = scenario == "baseline"
            ax.scatter(x[i, j], ice, s=9 if baseline else 24,
                       color=color if baseline else "none",
                       edgecolors="none" if baseline else color,
                       alpha=0.35 if baseline else 0.85, marker="o" if baseline else "^",
                       linewidths=0.9, zorder=2)
            model_lo = min(model_lo, float(np.nanmin(x[i, j])))
            model_hi = max(model_hi, float(np.nanmax(x[i, j])))
        spans[name] = (model_lo, model_hi)
        lo, hi = min(lo, model_lo), max(hi, model_hi)
    xs = np.linspace(lo - 1.0, hi + 1.0, 400)
    for name, entry in models.items():
        if entry.get("params") is None:
            continue
        excluded = entry.get("excluded")
        color = COLORS.get(name, "#666666")
        curve = region_response(entry["params"], xs, i, j, cells, area)
        style = "--" if excluded else "-"
        # Faint beyond the model's own training range: everything outside it is
        # the fitted plateau extrapolated, which is exactly UKESM's problem here.
        inside = (xs >= spans[name][0]) & (xs <= spans[name][1])
        ax.plot(xs, curve, style, color=color, linewidth=1.6, alpha=0.28, zorder=3)
        ax.plot(np.where(inside, xs, np.nan), np.where(inside, curve, np.nan), style,
                color=color, linewidth=2.0, zorder=4,
                label=f"{name}{' — rejected fit' if excluded else ''}")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(-0.03, 1.02)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel(f"{FOCUS_LAT_BAND}°N {FOCUS_LON_BAND} $tas$ anomaly (K)")
    ax.set_ylabel("September sea-ice fraction over fitted cells")
    ax.annotate(f"{int(cells.sum())} cells; faint = extrapolated beyond that model's\n"
                "own training range",
                (0.015, 0.03), xycoords="axes fraction", fontsize=7.5, color="#555555")
    ax.set_title(f"b  Fitted sigmoid, {FOCUS_LAT_BAND}°N {FOCUS_LON_BAND}",
                 loc="left", fontsize=10, fontweight="bold")
    _style(ax)


def _panel_coverage(ax, models, member, area):
    """(c) The span of regional anomalies each model contributes to the fit."""
    names = list(models)
    for row, name in enumerate(names):
        color = COLORS.get(name, "#666666")
        spans = [x.ravel()[np.isfinite(x.ravel())] for x, _ in models[name]["training"].values()]
        allx = np.concatenate(spans)
        lo, hi = float(allx.min()), float(allx.max())
        ax.barh(row, hi - lo, left=lo, height=0.40, color=color, alpha=0.28,
                edgecolor=color, linewidth=1.0, zorder=2)
        ax.annotate(f"{lo:+.1f} K", (lo, row), textcoords="offset points", xytext=(5, 11),
                    ha="left", va="center", fontsize=8, color=color, fontweight="bold")
    ax.axvline(0.0, color="#333333", linewidth=0.9, linestyle=":", zorder=1)
    ax.annotate("pre-industrial", (0.0, len(names) - 0.55), textcoords="offset points",
                xytext=(5, 0), ha="left", va="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_ylim(len(names) - 0.4, -0.7)
    ax.set_xlabel("Regional $tas$ anomaly spanned by the training set (K)")
    ax.set_title("c  Training coverage — UKESM never sees a cold Arctic",
                 loc="left", fontsize=10, fontweight="bold")
    _style(ax)


def _panel_exclusion_impact(ax, models, member, area):
    """(d) Shipped two-model mean against the three-model mean UKESM would give."""
    xs = np.linspace(0.0, 12.0, 200)
    responses, shipped = {}, []
    for name, entry in models.items():
        if entry.get("params") is None:
            continue
        responses[name] = uniform_response_area(entry["params"], xs, member, area)
        if not entry.get("excluded"):
            shipped.append(responses[name])
        ax.plot(xs, responses[name], color=COLORS.get(name, "#666666"), linewidth=1.0,
                alpha=0.6, linestyle="--" if entry.get("excluded") else "-",
                label=f"{name} alone", zorder=2)
    if not shipped:
        return
    mean_shipped = np.mean(shipped, axis=0)
    ax.plot(xs, mean_shipped, color="#222222", linewidth=2.4, zorder=5,
            label=f"shipped mean — {len(shipped)} models")
    excluded = [n for n, e in models.items() if e.get("excluded") and e.get("params") is not None]
    if excluded:
        mean_all = np.mean(shipped + [responses[n] for n in excluded], axis=0)
        ax.plot(xs, mean_all, color="#222222", linewidth=2.0, linestyle=":", zorder=5,
                label="mean including UKESM1-1-LL")
        for probe in (0.0, 6.0):
            idx = int(np.argmin(np.abs(xs - probe)))
            a, b = mean_shipped[idx], mean_all[idx]
            if a > 1e-6:
                ax.annotate(f"{b / a:.2f}×", (xs[idx], b), textcoords="offset points",
                            xytext=(7, 7), fontsize=8.5, color="#222222", fontweight="bold")
                ax.plot([xs[idx]] * 2, [a, b], color="#222222", linewidth=0.9, alpha=0.7,
                        zorder=4)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("Uniform $tas$ anomaly vs 1850-1900 applied to every region (K)")
    ax.annotate("x=0 is pre-industrial, not present day", (0.03, 0.10),
                xycoords="axes fraction", fontsize=7.5, color="#555555")
    ax.set_ylabel("Predicted September sea-ice area (10$^6$ km²)")
    ax.set_title("d  What including UKESM1-1-LL would do", loc="left", fontsize=10,
                 fontweight="bold")
    _style(ax)


def common_fitted_cells(models: dict, member: np.ndarray) -> np.ndarray:
    """Cells every model with parameters fitted, falling back PER REGION.

    The fallback matters: with the excluded model's fit included, 9 of the 24
    regions have no commonly-fitted cell at all. A global "did anything survive"
    fallback would leave those regions empty and any panel pointed at one would
    average over zero cells.
    """
    masks = [fitted_mask(e["params"]) for e in models.values() if e.get("params") is not None]
    if not masks:
        return member
    common = np.logical_and.reduce(masks) & member
    empty = ~common.any(axis=-1) & member.any(axis=-1)
    return np.where(empty[..., None], member, common)


def build_figure(models: dict, member: np.ndarray, area: np.ndarray, coords: dict,
                 output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 130, "axes.titlepad": 8,
                         "legend.handlelength": 1.6})
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6))
    _panel_emulator_vs_model(axes[0, 0], models, member, area)
    _panel_focus_region(axes[0, 1], models, member, area, coords,
                        common_fitted_cells(models, member))
    _panel_coverage(axes[1, 0], models, member, area)
    _panel_exclusion_impact(axes[1, 1], models, member, area)
    fig.suptitle(
        "Arctic sea-ice emulator — per-cell logistic fits of regional $tas$ anomaly "
        "(scripts/fit_icefrac_arctic.py)",
        fontsize=11, fontweight="bold", x=0.012, ha="left", y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="data/models")
    parser.add_argument("--output", default="img/icefrac_fit.png")
    parser.add_argument("--skip_excluded", action="store_true",
                        help="Do not fit the excluded models (panels b and d lose their "
                             "rejected-fit curve, and the script runs in seconds).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    member, area, coords = arctic_geometry(
        data_dir, MODEL_SCENARIOS[PRIMARY_MODEL]["baseline"]["file"]
    )
    fitted = load_fitted_params(Path(args.model_dir), coords)

    models: dict[str, dict] = {}
    for name, scenarios in MODEL_SCENARIOS.items():
        if name not in fitted:
            print(f"  {name}: no fitted sigmoids in the interpolator — skipping")
            continue
        print(f"  {name}: loading training data")
        models[name] = {"training": load_training_data(data_dir, scenarios),
                        "params": fitted[name], "excluded": False}

    for name, scenarios in EXCLUDED_MODEL_SCENARIOS.items():
        if not (data_dir / VAR_ICE / scenarios["baseline"]["file"]).exists():
            continue
        print(f"  {name}: loading training data (excluded from the shipped fit)")
        entry = {"training": load_training_data(data_dir, scenarios), "params": None,
                 "excluded": True}
        if not args.skip_excluded:
            print(f"  {name}: fitting in memory for the comparison panel")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                entry["params"] = _fit_one_model(VAR_ICE, data_dir, scenarios)
        models[name] = entry

    build_figure(models, member, area, coords, Path(args.output))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
