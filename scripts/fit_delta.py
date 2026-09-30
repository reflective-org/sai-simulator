"""Fit the per-variable SAI ``slope`` used by :mod:`backend.delta`.

For a given target variable, the fit walks a set of SAI **anchors** — each a
(SAI experiment, no-SAI baseline) pair at a known cooling magnitude — computes
the rebased experiment-vs-baseline delta per grid cell, and fits **one pooled
per-cell slope** (a through-origin least squares over all anchors of all
models together). The single slope field is saved as
``output_dir/<var>/interpolator.nc``.

**Weighting — read before adding anchors.** A pooled through-origin least
squares does NOT weight anchors equally: each anchor's influence is
``x_i^2 / sum x_j^2`` (the square of its cooling magnitude). With the current
six anchors that is 6.2 / 17.0 / 27.8% (CESM gauss-1.5/1.0/0.5),
5.6 / 28.0% (MIROC g6-1.5/g6-0.5) and 15.4% (UKESM g6-1.5, which cools
1.37 K — a high-ECS model pulled back to the same target) — CESM 51.1% /
MIROC 33.6% / UKESM 15.4% overall, while the no-SAI baseline map is a strict
equal-weight model mean. This asymmetry is a **deliberate choice**: the
alternative (fit per model, average the slopes equally — tried and reverted)
makes total influence per *model* equal at the cost of concentrating it per
*anchor* — MIROC's g6-0.5 run alone would carry ~40% of the fit, dominating
every variable from a single simulation. Until a model contributes a balanced
anchor set across cooling magnitudes, the pooled fit's data-driven weighting
is the lesser distortion. Note this means the SAI delta aggregates models at a
different stage (and weighting) than the no-SAI emulators and sea-ice
sigmoids, which average per-model results equally at prediction time.

Anchors (see ``ANCHORS``): the three CESM2-WACCM GAUSS cooling targets
(0.5 / 1.0 / 1.5 K) plus MIROC-ES2H's G6-1.5K and G6-0.5K SAI runs. Each anchor
is rebased against **the no-SAI run its own SAI experiment branched from**, so
inter-model climate bias — and any drift between two realizations of the same
scenario — doesn't leak into the SAI signal. For MIROC that is deliberately the
GeoMIP ``baseline`` run, *not* the ScenarioMIP ``ssp245`` run that the
climate-change-only fit in ``scripts/fit_map.py`` trains on (see
``backend.constants.REGIONAL_MODEL2PROJECTION_BASELINE_FILES``). All anchors' fields are on the common
CESM2-WACCM grid (MIROC is regridded by the consumer), so they stack cleanly.
Anchors whose files are absent for a variable are silently dropped; a model
whose anchors are all absent drops out of the ``model`` dim entirely (``so2``
has no MIROC equivalent, so it stays GAUSS-only).

The ``so2`` variable is handled separately because it is 1-D in latitude.

CLI entry: ``python scripts/fit_delta.py <var> <data_dir> <output_dir> [--ignore_existing]``.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

# ``fire`` is imported lazily inside the ``__main__`` block below so that
# importing this module (e.g. for unit tests of ``ANCHORS`` and the anchor
# helpers) does not require ``fire`` to be installed in the test environment.


@dataclass(frozen=True)
class Anchor:
    """One SAI training point: an experiment and its own-model baseline.

    ``exp_file`` / ``baseline_file`` are the per-experiment output filenames
    produced by ``process_monthly_gauss.py`` (the ``-miroc`` suffix marks the
    MIROC stores). Resolved relative to a variable's processed directory.
    ``model`` is the climate model the anchor belongs to (a value of the
    interpolator's ``model`` coord; matches ``REGIONAL_MODEL_NAMES``).
    """
    name: str
    exp_file: str
    baseline_file: str
    model: str


# CESM GAUSS cooling targets + MIROC G6 runs, each paired with its own baseline.
# Note the names are TARGET warming levels, so "0.5" is the DEEPEST cooling.
ANCHORS = [
    Anchor("gauss-1.5",    "output_gauss-1.5.nc",          "output_gauss-baseline.nc",       "CESM2-WACCM"),
    Anchor("gauss-1.0",    "output_gauss-1.0.nc",          "output_gauss-baseline.nc",       "CESM2-WACCM"),
    Anchor("gauss-0.5",    "output_gauss-0.5.nc",          "output_gauss-baseline.nc",       "CESM2-WACCM"),
    Anchor("g6-1.5-miroc", "output_gauss-g6-1.5-miroc.nc", "output_gauss-baseline-miroc.nc", "MIROC-ES2H"),
    Anchor("g6-0.5-miroc", "output_gauss-g6-0.5-miroc.nc", "output_gauss-baseline-miroc.nc", "MIROC-ES2H"),
    # UKESM contributes one G6-1.5K anchor (MACLOUD het-chem runs), rebased
    # against the het-chem SSP2-4.5 baseline it branched from. Each anchor joins
    # the pooled fit with x^2 leverage (see the module docstring), so this lone
    # ~1.5K run (the shallowest cooling) is the lowest-leverage kind of anchor a
    # model can contribute.
    Anchor("g6-1.5-ukesm", "output_gauss-g6-1.5-ukesm.nc", "output_gauss-baseline-ukesm.nc", "UKESM1-1-LL"),
]


def process_so2_data(paths: list[Path]) -> xr.DataArray:
    """Load and average the three-member GAUSS SO2 injection text files into a (lat, time) DataArray.

    Each file is a tab-separated table with timestamps as rows and per-latitude Tg columns
    (``30S(Tg)``, ``15S(Tg)``, ``15N(Tg)``, ``30N(Tg)``). The four latitude columns are
    extracted, sliced to 2035-2070, ensemble-averaged across members, and reshaped into
    an :class:`xr.DataArray` whose ``lat`` dimension holds the column headers and whose
    ``time`` dimension holds the timestamps.
    """
    latitudes = ["30S(Tg)", "15S(Tg)", "15N(Tg)", "30N(Tg)"]
    data = [pd.read_csv(path, sep=r"\s+").set_index("Timestamp")[latitudes].loc["2035":"2070"] for path in paths]
    # Slice each between 2035 and 2070
    data = [df.loc["2035":"2070"] for df in data]
    # Take mean of the three members
    data = sum(data) / len(data)
    # Slice to latitude values
    data = data[latitudes]
    # data is a pandas array with columns as latitudes and rows as time
    # Convert to an xarray dataset with dimensions lat and time
    data = xr.Dataset(data)
    data = data.rename({"Timestamp": "time"})
    # latitudes are data variables, so convert to coordinates
    data = data.to_array(dim="lat")
    return data


def temporal_average(
    data: xr.Dataset | xr.DataArray,
    start_year: str | int,
    end_year: str | int,
) -> xr.Dataset | xr.DataArray:
    """Slice ``data`` to ``[start_year, end_year]`` and take the time mean."""
    return data.sel(time=slice(start_year, end_year)).mean(dim="time")


def _global_mean(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Cos(lat)-weighted global mean over lat/lon."""
    return obj.weighted(np.cos(np.deg2rad(obj.lat))).mean(("lat", "lon"))


def _squeeze_model_ssp(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Drop the singleton ``model``/``ssp`` dims the producer adds.

    Their value differs per model (``CESM2-WACCM`` vs ``MIROC-ES2H``), so we
    can't ``.sel`` a fixed one — squeeze them out instead.
    """
    for dim in ("model", "ssp"):
        if dim in obj.dims:
            obj = obj.squeeze(dim, drop=True)
    return obj


def _load_anchor_2050_2069(
    var_dir: Path, anchor: Anchor
) -> tuple[xr.Dataset, xr.Dataset] | None:
    """Return an anchor's (experiment, baseline) 2050-2069 means for ``var_dir``.

    Returns ``None`` when either file is missing — i.e. the variable isn't
    available for that anchor's model, so the caller drops it.
    """
    exp_path = Path(var_dir) / anchor.exp_file
    base_path = Path(var_dir) / anchor.baseline_file
    if not exp_path.exists() or not base_path.exists():
        return None
    exp = temporal_average(xr.open_dataset(exp_path), "2050", "2069")
    base = temporal_average(xr.open_dataset(base_path), "2050", "2069")
    return exp, base


def _fit_delta_gridded(
    var: str, processed_dir: Path, tas_processed_dir: Path, interpolator_path: Path
) -> None:
    """Pooled anchor-based per-cell slope fit for 2-D (lat, lon) variables.

    All models' anchors enter one through-origin least squares; each anchor's
    influence is its ``x^2`` leverage (see the module docstring for the
    weighting discussion and why this is preferred over per-model averaging).
    """
    x_values: list[float] = []
    rebased_fields: list[xr.DataArray] = []

    for anchor in ANCHORS:
        tas_pair = _load_anchor_2050_2069(tas_processed_dir, anchor)
        var_pair = _load_anchor_2050_2069(processed_dir, anchor)
        if tas_pair is None or var_pair is None:
            print(f"  skipping anchor {anchor.name}: no {var}/tas files for this model")
            continue

        exp_tas, base_tas = tas_pair
        exp_var, base_var = var_pair

        # x: global-mean tas cooling of this anchor (exp - its own baseline).
        x = (_global_mean(exp_tas) - _global_mean(base_tas))["tas"].item()
        # y: per-cell var delta (exp - its own baseline), model/ssp squeezed.
        rebased = _squeeze_model_ssp(exp_var - base_var)[var]

        x_values.append(x)
        rebased_fields.append(rebased)
        print(f"  anchor {anchor.name}: global tas cooling = {x:.3f} K")

    if len(x_values) < 2:
        raise ValueError(
            f"Need >=2 anchors to fit {var}; only {len(x_values)} had data."
        )

    # Single-feature lstsq (slope through the origin) across all anchor points.
    X_numpy = np.array(x_values)[:, np.newaxis]
    Y = xr.concat(rebased_fields, dim="sample")
    Y_stacked = Y.stack(samples=("lat", "lon"))
    beta_numpy = np.linalg.lstsq(X_numpy, Y_stacked.values, rcond=None)[0]

    beta_reshaped = beta_numpy.reshape((1, Y.lat.size, Y.lon.size))
    if np.isnan(beta_reshaped).any():
        # A NaN slope cell would propagate silently into every SAI map —
        # refuse to ship such a fit.
        raise ValueError(f"NaN slope cells in the {var} fit.")
    beta_xr = xr.DataArray(
        beta_reshaped,
        dims=["features", "lat", "lon"],
        coords={"features": ["slope"], "lat": Y.lat.values, "lon": Y.lon.values},
    ).to_dataset(name=var)

    print(f"Writing interpolator ({len(x_values)} anchors) to {interpolator_path}")
    beta_xr.to_netcdf(interpolator_path)


def _fit_delta_so2(
    processed_dir: Path, tas_processed_dir: Path, interpolator_path: Path
) -> None:
    """so2 fit: 1-D-in-latitude. x from the GAUSS tas coolings, y from the
    GAUSS SO2 injection controllers (no baseline → rebased == raw). MIROC has
    no SO2, so this is GAUSS-only."""
    gauss_anchors = [a for a in ANCHORS if a.name.startswith("gauss-")]

    # x: GAUSS tas coolings (same anchors/files as the gridded path's tas dir).
    x_values: list[float] = []
    for anchor in gauss_anchors:
        pair = _load_anchor_2050_2069(tas_processed_dir, anchor)
        if pair is None:
            raise ValueError(f"Missing tas files for so2 anchor {anchor.name}")
        exp_tas, base_tas = pair
        x_values.append((_global_mean(exp_tas) - _global_mean(base_tas))["tas"].item())
    X_numpy = np.array(x_values)[:, np.newaxis]

    # y: SO2 injection at each cooling target, 2050-2069 mean (lat-only).
    so2_paths = {
        "gauss-1.5": [processed_dir / f"Controller_start_DefaultMA_{m}.txt" for m in ("001", "002", "003")],
        "gauss-1.0": [processed_dir / f"Controller_start_Lower-0.5-MA_{m}.txt" for m in ("001", "002", "003")],
        "gauss-0.5": [processed_dir / f"Controller_start_Lower-1.0-MA_{m}.txt" for m in ("001", "002", "003")],
    }
    rebased_fields = [
        temporal_average(process_so2_data(so2_paths[a.name]), "2050", "2069")
        for a in gauss_anchors
    ]
    Y = xr.concat(rebased_fields, dim="sample")
    beta_numpy = np.linalg.lstsq(X_numpy, Y.values, rcond=None)[0]

    beta_xr = xr.DataArray(
        beta_numpy,
        dims=["features", "lat"],
        coords={"features": ["slope"], "lat": Y.lat.values},
    ).to_dataset(name="so2")
    print(f"Writing interpolator (so2, {len(x_values)} anchors) to {interpolator_path}")
    beta_xr.to_netcdf(interpolator_path)


def fit_delta(
    var: str,
    data_dir: str | Path,
    output_dir: str | Path,
    ignore_existing: bool = False,
) -> None:
    """Fit the per-grid-cell linear ``slope`` mapping ``tas`` cooling to ``var`` delta.

    For each anchor (GAUSS 0.5/1.0/1.5 + MIROC G6-1.5/G6-0.5), averages the
    experiment and its own baseline to the 2050-2069 window, computes the
    global-mean tas cooling (x) and the rebased per-cell ``var`` delta (y), and
    fits a single-slope ``np.linalg.lstsq`` through the origin across all
    anchor points. The slope is saved as ``output_dir/<var>/interpolator.nc``.

    ``so2`` takes a 1-D-in-latitude path (GAUSS-only). Skips work when the
    interpolator already exists and ``ignore_existing`` is ``False``.
    """
    output_dir = Path(output_dir) / var
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_dir = Path(data_dir) / var
    tas_processed_dir = Path(data_dir) / "tas"

    interpolator_path = output_dir / "interpolator.nc"
    if interpolator_path.exists() and not ignore_existing:
        print(f"Interpolator {interpolator_path} already exists.")
        return

    if var == "so2":
        _fit_delta_so2(processed_dir, tas_processed_dir, interpolator_path)
    else:
        _fit_delta_gridded(var, processed_dir, tas_processed_dir, interpolator_path)


if __name__ == "__main__":
    import fire  # CLI-only; kept out of module-level imports (see top of file)
    fire.Fire(fit_delta)
