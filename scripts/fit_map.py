"""Fit the global-temperature → ScenarioMIP-regional-variable linear emulators.

For each ``(model, bootstrap)`` pair, this script trains a single-feature
:class:`sklearn.linear_model.LinearRegression` mapping a global-mean ``tas`` anomaly to a
flattened ``(lat × lon)`` regional ``var`` field. The set of bootstrap replicates is
saved as ``output_dir/<var>/fair_to_smip_<model>_<i>.joblib`` — these are the per-emulator
files concatenated by :func:`backend.baseline.get_regional_models` at request time.

This is the **climate-change-only** fit, so each model trains on its projection
baseline — the standard ScenarioMIP SSP2-4.5 run where one exists, resolved via
:func:`backend.constants.resolve_projection_baseline_file`. (The SAI delta fit is
separate and rebases against whichever no-SAI run each model's SAI experiments
branched from; see ``ANCHORS`` in ``scripts/fit_delta.py``.)

**Warming-level fit in each model's own coordinates.** Each model's fields are
regressed on **its own global-mean tas anomaly** (cos-lat weighted, rebased to
its own 1850-1900 mean, taken from the tas file of the *same run* the variable's
field comes from). Slope and intercept therefore come out jointly consistent:
the fitted line passes through the model's own training centroid, the slope is
the model's pattern per K of its own global warming (multi-model global tas
slope = 1 by construction), and the intercept needs no post-hoc anchoring. At
request time the backend feeds FaIR's global anomaly into every model's line
and takes the equal-weight model mean — since the emulators are linear, that
mean of predictions IS the equal-weight average of the fitted lines, which is the
final global-temp → grid-cell mapping. Under this anchoring the displayed
multi-model global mean tracks the FaIR driver with no constant offset (each
model's ``intercept − historical rebase`` cancels its own mean warming).

FaIR still sets the *magnitude* of warming at request time; each model
contributes only its per-K pattern. As of 2026-08 FaIR defaults to the AR6
calibrated-constrained ensemble (mean across members) rather than the
CESM-tuned parameters — see ``FAIR_CALIBRATION`` in ``backend/baseline.py``.
The warming-level fit here is calibration-agnostic (X is each model's own tas
anomaly), so the emulators need no refit when the calibration changes; only
the FaIR-dependent caches and goldens do.

Same-run rule: a variable that falls back to MIROC's GeoMIP run would take its
X from that run's tas, which is *skin* temperature (see the store attrs). As of
the ssp245 QFLX fill no fit_map variable takes that fallback, but the caveat
applies to any future one.

CLI entry: ``python scripts/fit_map.py <var> <data_dir> <output_dir>
[--num_bootstrap_replicates 100] [--ignore_existing]``.
"""
import sys
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from joblib import dump
from sklearn.linear_model import LinearRegression

# ``fire`` is imported lazily inside the ``__main__`` block below so that
# importing this module (e.g. for unit tests of the pattern normalization)
# does not require ``fire`` in the test environment.

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.constants import (  # noqa: E402  (needs the repo-root path above)
    regional_models_for_var,
    resolve_projection_baseline_file,
)


# Training window for the projection-era response (mirrors the FaIR-driven
# fit's historical window: FaIR runs 2015-2099, and training on earlier years
# would fold the aerosol-era historical pattern into the scenario response).
TRAIN_YEARS = (2015, 2099)

# Pre-industrial reference window for each model's own global-tas anomaly —
# matches the rebase used everywhere else (FaIR, backend/historical.py).
PREINDUSTRIAL_YEARS = (1850, 1900)


def _own_global_tas_anomaly(tas_path: Path, model: str, ssp: str) -> xr.DataArray:
    """The model's own global-mean tas anomaly series from one tas baseline file.

    Cos(lat)-weighted global mean, rebased to the file's own 1850-1900 mean
    (the merged historical part), returned for the full file so the caller can
    align it to the variable's training rows. This is the regression's X: the
    fit is done in the model's own warming coordinates, so slope and intercept
    come out jointly consistent (see the module docstring).
    """
    with xr.open_dataset(tas_path) as f:
        tas_ds = f.sel(model=model, ssp=ssp)
        weights = np.cos(np.deg2rad(tas_ds.lat))
        gm = tas_ds["tas"].weighted(weights).mean(("lat", "lon")).load()
    ref = gm.sel(time=slice(*PREINDUSTRIAL_YEARS)).mean()
    if not np.isfinite(ref):
        raise ValueError(
            f"No {PREINDUSTRIAL_YEARS} pre-industrial window in {tas_path} for "
            f"{model} — is the historical run merged into this file?"
        )
    anomaly = gm - ref
    if float(anomaly.sel(time=slice(*TRAIN_YEARS)).std()) < 0.1:
        raise ValueError(
            f"Implausibly flat global tas anomaly for {model} from {tas_path}."
        )
    return anomaly


def _fit_model(
    output: xr.Dataset,
    tas_path: Path,
    model: str,
    var: str,
    model_dir: Path,
    num_bootstrap_replicates: int,
    ignore_existing: bool,
) -> None:
    """Train ``model``'s bootstrap emulator set from its projection-baseline file.

    Regresses the regional fields on the model's **own** global-mean tas anomaly
    (from ``tas_path``, the same run's tas file), one bootstrapped
    :class:`LinearRegression` per replicate. Rows are aligned pointwise (inner
    join), so a gap in a model's series contributes no samples rather than NaNs.
    """
    output = output.sel(time=slice(*TRAIN_YEARS)).load()
    print(f"training window: {output.time.min().values} to {output.time.max().values}")

    # Train a (model's own global tas anomaly) -> regional var model
    #   a. X: cos-lat global mean of the same-run tas file, rebased to its own
    #      1850-1900 mean (see _own_global_tas_anomaly)
    #   b. output has variable var and dimensions (model, ssp, time, lat, lon)
    file_models = set(output.model.values)
    if model not in file_models:
        raise ValueError(
            f"Expected model {model} in the baseline file, found {sorted(file_models)}."
        )
    ssps = list(output.ssp.values)
    # X is identical across bootstrap replicates — compute it once per ssp
    # (also avoids re-opening the tas file inside the hot loop, which trips
    # HDF5's non-threadsafe handle caching when it is the same file as output).
    own_tas_by_ssp = {ssp: _own_global_tas_anomaly(tas_path, model, ssp) for ssp in ssps}
    print(f"Training models for {model} on its own global tas anomaly (X from {tas_path.name})")
    for i in tqdm(range(num_bootstrap_replicates), total=num_bootstrap_replicates):
        Xs, ys = [], []
        model_path = model_dir / f"fair_to_smip_{model}_{i}.joblib"
        if model_path.exists() and not ignore_existing:
            continue
        for ssp in ssps:
            scenariomip_model_data = output.sel(model=model, ssp=ssp)

            own_tas, scenariomip_model_data = xr.align(
                own_tas_by_ssp[ssp], scenariomip_model_data, join="inner"
            )

            X = own_tas.values # (time)
            X = X[:, np.newaxis] # (time, 1)
            y = scenariomip_model_data[var].values # (time, lat, lon)
            if np.isnan(y).sum() > 0:
                continue
            # Sample X and y with replacement
            bootstrap_inds = np.random.choice(X.shape[0], X.shape[0])
            X = X[bootstrap_inds]
            y = y[bootstrap_inds]
            # y -> (time, lat, lon) -> (time, lat * lon)
            y = y.reshape(y.shape[0], -1)
            Xs.append(X)
            ys.append(y)
        X = np.concatenate(Xs, axis=0) # (time * ssp, 1)
        y = np.concatenate(ys, axis=0) # (time * ssp, lat * lon)
        # Train a linear regression mapping global tas anomaly -> regional smip var
        reg = LinearRegression().fit(X, y)
        dump(reg, model_path)


def fit_map(
    var: str,
    data_dir: str | Path,
    output_dir: str | Path,
    num_bootstrap_replicates: int = 100,
    ignore_existing: bool = False,
) -> None:
    """Train bootstrapped global-tas-to-regional-``var`` linear emulators and dump each to disk.

    For every model in :func:`backend.constants.regional_models_for_var`, resolves that
    model's projection baseline under ``<data_dir>/<var>/`` (see
    :func:`backend.constants.resolve_projection_baseline_file`) and trains
    ``num_bootstrap_replicates`` regressors of the regional fields on the model's
    **own** global-mean tas anomaly, on a resampled (with-replacement) row stack.

    Bootstrap files already present are skipped unless ``ignore_existing`` is ``True``. The
    fitted regressors are persisted as ``output_dir/<var>/fair_to_smip_<model>_<i>.joblib``;
    SSP slices that contain any NaN in ``var`` are dropped from that bootstrap rather than
    poisoning the fit.

    Raises :class:`ValueError` if the primary model has no processed baseline for this
    variable — in that case ``process_monthly_gauss.py`` needs to run first. Secondary
    models are skipped with a warning, so the fit still works before their outputs are
    staged; a model that permanently lacks a variable must be declared in
    ``backend.constants.REGIONAL_MODEL_VAR_EXCLUSIONS`` so the runtime does not look
    for its emulators (UKESM1-1-LL / p-e is the current case).
    """
    data_dir = Path(data_dir)
    var_dir = data_dir / var

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / var
    model_dir.mkdir(exist_ok=True)

    for position, model in enumerate(regional_models_for_var(var)):
        baseline_path = resolve_projection_baseline_file(var_dir, model)
        if baseline_path is None:
            if position == 0:
                raise ValueError(
                    f"No processed projection baseline for {model} in {var_dir}. "
                    "Need to run process_monthly_gauss.py first."
                )
            print(f"Note: no processed baseline for {model} in {var_dir} — skipping it.")
            continue

        # The regression's X comes from the tas file of the SAME run this
        # variable's field comes from (a variable that fell back to another
        # no-SAI run must use that run's own warming, not the preferred run's).
        same_run_tas = data_dir / "tas" / baseline_path.name
        if not same_run_tas.exists():
            resolved_tas = resolve_projection_baseline_file(data_dir / "tas", model)
            if resolved_tas is None:
                raise ValueError(
                    f"No tas baseline for {model} in {data_dir / 'tas'} to supply the "
                    "own-warming regressor. Process tas first."
                )
            print(
                f"Note: no {baseline_path.name} in the tas dir — using "
                f"{resolved_tas.name} for {model}'s own-tas regressor."
            )
            same_run_tas = resolved_tas

        print(f"Fitting {model} from {baseline_path.name}")
        # Reseed per model so each model's bootstrap draws are reproducible on
        # their own, independent of which other models are present.
        np.random.seed(42)
        output = xr.open_dataset(baseline_path)
        _fit_model(
            output, same_run_tas, model, var, model_dir,
            num_bootstrap_replicates, ignore_existing,
        )


if __name__ == "__main__":
    import fire  # CLI-only; kept out of module-level imports (see top of file)
    fire.Fire(fit_map)
