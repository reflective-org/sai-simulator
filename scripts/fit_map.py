import fire
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import numba as nb
import os

njit = nb.njit

@njit(fastmath=True)
def stretched_sigmoid_x0(x, lam, beta, x0):
    """Vectorised stretched sigmoid (λ, β, x0 all scalars)."""
    z = -(x - x0) / lam    # z ≥ 0 for valid domain
    z = np.maximum(z, 0.0) # clip to domain  z ≥ 0
    return 1.0 - np.exp(-(z ** beta))

def _fit_one_column(y_col, x_data, p0, bounds, maxfev):
    """
    Returns (λ, β, x0) for a single 1-D y column.
    """
    popt, _ = curve_fit(
        stretched_sigmoid_x0,
        x_data,
        y_col,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev
    )
    return popt              # tuple length 3
    
def fit_stretched_sigmoid_columns(
        y, X,
        tiny_thr=1e-3,
        maxfev=10_000,
        n_jobs=None,
    ):
    """
    Parameters
    ----------
    y : ndarray, shape (n_time, n_cols)
        Your `y` matrix - each column is one time-series to fit.
    X : ndarray, shape (n_time,) or broadcastable to (n_time, 1)
        Predictor array (same for every column).
    tiny_thr : float
        Columns whose mean < tiny_thr are treated as "all zeros"
        and assigned (λ=∞, β=∞, x0=0).
    maxfev : int
        Max function evaluations passed to `curve_fit`.
    n_jobs : int or None
        Number of worker processes.  None → os.cpu_count().

    Returns
    -------
    reg : ndarray, shape (3, n_cols)
        Row 0 = λ̂, Row 1 = β̂, Row 2 = x̂0
    """
    # --- 1 set up shared x-data, initial guess, bounds ---------------------
    x_data  = np.ascontiguousarray(X.squeeze())
    x_min   = x_data.min()
    x_max   = x_data.max()
    p0      = (np.ptp(x_data) / 2.0, 1.0, x_min)
    bounds  = ((1e-9, 0.01, x_min - 1),
               (np.inf, 10.0, x_max + 2))

    n_cols  = y.shape[1]
    reg     = np.empty((3, n_cols), dtype=float)

    # --- 2 skip tiny columns in vectorised fashion -------------------------
    tiny_mask = y.mean(axis=0) < tiny_thr
    reg[:, tiny_mask] = np.array([np.inf, np.inf, 0.0]).reshape(3, 1)

    cols_to_fit = np.where(~tiny_mask)[0]
    if cols_to_fit.size == 0:        # nothing left
        return reg

    # --- 3 run the expensive fits in parallel ------------------------------
    _worker = partial(
        _fit_one_column,
        x_data=x_data,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev
    )

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = pool.map(_worker, (y[:, idx] for idx in cols_to_fit))
    reg[:, cols_to_fit] = np.array(list(results)).T
    return reg

def fit_map(var, data_dir, output_dir, num_bootstrap_replicates=100, ignore_existing=False):
    data_dir = Path(data_dir)
    input_fair_path = data_dir / "input_fair.nc"
    data_dir = data_dir / var
    output_path = data_dir / f"output_gauss-baseline.nc"
    if not output_path.exists():
        raise ValueError(f"output_path {output_path} does not exist. Need to run process_monthly_gauss.py first.")

    input_fair = xr.open_dataset(input_fair_path)

    output = xr.open_dataset(output_path)
    output = output.sel(time=input_fair.time)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    model_dir = output_dir / var
    model_dir.mkdir(exist_ok=True)

    np.random.seed(42)
    # Train a FaIR global tas -> ScenarioMIP regional tas model
    #   a. input_fair has variable tas and dimensions (ssp, time)
    #   b. output_scenario_mip has variable tas and dimensions (model, ssp, time, lat, lon)
    # Find overlapping ssp scenarios
    fair_ssps = set(input_fair.ssp.values)
    scenariomip_ssps = set(output.ssp.values)
    scenariomip_ssps = scenariomip_ssps & fair_ssps
    # Train a linear regression model for each model and bootstrap replicate
    scenariomip_models = set(output.model.values)
    model2bootstrapped_fair_emulators = defaultdict(list)
    for model in scenariomip_models:
        print(f"Training models for {model}")
        for i in tqdm(range(num_bootstrap_replicates), total=num_bootstrap_replicates):
            Xs, ys = [], []
            model_path = model_dir / f"fair_to_smip_{model}_{i}.joblib"
            if model_path.exists() and not ignore_existing:
                reg = load(model_path)
                model2bootstrapped_fair_emulators[model].append(reg)
                continue
            for ssp in scenariomip_ssps:
                fair_model_data = input_fair.sel(ssp=ssp)
                scenariomip_model_data = output.sel(model=model, ssp=ssp)
                X = fair_model_data.tas.values # (time)
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
            # For every climate model, train a linear regression that inputs global fair tas and outputs regional smip tas
            if var == "icefrac": # icefrac is a special case since the fit is not linear
                # fit a stretched sigmoid to the data
                reg = fit_stretched_sigmoid_columns(y, X, n_jobs=os.cpu_count())
            else:
                reg = LinearRegression().fit(X, y)
            model2bootstrapped_fair_emulators[model].append(reg)
            dump(reg, model_path)


if __name__ == "__main__":
    fire.Fire(fit_map)
