"""Fit per-grid-cell logistic regressions mapping Arctic ``tas`` anomaly to ``icefrac``.

For **each model** in :data:`MODEL_SCENARIOS` independently: stitches together that model's
baseline + SAI experiments, loads the matching Arctic temperature and ice-fraction fields,
filters out grid cells with too little signal or too few valid observations, then drives
:class:`backend.sigmoid.LogisticFitter` to fit a four-parameter decreasing logistic for every
cell in the Arctic. Temperature anomalies are computed against each model's **own** baseline
1850-1900 climatology so inter-model bias doesn't leak into the fit. The per-model parameter
arrays are stacked into one ``(model, features, lon_band, lat_band, lat, lon)`` DataArray
(non-Arctic = NaN) and saved as ``output_dir/<var>_arctic/interpolator.nc`` — this is the
``logistic_params`` file consumed by :func:`backend.icefrac.predict_icefrac`, which evaluates
every model's sigmoid and averages the predictions (mean of sigmoids, not sigmoid of pooled
data).

CLI entry: ``python scripts/fit_icefrac_arctic.py <var> <data_dir> <output_dir> [--ignore_existing]``.
"""
import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ``fire`` is imported lazily inside the ``__main__`` block below so that
# importing this module (e.g. for unit tests of ``MODEL_SCENARIOS`` and the
# fit helpers) does not require ``fire`` in the test environment.
from backend.sigmoid import LogisticFitter
from backend.icefrac import calculate_temperature_anomalies, create_arctic_dataset
from backend.utils import open_xarray_datasets, save_processed_data


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Per-model experiment scenarios. Outer keys are the ``model`` coord values the
# rest of the pipeline uses (see MODELS in process_monthly_gauss.py); inner keys
# are internal scenario names, values contain file names and time ranges. The
# "baseline" entry doubles as the source of each model's 1850-1900 anomaly
# reference, so it must be the merged historical+SSP2-4.5 file.
#
# MIROC uses its GeoMIP ``baseline`` rather than the ScenarioMIP ``ssp245`` run the
# projection emulator trains on: the sigmoid pairs each G6 SAI scenario with the
# run it branched from (as the delta fit does), keeping the temperature-anomaly
# axis and the ice response on one realization. (ssp245 ICEFRAC exists since the
# CMOR-siconc fill and could be ADDED as extra no-SAI training data later — a
# deliberate decision, since its regrid recipe differs slightly at the ice edge
# from the GeoMIP-derived stores; see the store's ICEFRAC note attr.) A model whose
# baseline file is missing is skipped with a warning (except the primary model,
# which is required); individual missing SAI files are skipped likewise.
MODEL_SCENARIOS = {
    "CESM2-WACCM": {
        "baseline": {"file": "output_gauss-baseline.nc", "start": 1900, "end": 2098},
        "sai_0_5":  {"file": "output_gauss-0.5.nc",      "start": 2050, "end": 2069},
        "sai_1_0":  {"file": "output_gauss-1.0.nc",      "start": 2050, "end": 2069},
        "sai_1_5":  {"file": "output_gauss-1.5.nc",      "start": 2050, "end": 2069},
    },
    "MIROC-ES2H": {
        "baseline":   {"file": "output_gauss-baseline-miroc.nc", "start": 1900, "end": 2098},
        "sai_g6_0_5": {"file": "output_gauss-g6-0.5-miroc.nc",   "start": 2050, "end": 2069},
        "sai_g6_1_5": {"file": "output_gauss-g6-1.5-miroc.nc",   "start": 2050, "end": 2069},
    },
    # UKESM's icefrac is a SINGLE realization (r3) in every experiment, because
    # ESGF publishes its historical sea ice for that one member — see
    # variable_members in process_monthly_gauss.MODELS. Its tas is still a
    # three-member mean, so this pairs a one-member ice record against an
    # ensemble-mean temperature; the forced response they share is what the
    # logistic fits, and the extra realization noise on the ice side widens the
    # scatter without biasing the curve.
    "UKESM1-1-LL": {
        "baseline":   {"file": "output_gauss-baseline-ukesm.nc", "start": 1900, "end": 2098},
        "sai_g6_1_5": {"file": "output_gauss-g6-1.5-ukesm.nc",   "start": 2050, "end": 2069},
    },
}

# Models whose scenarios are known-good but which are deliberately NOT fitted.
# Kept as real configuration (rather than commented-out code) so an exclusion is
# testable, and so ``scripts/plot_icefrac_fit.py`` can show what the rejected fit
# would look like. Nothing in the fitting path reads this dict.
#
# Empty today. UKESM1-1-LL sat here while its ICEFRAC began in 2015, which left
# the training window with no cold regime and the logistic's full-ice plateau
# extrapolated rather than fitted. ESGF supplied historical sea ice for the member
# this repo calls r3, so it now trains from 1900 like the others and has moved into
# MODEL_SCENARIOS above.
EXCLUDED_MODEL_SCENARIOS: dict[str, dict] = {}

# The model whose baseline file must exist (its grid is the common target grid
# every other model is regridded onto, and it supplies the output coords).
PRIMARY_MODEL = "CESM2-WACCM"

VAR_TEMP = "tas"

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def reshape_icefrac_data(
    dataset: xr.Dataset, 
    var_name: str = 'icefrac', 
    start_year: str = '2050', 
    end_year: str = '2069'
) -> np.ndarray:
    """
    Extracts a time slice of ice fraction data and flattens the spatial dimensions.
    
    Args:
        dataset: xarray Dataset (expected to be pre-processed/arctic-adjusted).
        var_name: The name of the data variable to access (default: 'icefrac').
        start_year: Start year for time slice. 
        end_year: End year for time slice.
    
    Returns:
        np.ndarray: Array with shape (time, spatial_points).
        
    Raises:
        KeyError: If var_name is not in the dataset.
        ValueError: If the time slice returns empty data.
    """
    # 1. Validation: Ensure variable exists
    if var_name not in dataset:
        raise KeyError(f"Variable '{var_name}' not found in dataset.")

    # 2. Slicing: Perform slicing lazily on the xarray object
    # Note: We assume the input dataset is already processed (e.g. create_arctic_dataset applied beforehand)
    # to maintain separation of concerns.
    subset = dataset[var_name].sel(time=slice(start_year, end_year))
    # No year is dropped here. This used to remove 2015 unconditionally, which
    # was a compensation for CESM2-WACCM (its processed tas has no 2015 while its
    # icefrac does). For a model carrying 2015 in BOTH files the drop shifted ice
    # one year against tas for every later year. _fit_one_model now pairs the two
    # on their common years instead, which handles CESM without special-casing it.
    
    # print(f"Subset shape: {subset.shape}")
    # print(f"Subset time range: {subset.time.values}")

    if subset.size == 0:
        raise ValueError(f"No data found for time range {start_year}-{end_year}")

    # 3. Reshaping: Use xarray stack to safely combine lat/lon (or last two dims)
    # This automatically handles dimension ordering better than raw reshape
    # Assuming the last two dimensions are the spatial ones we want to flatten
    spatial_dims = subset.dims[-2:] 
    stacked = subset.stack(spatial=spatial_dims)
    
    # 4. Conversion: Transpose to ensure (time, spatial) order, then load to numpy
    # '...' handles any leading dimensions (like ensemble members) automatically
    return stacked.transpose(..., "spatial").values.squeeze()


def filter_icefrac_data(
    data: np.ndarray, 
    min_valid_obs: int = 10, 
    intensity_threshold: float = 0.15
):
    """
    Filters ice fraction data based on observational quality and intensity thresholds.
    
    This function performs two filtering steps:
    1. Insufficient Data: Grid cells with fewer than `min_valid_obs` non-NaN values 
       along the time dimension are entirely masked (set to NaN).
    2. Low Intensity: Grid cells where all valid values are below 
       `intensity_threshold` are entirely masked (set to NaN).

    **Ice-free observations are KEPT.** An earlier version set every value <= 0.01
    to NaN as "noise", which censored the training set exactly where a decreasing
    logistic needs data: the zeros are what pin its warm asymptote. The cost scaled
    with how ice-free a model gets — CESM2-WACCM lost 1.8% of its observations,
    MIROC-ES2H 8.0%, UKESM1-1-LL 43.6% — and left the warm end unconstrained, so
    the fitted curve settled on a floor instead of decaying. Arctic-wide September
    ice bias against each model's own output improved for all three when the zeros
    were kept (UKESM +0.647 -> +0.001, MIROC +0.051 -> -0.017 million km^2).
    Marginal cells are still excluded, by `intensity_threshold` below, which is the
    guard the value threshold appeared to be aiming at.

    Note: This is a vectorized implementation. It avoids loops for performance and 
    fixes a logic issue in older versions where "Insufficient Data" cells were 
    incorrectly double-counted as "Low Intensity".

    Args:
        data (np.ndarray): Input array with shape (lon, lat, time, grid_cells).
        min_valid_obs (int, optional): Minimum required non-NaN observations per time series. 
            Defaults to 10.
        intensity_threshold (float, optional): If the maximum value in a time series 
            is below this, the series is considered low intensity. Defaults to 0.15.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - filtered_data: The processed data array (same shape as input).
            - count_insufficient: Array (lon, lat) counting cells dropped due to insufficient data.
            - count_low_intensity: Array (lon, lat) counting cells dropped due to low intensity.
    """
    # Work on a copy to ensure the input array is not mutated (no side effects)
    Y = data.copy()

    # ---------------------------------------------------------
    # Step 1: Analyze Time Series (Axis 2)
    # ---------------------------------------------------------
    # Count how many non-NaN values exist for every grid cell along the time axis
    valid_counts = np.count_nonzero(~np.isnan(Y), axis=2)
    
    # Calculate the max value per time series to check against intensity threshold.
    # We suppress RuntimeWarnings because some cells are already all-NaN, 
    # and np.nanmax raises a warning for empty slices (which is expected here).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_vals = np.nanmax(Y, axis=2)

    # ---------------------------------------------------------
    # Step 2: Create Boolean Masks (Shape: lon, lat, grid_cells)
    # ---------------------------------------------------------
    
    # Mask A: "Insufficient Data"
    # Cells that don't have enough data points.
    mask_too_few = valid_counts < min_valid_obs
    
    # Mask B: "Low Intensity"
    # Cells where the max value never exceeds the intensity threshold.
    # CRITICAL LOGIC: We use `& (~mask_too_few)` to ensure mutual exclusivity.
    # If a cell has too few obs, it is ONLY counted in mask A, not mask B.
    mask_low_intensity = (max_vals < intensity_threshold) & (~mask_too_few)

    # ---------------------------------------------------------
    # Step 3: Apply Masks and Aggregate Counts
    # ---------------------------------------------------------
    
    # Expand masks to match the 4D shape (lon, lat, time, grid_cells)
    # This broadcasts the 3D mask across the time dimension (axis 2)
    # Shape becomes: (lon, lat, 1, grid_cells) -> broadcasted to (lon, lat, time, grid_cells)
    
    # Apply Mask A
    mask_A_broadcast = np.expand_dims(mask_too_few, axis=2)
    Y = np.where(mask_A_broadcast, np.nan, Y)

    # Apply Mask B
    mask_B_broadcast = np.expand_dims(mask_low_intensity, axis=2)
    Y = np.where(mask_B_broadcast, np.nan, Y)

    # Sum the masks over the grid_cells dimension (axis 3 implies the last axis of the mask)
    # The masks are shape (lon, lat, grid_cells), so we sum over the last axis.
    count_insufficient_data = np.sum(mask_too_few, axis=-1)
    count_low_intensity = np.sum(mask_low_intensity, axis=-1)

    return Y, count_insufficient_data, count_low_intensity


def _window_years(path: Path, start: int, end: int) -> list[int]:
    """Integer years a processed file contributes to ``[start, end]``, in file order.

    The loaders below return bare arrays, so the caller needs this to know which
    year each time index corresponds to and pair ``tas`` with ``icefrac`` correctly.
    """
    ds = open_xarray_datasets(path)
    return [int(y) for y in ds.time.values if start <= int(y) <= end]


def _load_and_process_temp(
    data_dir: Path, 
    filename: str, 
    start_year: int, 
    end_year: int, 
    baseline_ref: xr.Dataset
) -> np.ndarray:
    """
    Loads temperature data, subsets the Arctic region, and calculates anomalies.

    Args:
        data_dir: Base directory containing data.
        filename: Name of the NetCDF file to load.
        start_year: Start year for time slicing.
        end_year: End year for time slicing.
        baseline_ref: The reference dataset used to calculate anomalies.

    Returns:
        np.ndarray: Flattened temperature anomalies for the specified time range.
    """
    # Load raw dataset
    ds = open_xarray_datasets(data_dir / VAR_TEMP / filename)
    
    # Subset to Arctic region
    arctic_ds = create_arctic_dataset(ds)
    
    # Calculate anomalies relative to the provided baseline reference
    anomalies = calculate_temperature_anomalies(
        arctic_ds[VAR_TEMP], 
        baseline_ref, 
        start_year=start_year, 
        end_year=end_year
    )

    # print(f"Anomalies shape: {anomalies[VAR_TEMP].shape}")
    # print(f"Anomalies time range: {anomalies[VAR_TEMP].time.values}")
    return anomalies[VAR_TEMP].values.squeeze()


def _load_and_process_ice(
    data_dir: Path, 
    var_name: str, 
    filename: str, 
    start_year: int, 
    end_year: int
) -> np.ndarray:
    """
    Loads ice fraction data, subsets the Arctic region, and reshapes for training.

    Args:
        data_dir: Base directory containing data.
        var_name: Variable name (e.g., 'icefrac').
        filename: Name of the NetCDF file to load.
        start_year: Start year for time slicing.
        end_year: End year for time slicing.

    Returns:
        np.ndarray: Reshaped ice fraction data suitable for model fitting.
    """
    ds = open_xarray_datasets(data_dir / var_name / filename)
    
    # Subset to Arctic region and reshape spatial dims
    reshaped_data = reshape_icefrac_data(
        create_arctic_dataset(ds), 
        var_name=var_name, 
        start_year=str(start_year), 
        end_year=str(end_year)
    )
    
    return reshaped_data


def _expand_params_to_global(
    params: np.ndarray,
    template_ds: xr.Dataset,
    arctic_ref: xr.Dataset,
) -> np.ndarray:
    """
    Expands one model's fitted parameters back to the full global grid.

    This handles the complexity of padding the non-Arctic regions with NaNs so the
    output matches the dimensions of the original global input files.

    Args:
        params: The fitted logistic parameters (features, lon_band, lat_band, spatial_flat).
        template_ds: A global dataset used to extract full lat/lon shapes.
        arctic_ref: The arctic-subset dataset used to recover the (lat, lon) shape.

    Returns:
        np.ndarray with shape (features, lon_band, lat_band, global_lat, global_lon).
    """
    # 1. Reshape params from flat spatial dim back to (lat, lon) for the arctic region
    # Expected shape: (features, lon_band, lat_band, arctic_lat, lon)
    params_reshaped = params.reshape(params.shape[:-1] + arctic_ref.tas.shape[-2:])

    # 2. Create a container for the full global grid filled with NaNs
    # Structure: (features, lon_band, lat_band, global_lat, global_lon)
    expanded_shape = list(params_reshaped.shape[:-2] + template_ds.tas.shape[-2:])
    expanded_params = np.full(expanded_shape, np.nan)

    # 3. Fill the Arctic portion of the grid
    # We assume the Arctic data corresponds to the highest latitudes (end of the array)
    lat_cutoff = params_reshaped.shape[-2]
    expanded_params[..., -lat_cutoff:, :] = params_reshaped

    return expanded_params


def _save_params_to_netcdf(
    model2expanded_params: dict[str, np.ndarray],
    template_ds: xr.Dataset,
    arctic_ref: xr.Dataset,
    output_path: Path,
) -> None:
    """
    Stacks per-model global parameter grids on a ``model`` dim and saves to NetCDF.

    Args:
        model2expanded_params: model name → global parameter grid from
            :func:`_expand_params_to_global`, one entry per fitted model.
        template_ds: A global dataset used to extract full lat/lon coordinates.
        arctic_ref: The arctic-subset dataset used to extract band coordinates.
        output_path: Path where the NetCDF file will be saved.
    """
    models = list(model2expanded_params)
    stacked = np.stack([model2expanded_params[m] for m in models], axis=0)

    params_xr = xr.DataArray(
        stacked,
        dims=['model', 'features', 'lon_band', 'lat_band', 'lat', 'lon'],
        coords={
            'model': models,
            'features': ['L', 'x0', 'k', 'b'],
            'lon_band': arctic_ref.lon_band,
            'lat_band': arctic_ref.lat_band,
            'lat': template_ds.lat,
            'lon': template_ds.lon,
        },
        name='logistic_params'
    )

    save_processed_data(params_xr, output_path, overwrite=True)


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------

def _fit_one_model(
    var: str,
    data_path: Path,
    scenarios: dict[str, dict],
) -> np.ndarray | None:
    """Fit one model's per-cell logistics from its own baseline + SAI scenarios.

    Temperature anomalies are computed against this model's own baseline
    1850-1900 climatology. Missing SAI files are skipped with a warning;
    returns ``None`` when the baseline file itself is missing (the model can't
    be fit at all).

    Returns:
        (features, lon_band, lat_band, spatial_flat) parameter array, or None.
    """
    baseline_file = scenarios["baseline"]["file"]
    if not (data_path / VAR_TEMP / baseline_file).exists():
        return None

    # --- 1. Prepare Baseline Reference ---
    # The model's own baseline temperature serves as the climatology reference
    # for anomaly calculations across all of its scenarios.
    baseline_raw = open_xarray_datasets(data_path / VAR_TEMP / baseline_file)
    baseline_arctic_ref = create_arctic_dataset(baseline_raw)

    # --- 2. Load & Process Data Scenarios ---
    X_arrays = []
    Y_arrays = []

    for name, config in scenarios.items():
        if not (data_path / VAR_TEMP / config["file"]).exists() or not (
            data_path / var / config["file"]
        ).exists():
            print(f"  scenario {name}: missing {config['file']} — skipping")
            continue

        # Process Temperature (X)
        x_data = _load_and_process_temp(
            data_path,
            config["file"],
            config["start"],
            config["end"],
            baseline_arctic_ref
        )
        X_arrays.append(x_data)

        # Process Ice Fraction (Y)
        y_data = _load_and_process_ice(
            data_path,
            var,
            config["file"],
            config["start"],
            config["end"]
        )

        # Pair the two on the years BOTH files actually provide. A model's tas and
        # icefrac files need not cover the same years: CESM2-WACCM's processed tas
        # has no 2015 while its icefrac does. Aligning explicitly keeps each year's
        # ice with its own year's temperature; relying on index position would
        # silently regress ice on the previous year's anomaly wherever they differ.
        temp_years = _window_years(
            data_path / VAR_TEMP / config["file"], config["start"], config["end"]
        )
        ice_years = _window_years(
            data_path / var / config["file"], config["start"], config["end"]
        )
        if x_data.shape[2] != len(temp_years) or y_data.shape[2] != len(ice_years):
            raise ValueError(
                f"scenario {name}: loader returned {x_data.shape[2]} tas and "
                f"{y_data.shape[2]} {var} steps for {len(temp_years)} and "
                f"{len(ice_years)} years — year bookkeeping is out of step"
            )
        shared = sorted(set(temp_years) & set(ice_years))
        if not shared:
            print(f"  scenario {name}: no year covered by both tas and {var} — skipping")
            X_arrays.pop()
            continue
        if len(shared) < len(temp_years) or len(shared) < len(ice_years):
            print(f"  scenario {name}: pairing on {len(shared)} shared years "
                  f"(tas {len(temp_years)}, {var} {len(ice_years)})")
        x_data = x_data[:, :, [temp_years.index(y) for y in shared]]
        y_data = y_data[:, :, [ice_years.index(y) for y in shared], :]
        X_arrays[-1] = x_data
        Y_arrays.append(y_data)

    if not X_arrays:
        return None

    # --- 3. Merge & Filter ---
    # Concatenate all scenarios along the time axis (axis 2)
    X = np.concatenate(X_arrays, axis=2)
    Y = np.concatenate(Y_arrays, axis=2)

    # Filter Y data (removes noise and insufficient data points)
    # Note: X is not filtered here; the fitter handles per-cell NaN masking
    Y, _, _ = filter_icefrac_data(Y)

    # --- 4. Model Fitting ---
    fitter = LogisticFitter(X, Y)

    # Fit models using all available cores (n_jobs=-1)
    params, _, _, _ = fitter.fit_all(n_jobs=-1)
    return params


def fit_arctic_logistic(
    var: str,
    data_dir: str,
    output_dir: str,
    ignore_existing: bool = False
) -> None:
    """
    Orchestrates the loading, transforming, and fitting of logistic models
    relating Arctic temperature anomalies to ice fraction, one fit per model.

    The workflow follows these steps:
    1. Checks if output exists (skips if so).
    2. For each model in :data:`MODEL_SCENARIOS` (against its own baseline
       reference): loads and processes Temperature (X) and Ice Fraction (Y)
       for every scenario, concatenates them into that model's training set,
       filters invalid or low-quality ice data, and fits per-cell logistics
       (in parallel).
    3. Stacks the per-model parameters on a ``model`` dim and saves them to a
       single NetCDF file, from which :func:`backend.icefrac.predict_icefrac`
       averages the per-model sigmoid predictions.

    Args:
        var: The variable name to fit (e.g., 'icefrac').
        data_dir: Path to the directory containing processed input data.
        output_dir: Path to the directory where model outputs will be saved.
        ignore_existing: If True, overwrites existing output files. Defaults to False.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir) / f"{var}_arctic"
    out_path.mkdir(exist_ok=True, parents=True)

    interpolator_path = out_path / "interpolator.nc"
    if interpolator_path.exists() and not ignore_existing:
        return

    model2expanded_params: dict[str, np.ndarray] = {}
    for model_name, scenarios in MODEL_SCENARIOS.items():
        print(f"--- Fitting {var} logistics for {model_name} ---")
        params = _fit_one_model(var, data_path, scenarios)
        if params is None:
            if model_name == PRIMARY_MODEL:
                raise ValueError(
                    f"Baseline file {scenarios['baseline']['file']} for primary model "
                    f"{PRIMARY_MODEL} is missing. Need to run process_monthly_gauss.py first."
                )
            print(f"\033[93m  no usable data for {model_name} — skipping this model\033[0m")
            continue

        # Expand to the global grid using this model's own baseline for the
        # arctic band shape (identical across models — all are regridded onto
        # the primary model's grid by process_monthly_gauss.py).
        baseline_raw = open_xarray_datasets(
            data_path / VAR_TEMP / scenarios["baseline"]["file"]
        )
        baseline_arctic_ref = create_arctic_dataset(baseline_raw)
        model2expanded_params[model_name] = _expand_params_to_global(
            params, baseline_raw, baseline_arctic_ref
        )

    # --- Save Output ---
    # Coords come from the primary model's baseline (the common target grid).
    primary_raw = open_xarray_datasets(
        data_path / VAR_TEMP / MODEL_SCENARIOS[PRIMARY_MODEL]["baseline"]["file"]
    )
    primary_arctic_ref = create_arctic_dataset(primary_raw)
    _save_params_to_netcdf(
        model2expanded_params, primary_raw, primary_arctic_ref, interpolator_path
    )


if __name__ == "__main__":
    import fire  # CLI-only; kept out of module-level imports (see top of file)
    fire.Fire(fit_arctic_logistic)