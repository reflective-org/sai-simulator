"""Fit the per-variable SO2 → variable slope used by :mod:`backend.delta`.

For a given target variable, this script runs the three GAUSS experiments (cooling targets
0.5 / 1.0 / 1.5 °C) and the matching no-SAI baseline, computes the rebased experiment-vs-baseline
deltas, and fits a per-grid-cell linear regression of those deltas against the global-mean
``tas`` difference at each experiment. The resulting ``slope`` field is saved as
``output_dir/<var>/interpolator.nc`` and is what the dashboard multiplies by ``temp_diff`` to
turn a desired cooling into a regional delta.

The ``so2`` variable is handled separately because it is 1-D in latitude rather than 2-D.

CLI entry: ``python scripts/fit_delta.py <var> <data_dir> <output_dir> [--ignore_existing]``.
"""
import fire
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


def process_so2_data(paths: list[Path]) -> xr.DataArray:
    """Load and average the three-member GAUSS SO2 injection text files into a (lat, time) DataArray.

    Each file is a tab-separated table with timestamps as rows and per-latitude Tg columns
    (``30S(Tg)``, ``15S(Tg)``, ``15N(Tg)``, ``30N(Tg)``). The four latitude columns are
    extracted, sliced to 2035-2070, ensemble-averaged across members, and reshaped into
    an :class:`xr.DataArray` whose ``lat`` dimension holds the column headers and whose
    ``time`` dimension holds the timestamps. The result feeds the ``so2`` branch of
    :func:`fit_delta` so that branch can fit a 1-D-in-latitude linear interpolator instead
    of a 2-D (lat, lon) one.
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


def load_experiment_data(
    processed_dir: Path,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    """Load the three GAUSS experiment files (1.5, 1.0, 0.5 cooling) for ``processed_dir``.

    For the ``so2`` variable the source is per-member ``Controller_start_*.txt`` files which
    are aggregated through :func:`process_so2_data`; for everything else the standard
    ``output_gauss-{target}.nc`` xarray files are opened. Returned in order: 1.5 (default
    target), 1.0 (lower-0.5), 0.5 (lower-1.0).
    """
    if "so2" in str(processed_dir):
        # output_dir = Path("feedback_suite")
        target2paths = {
            0.5: [processed_dir / f"Controller_start_Lower-1.0-MA_{member}.txt" for member in ["001", "002", "003"]],
            1.0: [processed_dir / f"Controller_start_Lower-0.5-MA_{member}.txt" for member in ["001", "002", "003"]],
            1.5: [processed_dir / f"Controller_start_DefaultMA_{member}.txt" for member in ["001", "002", "003"]]
        }
        output_gauss_1_5 = process_so2_data(target2paths[1.5])
        output_gauss_1_0 = process_so2_data(target2paths[1.0])
        output_gauss_0_5 = process_so2_data(target2paths[0.5])

    else:
        output_gauss_1_5 = xr.open_dataset(processed_dir / "output_gauss-1.5.nc") # SSP245-MA-GAUSS-DEFAULT
        output_gauss_1_0 = xr.open_dataset(processed_dir / "output_gauss-1.0.nc") # SSP245-MA-GAUSS-LOWER-0.5
        output_gauss_0_5 = xr.open_dataset(processed_dir / "output_gauss-0.5.nc") # SSP245-MA-GAUSS-LOWER-1.0

    return output_gauss_1_5, output_gauss_1_0, output_gauss_0_5


def load_baseline_data(processed_dir: Path) -> xr.Dataset:
    """Load the no-SAI baseline (``output_gauss-baseline.nc``), concatenating an ARISE historical prefix if present.

    The combined series is used to compute the experiment-vs-baseline deltas at the
    equivalent global-mean-temperature periods in :func:`get_temporal_averaged_baseline`.
    """
    output_gauss_baseline = xr.open_dataset(processed_dir / "output_gauss-baseline.nc") # MA-HISTORICAL ; MA-BASELINE

    historical_file = processed_dir / "output_arise-cmip_historical.nc"
    if historical_file.exists():
        output_arise_historical = xr.open_dataset(historical_file)
        # Concatenate the historical and baseline data
        output_arise_baseline = xr.concat([output_arise_historical, output_arise_baseline], dim="time")

    return output_gauss_baseline


def temporal_average(
    data: xr.Dataset | xr.DataArray,
    start_year: str | int,
    end_year: str | int,
) -> xr.Dataset | xr.DataArray:
    """Slice ``data`` to ``[start_year, end_year]`` and take the time mean. Helper for the fit-delta windows."""
    return data.sel(time=slice(start_year, end_year)).mean(dim="time")


def get_temporal_averaged_experiments(
    processed_dir: Path,
) -> tuple[xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray, xr.Dataset | xr.DataArray]:
    """Load the three GAUSS experiments and return their 2050-2069 temporal means.

    The fixed 2050-2069 window is where the SAI experiments are in equilibrium with their
    cooling targets, so taking the mean there gives a clean per-cell signal for the linear
    fit.
    """
    output_gauss_1_5, output_gauss_1_0, output_gauss_0_5 = load_experiment_data(processed_dir)

    # Slice sai exps to 2050 - 2069 and take the mean
    output_gauss_1_5 = temporal_average(output_gauss_1_5, "2050", "2069")
    output_gauss_1_0 = temporal_average(output_gauss_1_0, "2050", "2069")
    output_gauss_0_5 = temporal_average(output_gauss_0_5, "2050", "2069")

    return output_gauss_1_5, output_gauss_1_0, output_gauss_0_5,


def get_temporal_averaged_baseline(
    processed_dir: Path,
    equivalent: bool = True,
) -> tuple[xr.Dataset | xr.DataArray | int, xr.Dataset | xr.DataArray | int, xr.Dataset | xr.DataArray | int]:
    """Return the baseline 20-year temporal means matched to the 1.5 / 1.0 / 0.5 cooling targets.

    When ``equivalent=True`` (the default) the baselines are sliced to the historical periods
    whose global-mean temperatures match each target above pre-industrial (2020-2039, 2008-2027,
    1993-2012), so the experiment-vs-baseline difference isolates the SAI signal at equal
    warming. When ``equivalent=False`` all three baselines collapse to the same 2050-2069
    window (used for ``fit_delta``'s rebasing step). ``so2`` has no baseline series, so the
    function returns ``(0, 0, 0)`` for that case.
    """
    if "so2" in str(processed_dir):
        # No baseline data for SO2
        return 0, 0, 0

    else:
        output_gauss_baseline = load_baseline_data(processed_dir)

        if equivalent:
            # Slice no SAI baselines to equivalent global mean temps and take the mean
            output_gauss_baseline_1_5 = temporal_average(output_gauss_baseline, "2020", "2039") # equivalent to warming at 1.5 above PI
            output_gauss_baseline_1_0 = temporal_average(output_gauss_baseline, "2008", "2027") # equivalent to warming at 1.0 above PI
            output_gauss_baseline_0_5 = temporal_average(output_gauss_baseline, "1993", "2012") # equivalent to warming at 0.5 above PI
        else:
            # Slice no SAI baselines to 2050 - 2069 and take the mean
            output_gauss_baseline_1_5 = temporal_average(output_gauss_baseline, "2050", "2069")
            output_gauss_baseline_1_0 = output_gauss_baseline_1_5
            output_gauss_baseline_0_5 = output_gauss_baseline_1_5

        return output_gauss_baseline_1_5, output_gauss_baseline_1_0, output_gauss_baseline_0_5


def fit_delta(
    var: str,
    data_dir: str | Path,
    output_dir: str | Path,
    ignore_existing: bool = False,
) -> None:
    """Fit the per-grid-cell linear ``slope`` mapping ``tas`` difference to ``var`` delta.

    Walks the three GAUSS experiments, averages each to the 2050-2069 window, computes
    ``global_mean_tas_experiment - global_mean_tas_baseline`` at each cooling target, and
    uses those three x-values together with the rebased per-cell ``var`` deltas as the
    response. A single-feature ``np.linalg.lstsq`` fit per cell yields the slope, which is
    saved as ``output_dir/<var>/interpolator.nc``. Skips work when the interpolator already
    exists and ``ignore_existing`` is ``False``.

    The ``so2`` variable is handled in a 1-D-in-latitude path because its source files
    don't have a longitude axis.
    """
    output_dir = Path(output_dir) / var
    output_dir.mkdir(exist_ok=True, parents=True)

    processed_dir = Path(data_dir) / var
    tas_processed_dir = Path(data_dir) / "tas"

    interpolator_path = output_dir / "interpolator.nc"
    if interpolator_path.exists() and not ignore_existing:
        print(f"Interpolator {interpolator_path} already exists.")
        interpolator = xr.open_dataset(interpolator_path)
        return 

    # Get tas difference
    output_gauss_1_5_tas, output_gauss_1_0_tas, output_gauss_0_5_tas = get_temporal_averaged_experiments(tas_processed_dir)
    output_gauss_baseline_tas, _, _ = get_temporal_averaged_baseline(tas_processed_dir, equivalent=False)
    mean_diff_1_5 = (
        output_gauss_1_5_tas.weighted(np.cos(np.deg2rad(output_gauss_1_5_tas.lat))).mean(('lat', 'lon')) - 
        output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon'))
    )["tas"].item()
    mean_diff_1_0 = (
        output_gauss_1_0_tas.weighted(np.cos(np.deg2rad(output_gauss_1_0_tas.lat))).mean(('lat', 'lon')) - 
        output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon'))
    )["tas"].item()
    mean_diff_0_5 = (
        output_gauss_0_5_tas.weighted(np.cos(np.deg2rad(output_gauss_0_5_tas.lat))).mean(('lat', 'lon')) - 
        output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon'))
    )["tas"].item()

    # Fit a linear model to the three tas points
    x_values = np.array([mean_diff_1_5, mean_diff_1_0, mean_diff_0_5])

    # Create a DataArray for the design matrix X which includes a constant term for the intercept
    # x_values should be an array of the x-values corresponding to each DataArray, e.g., [0.5, 1.0, 1.5]
    X = xr.DataArray(x_values[:, np.newaxis], dims=['sample', 'features'])
    X_numpy = X.values

    # Get var data
    output_gauss_1_5, output_gauss_1_0, output_gauss_0_5 = get_temporal_averaged_experiments(processed_dir)
    output_gauss_baseline, _, _ = get_temporal_averaged_baseline(processed_dir, equivalent=False)

    # Get difference from the SAI exps to no SAI baselines at the same time periods
    output_gauss_1_5_rebased = output_gauss_1_5 - output_gauss_baseline
    output_gauss_1_0_rebased = output_gauss_1_0 - output_gauss_baseline
    output_gauss_0_5_rebased = output_gauss_0_5 - output_gauss_baseline

    # Stack Y across all lat and lon coordinates into a new 'samples' dimension
    if var == "so2":
        Y = xr.concat([output_gauss_1_5_rebased, output_gauss_1_0_rebased, output_gauss_0_5_rebased], dim='sample')
        beta_numpy = np.linalg.lstsq(X_numpy, Y.values, rcond=None)[0]
        beta_xr = xr.DataArray(beta_numpy, dims=['features', 'lat'], 
                               coords={'features': ['slope'],
                                       'lat': Y.lat.values})
    else:
        # Stack the data arrays along a new dimension ('sample'), aligning with the order of x_values
        Y = xr.concat(
            [output_gauss_1_5_rebased, output_gauss_1_0_rebased, output_gauss_0_5_rebased],
            dim='sample'
        ).sel(model="CESM2-WACCM", ssp="ssp245")[var]
        Y_stacked = Y.stack(samples=('lat', 'lon'))

        beta_numpy = np.linalg.lstsq(X_numpy, Y_stacked.values, rcond=None)[0]

        # Reshape the beta coefficients to have dimensions (features, lat, lon)
        beta_reshaped = beta_numpy.reshape((1, Y.lat.size, Y.lon.size))
        beta_xr = xr.DataArray(beta_reshaped, dims=['features', 'lat', 'lon'], 
                            coords={'features': ['slope'],
                                    'lat': Y.lat.values, 
                                    'lon': Y.lon.values})

    # Convert to a Dataset and save to disk
    beta_xr = beta_xr.to_dataset(name=var)
    print("Writing interpolator to", interpolator_path)
    beta_xr.to_netcdf(interpolator_path)


if __name__ == "__main__":
    fire.Fire(fit_delta)
