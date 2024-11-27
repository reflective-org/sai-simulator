import fire
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


def process_so2_data(paths):
    latitudes = ["30S(Tg)", "15S(Tg)", "15N(Tg)", "30N(Tg)"]
    data = [pd.read_csv(path, sep="\s+").set_index("Timestamp")[latitudes].loc["2035":"2070"] for path in paths]
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


def load_experiment_data(processed_dir):

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


def load_baseline_data(processed_dir):
    output_gauss_baseline = xr.open_dataset(processed_dir / "output_gauss-baseline.nc") # MA-HISTORICAL ; MA-BASELINE

    return output_gauss_baseline


def temporal_average(data, start_year, end_year):
    return data.sel(time=slice(start_year, end_year)).mean(dim="time")


def get_temporal_averaged_experiments(processed_dir):

    output_gauss_1_5, output_gauss_1_0, output_gauss_0_5 = load_experiment_data(processed_dir)

    # Slice sai exps to 2050 - 2069 and take the mean
    output_gauss_1_5 = temporal_average(output_gauss_1_5, "2050", "2069")
    output_gauss_1_0 = temporal_average(output_gauss_1_0, "2050", "2069")
    output_gauss_0_5 = temporal_average(output_gauss_0_5, "2050", "2069")

    return output_gauss_1_5, output_gauss_1_0, output_gauss_0_5,


def get_temporal_averaged_baseline(processed_dir, equivalent=True):

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


def fit_delta(var, data_dir, output_dir, ignore_existing=False):
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
    mean_diff_1_5 = (output_gauss_1_5_tas.weighted(np.cos(np.deg2rad(output_gauss_1_5_tas.lat))).mean(('lat', 'lon')) - output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon')))["tas"].item()
    mean_diff_1_0 = (output_gauss_1_0_tas.weighted(np.cos(np.deg2rad(output_gauss_1_0_tas.lat))).mean(('lat', 'lon')) - output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon')))["tas"].item()
    mean_diff_0_5 = (output_gauss_0_5_tas.weighted(np.cos(np.deg2rad(output_gauss_0_5_tas.lat))).mean(('lat', 'lon')) - output_gauss_baseline_tas.weighted(np.cos(np.deg2rad(output_gauss_baseline_tas.lat))).mean(('lat', 'lon')))["tas"].item()

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
        Y = xr.concat([output_gauss_1_5_rebased, output_gauss_1_0_rebased, output_gauss_0_5_rebased], dim='sample').sel(model="CESM2-WACCM", ssp="ssp245")[var]
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
