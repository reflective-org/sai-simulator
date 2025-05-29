### visualize_map_fit.py ###
import fire
import joblib
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


def visualize_variability(var, model, model_dir, data_dir):
    """
    Visualize the variabiiltiy of the emulation runs.
    """

    valid_vars = ["tas", "pr"]
    if var not in valid_vars:
        raise ValueError(f"var must be one of {valid_vars}")
    
    # Load Gauss data for lat, lon
    data_dir = Path(data_dir)
    try:
        ds = xr.open_dataarray(data_dir/var/"output_gauss-baseline.nc")
    except: 
        raise FileNotFoundError(f"Check that {data_dir} is a valid dir and contains data in dir {var}.")

    model_dir = Path(model_dir)
    NUM_EMULATORS=100
    coef_list = []
    intercept_list = []

    for i in range(NUM_EMULATORS):
        filepath = model_dir / var / f"fair_to_smip_{model}_{i}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist. Check that model={model} and {model_dir} is correct.")

        m = joblib.load(filepath)
        
        # Reshape the coefficients and intercepts
        coef_reshaped = m.coef_.reshape(ds.lat.size, ds.lon.size)
        intercept_reshaped = m.intercept_.reshape(ds.lat.size, ds.lon.size)

        # Stack the reshaped coefficients and intercepts along a new axis for emulator runs
        coef_list.append(coef_reshaped[np.newaxis, :, :])  # Add an extra dimension for 'emulator_run'
        intercept_list.append(intercept_reshaped[np.newaxis, :, :]) 

    print('Creating visualization!')
    # Concatenate along the 'emulator_run' dimension
    coef_dataset = xr.concat([xr.DataArray(coef, dims=['emulator_run', 'lat', 'lon'], 
                                        coords={'emulator_run': [i], 'lat': ds.lat.values, 'lon': ds.lon.values}) 
                            for i, coef in enumerate(coef_list)], dim='emulator_run')

    intercept_dataset = xr.concat([xr.DataArray(intercept, dims=['emulator_run', 'lat', 'lon'], 
                                            coords={'emulator_run': [i], 'lat': ds.lat.values, 'lon': ds.lon.values}) 
                                for i, intercept in enumerate(intercept_list)], dim='emulator_run')

    # Create a final Dataset containing both the coefficients and intercepts
    model_runs_ds = xr.Dataset({
        'model_coef': coef_dataset,
        'model_intercept': intercept_dataset
    })

    # Compute the range and IQR of the coef and intercept variables at each lat,lon
    coef_range = model_runs_ds.model_coef.max(dim='emulator_run') - model_runs_ds.model_coef.min(dim='emulator_run')
    coef_iqr = model_runs_ds.model_coef.quantile(0.75, dim='emulator_run') - model_runs_ds.model_coef.quantile(0.25, dim='emulator_run')
    intercept_range = model_runs_ds.model_intercept.max(dim='emulator_run') - model_runs_ds.model_intercept.min(dim='emulator_run')
    intercept_iqr = model_runs_ds.model_intercept.quantile(0.75, dim='emulator_run') - model_runs_ds.model_intercept.quantile(0.25, dim='emulator_run')
    coef_mean = model_runs_ds.model_coef.mean(dim='emulator_run')
    intercept_mean = model_runs_ds.model_intercept.mean(dim='emulator_run')

    stats_ds = xr.Dataset({
        'coef_range': coef_range,
        'intercept_range': intercept_range,
        'coef_iqr': coef_iqr,
        'intercept_iqr': intercept_iqr,
        'coef_mean': coef_mean,
        'intercept_mean': intercept_mean
    })

    fig = plt.figure(figsize=(12,8))
    for i, (name, data_var) in enumerate(stats_ds.data_vars.items()):
        ax = fig.add_subplot(3, 2, i+1, projection=ccrs.PlateCarree())
        data_var.plot(ax=ax, transform=ccrs.PlateCarree())

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)
    
    plt.suptitle(f"{var} {model} Linear Regression Visualization")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fire.Fire(visualize_variability)