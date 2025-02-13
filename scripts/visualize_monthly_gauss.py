import fire
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path


def visualize_gauss(var, year, data_dir):

    valid_vars = ["tas", "pr"]
    if var not in valid_vars:
        raise ValueError(f"var must be one of {valid_vars}")
    
    # Error if year is not between 2035 and 2071
    if not 2035 <= year <= 2070:
        raise ValueError(f"year must be between 2035 and 2070")
    
    scenarios = ["baseline", "1.5", "1.0", "0.5"]
    data = {}
    for scenario in scenarios:
        file_path = Path(data_dir) / var / f"output_gauss-{scenario}.nc"
        if file_path.exists():
            ds = xr.open_dataset(file_path)
            data[scenario] = ds
        else:
            print(f"file {file_path} does not exist")
    
    # Plot the four cases
    fig = plt.figure(figsize=(12, 8))

    # Create individual subplots with projection
    axes = []
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
        axes.append(ax)
    
    for ax, (scenario, ds) in zip(axes, data.items()):
        selected_data = ds[var].sel(time=year, ssp='ssp245', model='CESM2-WACCM')
        selected_data.plot(ax=ax, transform=ccrs.PlateCarree())

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.add_feature(cfeature.OCEAN, alpha=0.1)

        ax.set_title(f"{scenario}")

    fig.suptitle(f"{var} year={year} ssp245 CESM2-WACCM")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fire.Fire(visualize_gauss)