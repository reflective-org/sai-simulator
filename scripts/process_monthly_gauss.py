import fire
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def process_monthly(var, data_dir, output_dir):

    var2gauss_var = {
        "tas": "TREFHT",
        "pr": "PRECT",
        "aod": "AODVISstdn",
        "tasmin": "TREFHTMN",
        "tasmax": "TREFHTMX",
        "tas_above_35": "TREFHT_above_35",
        "tas_above_40": "TREFHT_above_40",
        "tas_below_0": "TREFHT_below_0",
        "pr_above_10": "PRECT_above_10",
        "pr_above_20": "PRECT_above_20",
        "p-e": "QFLX",
    }

    if var not in var2gauss_var:
        raise ValueError(f"var must be one of {list(var2gauss_var.keys())}")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir = output_dir / var
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get correct latitudes from CESM2-WACCM (center of bounds)
    correct_lat = np.load(output_dir.parent / "correct_lat.npy")
    common_grid = {
        'lon': np.linspace(0, 358.75, 288),
        'lat': correct_lat
    }
    exp2gauss_exp = {
        "WACCM-MA-1deg": "baseline",
        "SSP245-MA-GAUSS-DEFAULT": "1.5",
        "5": "1.0",
        "0": "0.5"
    }
    gauss_var = var2gauss_var[var]
    if gauss_var == "PRECT":
        # for MA-BASELINE-002, MA-BASELINE-003, and MA-HISTORICAL,
        # we need to load the variables PRECC and PRECL then sum them
        for member in ["002", "003"]:
            cdf_path_preccs = sorted(list(data_dir.glob(f"*1deg.{member}*PRECC*.nc"))) 
            cdf_path_precls = sorted(list(data_dir.glob(f"*1deg.{member}*PRECL*.nc")))
            if member == "002":
                # Only need to add these once
                cdf_path_preccs += sorted(list(data_dir.glob(f"*HIST*PRECC*.nc")))
                cdf_path_precls += sorted(list(data_dir.glob(f"*HIST*PRECL*.nc")))
            for cdf_path_precc, cdf_path_precl in zip(cdf_path_preccs, cdf_path_precls):
                x_precc = xr.open_dataset(cdf_path_precc, use_cftime=True)
                x_precl = xr.open_dataset(cdf_path_precl, use_cftime=True)
                x_precc = x_precc.rename_vars({"PRECC": "PRECT"})
                x_precl = x_precl.rename_vars({"PRECL": "PRECT"})
                x = x_precc[['PRECT']] + x_precl[['PRECT']]
                cdf_path = str(cdf_path_precc).replace("PRECC", "PRECT")
                print(f"Saving sum {cdf_path}")
                x.to_netcdf(cdf_path)

    exp2member2data = defaultdict(lambda: defaultdict(list))
    print("Loading GAUSS data...")
    cdf_paths = (
        list(data_dir.glob(f"*1deg.001*{gauss_var}.*.nc")) +
        list(data_dir.glob(f"*1deg.002*{gauss_var}.*.nc")) +
        list(data_dir.glob(f"*1deg.003*{gauss_var}.*.nc")) +
        list(data_dir.glob(f"*SSP245-MA-GAUSS-DEFAULT.*{gauss_var}.*.nc")) + 
        list(data_dir.glob(f"*SSP245-MA-GAUSS-LOWER-0.5*{gauss_var}.*.nc")) +
        list(data_dir.glob(f"*SSP245-MA-GAUSS-LOWER-1.0*{gauss_var}.*.nc")) + 
        list(data_dir.glob(f"*CMIP6-historical*{gauss_var}.*.nc"))
    )
    cdf_paths = [cdf_path for cdf_path in cdf_paths if "h1" not in str(cdf_path)]
    exp2member2paths = defaultdict(lambda: defaultdict(list))
    for cdf_path in tqdm(cdf_paths):
        split_path = cdf_path.stem.split(".")
        member = split_path[-5]
        if "CMIP6-historical" in cdf_path.stem:
            exp = "cmip_historical"
        elif "HIST." in cdf_path.stem:
            exp = "historical"
        else:
            exp = exp2gauss_exp[split_path[-6]]

        x = xr.open_dataset(cdf_path, use_cftime=True)[gauss_var]
        if "historical" in exp:
            # Slice to 1850-2014
            x = x.sel(time=slice("1850", "2014"))
        else:
            # Slice to 2015-2099
            x = x.sel(time=slice("2015", "2099"))
        if var == "pr":
            # Convert from m/s to kg m-2 s-1 (i.e. mm / s)
            x = x * 1000  
            # Convert from mm / s to mm / day
            x = x * 86400
        elif var == "p-e":
            # Convert from kg/m2/s to mm/day
            x = x * 86400
        # Drop the height coordinate if it exists
        if 'height' in x.coords:
            x = x.drop_vars('height')
        try:
            # Convert time to cftime datetime no leap
            x['time'] = x['time'].values.astype("datetime64[s]").astype("datetime64[ns]")
            exp2member2data[exp][member].append(x)
            exp2member2paths[exp][member].append(cdf_path)
        except Exception as e:
            print(f"Error processing {cdf_path}: {e}")

    # For each member, concatenate the xarrays across time using combine_by_coords
    exp2member2data_combined = defaultdict(lambda: defaultdict(list))
    print("Combining GAUSS data temporally and performing yearly agg...")
    for exp, member2data in tqdm(exp2member2data.items(), total=len(exp2member2data)):
        for member, data in member2data.items():
            exp2member2data_combined[exp][member] = xr.combine_by_coords(data, combine_attrs='drop_conflicts')
            # Assert data is continuous in the time dimension (by 1 month)
            assert np.all(np.diff(exp2member2data_combined[exp][member].time.values.astype("datetime64[M]")) == np.timedelta64(1, 'M'))
            # Compute yearly agg and slice data between 2021 and 2100
            resampled_data = exp2member2data_combined[exp][member].resample(time="1YE")
            if var == "tasmin":
                exp2member2data_combined[exp][member] = resampled_data.min()
            elif var == "tasmax":
                exp2member2data_combined[exp][member] = resampled_data.max()
            elif "above" in var or "below" in var:
                exp2member2data_combined[exp][member] = resampled_data.sum()
            else:
                exp2member2data_combined[exp][member] = resampled_data.mean()
            # Set time to years instead of dates
            exp2member2data_combined[exp][member]['time'] = [date.year for date in pd.DatetimeIndex(exp2member2data_combined[exp][member]['time'].values)]

    # Compute natural variability
    baseline = exp2member2data_combined['baseline']
    member_data = [data for data in baseline.values()]
    member_data = xr.concat(member_data, dim="member")
    # Compute std over time and ensemble members after taking lat-weighted global mean
    weights = np.cos(np.deg2rad(member_data.lat))
    global_mean = member_data[gauss_var].weighted(weights).mean(dim=("lat", "lon"))
    global_mean = global_mean.sel(time=slice("2025", "2099"))
    # Detrend before taking std
    trend = global_mean.polyfit(dim='time', deg=1)
    fitted_data = xr.polyval(global_mean['time'], trend.polyfit_coefficients)
    detrended_data = global_mean - fitted_data
    std = np.std(detrended_data)
    path = output_dir / "natural_variability.npy"
    print("Writing natural variability to", path)
    np.save(path, std.item())

    # Compute regional natural variability
    regional = member_data.sel(time=slice("2025", "2099"))
    decades = list(zip(range(2041, 2092, 10), range(2050, 2101, 10)))

    # Compute per-pixel trend and detrend efficiently
    trend = regional[gauss_var].polyfit(dim="time", deg=1)
    fitted_data = xr.polyval(regional["time"], trend.polyfit_coefficients)
    detrended_data = regional[gauss_var] - fitted_data

    # Compute standard deviation from 2010 to 2030
    regional_variability = detrended_data.sel(time=slice("2010", "2030")).std(dim=("time", "member"))
    # Convert back to dataset
    regional_variability = regional_variability.to_dataset(name=var)

    path = output_dir / "regional_natural_variability.nc"
    print(f"Writing regional natural variability to {path}")
    regional_variability.to_netcdf(path)

    # Concatenate exp to baseline data along the time dimension
    for member, data in exp2member2data_combined['historical'].items():
        exp2member2data_combined['baseline'][member] = xr.concat([data.sel(time=slice(None, "2014")), exp2member2data_combined['baseline'][member]], dim="time")
    del exp2member2data_combined['historical']

    print("Regridding ARISE data and aggregating over members...")
    for exp, member2data_combined in tqdm(exp2member2data_combined.items(), total=len(exp2member2data_combined)):
        output_gauss_data, members = [], []
        # Regrid to ensure centered lats
        try:
            x = member2data_combined[list(member2data_combined.keys())[0]]
            regridder = xe.Regridder(x, common_grid, 'bilinear', periodic=True)
        except:
            x['lat'] = np.linspace(-90, 90, len(x.lat))
            x['lon'] = np.linspace(0, 360, len(x.lon))
            regridder = xe.Regridder(x, common_grid, 'bilinear', periodic=True, ignore_degenerate=True)
        for member, data in member2data_combined.items():
            regridded_data = regridder(data)
            regridded_data = regridded_data.assign_coords(member=member)
            output_gauss_data.append(regridded_data)
            members.append(member)
        output_gauss_data = xr.concat(output_gauss_data, dim="member")
        output_gauss_data['member'] = members
        if exp == "cmip_historical":
            output_gauss_data = output_gauss_data.mean(dim="member")
        else:
            output_gauss_data = output_gauss_data.mean(dim="member")
        # Set xarray coordinate ssp to ssp245
        output_gauss_data = output_gauss_data.expand_dims({"ssp": 1})
        output_gauss_data = output_gauss_data.assign_coords(ssp=('ssp', ["ssp245"]))
        # Set xarray coordinate model to CESM2-WACCM
        output_gauss_data = output_gauss_data.expand_dims({"model": 1})
        output_gauss_data = output_gauss_data.assign_coords(model=('model', ["CESM2-WACCM"]))
        # Rename variable to match other data
        output_gauss_data = output_gauss_data.rename_vars({gauss_var: var})
        output_gauss_path = output_dir / f"output_gauss-{exp}.nc"
        output_gauss_data = output_gauss_data.assign_coords(lon=(output_gauss_data.lon % 360))
        if var == "p-e":
            output_gauss_data_pr = xr.open_dataset(output_dir.parent / "pr" / f"output_gauss-{exp}.nc")["pr"]
            output_gauss_data = output_gauss_data_pr - output_gauss_data

        print(f"Saving {output_gauss_path}")
        output_gauss_data.to_netcdf(output_gauss_path)


if __name__ == "__main__":
    fire.Fire(process_monthly)
