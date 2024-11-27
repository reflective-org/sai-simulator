import fire
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def day_to_month(var, data_dir, ignore_existing=False):

    var2gauss_var = {
        "tas": "TREFHT",
        "pr": "PRECT",
    }
    data_dir = Path(data_dir)

    if var not in var2gauss_var:
        raise ValueError(f"var must be one of {list(var2gauss_var.keys())}")

    var = var2gauss_var[var]

    cdf_paths = {
        "daily_historical": list(data_dir.glob(f"*CMIP6-historical*h1.{var}.*nc")),
        "daily_baseline": list(data_dir.glob(f"*1deg.001*h1.{var}.*nc")),
        "daily_1.5": list(data_dir.glob(f"*SSP245-MA-GAUSS-DEFAULT*h1*.{var}.*nc")),
        "daily_1.0": list(data_dir.glob(f"*SSP245-MA-GAUSS-LOWER-0.5*h1*.{var}.*nc")),
        "daily_0.5": list(data_dir.glob(f"*SSP245-MA-GAUSS-LOWER-1.0*h1*.{var}.*nc")),
    }

    key2path = {
        "daily_historical": "b.e21.BWHIST.f09_g17.CMIP6-historical-WACCM",
        "daily_baseline": "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg",
        "daily_1.5": "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-DEFAULT",
        "daily_1.0": "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-LOWER-0.5",
        "daily_0.5": "b.e21.BWSSP245.f09_g17.release-cesm2.1.3.WACCM-MA-1deg.SSP245-MA-GAUSS-LOWER-1.0",
    }

    for key, paths in cdf_paths.items():
        if len(paths) == 0:
            print(f"No data found for {key}")
            continue

        print(f"Processing {key} data")

        assert "daily" in key
        member2paths = defaultdict(list)
        for path in paths:
            if "historical" in key:
                index = 5
            elif "baseline" in key:
                index = 8
                if "50101" not in str(path):
                    continue
            elif "DEFAULT" in key2path[key]:
                index = 9
            else:
                index = 10
            member = path.name.split(".")[index]
            member2paths[member].append(path)

        members = list(member2paths.keys())
        # Load data if it exists
        for member in tqdm(members):
            paths = member2paths[member]

            cdfs = []
            for path in tqdm(paths):
                x = xr.open_dataset(path)
                # Drop lev and ilev dimensions
                try:
                    x = x.drop_dims(['lev', 'ilev'])
                except:
                    pass
                x = x[var]
                x['member'] = member
                cdfs.append(x)

            ds = xr.concat(cdfs, dim='time')
            ds = ds.sortby('time')

            if "historical" in key:
                ds = ds.sel(time=slice(None, '2014'))

            # Convert daily data to monthly data
            # For tas: max temp, min temp, num days above 40, num days above 35, num days below 0
            # For pr: num days above 10, num days above 20
            path = key2path[key]

            try:
                time1, time2 = ds.time[0].dt.strftime("%Y%m").values, ds.time[-1].dt.strftime("%Y%m").values
            except:
                time1, time2 = paths[0].name.split(".")[-2].split("-")[-2], paths[0].name.split(".")[-2].split("-")[-1]
                days = xr.cftime_range(start=time1+"01", end=time2+"31", freq="D", calendar="noleap")
                # Slice time dimension by length of days, then assign days to time dimension
                ds = ds.isel(time=slice(len(days)))
                ds['time'] = days

            if var == "TREFHT":
                # Find all time indices where the mean is 0 and remove them from ds
                ds = ds.where(ds.mean(dim=('lat', 'lon')) != 0, drop=True)
                # Compute maximum temperature per month
                max_var = var + "MX"
                max_path = data_dir / f"{path}.{member}.cam.h0.{max_var}.{time1}_{time2}.nc"
                if max_path.exists() and not ignore_existing:
                    print(f"Max path {max_path} exists, skipping")
                else:
                    max_monthly = ds.resample(time="1MS").max()
                    max_monthly = max_monthly.rename(var + "MX")
                    print(f"Saving max monthly data to {max_path}")
                    max_monthly.to_netcdf(max_path)

                # Compute minimum temperature per month
                min_var =   var + "MN"
                min_path = data_dir / f"{path}.{member}.cam.h0.{min_var}.{time1}_{time2}.nc"
                if min_path.exists() and not ignore_existing:
                    print(f"Min path {min_path} exists, skipping")
                else:
                    min_monthly = ds.resample(time="1MS").min()
                    min_monthly = min_monthly.rename(var + "MN")
                    print(f"Saving min monthly data to {min_path}")
                    min_monthly.to_netcdf(min_path)

                # Count number of days above 40C
                num_days_var = var + "_above_40"
                num_days_path = data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc"
                if num_days_path.exists() and not ignore_existing:
                    print(f"Num days above 40 path {num_days_path} exists, skipping")
                else:
                    num_days_above_40 = (ds > (40+273.15)).resample(time="1MS").sum()
                    num_days_above_40 = num_days_above_40.rename(var + "_above_40")
                    print(f"Saving num days above 40 data to {num_days_path}")
                    num_days_above_40.to_netcdf(data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc")

                # Count number of days above 35C
                num_days_var = var + "_above_35"
                num_days_path = data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc"
                if num_days_path.exists() and not ignore_existing:
                    print(f"Num days above 35 path {num_days_path} exists, skipping")
                else:
                    num_days_above_35 = (ds > (35+273.15)).resample(time="1MS").sum()
                    num_days_above_35 = num_days_above_35.rename(var + "_above_35")
                    print(f"Saving num days above 35 data to {num_days_path}")
                    num_days_above_35.to_netcdf(data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc")

                # Count number of days below 0C
                num_days_var = var + "_below_0"
                num_days_path = data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc"
                if num_days_path.exists() and not ignore_existing:
                    print(f"Num days below 0 path {num_days_path} exists, skipping")
                else:
                    num_days_below_0 = (ds < (0+273.15)).resample(time="1MS").sum()
                    num_days_below_0 = num_days_below_0.rename(var + "_below_0")
                    print(f"Saving num days below 0 data to {num_days_path}")
                    num_days_below_0.to_netcdf(data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc")

            elif var == "PRECT":
                # Count number of days above 10mm
                num_days_var = var + "_above_10"
                num_days_path = data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc"
                if num_days_path.exists() and not ignore_existing:
                    print(f"Num days above 10 path {num_days_path} exists, skipping")
                else:
                    num_days_precip_above_10 = (ds * 1000 * 86400 > 10).resample(time="1MS").sum()
                    num_days_precip_above_10 = num_days_precip_above_10.rename(var + "_above_10")
                    print(f"Saving num days above 10 data to {num_days_path}")
                    num_days_precip_above_10.to_netcdf(data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc")

                # Count number of days above 20mm
                num_days_var = var + "_above_20"
                num_days_path = data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc"
                if num_days_path.exists() and not ignore_existing:
                    print(f"Num days above 20 path {num_days_path} exists, skipping")
                else:
                    num_days_precip_above_20 = (ds * 1000 * 86400 > 20).resample(time="1MS").sum()
                    num_days_precip_above_20 = num_days_precip_above_20.rename(var + "_above_20")
                    print(f"Saving num days above 20 data to {num_days_path}")
                    num_days_precip_above_20.to_netcdf(data_dir / f"{path}.{member}.cam.h0.{num_days_var}.{time1}_{time2}.nc")


if __name__ == "__main__":
    fire.Fire(day_to_month)
