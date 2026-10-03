"""Open the simulator-inputs Zarr stores on the public R2 bucket with a uniform time axis.

Usage:
    from simstores import open_store, STORES
    ds = open_store("UKESM1-1-LL", "G6-1.5K-SAI")            # Mon/atmos_2d
    mx = open_store("CESM2-WACCM-MA", "baseline", "tasminmax")

What it fixes on read:
  * CESM stamps each month with the first day of the NEXT month (Jan 2015 is
    labelled 2015-02-01). We relabel so every model's stamp is the month itself.
  * Calendars differ (CESM/MIROC mostly proleptic_gregorian, MIROC G6-1.5K noleap,
    UKESM 360_day). For monthly data the day is meaningless, so time is rebuilt
    as numpy datetime64 first-of-month — stores from different models then align
    on `time` directly and `.dt.year` / `.dt.month` / resample just work.
  * CESM stores carry CAM housekeeping variables (hyam, gw, co2vmr, date, ...);
    only the (member, time, lat, lon) science fields are kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

BASE = "https://pub-d127229e707d485c80965ab41706e316.r2.dev/simulator-inputs"

STORES = {
    "CESM2-WACCM-MA": ["historical", "baseline", "0.5K-SAI", "1.0K-SAI", "1.5K-SAI"],
    "MIROC-ES2H": ["historical", "baseline", "ssp245", "G6-0.5K-SAI", "G6-1.5K-SAI"],
    "UKESM1-1-LL": ["historical", "baseline", "G6-1.5K-SAI"],
}

GROUPS = {
    "atmos_2d": "Mon/atmos_2d",              # TREFHT, PRECT, QFLX, ICEFRAC
    "tasminmax": "Mon/atmos_2d_tasminmax",   # TREFHTMN, TREFHTMX
    "derived": "Mon/atmos_2d_derived",       # TREFHT_above_35/40, TREFHT_below_0, PRECT_above_10/20
}


def store_url(model: str, experiment: str) -> str:
    if experiment not in STORES.get(model, []):
        raise ValueError(f"No store {model}_{experiment}; available: {STORES.get(model)}")
    return f"{BASE}/{model}_{experiment}.zarr"


def open_store(model: str, experiment: str, group: str = "atmos_2d",
               raw: bool = False) -> xr.Dataset:
    """Lazily open one store/group. raw=True skips all normalization."""
    ds = xr.open_zarr(store_url(model, experiment), group=GROUPS.get(group, group),
                      consolidated=True, use_cftime=True)
    if raw:
        return ds
    ds = ds[[v for v in ds.data_vars if ds[v].dims == ("member", "time", "lat", "lon")]]
    year = ds.time.dt.year.values
    month = ds.time.dt.month.values
    if model.startswith("CESM"):  # end-of-month stamping -> step back one month
        month = month - 1
        year = np.where(month == 0, year - 1, year)
        month = np.where(month == 0, 12, month)
    ds = ds.assign_coords(time=pd.to_datetime({"year": year, "month": month, "day": 1}).values)
    ds.attrs["time_normalized"] = "first-of-month datetime64; CESM shifted back one month"
    return ds
