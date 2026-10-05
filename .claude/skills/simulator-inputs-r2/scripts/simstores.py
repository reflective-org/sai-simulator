"""Open the simulator-inputs Zarr stores on the public R2 bucket with a uniform time axis.

Usage:
    from simstores import open_store, STORES
    ds = open_store("UKESM1-1-LL", "G6-1.5K-SAI")            # Mon/atmos_2d
    mx = open_store("CESM2-WACCM-MA", "baseline", "tasminmax")

What it fixes on read:
  * CESM groups that are raw CAM history output (Mon/atmos_2d in every
    experiment, Mon/atmos_2d_tasminmax in baseline and SAI) stamp each month with
    the first day of the NEXT month (Jan 2015 is labelled 2015-02-01). Those are
    relabelled so every stamp is the month itself. CESM's Mon/atmos_2d_derived and
    historical tasminmax are already stamped correctly and are left alone; the
    rule (see _cam_end_of_interval) is the one scripts/process_monthly_gauss.py uses.
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


def _cam_end_of_interval(ds: xr.Dataset) -> bool:
    """True for raw CAM history stamps: CAM's ``datesec`` present, every stamp the
    1st at 00:00, and the series running Feb-Jan (CESM runs start in January).
    A Jan-Dec series was corrected at the source and is not shifted again; any
    other span raises, as in scripts/process_monthly_gauss.py."""
    if "datesec" not in ds.variables:
        return False
    t = ds.time
    if not bool((t.dt.day == 1).all() and (t.dt.hour == 0).all() and (t.dt.minute == 0).all()):
        return False
    span = (int(t.dt.month[0]), int(t.dt.month[-1]))
    if span == (2, 1):
        return True
    if span == (1, 12):
        return False
    raise ValueError(f"CAM first-of-month stamps run month {span[0]} to {span[1]}; "
                     "cannot tell whether they are end-of-interval")


def open_store(model: str, experiment: str, group: str = "atmos_2d",
               raw: bool = False) -> xr.Dataset:
    """Lazily open one store/group. raw=True skips all normalization."""
    ds = xr.open_zarr(store_url(model, experiment), group=GROUPS.get(group, group),
                      consolidated=True, use_cftime=True)
    if raw:
        return ds
    shift = model.startswith("CESM") and _cam_end_of_interval(ds)
    ds = ds[[v for v in ds.data_vars if ds[v].dims == ("member", "time", "lat", "lon")]]
    year = ds.time.dt.year.values
    month = ds.time.dt.month.values
    if shift:  # end-of-interval stamping -> step back one month
        month = month - 1
        year = np.where(month == 0, year - 1, year)
        month = np.where(month == 0, 12, month)
    ds = ds.assign_coords(time=pd.to_datetime({"year": year, "month": month, "day": 1}).values)
    ds.attrs["time_normalized"] = ("first-of-month datetime64; CAM end-of-interval stamps "
                                   "shifted back one month" if shift else "first-of-month datetime64")
    return ds
