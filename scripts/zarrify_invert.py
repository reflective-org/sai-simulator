#!/usr/bin/env python3
"""Pack the four InVERT netCDFs into one Zarr store for R2.

The InVERT files (``TREFHT``, ``TREFHTMN``, ``TREFHTMX``, ``PRECT``; 100 members x 600
months of detrended anomalies on a 96 x 192 grid, ~4.4 GB float32 each) are what
:mod:`backend.invert_variability` turns into the dashboard's variability envelope. They
used to exist only as local copies under ``<data_dir>/invert/``. This script writes them,
unchanged, as the variables of one zarr-v3 store,
``simulator-inputs/InVERT_50yrs_SSP370_100members.zarr``, which the backend then reads
anonymously through :data:`version.ZARR_STORES_URI`.

The store is a byte-faithful copy: same variable names, dims, coordinates (including the
``month`` coord on ``time``) and float32 values, raw units (PRECT stays m/s). Each
variable records the netCDF it came from (name, size, SHA-256) so a cache built from the
store can be traced back to the file. Chunks are one member each, matching how
``build_annual_ensemble`` reads (``chunks={"ensemble": 1}``).

The store is written locally, one variable per run (so only one 4.4 GB netCDF needs to be
on disk at a time), and uploaded with rclone, which already holds the R2 credentials::

    python scripts/zarrify_invert.py write --src InVERT_TREFHT_50yrs_SSP370_100members.nc --out InVERT.zarr
    rclone copy InVERT.zarr r2:reflective-data-store/simulator-inputs/InVERT_50yrs_SSP370_100members.zarr
    # ...repeat for the other three files; then, against the public store:
    python scripts/zarrify_invert.py verify --src InVERT_TREFHT_50yrs_SSP370_100members.nc \\
        --store https://<public r2.dev host>/simulator-inputs/InVERT_50yrs_SSP370_100members.zarr

After a variable is uploaded its local chunks (``<out>/<VAR>/c/``) can be deleted; later
``write`` runs only need the store's metadata and coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

STORE_NAME = "InVERT_50yrs_SSP370_100members.zarr"
DIMS = ("ensemble", "time", "lat", "lon")
SIZES = {"ensemble": 100, "time": 600, "lat": 96, "lon": 192}
CHUNKS = {"ensemble": 1, "time": 600, "lat": 96, "lon": 192}

# Store attrs per variable. Values are detrended anomalies, so these are deliberately not
# the model-store ``VAR_ATTRS`` (those describe absolute monthly fields).
VAR_ATTRS = {
    "TREFHT": {"units": "K", "long_name": "InVERT 2 m air temperature anomaly, monthly"},
    "TREFHTMN": {"units": "K", "long_name": "InVERT minimum 2 m air temperature anomaly, monthly"},
    "TREFHTMX": {"units": "K", "long_name": "InVERT maximum 2 m air temperature anomaly, monthly"},
    "PRECT": {"units": "m/s", "long_name": "InVERT total precipitation rate anomaly, monthly"},
}
STORE_ATTRS = {
    "title": "InVERT internal-variability ensembles (100 members x 50 years, monthly)",
    "description": (
        "Detrended monthly internal-variability anomalies, seasonal cycle removed, on a "
        "96 x 192 Gaussian grid. One variable per source netCDF, values unchanged. The "
        "SSP370 label is informational: the forced signal was removed upstream."
    ),
    "consumer": "backend/invert_variability.py (sai-simulator)",
}

log = logging.getLogger("zarrify_invert")


def sha256_of(path: Path, block: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def open_source(src: Path) -> tuple[str, xr.Dataset]:
    """Open one InVERT netCDF and return ``(variable name, dataset)`` after shape checks."""
    ds = xr.open_dataset(src, chunks={"ensemble": 1})
    data_vars = list(ds.data_vars)
    if len(data_vars) != 1 or data_vars[0] not in VAR_ATTRS:
        raise ValueError(f"{src.name}: expected exactly one of {sorted(VAR_ATTRS)}, found {data_vars}")
    var = data_vars[0]
    da = ds[var]
    if da.dims != DIMS or dict(da.sizes) != SIZES:
        raise ValueError(f"{src.name}: {var} has dims {dict(da.sizes)}, expected {SIZES} in order {DIMS}")
    if da.dtype != np.float32:
        raise ValueError(f"{src.name}: {var} is {da.dtype}, expected float32")
    if set(ds.coords) != {"ensemble", "time", "lat", "lon", "month"}:
        raise ValueError(f"{src.name}: unexpected coordinates {sorted(ds.coords)}")
    return var, ds


def write(src: Path, out: Path) -> None:
    """Add the variable in ``src`` to the local store at ``out`` (creating it if needed)."""
    import zarr

    var, ds = open_source(src)
    ds[var].attrs = {
        **VAR_ATTRS[var],
        "source_file": src.name,
        "source_size_bytes": src.stat().st_size,
        "source_sha256": sha256_of(src),
    }
    for name in ds.variables:
        ds[name].encoding = {}
    ds = ds.chunk(CHUNKS)

    if (out / "zarr.json").exists():
        existing = xr.open_zarr(out, consolidated=False)
        if var in existing.data_vars:
            raise SystemExit(f"{out} already holds {var}; delete {out / var} first to rewrite it")
        # Coordinates must agree exactly with what is already in the store, or the
        # variables would silently disagree on the grid or month labels.
        for coord in ("ensemble", "time", "lat", "lon", "month"):
            if not np.array_equal(existing[coord].values, ds[coord].values):
                raise ValueError(f"{src.name}: coordinate {coord!r} differs from the store's")
        log.info("appending %s to %s", var, out)
        ds[[var]].drop_vars(["ensemble", "time", "lat", "lon", "month"]).to_zarr(
            out, mode="a", zarr_format=3, consolidated=False
        )
    else:
        log.info("creating %s with %s", out, var)
        ds.attrs = dict(STORE_ATTRS)
        ds.to_zarr(out, mode="w", zarr_format=3, consolidated=False)
    zarr.consolidate_metadata(str(out))


def verify(src: Path, store: str) -> list[str]:
    """Compare the store's copy of the variable in ``src`` with the netCDF, member by member."""
    var, ds = open_source(src)
    failures = []
    try:
        zs = xr.open_zarr(store, consolidated=True)
    except Exception as exc:  # noqa: BLE001 -- report any open failure as a finding
        return [f"open failed: {type(exc).__name__}: {exc}"]
    if var not in zs.data_vars:
        return [f"{var} missing from store (has {sorted(zs.data_vars)})"]
    zda = zs[var]
    if zda.dims != DIMS or dict(zda.sizes) != SIZES or zda.dtype != np.float32:
        failures.append(f"{var}: store has {dict(zda.sizes)} {zda.dtype}")
    for coord in ("ensemble", "time", "lat", "lon", "month"):
        if not np.array_equal(zs[coord].values, ds[coord].values):
            failures.append(f"{var}: coordinate {coord!r} differs")
    attrs = zda.attrs
    if attrs.get("source_size_bytes") != src.stat().st_size:
        failures.append(f"{var}: source_size_bytes {attrs.get('source_size_bytes')} != {src.stat().st_size}")
    if attrs.get("units") != VAR_ATTRS[var]["units"]:
        failures.append(f"{var}: units {attrs.get('units')!r}")
    if failures:
        return failures
    for member in range(SIZES["ensemble"]):
        a = ds[var].isel(ensemble=member).values
        b = zda.isel(ensemble=member).values
        if not np.array_equal(a, b, equal_nan=True):
            failures.append(f"{var}: member {member} differs (max |diff| {np.nanmax(np.abs(a - b))})")
        if member % 20 == 19:
            log.info("%s: verified members 0-%d", var, member)
    if attrs.get("source_sha256") != sha256_of(src):
        failures.append(f"{var}: source_sha256 does not match {src.name}")
    return failures


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="add one InVERT netCDF's variable to a local store")
    w.add_argument("--src", type=Path, required=True)
    w.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify", help="check a store's copy of one variable against its netCDF")
    v.add_argument("--src", type=Path, required=True)
    v.add_argument("--store", required=True, help="local path or URL of the store")
    args = parser.parse_args(argv)

    if args.cmd == "write":
        write(args.src, args.out)
        return
    failures = verify(args.src, args.store)
    for line in failures:
        print(f"FAIL {line}")
    if failures:
        raise SystemExit(1)
    print(f"OK {args.src.name} matches {args.store}")


if __name__ == "__main__":
    main()
