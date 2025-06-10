import xarray as xr
import numpy as np
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=32)
def get_area(data_dir: Path):
    area = xr.open_dataset(data_dir / "area.nc", autoclose=True).AREA
    area = area.sel(time=area.time[-1]).drop('time') * 1e-6 * 1e-6 # Convert to million km^2
    return area

