"""Write simplified Natural Earth 110m coastlines as one NaN-separated line list (Plotly overlay).

Run from the repo root:
    python model-comparison/pipeline/coastlines.py model-comparison/data/coastlines.json
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
from cartopy.io import shapereader

# cartopy returns its local copy of the shapefile, downloading it on first use.
SHP = shapereader.natural_earth(resolution="110m", category="physical", name="coastline")

gdf = gpd.read_file(SHP)
lon, lat = [], []
for g in gdf.geometry:
    for line in getattr(g, "geoms", [g]):
        xs, ys = line.coords.xy
        lon += [round(float(x), 2) for x in xs] + [None]
        lat += [round(float(y), 2) for y in ys] + [None]
Path(sys.argv[1]).write_text(json.dumps({"lon": lon, "lat": lat}, separators=(",", ":")))
