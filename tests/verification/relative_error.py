#!/usr/bin/env python3
"""
Compute relative error metrics between two datasets.

Supported file formats:
- .tif / .tiff
- .geojson
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

def _load_tiff(path: Path) -> np.ndarray:
    """Load all bands from a GeoTIFF as a float array of shape ``(band, row, col)``."""
    with rasterio.open(path) as src:
        # Returns shape (band, row, col) for multi-band rasters.
        return src.read().astype(float, copy=False)


def _load_geojson(path: Path, columns: list[str] | None = None) -> np.ndarray:
    """Load numeric GeoJSON properties as a 2D float array (optionally subset ``columns``)."""
    gdf = gpd.read_file(path)
    numeric = gdf.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError(f"No numeric properties found in GeoJSON: {path}")
    if columns is not None:
        missing = [col for col in columns if col not in numeric.columns]
        if missing:
            raise ValueError(
                f"Requested GeoJSON columns not found in {path}: {missing}. "
                f"Available numeric columns: {list(numeric.columns)}"
            )
        numeric = numeric[columns]
    return numeric.to_numpy(dtype=float, copy=True)


def load_dataset(path: Path, geojson_columns: list[str] | None = None) -> np.ndarray:
    """Load a ``.tif`` / ``.tiff`` or ``.geojson`` dataset as a float :class:`numpy.ndarray`."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _load_tiff(path)
    if suffix == ".geojson":
        return _load_geojson(path, columns=geojson_columns)

    raise ValueError(
        f"Unsupported file type '{suffix}' for {path}. Use .tif/.tiff or .geojson."
    )


def compute_relative_error(
    baseline: np.ndarray, comparison: np.ndarray, epsilon: float
) -> np.ndarray:
    """Element-wise ``|comparison - baseline| / max(|baseline|, epsilon)``."""
    if baseline.shape != comparison.shape:
        raise ValueError(
            f"Shape mismatch: baseline {baseline.shape} vs comparison {comparison.shape}"
        )

    baseline = baseline.astype(float, copy=False)
    comparison = comparison.astype(float, copy=False)
    denominator = np.maximum(np.abs(baseline), epsilon)
    return np.abs(comparison - baseline) / denominator
