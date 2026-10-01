"""Uncertainty band for multi-model mean-over-time series.

The dashboard draws one band around the multi-model mean (see
``docs/pvalues-uncertainty-histograms-review.md``, "Uncertainty bands"): the **internal
variability of that mean**, ``±Z · σ_internal(mean)`` with

* ``σ²_internal(mean) = (1/M²) · Σ_m σ²_int,m`` — the mean of ``M`` independent models varies
  less than any one of them; each ``σ_int,m`` is that model's own internal variability
  (``model_internal_variability_<model>.npy`` or the per-region geojson column).

The spread between the models is deliberately NOT in the band (decision 2026-09-28; the
earlier band ``Z · sqrt(σ²_model + σ²_internal(mean))`` is preserved on the branch
``aa-feature/multi-model-mean-uncertainty-both-internal-and-modelspread``). ``σ_model``, the
standard deviation across the per-model series at each time, is still computed and returned
as a diagnostic: with fewer than three models the population estimator (``ddof=0``) is used,
so for two models it is exactly half the gap between them; from three models on, ``ddof=1``.

Half-widths are ``±Z·σ`` with ``Z = 2`` (≈95% for a normal).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

Z = 2.0  # half-width multiplier
MIN_MODELS_FOR_UNBIASED = 3


def model_spread_ddof(n_models: int) -> int:
    """``ddof`` for the across-model std: 0 while ``n_models < 3``, else 1."""
    return 1 if n_models >= MIN_MODELS_FOR_UNBIASED else 0


def internal_sigma_of_mean(sigmas: Sequence[float]) -> float:
    """``sqrt(Σ σ²_m) / M``: internal variability of the equal-weight mean of ``M`` models."""
    arr = np.asarray(list(sigmas), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.nansum(arr**2)) / arr.size)


@dataclass(frozen=True)
class BandResult:
    """Per-time band components for one series (half-widths are ``±Z·σ``)."""

    mean: xr.DataArray
    internal: xr.DataArray  # the drawn band's half-width, Z·σ_internal(mean)
    sigma_model: xr.DataArray  # diagnostic only, not part of the band
    sigma_internal_mean: float
    n_models: int
    ddof: int

    def as_dict(self, prefix: str) -> dict:
        """Flatten into ``mean_over_time``-style keys, e.g. ``band_internal_<prefix>``."""
        return {
            f"band_internal_{prefix}": self.internal,
            f"sigma_model_{prefix}": self.sigma_model,
        }


def combine_series(
    per_model: xr.DataArray,
    sigma_internal: Mapping[str, float] | Sequence[float],
    dim: str = "model",
) -> BandResult:
    """Mean, band half-width and model spread from per-model series and per-model internal σ.

    ``per_model`` has a ``dim`` axis (default ``model``) plus ``time``. ``sigma_internal`` is
    either a mapping ``{model_name: σ}`` (looked up by the ``dim`` coordinate) or a sequence
    in the same order as ``per_model[dim]``. NaNs in a model's series (e.g. a shorter
    historical record) are skipped in the mean and spread at those times.
    """
    if dim not in per_model.dims:
        raise ValueError(f"per_model has no {dim!r} dim; dims={per_model.dims}")
    models = [str(m) for m in per_model[dim].values]
    n_models = len(models)
    if isinstance(sigma_internal, Mapping):
        missing = [m for m in models if m not in sigma_internal]
        if missing:
            raise KeyError(f"no internal sigma for models {missing}")
        sigmas = [float(sigma_internal[m]) for m in models]
    else:
        sigmas = [float(s) for s in sigma_internal]
        if len(sigmas) != n_models:
            raise ValueError(
                f"sigma_internal has {len(sigmas)} entries for {n_models} models"
            )

    ddof = model_spread_ddof(n_models)
    mean = per_model.mean(dim, skipna=True)
    if n_models == 1:
        sigma_model = xr.zeros_like(mean)
    else:
        sigma_model = per_model.std(dim, ddof=ddof, skipna=True).fillna(0.0)
    sigma_int_mean = internal_sigma_of_mean(sigmas)

    internal = (xr.zeros_like(mean) + Z * sigma_int_mean).where(mean.notnull())

    return BandResult(
        mean=mean.rename("mean"),
        internal=internal.rename("band_internal"),
        sigma_model=sigma_model.rename("sigma_model"),
        sigma_internal_mean=sigma_int_mean,
        n_models=n_models,
        ddof=ddof,
    )


def band_label(n_models: int) -> str:
    """Legend text naming what the band is and how many models are in it."""
    return f"±2σ internal variability of the multi-model mean, M={n_models}"
