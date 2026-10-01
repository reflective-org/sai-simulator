"""Grid-level significance of the projected change against the current-climate reference.

The question each cell answers (review doc, "P-values"): **is this decade's mean distinguishable
from the 1995-2014 reference mean, given internal variability?** It is a Welch two-sample
t-test of the two means, each with its own noise:

    Δ   = mean over the decade window of the forced field − the reference mean
    v_d = noise variance of a k-year mean     (decade side, :data:`DecadeNoise`)
    v_r = noise variance of the reference mean (reference side, :data:`ReferenceNoise`)
    t   = Δ / sqrt(v_d + v_r),  df = (v_d + v_r)² / (v_d²/dof_d + v_r²/dof_r)  (Welch-Satterthwaite)
    p   = 2 · P(T_df > |t|)

The two noise terms come from interchangeable sources, so a different estimate of internal
variability can replace one without touching the test:

* **decade** — default :func:`member_decade_noise`: the emulated decade is a smooth forced
  response with no samples of its own, so its noise is that of one realization,
  ``c_ρ · σ² / k`` with ``σ`` pooled over models (``sqrt(mean_m σ²_m)`` from each model's
  ``grid_level_model_internal_variability_<model>.nc``, 2025-2039 residuals) and
  ``c_ρ = (1+ρ)/(1−ρ)`` from the per-cell lag-1 autocorrelation; ``dof`` = the samples behind
  σ. A variable-specific source (e.g. a stochastic emulator's members) is registered in
  :data:`DECADE_NOISE_SOURCES` and used instead.
* **reference** — default :func:`member_reference_noise`: the reference is the multi-model
  mean of each model's ensemble-mean historical, so its noise is measured from the members
  themselves over 1995-2014 (``grid_level_reference_variability_<model>.nc``):
  ``(1/M²) · Σ_m c_ρ,m · s²_m / (n_years · N_members,m)``.

Cells with zero total noise keep the rule the dashboard always had (decision): p = 1 where
the field is unchanged (Δ = 0 — e.g. no population, or a threshold never crossed), p = 0 where
it changed against zero noise. NaN only where Δ or the noise is undefined.

Multiple comparisons: **no correction is applied** (decision, review doc P4) — the returned
values are raw per-cell p-values and ``SIGNIFICANCE_THRESHOLD`` (0.05) is applied per cell,
so about 5% of truly unchanged cells will be marked significant by chance.
:func:`benjamini_hochberg` is provided (and tested) for anyone who wants the FDR-adjusted
alternative via ``fdr=True``; the review doc quantifies its effect.

Exposure variables are tested in exposure units: every σ is scaled by the population grid
exactly as the fields are.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
from scipy.stats import t as student_t

from .constants import (
    DISPLAY_BASELINE_END,
    DISPLAY_BASELINE_START,
    exposurevar2var,
    regional_models_for_var,
)
from .utils import clip_to_land, clip_to_ocean
from .population import get_population_xr

# Decade windows the dashboard offers, as inclusive (start, end) years.
DECADES = list(zip(range(2041, 2092, 10), range(2050, 2101, 10)))

# Maximum |ρ| used in the effective-sample-size factor; keeps c_ρ finite for degenerate cells.
RHO_CLIP = 0.9


def _artifact_dir(var: str, data_dir: Path) -> tuple[str, Path]:
    daily_var = exposurevar2var[var] if "exposure" in var else var
    return daily_var, Path(data_dir) / daily_var


def _population_scale(var: str, data_dir: Path, shape: tuple[int, int]) -> np.ndarray | float:
    """Population grid (millions) for exposure variables, as plain values on the σ grid."""
    if "exposure" not in var:
        return 1.0
    population = get_population_xr(data_dir)
    values = population.transpose("lat", "lon").values if set(population.dims) == {"lat", "lon"} else population.values
    if values.shape != shape:
        raise ValueError(f"population grid {values.shape} does not match sigma grid {shape}")
    return values / 1.0e6


def _models(var: str, models: list[str] | None) -> list[str]:
    return list(models) if models else regional_models_for_var(var)


def _load_artifact(var: str, data_dir: Path, stem: str, model: str) -> xr.DataArray:
    _, art_dir = _artifact_dir(var, data_dir)
    path = art_dir / f"{stem}_{model}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run `python scripts/variability.py --var_name "
            f"{art_dir.name} --data_dir {data_dir} --model_dir <model_dir>`."
        )
    with xr.open_dataarray(path) as da:
        return da.load()


def load_pooled_sigma(var: str, data_dir: Path, models: list[str] | None = None) -> np.ndarray:
    """``sqrt(mean_m σ²_m)`` over ``models`` from the per-model grid-σ artifacts, in the field's units.

    ``models`` defaults to the variable's roster (:func:`backend.constants.regional_models_for_var`).
    """
    sigmas = [
        _load_artifact(var, data_dir, "grid_level_model_internal_variability", m).values
        for m in _models(var, models)
    ]
    pooled = np.sqrt(np.nanmean(np.square(np.stack(sigmas)), axis=0))
    return pooled * _population_scale(var, data_dir, pooled.shape)


def load_sigma_dof(var: str, data_dir: Path, models: list[str] | None = None) -> float:
    """Degrees of freedom behind the pooled σ: the sum over models of each artifact's ``dof``.

    Artifacts written before the attribute existed count as infinitely many (σ treated as
    known, the earlier behaviour).
    """
    total = 0.0
    for m in _models(var, models):
        dof = _load_artifact(var, data_dir, "grid_level_model_internal_variability", m).attrs.get("dof")
        if dof is None:
            return np.inf
        total += float(dof)
    return total


def _clip_rho(rho: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(rho, nan=0.0), -RHO_CLIP, RHO_CLIP)


def load_mean_rho(var: str, data_dir: Path, models: list[str] | None = None) -> np.ndarray:
    """Per-cell lag-1 autocorrelation averaged over ``models`` (default: the variable's roster)."""
    rhos = [_load_artifact(var, data_dir, "grid_level_lag1_autocorr", m).values for m in _models(var, models)]
    return _clip_rho(np.nanmean(np.stack(rhos), axis=0))


def _correlation_factor(rho: np.ndarray | float) -> np.ndarray:
    rho = np.clip(np.asarray(rho, dtype=float), -RHO_CLIP, RHO_CLIP)
    return (1.0 + rho) / (1.0 - rho)


@dataclass(frozen=True)
class MeanNoise:
    """Noise of one side's mean, per cell: its variance and the dof of that variance estimate."""

    variance: np.ndarray
    dof: np.ndarray | float


# (var, data_dir, k) -> noise of a k-year mean of one realization
DecadeNoise = Callable[[str, Path, int], MeanNoise]
# (var, data_dir) -> noise of the reference-period mean
ReferenceNoise = Callable[[str, Path], MeanNoise]


def member_decade_noise(var: str, data_dir: Path, k: int, models: list[str] | None = None) -> MeanNoise:
    """One realization's k-year-mean noise from the models' members: ``c_ρ · σ²_pooled / k``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sigma = load_pooled_sigma(var, data_dir, models)
    c = _correlation_factor(load_mean_rho(var, data_dir, models))
    return MeanNoise(variance=c * sigma**2 / k, dof=load_sigma_dof(var, data_dir, models))


def member_reference_noise(var: str, data_dir: Path, models: list[str] | None = None) -> MeanNoise:
    """Noise of the multi-model reference mean, measured from the members over 1995-2014.

    Model ``m``'s ensemble-mean historical averaged over ``n`` years has variance
    ``c_ρ,m · s²_m / (n · N_m)``; the multi-model mean averages ``M`` independent models,
    ``(1/M²) Σ_m``. Its dof combines the per-model estimates (Welch-Satterthwaite).
    """
    names = _models(var, models)
    terms, dofs = [], []
    for m in names:
        scatter = _load_artifact(var, data_dir, "grid_level_reference_variability", m)
        s = scatter.values * _population_scale(var, data_dir, scatter.shape)
        rho = _clip_rho(_load_artifact(var, data_dir, "grid_level_lag1_autocorr", m).values)
        n_years, n_members = int(scatter.attrs["n_years"]), int(scatter.attrs["n_members"])
        terms.append(_correlation_factor(rho) * s**2 / (n_years * n_members) / len(names) ** 2)
        dofs.append(float(scatter.attrs["dof"]))
    terms = np.stack(terms)
    variance = np.nansum(terms, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dof = variance**2 / np.nansum(terms**2 / np.asarray(dofs)[:, None, None], axis=0)
    return MeanNoise(variance=variance, dof=dof)


# Variable-specific decade-noise sources that replace :func:`member_decade_noise`, e.g. a
# stochastic emulator's members for the variables it covers. Empty by default.
DECADE_NOISE_SOURCES: dict[str, DecadeNoise] = {}


def decade_noise_for(var: str, models: list[str] | None = None) -> DecadeNoise:
    """The decade-noise source for ``var``: a registered one, else the members' σ."""
    if var in DECADE_NOISE_SOURCES:
        return DECADE_NOISE_SOURCES[var]
    return lambda v, data_dir, k: member_decade_noise(v, data_dir, k, models)


def welch_pvalues(delta: np.ndarray, decade: MeanNoise, reference: MeanNoise) -> np.ndarray:
    """Two-sided Welch p for a difference of means ``delta`` given each side's noise.

    Zero total noise follows the dashboard's long-standing rule: ``p = 1`` if ``delta == 0``,
    ``p = 0`` otherwise. Non-finite ``delta`` or noise gives NaN. With an infinite dof on both
    sides the t-distribution is the normal, so a known-σ z-test is the special case
    ``reference=MeanNoise(0, inf)``, ``decade.dof=inf``.
    """
    delta = np.asarray(delta, dtype=float)
    v_d = np.broadcast_to(np.asarray(decade.variance, dtype=float), delta.shape)
    v_r = np.broadcast_to(np.asarray(reference.variance, dtype=float), delta.shape)
    d_d = np.broadcast_to(np.asarray(decade.dof, dtype=float), delta.shape)
    d_r = np.broadcast_to(np.asarray(reference.dof, dtype=float), delta.shape)
    v = v_d + v_r
    defined = np.isfinite(delta) & np.isfinite(v)
    positive = defined & (v > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(v_d > 0, v_d**2 / d_d, 0.0) + np.where(v_r > 0, v_r**2 / d_r, 0.0)
        df = np.where(denom > 0, v**2 / denom, np.inf)
        stat = np.where(positive, delta / np.sqrt(np.where(positive, v, 1.0)), 0.0)
    p = 2.0 * student_t.sf(np.abs(stat), np.where(positive, df, np.inf))
    zero_noise = defined & ~positive
    p = np.where(zero_noise, np.where(delta == 0, 1.0, 0.0), p)
    return np.where(defined, p, np.nan)


def detectability_pvalues(
    delta: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray | float,
    k: int,
) -> np.ndarray:
    """Known-σ special case of :func:`welch_pvalues`: decade noise only, normal distribution.

    ``SE = sigma * sqrt(((1 + rho) / (1 - rho)) / k)``; zero-noise cells follow the same rule.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sigma = np.asarray(sigma, dtype=float)
    decade = MeanNoise(variance=_correlation_factor(rho) * sigma**2 / k, dof=np.inf)
    return welch_pvalues(delta, decade, MeanNoise(variance=0.0, dof=np.inf))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """FDR-adjusted p-values (q-values) over the finite entries of ``p``; NaNs stay NaN.

    ``q_(i) = min_{j >= i} m · p_(j) / j`` for the sorted finite p-values, clipped to 1.
    Thresholding the result at ``q`` controls the expected false-discovery rate at ``q``.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    flat = p.ravel()
    valid = np.isfinite(flat)
    m = int(valid.sum())
    if m == 0:
        return out
    pv = flat[valid]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    out_flat = out.ravel()
    out_flat[valid] = adjusted
    return out_flat.reshape(p.shape)


def _clip_to_domain(var: str, data_dir: Path, field: xr.DataArray) -> xr.DataArray:
    """Ocean north of 60°N for icefrac, land for everything else — the cells the dashboard shows."""
    if var == "icefrac":
        field = clip_to_ocean(data_dir, field)
        return field.where(field.lat >= 60, np.nan)
    return clip_to_land(data_dir, field)


def get_grid_level_p_values(
    var: str,
    data_dir: Path,
    reference: xr.DataArray,
    comparison: xr.DataArray,
    models: list[str] | None = None,
    fdr: bool = False,
    decade_noise: DecadeNoise | None = None,
    reference_noise: ReferenceNoise | None = None,
) -> xr.DataArray:
    """Decadal grid-level significance of ``comparison`` against the ``reference`` field.

    For each window in :data:`DECADES`: ``Δ`` is the window mean of ``comparison`` minus
    ``reference``, tested with :func:`welch_pvalues`. The decade side's noise comes from
    ``decade_noise`` (default :func:`decade_noise_for`, i.e. a registered source or the
    members' σ of ``models``) with ``k`` = the years the window actually contains; the
    reference side's from ``reference_noise`` (default :func:`member_reference_noise`).
    ``models`` defaults to the variable's roster. The map is clipped to the dashboard's
    domain; with ``fdr=True`` (off by default, see the module docstring) it is then adjusted
    with :func:`benjamini_hochberg` over its valid cells. Returned stamped with the window's
    end year on a ``time`` dim, exactly like before.
    """
    decade_noise = decade_noise or decade_noise_for(var, models)
    reference_noise = reference_noise or (lambda v, d: member_reference_noise(v, d, models))
    ref_noise = reference_noise(var, data_dir)
    reference_np = reference.values

    per_decade = []
    for start_year, end_year in DECADES:
        window = comparison.sel(time=slice(start_year, end_year))
        k = int(window.sizes["time"])
        if k == 0:
            # No years of ``comparison`` fall in this window: emit an explicit not-testable
            # map rather than dropping the decade, so callers can still select it.
            p = np.full(reference_np.shape, np.nan)
        else:
            delta = window.mean(dim="time").values - reference_np
            p = welch_pvalues(delta, decade_noise(var, data_dir, k), ref_noise)
        p_da = xr.DataArray(p, dims=("lat", "lon"), coords={"lat": reference.lat, "lon": reference.lon})
        p_da = _clip_to_domain(var, data_dir, p_da)
        if fdr:
            p_da = p_da.copy(data=benjamini_hochberg(p_da.values))
        per_decade.append(p_da.expand_dims(time=[end_year]))

    return xr.concat(per_decade, dim="time").sortby("time")


def _significance_reference_tag() -> str:
    """Filename tag naming the period p-values are tested against (the display baseline)."""
    return f"ref{DISPLAY_BASELINE_START}_{DISPLAY_BASELINE_END}"


def no_sai_p_values_cache_path(cache_dir: Path, simple_ssp: str, var: str) -> Path:
    """Cache path for the no-SAI grid-level p-values of ``var`` under ``simple_ssp``.

    The significance reference period is part of the filename, so p-values computed against a
    different baseline are never served from (or overwrite) a stale file.
    """
    return cache_dir / f"{simple_ssp}_regional_no_sai_p_values_{var}_{_significance_reference_tag()}.nc"


def sai_p_values_cache_path(cache_dir: Path, simple_ssp: str, temp_target: float, var: str) -> Path:
    """Cache path for the with-SAI grid-level p-values; see :func:`no_sai_p_values_cache_path`."""
    return (
        cache_dir
        / f"{simple_ssp}_{temp_target:.1f}_regional_sai_p_values_{var}_{_significance_reference_tag()}.nc"
    )
