"""Solve the SO2 injection that holds the dashboard's global-mean temperature at a target.

The dashboard offers deployments defined by a temperature target and a latitude pattern
(e.g. "Polar", "Northern Hemisphere only"). Those patterns run through the custom-injection
(CIDER) path, which needs an injection, so the target has to be inverted into one. This module
does that inversion on the backend, with the same CIDER parameters the custom-injection path
loads (:data:`backend.variable.CESM_PARAMS_FILE`) and against the same no-SAI line the
dashboard draws, so a solved scenario shows its target on the chart exactly as GAUSS does.

**What is held.** With-SAI global-mean tas = the dashboard's no-SAI global mean (the
cos-latitude mean of the three-model ``tas`` map, :func:`backend.get_outputs.get_temp_diff`)
plus the CIDER global-mean ``tas`` delta of the injection. From ``start_year`` the cooling ramps
linearly over ``ramp_up`` years to ``max(0, no_sai - target)`` and is held there; years whose
no-SAI warming is already below the target get no injection.

**Global-mean CIDER.** In :func:`backend.variable.pattern_from_all_injections` each latitude's
response is a time series times a fixed spatial pattern, so the cos-latitude global mean of the
whole map is ``Σ_i r_i(t) · gm(pattern_i)``: the solver works on that 7-latitude, 1-D form
(:func:`global_mean_tas_delta`), which equals the global mean of the full map path to rounding.
The AOD step is kept monthly, exactly as the map path computes it.

**The inversion** fixes the latitude split (``weights``, normalised to sum to 1) and solves the
yearly total ``M(year)`` as a bounded least-squares problem: the squared miss of the desired
cooling over every year from ``start_year``, plus a small penalty on year-to-year changes in
``M``, with ``0 <= M <= max_total_tg`` (the editor's ceiling). Each year's injection cools that
same year less than the next one (the AOD and the response both build up over months), so an
exact year-by-year inversion alternates and grows; the smoothness penalty is what keeps the
injection a plain ramp-and-follow curve at the cost of a few mK. The AOD decay rate depends on
the injection, so the problem is solved by Gauss-Newton: linearise, solve the bounded linear
problem, repeat (3 to 8 steps, at most ~40 ms).

When the target needs more than the ceiling, the years that cannot reach it are pinned at the
ceiling and taken out of the fit: the line holds the target for as long as the ceiling allows,
then rises with the injection at the ceiling, and the solution comes back with
``reached = False``. Without the pinning, least squares would cool the earlier years below the
target to make up a little of the later shortfall.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io
from scipy.optimize import lsq_linear
from scipy.special import erfc

from .variable import CESM_PARAMS_FILE

N_LATS = 7  # injection rows, 60N -> 60S (backend.constants.INJECTION_LATS)
FIRST_YEAR, LAST_YEAR = 2035, 2100  # injection columns (the API's VI_N_YEARS = 66)
N_YEARS = LAST_YEAR - FIRST_YEAR + 1
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)
MONTHS_PER_YEAR = 12
DEFAULT_MAX_TOTAL_TG = 28.0  # Tg SO2 / yr summed over latitudes; the injection editor's ceiling
HOLD_TOLERANCE_K = 0.05  # largest |with-SAI - target| after the ramp that still counts as reached
# Weight of the year-to-year change penalty, in K^2 per (Tg/yr)^2. Measured over the 8 scenarios,
# targets 0.5-2.0 K, starts 2035 (10-yr ramp) and 2050 (5-yr ramp) and the 6 dashboard latitude
# patterns (384 solves): at 1e-4 the hold error of the reached cases has median 0.008 K and 90th
# percentile 0.014 K, no year is colder than desired by more than 0.017 K, and the injection
# reverses direction by more than 1 Tg/yr 0.2 times per solve on average. At 1e-5 the hold error
# is a little smaller (median 0.006 K) but the injection zig-zags after fast ramps (1.5 reversals).
SMOOTHNESS = 1e-4
# A year at the ceiling still warmer than desired by more than this is pinned there (see the solver).
PIN_TOLERANCE_K = 1e-3
MAX_ITERATIONS = 30  # Gauss-Newton steps; 3 to 8 are needed in practice
STEP_TOLERANCE_TG = 1e-4  # stop once no year's injection moves by more than this
FLOOR_RESOLUTION_K = 0.01  # precision of lowest_reachable_target

N_MONTHS = N_YEARS * MONTHS_PER_YEAR


@dataclass(frozen=True)
class GlobalMeanCider:
    """The tas parts of the CIDER parameter file the solver needs, per injection latitude."""

    param_aod: np.ndarray  # (7, 3) [beta, alpha, gamma]
    param_tas: np.ndarray  # (7, 2) [tau, mu]
    pattern_global_mean: np.ndarray  # (7,) cos-latitude global mean of each latitude's tas pattern
    response_kernel: np.ndarray  # (7, N_MONTHS) climate impulse response at lags 1..N_MONTHS
    # (7, N_YEARS, N_MONTHS): monthly AOD -> yearly global-mean tas, i.e. the convolution with
    # response_kernel, averaged to years and scaled by pattern_global_mean
    yearly_operator: np.ndarray


def _semi_infinite_diffusion_kernel(tau: float, mu: float, n: int) -> np.ndarray:
    """The impulse response of :func:`backend.variable.response_from_1_forcing_vectorized` at
    lags ``t = 1..n`` (same expression and fallback), so the response is a plain convolution."""
    t = np.arange(1, n + 1, dtype=float)
    sqrt_term = np.sqrt(t / tau)
    with np.errstate(over="ignore", invalid="ignore"):
        partial = np.exp(t / tau) * erfc(sqrt_term)
    h = mu * (1.0 / np.sqrt(np.pi * t / tau) - partial)
    fallback = mu * (1.0 / np.sqrt(np.pi * t / tau) - 2.0 / np.sqrt(np.pi) / (sqrt_term + np.sqrt(sqrt_term**2 + 2.0)))
    bad = np.isinf(partial) | np.isnan(partial)
    h[bad] = fallback[bad]
    return h


def _yearly_operator(kernel: np.ndarray, pattern_global_mean: float) -> np.ndarray:
    """``(N_YEARS, N_MONTHS)`` matrix taking a monthly AOD series to the yearly global-mean tas
    change: the causal convolution with ``kernel`` (``np.convolve(aod, kernel)[:N_MONTHS]``),
    averaged over each year's months and scaled by the latitude's global-mean pattern."""
    lag = np.arange(N_MONTHS)[:, None] - np.arange(N_MONTHS)[None, :]  # month k minus source month j
    toeplitz = np.where(lag >= 0, kernel[np.clip(lag, 0, None)], 0.0)
    return toeplitz.reshape(N_YEARS, MONTHS_PER_YEAR, N_MONTHS).mean(axis=1) * pattern_global_mean


def _aod_by_source_year(param_aod: np.ndarray, yearly_tg: np.ndarray) -> np.ndarray:
    """Monthly AOD of :func:`backend.variable.AOD_from_injection_vectorized` for an injection that
    is constant within each year (``yearly_tg / 12`` per month), split by the year it came from:
    ``(N_MONTHS, N_YEARS)``, summing over the columns gives the AOD.

    The reference sums ``q_s β exp(-(α + γ q_s)(k - s + 1))`` over every past month ``s``. Within a
    year ``q_s`` and the decay rate are constant, so each year's 12 terms are a geometric series:
    the same sum in a different order, ``N_YEARS`` terms per month instead of ``N_MONTHS``. Each
    column depends on that year's injection only, which is what makes the Jacobian cheap.
    """
    beta, alpha, gamma = param_aod
    q = np.asarray(yearly_tg, dtype=float) / MONTHS_PER_YEAR  # (Y,) Tg per month
    rate = alpha + gamma * q  # (Y,)
    k = np.arange(N_MONTHS, dtype=float)[:, None]  # (K, 1)
    start = (np.arange(N_YEARS) * MONTHS_PER_YEAR).astype(float)[None, :]  # (1, Y)
    r = rate[None, :]
    one_minus = -np.expm1(-r)  # 1 - e^{-r}
    # within the source year (start <= k < start + 12): lags 1 .. k - start + 1
    within = np.exp(-r) * (-np.expm1(-r * (k - start + 1))) / one_minus
    # after the source year (k >= start + 12): lags k - start - 10 .. k - start + 1
    after = np.exp(-r * (k - start - 10)) * (-np.expm1(-MONTHS_PER_YEAR * r)) / one_minus
    terms = np.where(k < start, 0.0, np.where(k < start + MONTHS_PER_YEAR, within, after))
    return terms * (q * beta)[None, :]


def _aod_from_yearly_injection(param_aod: np.ndarray, yearly_tg: np.ndarray) -> np.ndarray:
    """Monthly AOD (``N_MONTHS``,) of a yearly-constant injection; see :func:`_aod_by_source_year`."""
    return _aod_by_source_year(param_aod, yearly_tg).sum(axis=1)


@lru_cache(maxsize=4)
def load_global_mean_cider(data_dir: str | Path) -> GlobalMeanCider:
    """Read :data:`CESM_PARAMS_FILE` and reduce each latitude's tas pattern to its global mean."""
    data_dir = Path(data_dir)
    mat = scipy.io.loadmat(data_dir / CESM_PARAMS_FILE)
    lat = np.load(data_dir / "correct_lat.npy")
    pattern = np.asarray(mat["pattern_T_all"], dtype=float)  # (lon, lat, 8): 7 latitudes + CO2
    weights = np.cos(np.deg2rad(lat))[np.newaxis, :]  # the weighting of get_temp_diff
    gm = np.array([
        np.nansum(pattern[:, :, i] * weights) / np.sum(np.where(np.isnan(pattern[:, :, i]), 0.0, weights * np.ones_like(pattern[:, :, i])))
        for i in range(N_LATS)
    ])
    param_tas = np.asarray(mat["param_T_all"], dtype=float)[:N_LATS]
    kernels = np.stack([_semi_infinite_diffusion_kernel(tau, mu, N_MONTHS) for tau, mu in param_tas])
    return GlobalMeanCider(
        param_aod=np.asarray(mat["param_AOD_all"], dtype=float)[:N_LATS],
        param_tas=param_tas,
        pattern_global_mean=gm,
        response_kernel=kernels,
        yearly_operator=np.stack([_yearly_operator(kernels[i], gm[i]) for i in range(N_LATS)]),
    )


def global_mean_tas_delta(cider: GlobalMeanCider, injection: np.ndarray) -> np.ndarray:
    """Yearly global-mean tas change (K, negative = cooling) of a ``(7, 66)`` Tg/yr injection.

    Mirrors :func:`backend.variable.get_variable_regional_delta`: each year's Tg is spread over its
    12 months, run through the AOD step and the climate response monthly, scaled by the
    latitude's global-mean pattern, summed over latitudes and averaged back to years. The AOD
    step is evaluated in closed form per year (:func:`_aod_by_source_year`) and the response
    with the precomputed yearly operator; both are the reference sums in a different order.
    """
    return _contributions(cider, injection).sum(axis=1)


def _contributions(cider: GlobalMeanCider, injection: np.ndarray) -> np.ndarray:
    """``(N_YEARS, N_YEARS)``: the yearly global-mean tas change (rows) caused by each year's
    injection (columns). Rows sum to :func:`global_mean_tas_delta`."""
    injection = np.asarray(injection, dtype=float)
    if injection.shape != (N_LATS, N_YEARS):
        raise ValueError(f"injection must be ({N_LATS}, {N_YEARS}); got {injection.shape}")
    out = np.zeros((N_YEARS, N_YEARS))
    for i in range(N_LATS):
        if np.any(injection[i]):
            out += cider.yearly_operator[i] @ _aod_by_source_year(cider.param_aod[i], injection[i])
    return out


@dataclass(frozen=True)
class HoldTargetSolution:
    """A solved injection and how well it holds the target."""

    injection: np.ndarray  # (7, 66) Tg SO2 / yr, rows 60N -> 60S, columns 2035-2100
    total_tg: np.ndarray  # (66,) Tg / yr summed over latitudes
    with_sai: np.ndarray  # (66,) the resulting with-SAI global-mean tas (K vs 1850-1900)
    peak_tg: float
    max_hold_error: float  # largest |with_sai - target| over the hold years (after the ramp)
    reached: bool
    iterations: int
    # the first year (from 2035, whatever the start year) whose no-SAI line is above the target,
    # i.e. the first year SAI is needed at all (None if never). Later than start_year: SAI could
    # start later; earlier: it could start earlier than start_year.
    first_needed_year: int | None = None


def _normalised_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (N_LATS,):
        raise ValueError(f"weights must have {N_LATS} entries (60N -> 60S); got shape {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("weights must not all be zero")
    return w / w.sum()


def solve_hold_target(
    no_sai_global: np.ndarray,
    temp_target: float,
    weights,
    cider: GlobalMeanCider,
    start_year: int = FIRST_YEAR,
    ramp_up: int = 10,
    max_total_tg: float = DEFAULT_MAX_TOTAL_TG,
) -> HoldTargetSolution:
    """Solve the yearly injection that holds ``no_sai_global + delta`` at ``temp_target``.

    ``no_sai_global`` is the no-SAI global-mean tas over 2035-2100 (K vs 1850-1900). Before
    ``start_year`` nothing is injected; over the next ``ramp_up`` years the required cooling ramps
    linearly to the full ``no_sai - target`` (like the GAUSS ramp), then is held.
    """
    no_sai = np.asarray(no_sai_global, dtype=float)
    if no_sai.shape != (N_YEARS,):
        raise ValueError(f"no_sai_global must cover {FIRST_YEAR}-{LAST_YEAR} ({N_YEARS} years)")
    if not FIRST_YEAR <= start_year <= LAST_YEAR:
        raise ValueError(f"start_year must be within {FIRST_YEAR}-{LAST_YEAR}")
    ramp_up = max(1, int(ramp_up))
    ratio = _normalised_weights(weights)
    start = int(start_year - FIRST_YEAR)
    hold_start = min(N_YEARS, start + ramp_up)

    # desired change: 0 before the start, a linear ramp, then the full shortfall (negative = cooling)
    frac = np.zeros(N_YEARS)
    frac[start:hold_start] = (np.arange(start, hold_start) - start + 1) / ramp_up
    frac[hold_start:] = 1.0
    desired = -frac * np.maximum(0.0, no_sai - temp_target)

    def contributions(total):
        return _contributions(cider, np.outer(ratio, total))

    # Gauss-Newton on: min ||delta(M) - desired||^2 + SMOOTHNESS ||diff(M)||^2, 0 <= M <= cap, over
    # the years from the start. Column s of the Jacobian depends on M[s] only (_aod_by_source_year),
    # so one perturbed evaluation of every year at once gives the whole matrix.
    #
    # Years at the ceiling that are still warmer than desired are pinned at the ceiling and leave the
    # fit. A year's temperature depends only on the injections up to it, so nothing can help those
    # years except cooling the earlier ones below the target, which is what plain least squares
    # would do; with them out of the fit the earlier years simply hold the target.
    n = N_YEARS - start
    smooth = np.sqrt(SMOOTHNESS) * np.diff(np.eye(n), axis=0)
    total = np.zeros(N_YEARS)
    pinned = np.zeros(n, dtype=bool)
    it = 0
    for it in range(1, MAX_ITERATIONS + 1):
        current = contributions(total)
        residual = (current.sum(axis=1) - desired)[start:]  # > 0: warmer than desired
        newly_pinned = (total[start:] >= max_total_tg - 1e-6) & (residual > PIN_TOLERANCE_K) & ~pinned
        pinned |= newly_pinned
        step = 1e-4 * np.maximum(1.0, total)
        jac = ((contributions(total + step) - current) / step[None, :])[start:, start:]
        # linearised: delta(M_new) ~ delta(M) + J (M_new - M) = desired, i.e. J M_new = rhs
        rhs = jac @ total[start:] - residual
        free = ~pinned
        fixed = np.where(pinned, max_total_tg, 0.0)
        solved = fixed.copy()
        if free.any():
            solved[free] = lsq_linear(
                np.vstack([jac[free][:, free], smooth[:, free]]),
                np.concatenate([(rhs - jac @ fixed)[free], -smooth @ fixed]),
                bounds=(0.0, max_total_tg), method="bvls",
            ).x
        moved = float(np.max(np.abs(solved - total[start:])))
        total[start:] = solved
        if moved < STEP_TOLERANCE_TG and not newly_pinned.any():
            break

    injection = np.outer(ratio, total)
    with_sai = no_sai + global_mean_tas_delta(cider, injection)
    hold_err = float(np.max(np.abs(with_sai[hold_start:] - np.minimum(no_sai[hold_start:], temp_target)))) if hold_start < N_YEARS else 0.0
    return HoldTargetSolution(
        injection=injection,
        total_tg=total.copy(),
        with_sai=with_sai,
        peak_tg=float(total.max()),
        max_hold_error=hold_err,
        reached=hold_err <= HOLD_TOLERANCE_K,
        iterations=it,
        first_needed_year=_first_needed_year(no_sai, temp_target),
    )


def _first_needed_year(no_sai: np.ndarray, temp_target: float) -> int | None:
    above = np.flatnonzero(no_sai > temp_target)
    return int(YEARS[above[0]]) if above.size else None


def solve_hold_target_injection(
    ssp_scenario: str,
    temp_target: float,
    weights,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    start_year: int = FIRST_YEAR,
    ramp_up: int = 10,
    max_total_tg: float = DEFAULT_MAX_TOTAL_TG,
) -> HoldTargetSolution:
    """:func:`solve_hold_target` against the dashboard's own no-SAI line for ``ssp_scenario``."""
    return solve_hold_target(
        _dashboard_no_sai(ssp_scenario, Path(data_dir), Path(model_dir), Path(cache_dir)),
        temp_target, weights, load_global_mean_cider(str(data_dir)),
        start_year=start_year, ramp_up=ramp_up, max_total_tg=max_total_tg,
    )


def _dashboard_no_sai(ssp_scenario: str, data_dir: Path, model_dir: Path, cache_dir: Path) -> np.ndarray:
    """The dashboard's no-SAI global-mean tas over 2035-2100 (K vs 1850-1900): the cos-latitude
    mean of the three-model tas map, as :func:`backend.get_outputs.get_temp_diff` computes it."""
    from .get_outputs import get_temp_diff  # local: get_outputs imports a lot

    # get_temp_diff returns target - no_sai; any target gives the same line
    return (0.0 - get_temp_diff(data_dir, model_dir, cache_dir, ssp_scenario, 0.0)).sel(time=slice(FIRST_YEAR, LAST_YEAR)).values


def lowest_reachable_target(
    no_sai_global: np.ndarray,
    weights,
    cider: GlobalMeanCider,
    start_year: int = FIRST_YEAR,
    ramp_up: int = 10,
    max_total_tg: float = DEFAULT_MAX_TOTAL_TG,
) -> float:
    """The lowest target :func:`solve_hold_target` reaches under the ceiling, to
    :data:`FLOOR_RESOLUTION_K` (K vs 1850-1900).

    A higher target needs less cooling every year, so "reached" switches once, from no to yes, as
    the target rises, and the floor is found by bisection (about 8 solves) between two bounds:
    - below, the warmest hold year of the line with the injection at the ceiling from the start
      on, since no injection within the ceiling cools any year more. It is optimistic: holding a
      flat target cannot use the ceiling early on without cooling those years below it, so less
      cooling builds up for the later ones;
    - above, the warmest hold year of the no-SAI line, which needs no injection at all.
    """
    no_sai = np.asarray(no_sai_global, dtype=float)
    start = int(start_year - FIRST_YEAR)
    hold_start = min(N_YEARS - 1, start + max(1, int(ramp_up)))
    capped = np.zeros(N_YEARS)
    capped[start:] = max_total_tg
    coldest = no_sai + global_mean_tas_delta(cider, np.outer(_normalised_weights(weights), capped))
    low, high = float(np.max(coldest[hold_start:])), float(np.max(no_sai[hold_start:]))

    def reached(target):
        return solve_hold_target(no_sai, target, weights, cider, start_year, ramp_up, max_total_tg).reached

    if high - low <= FLOOR_RESOLUTION_K or reached(low):
        return low
    while high - low > FLOOR_RESOLUTION_K:
        middle = 0.5 * (low + high)
        if reached(middle):
            high = middle
        else:
            low = middle
    return high


@lru_cache(maxsize=256)
def _lowest_reachable_target_cached(
    ssp_scenario: str, weights: tuple, data_dir: str, model_dir: str, cache_dir: str,
    start_year: int, ramp_up: int, max_total_tg: float,
) -> float:
    return lowest_reachable_target(
        _dashboard_no_sai(ssp_scenario, Path(data_dir), Path(model_dir), Path(cache_dir)),
        list(weights), load_global_mean_cider(data_dir), start_year, ramp_up, max_total_tg,
    )


def lowest_reachable_target_for_scenario(
    ssp_scenario: str,
    weights,
    data_dir: Path,
    model_dir: Path,
    cache_dir: Path,
    start_year: int = FIRST_YEAR,
    ramp_up: int = 10,
    max_total_tg: float = DEFAULT_MAX_TOTAL_TG,
) -> float:
    """:func:`lowest_reachable_target` against the dashboard's no-SAI line for ``ssp_scenario``.
    It does not depend on the target, so it is cached per scenario, split and timeline."""
    return _lowest_reachable_target_cached(
        ssp_scenario, tuple(float(w) for w in _normalised_weights(weights)), str(data_dir), str(model_dir),
        str(cache_dir), int(start_year), int(ramp_up), float(max_total_tg),
    )
