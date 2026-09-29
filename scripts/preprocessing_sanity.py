"""Preprocessing-time sanity checks shared between the daily/monthly processing scripts.

Builds plausibility bounds from historical minima and the SSP2-4.5 baseline maxima and
asserts that derived variables stay within them; checks for NaNs and out-of-range exposure
or icefrac values; validates that yearly time series are continuous (with a configurable
maximum gap for the historical→SSP245 seam). All checks raise :class:`SanityCheckError`
(a :class:`ValueError` subclass) so callers can distinguish data-quality failures from
generic errors.
"""
import numpy as np
import xarray as xr


class SanityCheckError(ValueError):
    """Raised when a data sanity validation fails."""


# Inclusive (min, max) allowed annual lengths after ``get_yearly_aggregate`` drops
# incomplete years. Nominal maxima: historical 165, baseline SSP2-4.5 ~85, SAI ~35.
# Ranges tolerate common edge drops (e.g. 1850, 2015, 2035, 2070) and shorter LENS
# members (007–009) under ``baseline`` / ``tasmax`` / ``tasmin``.
EXPECTED_YEARS_BY_EXPERIMENT = {
    "historical": (164, 165),
    "baseline": (50, 85),
    # ScenarioMIP SSP2-4.5 (MIROC): 2015-2100 for most variables (85-86 yr), but
    # p-e is limited to the store's QFLX fill window 2020-2084 (65 yr; the NaN-
    # padded months drop as incomplete years). Lower bound set to catch gross
    # truncation without failing the p-e case.
    "ssp245": (63, 86),
    "0.5": (33, 36),
    "1.0": (33, 36),
    "1.5": (33, 36),
}

NON_NEGATIVE_VARIABLES = {"tas", "pr", "icefrac", "tasmin", "tasmax"}
TEMP_BOUND_VARIABLES = {"tas", "tasmin", "tasmax"}
PR_BOUND_VARIABLES = {"pr"}
EXPOSURE_VARIABLES = {
    "tas_above_35",
    "tas_above_40",
    "tas_below_0",
    "pr_above_10",
    "pr_above_20",
}

# After yearly resampling, cumulative "days per month" metrics are summed over months;
# upper bound is ~12 × 31 days/year (leap / calendar quirks), not 365.
EXPOSURE_YEARLY_SUM_MAX = 372

# Temperature lower-bound margin (K) below the per-cell historical minimum for
# the monthly plausibility check. The check exists to catch gross unit/scale
# blunders (°C-shipped-as-K ≈ 273 K offset, scale errors are far larger), so the
# margin only needs to clear genuine climate variability. 10 K was enough for
# CESM, but MIROC-ES2H's SSP2-4.5 runs contain spatially coherent Antarctic
# winter cold events reaching well below the 165-year per-cell historical
# minimum — deepest measured dip 19.6 K in the GeoMIP baseline (tasmin; Ross Ice
# Shelf, member r01, April 2033) and 21.3 K in the ScenarioMIP ssp245 run. Each
# additional realization has pushed that maximum up, so 30 K leaves headroom for
# the next one while remaining an order of magnitude below a unit-error signature.
TEMP_MONTHLY_LOW_MARGIN_K = 30.0

# Per-cell yearly-mean precipitation (mm/day) tops out around ~30 mm/day even in
# the wettest places on Earth. A generous 100 mm/day cap catches gross unit/scale
# blunders — e.g. precip shipped in kg/m²/s (≈ mm/s) instead of m/s, which the
# consumer's m/s→mm/day scaler then inflates 1000× — without false-positiving on
# genuinely wet cells. This is the model-agnostic guard for precip magnitude;
# unlike the monthly cell-bound check it does not need a historical experiment,
# so it also protects models (e.g. MIROC) that have none.
PR_YEARLY_CELL_MAX_MM_PER_DAY = 100


def expected_year_count_for_experiment(exp: str) -> tuple[int, int] | None:
    """Return inclusive (min, max) allowed annual length for an experiment, if constrained."""
    return EXPECTED_YEARS_BY_EXPERIMENT.get(exp)


def validate_year_count_for_experiment(
    exp: str, year_count: int, var_name: str, member: str
) -> None:
    """Validate that an experiment/member has the expected annual count."""
    expected = expected_year_count_for_experiment(exp)
    if expected is None:
        return
    lo, hi = expected
    if not (lo <= year_count <= hi):
        raise SanityCheckError(
            f"Unexpected yearly count for var={var_name}, exp={exp}, member={member}: "
            f"expected between {lo} and {hi} inclusive, got {year_count}."
        )


def _reduced_bool_sum(mask: xr.DataArray | xr.Dataset) -> int:
    """Count True values in a boolean mask (handles Dataset from broadcasting)."""
    if isinstance(mask, xr.Dataset):
        mask = mask.to_array()
    total = mask.sum()
    return int(np.asarray(total.values).sum())


def validate_no_nans(
    data: xr.DataArray | xr.Dataset, var_name: str, exp: str, member: str, stage: str
) -> None:
    """Raise if any NaNs are present in a DataArray or Dataset."""
    if hasattr(data, "data_vars"):
        # Dataset -> stack variables into one array for a single NaN check.
        array_like = data.to_array()
    else:
        # DataArray
        array_like = data

    nan_count = _reduced_bool_sum(array_like.isnull())
    if nan_count > 0:
        raise SanityCheckError(
            f"NaN sanity check failed at {stage} for var={var_name}, exp={exp}, "
            f"member={member}: found {nan_count} NaN values."
        )


def requires_non_negative_check(var_name: str) -> bool:
    """Return whether a variable must be constrained to non-negative values."""
    return var_name in NON_NEGATIVE_VARIABLES


def validate_no_negative_values(
    data: xr.DataArray | xr.Dataset, var_name: str, exp: str, member: str, stage: str
) -> None:
    """Raise if any negative values are present for required variables."""
    if not requires_non_negative_check(var_name):
        return

    if hasattr(data, "data_vars"):
        array_like = data.to_array()
    else:
        array_like = data

    negative_count = _reduced_bool_sum(array_like < 0)
    if negative_count > 0:
        raise SanityCheckError(
            f"Negative-value sanity check failed at {stage} for var={var_name}, exp={exp}, "
            f"member={member}: found {negative_count} negative values."
        )


def requires_monthly_cell_bound_check(var_name: str) -> bool:
    """Return whether a variable should receive historical/baseline bound checks."""
    return var_name in TEMP_BOUND_VARIABLES or var_name in PR_BOUND_VARIABLES


def _concat_member_time_series(data_list: list[xr.DataArray]) -> xr.DataArray:
    """Concatenate chunks for one member along time in chronological order."""
    combined = xr.concat(data_list, dim="time", join="outer", coords="minimal").sortby("time")
    times = combined["time"].values
    if len(times) != len(np.unique(times)):
        combined = combined.groupby("time", squeeze=False).mean(keep_attrs=True)
    return combined


def _stack_experiment_members(member2data: dict[str, list[xr.DataArray]]) -> xr.DataArray:
    """Stack all member monthly series for one experiment on `member` dim."""
    member_arrays = []
    for data_list in member2data.values():
        if not data_list:
            continue
        member_arrays.append(_concat_member_time_series(data_list))

    if not member_arrays:
        raise SanityCheckError("Cannot build monthly bounds: experiment has no data.")

    return xr.concat(member_arrays, dim="member", join="outer", coords="minimal")


def build_monthly_cell_bounds(
    var_name: str,
    exp2member2data: dict[str, dict[str, list[xr.DataArray]]],
    no_sai_experiments: tuple[str, ...] = ("baseline",),
) -> dict[str, xr.DataArray] | None:
    """
    Build per-grid-cell monthly bounds.
    - Min bound from historical minimum (temperature only, minus
      ``TEMP_MONTHLY_LOW_MARGIN_K``).
    - Max bound from the no-SAI SSP2-4.5 maxima (all checked vars, plus 10). A model
      publishing more than one no-SAI run (e.g. a GeoMIP branch point alongside the
      ScenarioMIP run) contributes the elementwise max over all of them, so each run's
      own extremes sit inside the envelope it is later checked against.
    - For precipitation, min bound is fixed at 0.
    """
    if not requires_monthly_cell_bound_check(var_name):
        return None

    historical = exp2member2data.get("historical", {})
    reference_members = [
        exp2member2data[exp] for exp in no_sai_experiments if exp2member2data.get(exp)
    ]
    if not historical:
        raise SanityCheckError("Cannot build monthly bounds: missing historical data.")
    if not reference_members:
        raise SanityCheckError("Cannot build monthly bounds: missing baseline (SSP2-4.5) data.")

    historical_stack = _stack_experiment_members(historical)

    historical_min = historical_stack.min(dim=("member", "time"))
    baseline_max = xr.concat(
        [
            _stack_experiment_members(members).max(dim=("member", "time"))
            for members in reference_members
        ],
        dim="reference",
    ).max(dim="reference")

    if var_name in TEMP_BOUND_VARIABLES:
        min_allowed = historical_min - TEMP_MONTHLY_LOW_MARGIN_K
    else:
        min_allowed = xr.zeros_like(baseline_max)

    max_allowed = baseline_max + 10.0
    return {"min": min_allowed, "max": max_allowed}


def validate_monthly_cell_bounds(
    data: xr.DataArray | xr.Dataset,
    var_name: str,
    exp: str,
    member: str,
    bounds: dict[str, xr.DataArray] | None,
    no_sai_experiments: tuple[str, ...] = ("baseline",),
) -> None:
    """
    Plausibility bounds from historical minima and SSP2-4.5 no-SAI maxima.

    Historical monthly data can exceed future SSP245 maxima (e.g. heavy rain); SAI
    scenarios can fall outside the historical / SSP245 envelope by design. We
    therefore enforce temperature **lower** bounds (vs historical min − margin) on
    ``historical`` and the no-SAI runs only, **upper** bounds (vs the no-SAI max +
    margin) on those same no-SAI runs only, and ``pr >= 0`` on all experiments.
    """
    if bounds is None:
        return

    # SAI experiments: the GAUSS cooling labels plus any G6 run ("g6-1.5",
    # "g6-0.5", ...). They may cool below the historical envelope by design,
    # so the temperature lower bound does not apply to them.
    is_sai_exp = exp in ("0.5", "1.0", "1.5") or exp.startswith("g6-")
    if var_name in TEMP_BOUND_VARIABLES:
        low_count = (
            0
            if is_sai_exp
            else _reduced_bool_sum(data < bounds["min"])
        )
        high_count = (
            0
            if exp not in no_sai_experiments
            else _reduced_bool_sum(data > bounds["max"])
        )
    elif var_name in PR_BOUND_VARIABLES:
        low_count = _reduced_bool_sum(data < 0.0)
        high_count = (
            0
            if exp not in no_sai_experiments
            else _reduced_bool_sum(data > bounds["max"])
        )
    else:
        low_count = _reduced_bool_sum(data < bounds["min"])
        high_count = _reduced_bool_sum(data > bounds["max"])

    if low_count == 0 and high_count == 0:
        return

    raise SanityCheckError(
        f"Monthly grid-cell bound check failed for var={var_name}, exp={exp}, member={member}: "
        f"{low_count} values below min bound and {high_count} values above max bound."
    )


def validate_variable_specific_ranges(
    data: xr.DataArray | xr.Dataset, var_name: str, exp: str, member: str, stage: str
) -> None:
    """Apply strict variable-specific range checks."""
    if hasattr(data, "data_vars"):
        array_like = data.to_array()
    else:
        array_like = data

    if var_name == "icefrac":
        below_zero = _reduced_bool_sum(array_like < 0)
        above_one = _reduced_bool_sum(array_like > 1)
        if below_zero > 0 or above_one > 0:
            raise SanityCheckError(
                f"Range sanity check failed at {stage} for var={var_name}, exp={exp}, "
                f"member={member}: {below_zero} values below 0 and {above_one} values above 1."
            )

    if var_name in EXPOSURE_VARIABLES:
        below_zero = _reduced_bool_sum(array_like < 0)
        above_max = _reduced_bool_sum(array_like > EXPOSURE_YEARLY_SUM_MAX)
        if below_zero > 0 or above_max > 0:
            raise SanityCheckError(
                f"Range sanity check failed at {stage} for var={var_name}, exp={exp}, "
                f"member={member}: {below_zero} values below 0 and {above_max} values "
                f"above {EXPOSURE_YEARLY_SUM_MAX}."
            )

    if var_name in PR_BOUND_VARIABLES:
        # Upper-bound guard against unit/scale errors (negatives are caught by
        # validate_no_negative_values). See PR_YEARLY_CELL_MAX_MM_PER_DAY.
        above_max = _reduced_bool_sum(array_like > PR_YEARLY_CELL_MAX_MM_PER_DAY)
        if above_max > 0:
            raise SanityCheckError(
                f"Range sanity check failed at {stage} for var={var_name}, exp={exp}, "
                f"member={member}: {above_max} values above {PR_YEARLY_CELL_MAX_MM_PER_DAY} "
                f"mm/day (likely a unit/scale error, e.g. kg/m²/s vs m/s)."
            )


def validate_year_continuity_strict(
    years: np.ndarray, var_name: str, exp: str, member: str, stage: str
) -> None:
    """Require strictly continuous yearly sequence (step=1)."""
    if len(years) <= 1:
        return
    diffs = years[1:] - years[:-1]
    if (diffs != 1).any():
        raise SanityCheckError(
            f"Continuity check failed at {stage} for var={var_name}, exp={exp}, member={member}: "
            "yearly data is not strictly continuous."
        )


def validate_year_continuity_with_max_gap(
    years: np.ndarray, var_name: str, exp: str, member: str, stage: str, max_gap_years: int
) -> None:
    """Allow gaps, but enforce maximum allowed gap size in years."""
    if len(years) <= 1:
        return
    diffs = years[1:] - years[:-1]
    max_observed_gap = int(diffs.max())
    if max_observed_gap > max_gap_years:
        raise SanityCheckError(
            f"Continuity check failed at {stage} for var={var_name}, exp={exp}, member={member}: "
            f"maximum year gap is {max_observed_gap}, allowed {max_gap_years}."
        )
