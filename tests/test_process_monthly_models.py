"""Unit tests for the multi-model config in :mod:`scripts.process_monthly_gauss`.

Covers the ``MODELS`` table (CESM2-WACCM-MA + MIROC-ES2H, with MIROC's
no-variability / ``-miroc`` suffix quirks), the variable→Zarr group routing in
``_group_for_var``, the ``var2esm_var`` name map, and the ``process_monthly``
argument guards. Importing this module must not require a compiled ESMF
library — ``xesmf`` is imported lazily in the regridding functions — so these
run in a plain CI environment.
"""
import pytest

import scripts.process_monthly_gauss as pmg


def test_models_are_cesm_miroc_and_ukesm() -> None:
    assert set(pmg.MODELS) == {"CESM2-WACCM-MA", "MIROC-ES2H", "UKESM1-1-LL"}


def test_cesm_config() -> None:
    cfg = pmg.MODELS["CESM2-WACCM-MA"]
    assert cfg["model_coord"] == "CESM2-WACCM"
    assert cfg["output_suffix"] == ""
    assert cfg["has_historical"] is True
    assert cfg["compute_variability"] is True
    labels = [label for _, label in cfg["experiments"]]
    assert "historical" in labels
    assert "baseline" in labels
    # One run serves as both the GAUSS branch point and the projection baseline.
    assert cfg["no_sai_experiments"] == ("baseline",)
    assert cfg["experiment_missing_vars"] == {}


def test_miroc_config_isolates_its_quirks() -> None:
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["model_coord"] == "MIROC-ES2H"
    assert cfg["output_suffix"] == "-miroc"
    assert cfg["has_historical"] is True
    assert cfg["compute_variability"] is False
    labels = [label for _, label in cfg["experiments"]]
    assert labels == ["baseline", "ssp245", "historical", "g6-1.5", "g6-0.5"]


def test_miroc_publishes_two_no_sai_runs() -> None:
    """The GeoMIP branch point and the ScenarioMIP run are both no-SAI references."""
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["no_sai_experiments"] == ("baseline", "ssp245")
    # Historical is stitched into both so each carries the full 1850- series.
    assert "historical" not in cfg["no_sai_experiments"]


def test_miroc_ssp245_has_no_missing_variables() -> None:
    """The ssp245 store is complete: QFLX filled from the G6 baseline hfls
    (2020-2084) and ICEFRAC regridded from the run's own CMOR siconc (2015-2100)."""
    cfg = pmg.MODELS["MIROC-ES2H"]
    assert cfg["experiment_missing_vars"] == {}


def test_ukesm_config_isolates_its_quirks() -> None:
    cfg = pmg.MODELS["UKESM1-1-LL"]
    assert cfg["model_coord"] == "UKESM1-1-LL"
    assert cfg["output_suffix"] == "-ukesm"
    assert cfg["has_historical"] is True
    assert cfg["compute_variability"] is False
    labels = [label for _, label in cfg["experiments"]]
    assert labels == ["baseline", "historical", "g6-1.5"]
    # One het-chem SSP2-4.5 run is both the G6 branch point and the projection
    # baseline (CESM-style).
    assert cfg["no_sai_experiments"] == ("baseline",)


def test_ukesm_historical_has_nothing_left_to_skip() -> None:
    """MACLOUD publishes neither sea ice nor evaporative flux for the historical
    suites, but ESGF publishes both (siconca, hfls), so every variable now stitches
    across 2015. An entry reappearing here means a store lost its ESGF-backed
    array and some variable silently went back to a 2015 start."""
    assert pmg.MODELS["UKESM1-1-LL"]["experiment_missing_vars"] == {}


def test_ukesm_esgf_backed_vars_use_one_realization_everywhere() -> None:
    """ESGF publishes UKESM historical sea ice and hfls for one member, the run
    this repo calls r3. Restricting both to r3 in every experiment keeps each
    series on a single realization; a three-member mean after 2015 against one
    member before it would step the variance at the stitch, and the sea-ice
    sigmoid trains on exactly that series."""
    cfg = pmg.MODELS["UKESM1-1-LL"]
    assert cfg["variable_members"] == {"icefrac": ("r3",), "p-e": ("r3",)}
    # The restriction is per variable: tas, pr and the rest keep all three members.
    assert set(cfg["variable_members"]) == {"icefrac", "p-e"}


def test_pe_member_restriction_covers_only_the_qflx_term() -> None:
    """p-e is assembled as (the model's pr output) - QFLX, and variable_members is
    keyed on the OUTPUT variable, so restricting "p-e" restricts the QFLX term
    alone — the pr file it subtracts from stays a three-member mean. That is
    intended (both terms are uniform across the stitch, and the ensemble mean is
    the better estimate of forced pr), but it means p-e is not single-member the
    way icefrac is, and anyone reading the config should not assume it is."""
    cfg = pmg.MODELS["UKESM1-1-LL"]
    assert cfg["variable_members"]["p-e"] == ("r3",)
    assert "pr" not in cfg["variable_members"]
    assert pmg.var2esm_var["p-e"] == "QFLX"


def test_experiment_uris_use_zarr_root() -> None:
    for cfg in pmg.MODELS.values():
        for uri, _label in cfg["experiments"]:
            assert uri.startswith(pmg.ZARR_STORES_URI)
            assert uri.endswith(".zarr")


@pytest.mark.parametrize(
    ("var", "group"),
    [
        ("tas", "Mon/atmos_2d"),
        ("pr", "Mon/atmos_2d"),
        ("p-e", "Mon/atmos_2d"),
        ("icefrac", "Mon/atmos_2d"),
        ("tasmin", "Mon/atmos_2d_tasminmax"),
        ("tasmax", "Mon/atmos_2d_tasminmax"),
        ("tas_above_35", "Mon/atmos_2d_derived"),
        ("tas_above_40", "Mon/atmos_2d_derived"),
        ("tas_below_0", "Mon/atmos_2d_derived"),
        ("pr_above_10", "Mon/atmos_2d_derived"),
        ("pr_above_20", "Mon/atmos_2d_derived"),
    ],
)
def test_group_for_var(var: str, group: str) -> None:
    assert pmg._group_for_var(var) == group


@pytest.mark.parametrize(
    ("var", "esm_var"),
    [
        ("tas", "TREFHT"),
        ("pr", "PRECT"),
        ("tasmin", "TREFHTMN"),
        ("tasmax", "TREFHTMX"),
        ("p-e", "QFLX"),
        ("icefrac", "ICEFRAC"),
        ("pr_above_20", "PRECT_above_20"),
    ],
)
def test_var2esm_var_mapping(var: str, esm_var: str) -> None:
    assert pmg.var2esm_var[var] == esm_var


def test_prect_unit_scaler_is_m_per_s_to_mm_per_day() -> None:
    # CESM PRECT is m/s; MIROC is normalised to m/s in the producer so this
    # single scaler is correct for both. 86400 * 1000 = m/s -> mm/day.
    assert pmg.UNIT_SCALERS["PRECT"] == 86400 * 1000


def test_process_monthly_rejects_unknown_var() -> None:
    with pytest.raises(ValueError, match="var must be one of"):
        pmg.process_monthly(var="not_a_var", output_dir="/tmp/should-not-be-created")


def test_process_monthly_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="model must be 'all' or one of"):
        pmg.process_monthly(
            var="tas", model="NOPE", output_dir="/tmp/should-not-be-created"
        )


# --- CAM end-of-interval time stamps ---------------------------------------


def _monthly_da(stamps, *, cam_housekeeping: bool):
    """Two-year (member-less) monthly DataArray inside a Dataset, CESM-style."""
    import numpy as np
    import xarray as xr

    n = len(stamps)
    ds = xr.Dataset(
        {"ICEFRAC": (("time", "lat", "lon"), np.arange(n, dtype=float).reshape(n, 1, 1))},
        coords={"time": stamps, "lat": [80.0], "lon": [0.0]},
    )
    if cam_housekeeping:
        ds["datesec"] = ("time", np.zeros(n, dtype=int))
    return ds


def _cftime_stamps(first_year, first_month, n):
    import cftime

    out, y, m = [], first_year, first_month
    for _ in range(n):
        out.append(cftime.datetime(y, m, 1, calendar="proleptic_gregorian"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def test_cesm_opts_into_cam_time_fix_and_others_do_not() -> None:
    assert pmg.MODELS["CESM2-WACCM-MA"]["cam_history_time"] is True
    assert not pmg.MODELS["MIROC-ES2H"].get("cam_history_time", False)
    assert not pmg.MODELS["UKESM1-1-LL"].get("cam_history_time", False)


def test_cam_stamping_detected_only_with_housekeeping_and_first_of_month() -> None:
    import cftime

    stamps = _cftime_stamps(2015, 2, 24)  # Jan 2015 stamped 2015-02-01, CAM-style
    assert pmg._is_cam_end_of_interval_stamping(_monthly_da(stamps, cam_housekeeping=True))
    # Same stamps without CAM's datesec (e.g. the derived day-count group): no shift.
    assert not pmg._is_cam_end_of_interval_stamping(_monthly_da(stamps, cam_housekeeping=False))
    # Mid-month stamps (CESM's CMOR-style historical tasminmax): no shift.
    mid = [cftime.datetime(t.year, t.month, 16, 12, calendar=t.calendar) for t in stamps]
    assert not pmg._is_cam_end_of_interval_stamping(_monthly_da(mid, cam_housekeeping=True))


def test_store_corrected_at_the_source_is_not_shifted_twice() -> None:
    # Re-stamped to the month itself but still carrying CAM's datesec: Jan 2015 .. Dec 2016.
    ds = _monthly_da(_cftime_stamps(2015, 1, 24), cam_housekeeping=True)
    assert not pmg._is_cam_end_of_interval_stamping(ds)
    out = pmg._process_member_data(
        ds["ICEFRAC"], "ICEFRAC", "baseline",
        shift_back_one_month=pmg._is_cam_end_of_interval_stamping(ds),
    )
    # Same true-September values (8 and 20) as the shifted CAM store gives.
    assert out.dropna("time").values.ravel().tolist() == [8.0, 20.0]


def test_cam_stamps_with_an_unexpected_span_raise() -> None:
    # Starts in March: neither CAM's Feb-Jan nor a corrected Jan-Dec.
    ds = _monthly_da(_cftime_stamps(2015, 3, 24), cam_housekeeping=True)
    with pytest.raises(ValueError, match="Cannot tell whether"):
        pmg._is_cam_end_of_interval_stamping(ds)


def test_shifted_cam_baseline_keeps_2015_and_masks_true_september() -> None:
    # Jan 2015 .. Dec 2016, stamped one month late as the CESM stores are.
    ds = _monthly_da(_cftime_stamps(2015, 2, 24), cam_housekeeping=True)
    out = pmg._process_member_data(ds["ICEFRAC"], "ICEFRAC", "baseline", shift_back_one_month=True)
    years = out.time.dt.year.values
    assert (years == 2015).sum() == 12 and (years == 2016).sum() == 12
    # Value i is calendar month i (0 = Jan 2015), so the surviving September values
    # must be 8 and 20; without the shift they would be 7 and 19 (August).
    assert out.dropna("time").values.ravel().tolist() == [8.0, 20.0]


def test_unshifted_cam_baseline_reproduces_the_old_bug() -> None:
    ds = _monthly_da(_cftime_stamps(2015, 2, 24), cam_housekeeping=True)
    out = pmg._process_member_data(ds["ICEFRAC"], "ICEFRAC", "baseline")
    assert out.dropna("time").values.ravel().tolist() == [7.0, 19.0]
    assert (out.time.dt.year.values == 2015).sum() == 11


# --- Store opening: retry transient errors, never skip a store silently -----


class _HTTPError(Exception):
    """Stand-in for aiohttp.ClientResponseError, which carries ``.status``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"{status}, message='...'")
        self.status = status


def _wrapped(status: int) -> ValueError:
    # open_xarray_datasets re-raises every failure as ValueError from the original.
    try:
        raise ValueError("Failed to open zarr store") from _HTTPError(status)
    except ValueError as e:
        return e


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_statuses_are_retried(status: int) -> None:
    assert pmg._is_transient_open_error(_wrapped(status))


@pytest.mark.parametrize("status", [403, 404])
def test_permanent_http_statuses_are_not_retried(status: int) -> None:
    assert not pmg._is_transient_open_error(_wrapped(status))


def test_open_retries_with_backoff_then_succeeds(monkeypatch) -> None:
    calls, sleeps = [], []

    def fake_open(url, group=None):
        calls.append(url)
        if len(calls) < 3:
            raise _wrapped(429)
        return "dataset"

    monkeypatch.setattr(pmg, "open_xarray_datasets", fake_open)
    out = pmg._open_store_with_retry("u", "g", attempts=5, first_delay_s=2, sleep=sleeps.append)
    assert out == "dataset"
    assert len(calls) == 3 and sleeps == [2, 4]


def test_open_gives_up_after_last_attempt(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(pmg, "open_xarray_datasets", lambda url, group=None: (_ for _ in ()).throw(_wrapped(429)))
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        pmg._open_store_with_retry("u", "g", attempts=3, first_delay_s=1, sleep=sleeps.append)
    assert sleeps == [1, 2]


def test_open_does_not_retry_a_permanent_error(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(pmg, "open_xarray_datasets", lambda url, group=None: (_ for _ in ()).throw(_wrapped(404)))
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        pmg._open_store_with_retry("u", "g", sleep=sleeps.append)
    assert sleeps == []


def test_unopenable_store_fails_the_run_instead_of_being_skipped(monkeypatch, tmp_path) -> None:
    # The old loop printed "failed to open" and carried on, writing outputs that
    # silently lacked that experiment.
    monkeypatch.setattr(pmg, "STORE_OPEN_ATTEMPTS", 2)
    monkeypatch.setattr(pmg, "STORE_OPEN_FIRST_DELAY_S", 0.0)
    monkeypatch.setattr(pmg, "open_xarray_datasets", lambda url, group=None: (_ for _ in ()).throw(_wrapped(429)))
    with pytest.raises(RuntimeError, match="Could not open"):
        pmg.process_monthly(var="tas", model="CESM2-WACCM-MA", output_dir=tmp_path)


def test_store_missing_the_variable_fails_the_run(monkeypatch, tmp_path) -> None:
    import xarray as xr

    monkeypatch.setattr(pmg, "open_xarray_datasets", lambda url, group=None: xr.Dataset())
    with pytest.raises(ValueError, match="experiment_missing_vars"):
        pmg.process_monthly(var="tas", model="CESM2-WACCM-MA", output_dir=tmp_path)
