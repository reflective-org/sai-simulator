"""Persist the harness's intermediates and the three audited quantities.

Two stages, so the expensive part runs once per config and the audited part re-runs per
code revision:

* :func:`dump_intermediates` — everything ``get_outputs`` computes *before* the p-values,
  bands and histograms: the projection map, the SAI delta, the historical series and
  rebase, and (icefrac) the per-model sigmoid predictions and cell areas. Written under
  ``<root>/<config>/intermediates/<params.key>/<var>/``. FaIR- and emulator-dependent, code
  under audit-independent.
* :func:`compute_quantities` — from those intermediates, the decadal grid-level p-values
  (no-SAI and with-SAI, **all** decades), the mean-over-time series with whatever band the
  current code produces, the internal-variability scalar, and the histogram. Written under
  ``<root>/<config>/<git-sha>/<params.key>/<var>/``. This deliberately bypasses the
  production cache (``get_outputs`` prefers cached p-value ``.nc`` files, so a p-value fix
  would otherwise be invisible until the cache is regenerated).

The calls mirror ``backend.get_outputs.get_outputs`` step for step; when that orchestration
changes, this module must follow.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.baseline import get_regional_map  # noqa: E402
from backend.constants import MIN_SAI_START, VAR2INFO  # noqa: E402
from backend.get_outputs import (  # noqa: E402
    _compute_distribution,
    _compute_internal_variability,
    _compute_mean_over_time,
    _compute_regional_delta,
    _internal_variability_per_model,
    _load_variable_fields,
    get_temp_diff,
)
from backend.p_values import get_grid_level_p_values  # noqa: E402
from fair_wrap.fair_utils import REVERSE_FANCY_SSP_TITLES, SIM_END_YEAR  # noqa: E402

from .configs import CONFIGS, POOLED_DELTA_CAVEAT, active_models, build_model_dir  # noqa: E402

# Variables whose projection map get_regional_map returns *rebased* to the 1850-1900 mean;
# the p-value comparison needs the absolute field back (mirrors _compute_sai_p_values).
REBASED_VARS = ("tas", "tasmin", "tasmax", "pr", "p-e")

ALL_VARIABLES: tuple[str, ...] = tuple(VAR2INFO)
ALL_SCENARIOS: tuple[str, ...] = tuple(REVERSE_FANCY_SSP_TITLES)

_ENCODING = {"zlib": True, "complevel": 4}


@dataclass(frozen=True)
class RunParams:
    """One dashboard request. Defaults are the dashboard's own (launch_gradio.py:531-537)."""

    ssp_scenario: str = "SSP2-4.5"
    temp_target: float = 1.0
    start_year: int = MIN_SAI_START
    ramp_up: int = 10
    decade_start_year: int = 2091
    decade_end_year: int = 2100

    @property
    def simple_ssp(self) -> str:
        return REVERSE_FANCY_SSP_TITLES[self.ssp_scenario]

    @property
    def key(self) -> str:
        return (
            f"{self.simple_ssp}_t{self.temp_target:.1f}_y{self.start_year}"
            f"_r{self.ramp_up}_d{self.decade_start_year}-{self.decade_end_year}"
        )


def git_sha() -> str:
    """Short HEAD sha, with ``-dirty`` appended when tracked files have local changes."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
    ).strip()
    return f"{sha}-dirty" if dirty else sha


def intermediates_dir(root: Path, config: str, params: RunParams, var: str) -> Path:
    return Path(root) / config / "intermediates" / params.key / var


def quantities_dir(root: Path, config: str, sha: str, params: RunParams, var: str) -> Path:
    return Path(root) / config / sha / params.key / var


def _save(out: Path, name: str, da: xr.DataArray | None) -> None:
    if da is None:
        return
    da.to_dataset(name=name).to_netcdf(out / f"{name}.nc", encoding={name: _ENCODING})


def _load(out: Path, name: str, required: bool = True) -> xr.DataArray | None:
    path = out / f"{name}.nc"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing intermediate {path}; run dump_intermediates first")
        return None
    with xr.open_dataarray(path) as da:
        return da.load()


def dump_intermediates(
    config: str,
    var: str,
    params: RunParams,
    data_dir: Path,
    src_model_dir: Path,
    root: Path,
    force: bool = False,
) -> Path:
    """Run the pre-audit part of ``get_outputs`` under ``config`` and persist its fields."""
    models = CONFIGS[config]
    model_dir = build_model_dir(src_model_dir, Path(root) / config / "model_dir", models)
    cache_dir = Path(root) / config / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = intermediates_dir(root, config, params, var)
    if (out / "regional_map.nc").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)

    with active_models(models):
        regional_temp, regional_temp_p, regional_temp_per_model = get_regional_map(
            "tas", data_dir, model_dir, cache_dir, params.ssp_scenario
        )
        temp_diff = get_temp_diff(
            data_dir, model_dir, cache_dir, params.ssp_scenario, params.temp_target
        )
        fields = _load_variable_fields(
            var,
            ssp_scenario=params.ssp_scenario,
            data_dir=data_dir,
            model_dir=model_dir,
            cache_dir=cache_dir,
            regional_temp=regional_temp,
            regional_temp_no_sai_p_values=regional_temp_p,
            regional_temp_per_model=regional_temp_per_model,
        )
        delta = _compute_regional_delta(
            var,
            fields=fields,
            regional_temp=regional_temp,
            ssp_scenario=params.ssp_scenario,
            temp_target=params.temp_target,
            ramp_up=params.ramp_up,
            start_year=params.start_year,
            temp_diff=temp_diff,
            variable_injection=None,
            data_dir=data_dir,
            model_dir=model_dir,
            cache_dir=cache_dir,
            regional_temp_per_model=regional_temp_per_model,
        )

    _save(out, "regional_map", fields.regional_map)
    _save(out, "regional_map_per_model", fields.regional_map_per_model)
    _save(out, "historical_model_per_model", fields.historical_model_per_model)
    _save(out, "regional_delta", delta.regional_delta)
    _save(out, "historical_model", fields.historical_model)
    _save(out, "historical_rebase", fields.historical_rebase)
    _save(out, "area", fields.area)
    _save(out, "icefrac_per_model", fields.icefrac_per_model)
    _save(out, "icefrac_with_sai", delta.icefrac_with_sai)
    _save(out, "icefrac_with_sai_per_model", delta.icefrac_with_sai_per_model)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "config": config,
                "models": list(models),
                "var": var,
                "params": asdict(params),
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pooled_delta_caveat": POOLED_DELTA_CAVEAT,
            },
            indent=2,
        )
        + "\n"
    )
    return out


def compute_quantities(
    config: str,
    var: str,
    params: RunParams,
    data_dir: Path,
    root: Path,
    sha: str | None = None,
) -> Path:
    """Recompute p-values, bands and histogram from the intermediates with the current code."""
    sha = sha or git_sha()
    inter = intermediates_dir(root, config, params, var)
    out = quantities_dir(root, config, sha, params, var)
    out.mkdir(parents=True, exist_ok=True)

    regional_map = _load(inter, "regional_map")
    regional_delta = _load(inter, "regional_delta")
    historical_model = _load(inter, "historical_model", required=False)
    historical_rebase = _load(inter, "historical_rebase", required=False)
    area = _load(inter, "area", required=False)
    icefrac_per_model = _load(inter, "icefrac_per_model", required=False)
    icefrac_with_sai = _load(inter, "icefrac_with_sai", required=False)
    icefrac_with_sai_per_model = _load(inter, "icefrac_with_sai_per_model", required=False)
    regional_map_per_model = _load(inter, "regional_map_per_model", required=False)
    historical_model_per_model = _load(inter, "historical_model_per_model", required=False)

    # --- p-values, all decades. No-SAI mirrors backend.baseline.get_regional_map_from_global_temp
    # (absolute field vs 1850-1900 mean); with-SAI mirrors backend.get_outputs._compute_sai_p_values.
    # Zero delta before the SAI start, as get_outputs does (with-SAI == no-SAI there).
    regional_delta = regional_delta.reindex(time=regional_map.time, fill_value=0.0)
    if icefrac_with_sai is not None:
        icefrac_with_sai = icefrac_with_sai.combine_first(regional_map)
    if historical_rebase is not None:
        if var in REBASED_VARS:
            absolute_no_sai = regional_map + historical_rebase
            absolute_sai = regional_map + regional_delta + historical_rebase
        elif var == "icefrac":
            absolute_no_sai = regional_map
            absolute_sai = icefrac_with_sai
        else:
            absolute_no_sai = regional_map
            absolute_sai = regional_map + regional_delta
        _save(out, "p_no_sai", get_grid_level_p_values(var, data_dir, historical_rebase, absolute_no_sai))
        _save(out, "p_sai", get_grid_level_p_values(var, data_dir, historical_rebase, absolute_sai))

    # --- mean over time + band + internal variability + histogram (mirrors get_outputs, global mode)
    start_year = params.start_year
    recent_regional_map = regional_map.sel(time=slice(2015, start_year - 1))
    regional_map_post = regional_map.sel(time=slice(start_year, SIM_END_YEAR))

    per_model_field = icefrac_per_model if var == "icefrac" else regional_map_per_model
    sigma_internal_per_model = None
    if per_model_field is not None and "model" in per_model_field.dims:
        try:
            sigma_internal_per_model = _internal_variability_per_model(
                var, [str(m) for m in per_model_field.model.values],
                regional_map=regional_map_post, spatial_gdf=None, spatial_item=None,
                data_dir=data_dir,
            )
        except FileNotFoundError as e:  # per-model artifacts not generated yet: no bands
            print(f"  (no combined bands: {e})")

    mot = _compute_mean_over_time(
        var,
        regional_map=regional_map_post,
        regional_delta=regional_delta,
        recent_regional_map=recent_regional_map,
        historical_model=historical_model,
        area=area,
        spatial_gdf=None,
        spatial_item=None,
        start_year=start_year,
        variable_injection=None,
        data_dir=data_dir,
        icefrac_per_model=icefrac_per_model,
        icefrac_with_sai_per_model=icefrac_with_sai_per_model,
        regional_map_per_model=regional_map_per_model,
        historical_model_per_model=historical_model_per_model,
        sigma_internal_per_model=sigma_internal_per_model,
    )
    series = {
        "no_sai": mot.mean_no_sai,
        "with_sai": mot.mean_with_sai,
        "historical_model": mot.historical_model_global_mean,
    }
    band_scalars: dict = {}
    if mot.bands is not None:
        for k, v in mot.bands.items():
            if isinstance(v, xr.DataArray):
                series[k] = v
            elif k == "per_model_series":
                xr.Dataset({name: da for name, da in v.items()}).to_netcdf(out / "per_model_series.nc")
            else:
                band_scalars[k] = v
    xr.Dataset({k: v for k, v in series.items() if v is not None}).to_netcdf(
        out / "mean_over_time.nc"
    )

    internal_variability = _compute_internal_variability(
        var, regional_map=regional_map_post, spatial_gdf=None, spatial_item=None, data_dir=data_dir
    )
    distribution = _compute_distribution(
        var,
        regional_map=regional_map_post,
        regional_delta=regional_delta,
        historical_model=historical_model,
        masked_regional_map=None,
        masked_regional_delta=None,
        masked_historical_model=None,
        decade_start_year=params.decade_start_year,
        decade_end_year=params.decade_end_year,
        spatial_gdf=None,
        spatial_item=None,
    )
    (out / "quantities.json").write_text(
        json.dumps(
            {
                "config": config,
                "sha": sha,  # label used for the directory (may be a name like "baseline")
                "git_head": git_sha(),  # actual code revision, for provenance
                "var": var,
                "params": asdict(params),
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "internal_variability": float(np.asarray(internal_variability)),
                "band_keys": sorted(k for k in series if k.startswith("band_") or k.startswith("sigma_model")),
                "bands": band_scalars,
                "distribution": distribution,
            },
            indent=2,
        )
        + "\n"
    )
    return out
