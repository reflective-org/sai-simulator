"""Model-subset configs for the comparison harness.

A *config* names which climate models feed the backend: ``combined`` (both), ``cesm-only``,
``miroc-only``. The backend has no runtime switch for this, so the harness does two things
per config — neither touches production code:

1. **A reduced ``model_dir``** (:func:`build_model_dir`): symlinks to only the selected
   models' emulator joblibs, the pooled SAI-delta ``interpolator.nc`` files as-is, and an
   icefrac interpolator re-cut to the selected models.
2. **Rebinding the model list** (:func:`active_models`): ``backend.baseline`` and
   ``backend.historical`` capture ``REGIONAL_MODEL_NAMES`` at import time, so for the
   duration of a run the harness rebinds those module attributes and clears every
   ``lru_cache`` loader that could otherwise serve another config's arrays.

The reduced directory is the safety net: if the rebinding ever misses a consumer, the loader
asks for a joblib the directory does not contain and fails with ``FileNotFoundError``
instead of silently mixing models.

What a config **cannot** isolate: the SAI delta. ``scripts/fit_delta.py`` fits one pooled
slope over every model's anchors and writes no ``model`` dim, so every config — including
``cesm-only`` and ``miroc-only`` — applies the same combined delta. See
:data:`POOLED_DELTA_CAVEAT`.
"""
from __future__ import annotations

import json
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import xarray as xr

ALL_MODELS: tuple[str, ...] = ("CESM2-WACCM", "MIROC-ES2H")

CONFIGS: dict[str, tuple[str, ...]] = {
    "combined": ALL_MODELS,
    "cesm-only": ("CESM2-WACCM",),
    "miroc-only": ("MIROC-ES2H",),
}

POOLED_DELTA_CAVEAT = (
    "NOTE: the SAI delta slope (data/models/<var>/interpolator.nc) is ONE pooled fit over "
    "all models' anchors with no model dimension, so it is identical in every config. "
    "cesm-only / miroc-only isolate the no-SAI projection map, the historical reference and "
    "the sea-ice sigmoids -- NOT the SAI response. See scripts/fit_delta.py."
)

_JOBLIB_RE = re.compile(r"^fair_to_smip_(?P<model>.+)_(?P<i>\d+)\.joblib$")
_MARKER = "config.json"


def _recut_icefrac_interpolator(src: Path, dst: Path, models: Sequence[str]) -> None:
    """Write ``src`` restricted to ``models`` along its ``model`` dim.

    A legacy single-model file (no ``model`` dim) is CESM-only by construction
    (see ``backend.icefrac.load_icefrac_params``) and can therefore only serve a
    CESM-only selection.
    """
    with xr.open_dataarray(src) as da:
        da = da.load()
    if "model" not in da.dims:
        if tuple(models) != ("CESM2-WACCM",):
            raise ValueError(
                f"{src} is a legacy single-model (CESM2-WACCM) icefrac interpolator and "
                f"cannot serve models={list(models)}. Re-run scripts/fit_icefrac_arctic.py."
            )
        shutil.copy2(src, dst)
        return
    present = {str(m) for m in da.model.values}
    missing = set(models) - present
    if missing:
        raise ValueError(
            f"{src} has models {sorted(present)}; requested {sorted(missing)} are absent."
        )
    da.sel(model=list(models)).to_netcdf(dst)


def build_model_dir(
    src_model_dir: str | Path,
    dst_model_dir: str | Path,
    models: Sequence[str],
    force: bool = False,
) -> Path:
    """Materialise a ``model_dir`` that contains only ``models``' artifacts.

    Emulator joblibs are symlinked per model; ``icefrac_arctic/interpolator.nc`` is re-cut;
    everything else (pooled delta interpolators, so2) is symlinked unchanged.

    The build is an **idempotent sync**: every call walks the source and adds whatever the
    destination lacks, so fitting a new variable after the first build (or re-fitting the
    icefrac interpolator) is picked up on the next run without rebuilding. Nothing is ever
    removed unless ``force`` is set, which wipes and rebuilds. A ``config.json`` marker records
    the selection; a marker for a different selection raises rather than silently serving
    the wrong models.
    """
    src = Path(src_model_dir).resolve()
    dst = Path(dst_model_dir)
    models = list(models)
    marker = dst / _MARKER
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if (recorded.get("models"), recorded.get("source")) != (models, str(src)):
            if not force:
                raise FileExistsError(
                    f"{dst} was built for {recorded}; requested models={models} from {src}. "
                    "Pass force=True to rebuild."
                )
    if force and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        out = dst / sub.name
        out.mkdir(exist_ok=True)
        for f in sorted(sub.iterdir()):
            target = out / f.name
            m = _JOBLIB_RE.match(f.name)
            if m:
                if m["model"] in models and not target.is_symlink():
                    target.symlink_to(f)
                continue
            if sub.name == "icefrac_arctic" and f.name == "interpolator.nc":
                # Re-cut when missing or when the source interpolator is newer (re-fit).
                if not target.exists() or f.stat().st_mtime > target.stat().st_mtime:
                    _recut_icefrac_interpolator(f, target, models)
                continue
            if not target.is_symlink():
                target.symlink_to(f)

    marker.write_text(json.dumps({"models": models, "source": str(src)}, indent=2) + "\n")
    return dst


def clear_backend_caches() -> None:
    """Clear every ``functools.lru_cache`` in the backend modules the harness drives.

    Path-keyed caches would mostly separate by the per-config ``model_dir``/``cache_dir``
    anyway; clearing all of them makes the config switch independent of that detail.
    """
    import backend.baseline
    import backend.get_outputs
    import backend.historical
    import backend.icefrac
    import backend.population
    import backend.utils

    for module in (
        backend.baseline,
        backend.get_outputs,
        backend.historical,
        backend.icefrac,
        backend.population,
        backend.utils,
    ):
        for obj in vars(module).values():
            if callable(getattr(obj, "cache_clear", None)):
                obj.cache_clear()


@contextmanager
def active_models(models: Sequence[str]) -> Iterator[list[str]]:
    """Rebind ``REGIONAL_MODEL_NAMES`` everywhere the backend captured it, for one run.

    ``backend.constants`` is rebound for anything that reads it live; ``backend.baseline``
    and ``backend.historical`` are rebound because they imported the name at module load
    (``get_regional_models`` iterates it, ``get_smip`` reshapes by its length,
    ``get_regional_map_from_global_temp`` uses it as the ``model`` coord, and
    ``get_historical_model`` passes it to ``_load_multi_model_baseline``). Caches are
    cleared on entry and exit.
    """
    import backend.baseline
    import backend.constants
    import backend.historical

    targets = (backend.constants, backend.baseline, backend.historical)
    saved = {mod: mod.REGIONAL_MODEL_NAMES for mod in targets}
    models = list(models)
    for mod in targets:
        mod.REGIONAL_MODEL_NAMES = models
    clear_backend_caches()
    try:
        yield models
    finally:
        for mod, original in saved.items():
            mod.REGIONAL_MODEL_NAMES = original
        clear_backend_caches()
