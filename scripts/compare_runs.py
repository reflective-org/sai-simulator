"""CLI for the before/after comparison harness (see ``scripts/compare/``).

Examples::

    # Persist intermediates + audited quantities for every dashboard variable, one config
    python scripts/compare_runs.py dump --config combined
    python scripts/compare_runs.py dump --config cesm-only --variables tas,icefrac

    # Re-run only the audited quantities against the stored intermediates (after a code change)
    python scripts/compare_runs.py dump --config combined --stage quantities

    # Figures: several configs at one sha (model isolation) ...
    python scripts/compare_runs.py render_matrix --configs cesm-only,miroc-only,combined --step baseline
    # ... or several shas of one config (before/after a fix)
    python scripts/compare_runs.py render_matrix --configs combined --shas <before>,<after> --step fix-U1

    # Each SAI anchor's weight in the pooled delta slope, from the real processed files
    python scripts/compare_runs.py leverage

Defaults are the dashboard's own request (SSP2-4.5, target 1.0, start 2035, ramp-up 10,
decade 2091-2100). ``--scenarios all`` covers every SSP; ``--variables all`` every
``VAR2INFO`` entry. Outputs land under ``--root`` (default ``harness/``, gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare.configs import CONFIGS, POOLED_DELTA_CAVEAT  # noqa: E402
from scripts.compare.dump import (  # noqa: E402
    ALL_SCENARIOS,
    ALL_VARIABLES,
    RunParams,
    compute_quantities,
    dump_intermediates,
    git_sha,
)


def _split(value: str | list | tuple, everything: tuple[str, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if value == "all":
        return list(everything)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def dump(
    config: str = "combined",
    variables: str = "all",
    scenarios: str = "SSP2-4.5",
    temp_target: float = 1.0,
    start_year: int = 2035,
    ramp_up: int = 10,
    decade: str = "2091-2100",
    data_dir: str = "data/processed",
    model_dir: str = "data/models",
    root: str = "harness",
    stage: str = "both",
    force: bool = False,
    sha: str | None = None,
) -> None:
    """Persist intermediates and/or audited quantities for one config.

    ``stage`` is ``intermediates``, ``quantities`` or ``both``. ``force`` recomputes
    intermediates that already exist. ``sha`` overrides the git sha used to label the
    quantities directory (default: current HEAD, ``-dirty`` when tracked files are modified).
    """
    if config not in CONFIGS:
        raise SystemExit(f"config must be one of {sorted(CONFIGS)}, got {config!r}")
    if stage not in ("intermediates", "quantities", "both"):
        raise SystemExit(f"stage must be intermediates|quantities|both, got {stage!r}")
    decade_start, decade_end = (int(y) for y in str(decade).split("-"))
    data_dir_p, model_dir_p, root_p = Path(data_dir), Path(model_dir), Path(root)
    label = sha or git_sha()

    print(POOLED_DELTA_CAVEAT)
    print(f"config={config} models={list(CONFIGS[config])} sha={label} root={root_p}")

    for scenario in _split(scenarios, ALL_SCENARIOS):
        params = RunParams(
            ssp_scenario=scenario,
            temp_target=float(temp_target),
            start_year=int(start_year),
            ramp_up=int(ramp_up),
            decade_start_year=decade_start,
            decade_end_year=decade_end,
        )
        for var in _split(variables, ALL_VARIABLES):
            if stage in ("intermediates", "both"):
                out = dump_intermediates(config, var, params, data_dir_p, model_dir_p, root_p, force=force)
                print(f"  intermediates  {params.key:36s} {var:20s} -> {out}")
            if stage in ("quantities", "both"):
                out = compute_quantities(config, var, params, data_dir_p, root_p, sha=label)
                print(f"  quantities     {params.key:36s} {var:20s} -> {out}")


def render(
    inputs: str,
    out_dir: str,
    labels: str | None = None,
) -> None:
    """Render the figure set for N quantity dirs (same variable) into ``out_dir``.

    ``inputs`` is a comma-separated list of ``<root>/<config>/<sha>/<key>/<var>`` dirs;
    ``labels`` an optional comma-separated list of the same length (default: config@sha).
    Figures: ``<var>_pvalues_fraction.png``, ``<var>_pvalues_masks.png``,
    ``<var>_series.png``, ``<var>_histogram.png``, plus ``<var>_summary.json``.
    """
    from scripts.compare.render import render as _render

    dirs = _split(inputs, ())
    labs = _split(labels, ()) if labels else None
    out = _render(dirs, out_dir, labels=labs)
    print(f"wrote {sorted(p.name for p in out.iterdir())} -> {out}")


def render_matrix(
    variables: str = "all",
    configs: str = "combined",
    shas: str | None = None,
    key: str = "ssp245_t1.0_y2035_r10_d2091-2100",
    root: str = "harness",
    out_dir: str = "figs",
    step: str = "compare",
) -> None:
    """Render one figure set per variable, comparing configs and/or shas.

    Exactly one of ``configs`` / ``shas`` should list more than one entry: several configs at
    one sha (model isolation) or several shas of one config (before/after a fix). ``shas``
    defaults to the current HEAD. Output: ``<out_dir>/<step>/<var>/``.
    """
    from scripts.compare.render import render as _render

    cfgs = _split(configs, tuple(CONFIGS))
    sha_list = _split(shas, ()) if shas else [git_sha()]
    for var in _split(variables, ALL_VARIABLES):
        dirs, labs = [], []
        for cfg in cfgs:
            for sha in sha_list:
                d = Path(root) / cfg / sha / key / var
                if not (d / "quantities.json").exists():
                    print(f"  skip {var}: no dump at {d}")
                    continue
                dirs.append(str(d))
                labs.append(cfg if len(sha_list) == 1 else (sha if len(cfgs) == 1 else f"{cfg}@{sha}"))
        if dirs:
            out = _render(dirs, Path(out_dir) / step / var, labels=labs)
            print(f"  {var:20s} -> {out}")


def sigma_options(
    variables: str = "tas,tasmax,p-e,exposure_above_35,exposure_above_10,icefrac",
    config: str = "combined",
    baseline_sha: str = "baseline",
    key: str = "ssp245_t1.0_y2035_r10_d2091-2100",
    root: str = "harness",
    out_dir: str = "figs/p3-sigma-options",
    data_dir: str = "data/processed",
) -> None:
    """Per-variable what-if maps for the p-value test sigma (internal vs between-model)."""
    from scripts.compare.sigma_options import sigma_options as _sigma_options

    for var in _split(variables, ALL_VARIABLES):
        summary = _sigma_options(
            var, Path(root) / config / "intermediates" / key / var,
            Path(root) / config / baseline_sha / key / var, out_dir, data_dir=data_dir,
        )
        pct = summary["percent_significant"]
        print(f"{var:18s} " + " | ".join(f"{k.split(':')[0]}={v:5.1f}%" for k, v in pct.items()))


def leverage(data_dir: str = "data/processed") -> None:
    """Print each anchor's cooling and its x^2 share of the pooled SAI-delta slope."""
    from scripts.compare.leverage import anchor_leverages, format_table

    rows, per_model = anchor_leverages(data_dir)
    print(format_table(rows, per_model))


if __name__ == "__main__":
    import fire  # CLI-only, matching the other scripts

    fire.Fire({"dump": dump, "render": render, "render_matrix": render_matrix, "sigma_options": sigma_options, "leverage": leverage})
