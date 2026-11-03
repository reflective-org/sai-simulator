"""Recompute each SAI anchor's share of the pooled delta slope from the processed tas files.

``scripts/fit_delta.py`` fits ``beta = sum(x_i y_i) / sum(x_j^2)`` through the origin over all
anchors, so anchor *i* carries weight ``x_i^2 / sum(x_j^2)`` in the final slope, where ``x_i``
is its global-mean tas cooling. That weight is what "influence" means throughout the review.
The same ``x`` values apply to every variable, since ``x`` is always the tas cooling.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_delta import ANCHORS, _global_mean, _load_anchor_2050_2069  # noqa: E402


def anchor_leverages(data_dir: str | Path) -> tuple[list[dict], dict[str, float]]:
    """Return per-anchor rows (name, model, cooling_K, share) and per-model share totals.

    Anchors whose tas files are absent are skipped, exactly as the fit skips them.
    """
    tas_dir = Path(data_dir) / "tas"
    rows: list[dict] = []
    for anchor in ANCHORS:
        pair = _load_anchor_2050_2069(tas_dir, anchor)
        if pair is None:
            continue
        exp_tas, base_tas = pair
        x = float((_global_mean(exp_tas) - _global_mean(base_tas))["tas"].item())
        rows.append({"anchor": anchor.name, "model": anchor.model, "cooling_K": x})
    total = sum(r["cooling_K"] ** 2 for r in rows)
    for r in rows:
        r["share"] = r["cooling_K"] ** 2 / total if total else float("nan")
    per_model: dict[str, float] = {}
    for r in rows:
        per_model[r["model"]] = per_model.get(r["model"], 0.0) + r["share"]
    return rows, per_model


def format_table(rows: list[dict], per_model: dict[str, float]) -> str:
    lines = [f"{'anchor':16s} {'model':12s} {'cooling (K)':>12s} {'share':>8s}"]
    for r in rows:
        lines.append(
            f"{r['anchor']:16s} {r['model']:12s} {r['cooling_K']:12.3f} {100 * r['share']:7.1f}%"
        )
    lines.append("")
    for model, share in per_model.items():
        lines.append(f"{model:29s} {100 * share:7.1f}%")
    return "\n".join(lines)
