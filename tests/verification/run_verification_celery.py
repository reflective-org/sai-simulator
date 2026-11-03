#!/usr/bin/env python3
"""
Dispatch verification cases to Celery workers.

Default path for parallel S3 verification (bounded Celery worker concurrency).
Run fewer workers than CPU count to avoid OOM.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from tests.verification.repo_root import ensure_repo_on_sys_path

ensure_repo_on_sys_path()

# Imports below intentionally come after repo-root path setup.
from tests.verification.celery_tasks import config_to_payload, validate_case_task  # noqa: E402
from tests.verification.celery_app import celery_app  # noqa: E402
from tests.verification.env import env_float  # noqa: E402
from tests.verification.env import env_int  # noqa: E402
from tests.verification.env import env_path  # noqa: E402
from tests.verification.env import env_str  # noqa: E402
from tests.verification.env import parse_verification_seed  # noqa: E402
from tests.verification.validate_outputs import VerificationConfig  # noqa: E402
from tests.verification.verification_runner import resolve_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments for the Celery verification dispatcher."""
    parser = argparse.ArgumentParser(description="Run S3 verification with Celery workers.")
    parser.add_argument("--bucket", required=True, help="S3 bucket for reference artifacts.")
    parser.add_argument("--prefix", default=(os.environ.get("VALIDATION_S3_PREFIX") or "").strip())
    parser.add_argument("--filetype", default=env_str("VALIDATION_S3_FILETYPE", "geotiffs"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=env_path("VALIDATION_DATA_DIR", Path("data/processed")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=env_path("VALIDATION_MODEL_DIR", Path("data/models")),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=env_path("VALIDATION_CACHE_DIR", Path("data/cache")),
    )
    parser.add_argument("--decade", default=env_str("VALIDATION_DECADE", "2091-2100"))
    parser.add_argument("--ramp-up", type=int, default=env_int("VALIDATION_RAMP_UP", 10))
    parser.add_argument("--epsilon", type=float, default=env_float("VALIDATION_EPSILON", 1e-12))
    parser.add_argument("--variables", default=env_str("VALIDATION_VARIABLES", "tas"))
    parser.add_argument("--cases-json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=env_int("VALIDATION_LIMIT", 1))
    parser.add_argument("--generate-count", type=int, default=None)
    parser.add_argument("--seed", default=env_str("VALIDATION_SEED", "42"))
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--no-progress-timeout-seconds",
        type=float,
        default=env_float("VALIDATION_NO_PROGRESS_TIMEOUT_SECONDS", 600.0),
        help="Fail if no task completes within this many seconds.",
    )
    parser.add_argument(
        "--dispatch-chunk",
        type=int,
        default=env_int("VALIDATION_DISPATCH_CHUNK", 80),
        help="Max tasks queued at once (avoids huge broker/result spikes for large LIMIT).",
    )
    parser.add_argument("--output-json", type=Path, default=Path("validation_celery_report.json"))
    return parser.parse_args()


def _parse_variables(raw: str) -> list[str] | None:
    """Split comma-separated variable names, or return ``None`` when the list is empty."""
    vals = [v.strip() for v in str(raw).split(",") if v.strip()]
    return vals or None


def _inspect_summary_wall_limited(wall_s: float = 6.0) -> str:
    """Run Celery inspect calls without letting a stuck broker block the dispatcher forever."""

    def _gather() -> str:
        inspect = celery_app.control.inspect(timeout=2.0)
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        stats = inspect.stats() or {}
        return (
            f"workers={len(stats)} "
            f"active={sum(len(v) for v in active.values())} "
            f"reserved={sum(len(v) for v in reserved.values())}"
        )

    ex: ThreadPoolExecutor | None = None
    try:
        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="celery-inspect")
        fut = ex.submit(_gather)
        return fut.result(timeout=wall_s)
    except FuturesTimeout:
        return f"inspect_wall_timeout({wall_s}s)"
    except Exception as exc:  # pragma: no cover
        return f"inspect_error:{exc!r}"
    finally:
        if ex is not None:
            ex.shutdown(wait=False)


def _build_config(args: argparse.Namespace) -> VerificationConfig:
    """Construct :class:`~tests.verification.validate_outputs.VerificationConfig` from parsed CLI args."""
    return VerificationConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_dir=args.cache_dir,
        s3_bucket=args.bucket,
        s3_prefix=args.prefix,
        s3_filetype=args.filetype,
        decade_visualization=args.decade,
        ramp_up=args.ramp_up,
        variables=_parse_variables(args.variables),
        epsilon=args.epsilon,
    )


def main() -> None:
    """Dispatch all cases to workers, wait for results, write JSON report, exit non-zero on failure."""
    args = parse_args()
    print(
        "run_verification_celery: starting "
        f"(limit={args.limit}, dispatch_chunk={args.dispatch_chunk}, "
        f"task_default_queue={celery_app.conf.task_default_queue!r}, "
        f"broker_url_set={bool(env_str('VALIDATION_CELERY_BROKER_URL', '').strip())})",
        flush=True,
    )
    print("run_verification_celery: resolving cases...", flush=True)
    cases = resolve_cases(
        cases_path=args.cases_json,
        generate_count=args.generate_count,
        limit=args.limit,
        seed=parse_verification_seed(args.seed),
    )
    if not cases:
        raise SystemExit("No verification cases to run.")
    print(f"run_verification_celery: {len(cases)} case(s) to dispatch.", flush=True)

    cfg = _build_config(args)
    payload = config_to_payload(cfg)

    results: list[dict[str, Any] | None] = [None] * len(cases)
    failed = 0
    completed = 0
    chunk = max(1, int(args.dispatch_chunk))

    for chunk_start in range(0, len(cases), chunk):
        chunk_cases = cases[chunk_start : chunk_start + chunk]
        pending: list[tuple[int, dict[str, Any], Any]] = []
        for offset, case in enumerate(chunk_cases):
            idx = chunk_start + offset
            pending.append((idx, case, validate_case_task.delay(case, payload)))

        print(
            f"Dispatched chunk [{chunk_start}:{chunk_start + len(chunk_cases) - 1}] "
            f"({len(pending)} tasks), total_cases={len(cases)}.",
            flush=True,
        )

        last_heartbeat = time.time()
        last_completion = time.time()
        while pending:
            next_pending: list[tuple[int, dict[str, Any], Any]] = []
            completed_this_round = 0

            for idx, case, ar in pending:
                if not ar.ready():
                    next_pending.append((idx, case, ar))
                    continue

                row = ar.get(timeout=0, propagate=False)
                completed += 1
                completed_this_round += 1

                if ar.failed():
                    failed += 1
                    results[idx] = {
                        "case": case,
                        "ok": False,
                        "errors": [str(row)],
                        "comparisons": [],
                    }
                else:
                    results[idx] = row
                    if not row.get("ok", False):
                        failed += 1

                if completed % 25 == 0 or completed == len(cases):
                    print(
                        f"Progress: {completed}/{len(cases)} complete, failed={failed}",
                        flush=True,
                    )
                last_completion = time.time()

            pending = next_pending

            now = time.time()
            # Enforce stalled-task timeout *before* heartbeat/inspect. Inspect can block on a
            # wedged broker; running it first would prevent this check from ever firing.
            if pending and (now - last_completion >= args.no_progress_timeout_seconds):
                raise RuntimeError(
                    "No task completions within timeout window. "
                    "Likely worker/broker/result-backend mismatch, blocked tasks, or another "
                    "service consuming the same Redis queue. Ensure the worker listens on "
                    f"task_default_queue={celery_app.conf.task_default_queue!r} "
                    "(see VALIDATION_CELERY_QUEUE / celery worker -Q …)."
                )

            if pending and completed_this_round == 0 and (
                now - last_heartbeat >= args.heartbeat_seconds
            ):
                sample_states = []
                for _idx, _case, _ar in pending[:5]:
                    try:
                        sample_states.append(_ar.state)
                    except Exception as exc:  # pragma: no cover
                        sample_states.append(f"state_error:{exc!r}")

                inspect_summary = _inspect_summary_wall_limited(wall_s=6.0)

                print(
                    f"Heartbeat: chunk pending={len(pending)}; "
                    f"completed={completed}/{len(cases)} failed={failed}; "
                    f"sample_states={sample_states}; {inspect_summary}",
                    flush=True,
                )
                last_heartbeat = now

            if pending:
                time.sleep(max(args.poll_seconds, 0.1))

    report = {
        "total_cases": len(cases),
        "failed_cases": failed,
        "config": payload,
        "results": [r for r in results if r is not None],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report: {args.output_json}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
