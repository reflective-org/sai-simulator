"""
Fixtures for verification tests. Configure via environment variables.

**S3 verification** (``test_verification_s3``):

- ``VALIDATION_S3_BUCKET`` — required to run (otherwise test is skipped)
- ``VALIDATION_S3_PREFIX`` — key prefix inside the bucket (default empty)
- ``VALIDATION_S3_FILETYPE`` — ``geotiffs`` or ``plot_data`` (default ``geotiffs``)
- ``VALIDATION_DATA_DIR``, ``VALIDATION_MODEL_DIR``, ``VALIDATION_CACHE_DIR`` — paths (defaults: ``data/processed``, ``data/models``, ``data/cache``)
- ``VALIDATION_CASES_JSON`` — optional path to a JSON case list; if unset, cases are **generated in memory** on each test run via :func:`tests.verification.generate_test_cases.generate_test_cases` (never written to disk).
- ``VALIDATION_LIMIT`` — how many random cases to generate (default ``1``) when not using a JSON file
- ``VALIDATION_GENERATE_COUNT`` — if set, overrides ``VALIDATION_LIMIT`` as the number of cases to generate
- ``VALIDATION_SEED`` — RNG seed for sampling cases (default ``42``). Set to ``random`` for a new seed each pytest run.
- ``VALIDATION_RAMP_UP`` — default ``10``

**Pytest flags** (override env defaults):

- ``--validation-limit N``
- ``--validation-generate-count N``
- ``--validation-seed N``
- ``--validation-print-timing`` — print total + per-case seconds (also set ``VALIDATION_PRINT_TIMING=1``)
- ``--validation-s3-filetype {geotiffs|plot_data}``

**Parallel verification (default: Celery)**

For multiple cases in parallel with bounded worker memory, use Celery (not pytest workers):

1. Start Redis (or your broker) and a worker with limited concurrency, e.g.::

    celery -A tests.verification.celery_app worker --loglevel=INFO --concurrency=2

2. Dispatch cases (set broker URLs via ``VALIDATION_CELERY_BROKER_URL`` / ``VALIDATION_CELERY_RESULT_BACKEND`` if needed)::

    python tests/verification/run_verification_celery.py --bucket <bucket> --limit 20

**Pytest** runs verification **sequentially** (parametrized ``s3_verification`` items or ``s3_verification_batch``). Use that for local debugging or CI without a Celery stack.

**Pytest + Celery task path** (``s3_verification_celery``):

Runs the same ``verification.verify_case_task`` as the worker CLI. If the task fails (verification error or exception), pytest fails.

- With a **running worker** + Redis: leave ``VALIDATION_CELERY_EAGER`` unset (default).
- **Without** a worker (CI / quick check): set ``VALIDATION_CELERY_EAGER=1`` so tasks run in-process and errors still propagate to pytest.

Example::

    VALIDATION_S3_BUCKET=my-bucket VALIDATION_CELERY_EAGER=1 \\
      pytest tests/verification/test_verification_s3_celery.py -m s3_verification_celery -v

Batch (one test, all cases)::

    pytest tests/verification/test_verification_s3_celery.py -m s3_verification_celery_batch -v

"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import pytest

from tests.verification.env import (
    env_float,
    env_int,
    env_optional_int,
    env_path,
    env_str,
    parse_verification_seed,
)
from tests.verification.repo_root import REPO_ROOT
from tests.verification.validate_outputs import VALIDATION_VARIABLE_POOL, VerificationConfig


def _resolve_verification_seed(config: pytest.Config) -> int:
    """Integer seed for ``generate_test_cases``. Use ``random`` / ``rand`` for a new seed each run."""
    cli = config.getoption("--validation-seed")
    env = env_str("VALIDATION_SEED", "42")
    raw = cli if cli is not None else env
    return parse_verification_seed(str(raw))


def _load_cases_for_config(config: pytest.Config) -> list[dict[str, Any]]:
    """Load or generate cases once per pytest session (parametrization / batch fixture)."""
    from tests.verification.verification_runner import resolve_cases

    cases_path_env = os.environ.get("VALIDATION_CASES_JSON")
    cases_path = Path(cases_path_env) if cases_path_env else None

    cli_gen = config.getoption("--validation-generate-count")
    if cli_gen is not None:
        gen_n = int(cli_gen)
    else:
        gen_n = env_optional_int("VALIDATION_GENERATE_COUNT")

    cli_limit = config.getoption("--validation-limit")
    env_limit = env_str("VALIDATION_LIMIT", "1")
    limit_n = int(cli_limit if cli_limit is not None else env_limit)

    seed = _resolve_verification_seed(config)

    return resolve_cases(
        cases_path=cases_path,
        generate_count=gen_n,
        limit=limit_n,
        seed=seed,
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize single-case S3 verification tests with the resolved case list."""
    _CELERY_PARAMETRIZED = (
        "test_get_outputs_matches_s3_reference_single_case",
        "test_get_outputs_matches_s3_reference_celery_single_case",
    )
    if metafunc.definition.name not in _CELERY_PARAMETRIZED:
        return
    cases = _load_cases_for_config(metafunc.config)
    ids = [
        f"{i}-{c['ssp_scenario']}-{c['temp_target']}-{c['start_year']}"
        for i, c in enumerate(cases)
    ]
    metafunc.parametrize("verification_case", cases, ids=ids)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for verification limits, seed, timing, and S3 filetype."""
    group = parser.getgroup("validation")
    group.addoption(
        "--validation-limit",
        action="store",
        default=None,
        help="Max verification cases to run (overrides VALIDATION_LIMIT).",
    )
    group.addoption(
        "--validation-generate-count",
        action="store",
        default=None,
        help="Generate N random cases (overrides VALIDATION_GENERATE_COUNT).",
    )
    group.addoption(
        "--validation-seed",
        action="store",
        default=None,
        help="Seed for generated cases (overrides VALIDATION_SEED).",
    )
    group.addoption(
        "--validation-print-timing",
        action="store_true",
        default=False,
        help="Print total + per-case timing summary.",
    )
    group.addoption(
        "--validation-s3-filetype",
        action="store",
        default=None,
        help="geotiffs or plot_data (overrides VALIDATION_S3_FILETYPE).",
    )


@pytest.fixture
def repo_root() -> Path:
    """Absolute filesystem path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def verification_config(
    request: pytest.FixtureRequest, repo_root: Path
) -> VerificationConfig:
    """Build :class:`~tests.verification.validate_outputs.VerificationConfig` from env and CLI (or skip)."""
    bucket = os.environ.get("VALIDATION_S3_BUCKET")
    if not bucket:
        pytest.skip("Set VALIDATION_S3_BUCKET to run S3 verification tests")

    data_dir = env_path("VALIDATION_DATA_DIR", repo_root / "data/processed")
    model_dir = env_path("VALIDATION_MODEL_DIR", repo_root / "data/models")
    cache_dir = env_path("VALIDATION_CACHE_DIR", repo_root / "data/cache")

    for label, p in ("data", data_dir), ("model", model_dir), ("cache", cache_dir):
        if not p.is_dir():
            pytest.skip(f"Missing {label} directory: {p}")

    prefix = (os.environ.get("VALIDATION_S3_PREFIX") or "").strip()
    cli_ft = request.config.getoption("--validation-s3-filetype")
    s3_filetype = cli_ft or env_str("VALIDATION_S3_FILETYPE", "geotiffs")
    variables = [random.choice(VALIDATION_VARIABLE_POOL)]
    decade = "2091-2100"
    ramp_up = env_int("VALIDATION_RAMP_UP", 10)
    epsilon = env_float("VALIDATION_EPSILON", 1e-12)

    return VerificationConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        cache_dir=cache_dir,
        s3_bucket=bucket,
        s3_prefix=prefix,
        s3_filetype=s3_filetype,
        decade_visualization=decade,
        ramp_up=ramp_up,
        variables=variables,
        epsilon=epsilon,
    )


@pytest.fixture
def verification_cases(request: pytest.FixtureRequest) -> list[dict[str, Any]]:
    """All verification parameter dicts for the current session (same resolution as parametrization)."""
    request.getfixturevalue("verification_config")
    return _load_cases_for_config(request.config)
