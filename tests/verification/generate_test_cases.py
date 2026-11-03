"""
Generate random test cases from PARAMS with optional uniqueness constraints.

Import :func:`generate_test_cases` from tests (e.g. ``from tests.verification.generate_test_cases import generate_test_cases``).
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from tests.verification.params import PARAMS as DEFAULT_PARAMS


def _all_combinations(
    params: dict[str, list[Any]],
) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Return parameter keys and the Cartesian product of their value lists."""
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    combos = list(itertools.product(*value_lists))
    return keys, combos


def _sample_cases(
    keys: list[str],
    combos: list[tuple[Any, ...]],
    count: int,
    seed: int,
    allow_repeats: bool,
) -> list[dict[str, Any]]:
    """Draw ``count`` random combinations from ``combos`` and map them to dicts keyed by ``keys``."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    random.seed(seed)
    total_unique = len(combos)

    if not allow_repeats and count > total_unique:
        raise ValueError(
            f"Requested {count} unique cases but only {total_unique} unique "
            "combinations exist. Reduce count or pass allow_repeats=True."
        )

    if allow_repeats:
        selected = [random.choice(combos) for _ in range(count)]
    else:
        selected = random.sample(combos, k=count)

    return [dict(zip(keys, combo)) for combo in selected]


def generate_test_cases(
    count: int,
    *,
    seed: int = 42,
    allow_repeats: bool = False,
    params: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Return a list of randomly sampled parameter dicts from ``params``.

    Each dict has one value per key in the same order as ``params`` (typically
    ``ssp_scenario``, ``temp_target``, ``start_year``).

    Args:
        count: Number of test cases to return.
        seed: Random seed for reproducible sampling.
        allow_repeats: If True, allow duplicate combinations when ``count``
            exceeds the number of unique combinations.
        params: Parameter grid; defaults to :file:`tests/verification/params.py`.

    Returns:
        List of dictionaries, one per sampled combination.
    """
    grid = DEFAULT_PARAMS if params is None else params
    keys, combos = _all_combinations(grid)
    return _sample_cases(
        keys=keys,
        combos=combos,
        count=count,
        seed=seed,
        allow_repeats=allow_repeats,
    )
