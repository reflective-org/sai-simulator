"""Repository root and ``sys.path`` bootstrap for verification entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_sys_path() -> None:
    """Prepend ``REPO_ROOT`` to ``sys.path`` if missing so package imports (e.g. ``backend``) resolve."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
