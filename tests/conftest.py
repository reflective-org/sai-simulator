"""Root pytest config for the SAI test suite.

Prepends the repository root to :data:`sys.path` so the ``backend`` / ``scripts`` /
``fair_wrap`` packages resolve when running ``pytest`` from anywhere in the tree.
"""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
