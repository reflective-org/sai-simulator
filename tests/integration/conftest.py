import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.constants import REGIONAL_MODEL_NAMES, resolve_projection_baseline_file  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "data" / "models"

# The artifacts every test here needs, for every model in the roster: processed tas,
# per-model variability and the fitted emulators. Anything missing => skip the whole
# module set (e.g. a data dir regenerated before a model joined the roster).
_REQUIRED = [
    MODEL_DIR / "icefrac_arctic" / "interpolator.nc",
    DATA_DIR / "tas" / "output_gauss-baseline.nc",
]
for _model in REGIONAL_MODEL_NAMES:
    _REQUIRED += [
        resolve_projection_baseline_file(DATA_DIR / "tas", _model) or DATA_DIR / "tas" / f"<no baseline for {_model}>",
        DATA_DIR / "tas" / f"model_internal_variability_{_model}.npy",
        DATA_DIR / "tas" / f"grid_level_lag1_autocorr_{_model}.nc",
        MODEL_DIR / "tas" / f"fair_to_smip_{_model}_0.joblib",
    ]


def pytest_collection_modifyitems(config, items):
    missing = [p for p in _REQUIRED if not p.exists()]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"real data not present locally (missing e.g. {missing[0].relative_to(REPO_ROOT)})")
    for item in items:
        if Path(str(item.fspath)).is_relative_to(Path(__file__).parent):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return MODEL_DIR


@pytest.fixture(scope="session")
def scratch_cache(tmp_path_factory) -> Path:
    """A fresh cache dir; validate_cache_dir stamps it for this process."""
    return tmp_path_factory.mktemp("cache")
