"""get_regional_models averages each model's bootstrap emulators by coefficients: for linear
regressors that equals averaging their predictions, at 1/NUM_EMULATORS of the memory."""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

import backend.baseline as baseline
import backend.constants as constants


def _fit(rng, n_cells):
    x = rng.normal(size=(30, 1))
    y = x * rng.normal(size=(1, n_cells)) + rng.normal(size=(30, n_cells))
    return LinearRegression().fit(x, y)


def test_emulator_mean_by_coefficients_equals_mean_of_predictions(tmp_path, monkeypatch):
    n_emulators, n_cells = 4, 6
    monkeypatch.setattr(constants, "REGIONAL_MODEL_NAMES", ["A", "B"])
    monkeypatch.setattr(baseline, "NUM_EMULATORS", n_emulators)
    var_dir = tmp_path / "tas"
    var_dir.mkdir()
    rng = np.random.default_rng(0)
    fits = {m: [_fit(rng, n_cells) for _ in range(n_emulators)] for m in ("A", "B")}
    for m, regs in fits.items():
        for i, reg in enumerate(regs):
            joblib.dump(reg, var_dir / f"fair_to_smip_{m}_{i}.joblib")

    baseline.get_regional_models.cache_clear()
    stacked = baseline.get_regional_models(var_dir)
    x = rng.normal(size=(5, 1))
    got = stacked.predict(x).reshape(5, 2, n_cells)
    for j, m in enumerate(("A", "B")):
        expected = np.mean([reg.predict(x) for reg in fits[m]], axis=0)
        np.testing.assert_allclose(got[:, j], expected, rtol=1e-12, atol=1e-12)
    # one regression per model, not one per emulator
    assert stacked.coef_.shape[0] == 2 * n_cells
    baseline.get_regional_models.cache_clear()
