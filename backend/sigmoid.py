"""Decreasing-logistic fitting utilities for the icefrac response model.

Provides :func:`logistic` (the four-parameter decreasing logistic used to relate Arctic
temperature anomalies to sea-ice fraction) together with the :class:`LogisticFitter` driver
that fits one curve per ``(lon, lat, cell)`` triple and the :class:`LogisticFitResult`
container used to ship per-cell fit diagnostics back to the orchestrator.
"""
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from scipy.special import expit


def logistic(
    x: np.ndarray | float,
    L: float,
    x0: float,
    k: float,
    b: float,
) -> np.ndarray | float:
    """Evaluate the four-parameter decreasing logistic ``b + L * sigmoid(-(x - x0) * k)``.

    Uses :func:`scipy.special.expit` so the sigmoid remains numerically stable for large
    ``|x - x0| * k``. The minus sign on the exponent makes the curve decreasing for ``k > 0``,
    matching the temperature → ice-fraction relationship.
    """
    # expit(z) = 1/(1+exp(-z)) is stable for large |z|
    return b + L * expit(-(x - x0) * k) # minus gives decreasing curve for k>0

@dataclass
class LogisticFitResult:
    """Per-cell logistic-fit outcome returned by :meth:`LogisticFitter._fit_one`.

    Carries the grid indices ``(i, j, c)``, the fitted ``params`` (``[L, x0, k, b]``), the
    error metrics (``rmse``, ``mae``, ``r2``), an ``ok`` flag, and an optional ``error``
    description when the fit failed (e.g., ``"insufficient_points"`` or a stringified
    exception).
    """
    i: int
    j: int
    c: int
    params: np.ndarray
    rmse: float
    mae: float
    r2: float
    ok: bool
    error: str | None = None

class LogisticFitter:
    """Driver that fits a decreasing logistic to every ``(lon, lat, cell)`` slot in a 4-D grid.

    Owns the input arrays plus pre-allocated output buffers (``params``, ``rmse``, ``mae``,
    ``r2``) and runs the fits either serially or via a thread pool. Designed so the per-cell
    work (:meth:`_fit_one`) is trivially parallelizable since each cell is independent.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        X: (n_lon, n_lat, n_time)
        Y: (n_lon, n_lat, n_time, n_cells)
        """
        self.X = X
        self.Y = Y
        self.dtype = dtype

        self.n_lon, self.n_lat, self.n_time = X.shape
        assert Y.shape[:3] == (self.n_lon, self.n_lat, self.n_time)
        self.n_cells = Y.shape[3]

        # allocate outputs
        self.params = np.full((4, self.n_lon, self.n_lat, self.n_cells), np.nan, dtype=dtype)
        self.rmse   = np.full((self.n_lon, self.n_lat, self.n_cells), np.nan, dtype=dtype)
        self.mae    = np.full((self.n_lon, self.n_lat, self.n_cells), np.nan, dtype=dtype)
        self.r2     = np.full((self.n_lon, self.n_lat, self.n_cells), np.nan, dtype=dtype)

    @staticmethod
    def _initial_guess(x: np.ndarray, y: np.ndarray) -> list[float]:
        """Build a robust starting point ``[L, x0, k, b]`` for :func:`scipy.optimize.curve_fit`.

        ``b`` starts at ``min(y)``, ``L`` at the y-range (floored at ``1e-6`` so the fit isn't
        born stuck at zero amplitude), ``x0`` at the median of ``x``, and ``k`` at the inverse
        of ``std(x)`` so the sigmoid's transition width matches the data's x-spread.
        """
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        b0 = y_min
        L0 = max(y_max - y_min, 1e-6)
        x00 = float(np.nanmedian(x))
        x_std = float(np.nanstd(x))
        k0 = 1.0 / (x_std + 1e-6)
        return [L0, x00, k0, b0]

    def _fit_one(self, i: int, j: int, c: int) -> LogisticFitResult:
        """Fit the logistic for a single ``(i, j, c)`` grid cell and return its diagnostics.

        NaN-masks the per-cell time series, falls back to a no-fit result tagged
        ``"insufficient_points"`` when fewer than 10 valid samples remain, and otherwise
        runs a bounded :func:`scipy.optimize.curve_fit` (L ∈ [0, 1], x0 ∈ [min(x), max(x)],
        k ≥ 0, b ∈ [0, 1]). The ``ok=False`` branch also catches generic exceptions and
        stores their stringified message so the caller can decide how to surface failures.
        """
        try:
            y = self.Y[i, j, :, c]
            x = self.X[i, j, :]

            mask = (~np.isnan(y)) & (~np.isnan(x))
            x_clean, y_clean = x[mask], y[mask]

            if x_clean.size < 10:
                return LogisticFitResult(i, j, c, np.full(4, np.nan), np.nan, np.nan, np.nan, False, "insufficient_points")

            p0 = self._initial_guess(x_clean, y_clean)

            lower = [0.0, float(np.min(x_clean)), 0.0, 0.0]
            upper = [1.0, float(np.max(x_clean)), np.inf, 1.0]

            popt, _ = curve_fit(
                logistic, x_clean, y_clean,
                p0=p0, bounds=(lower, upper),
                maxfev=100000
            )

            y_pred = logistic(x_clean, *popt)

            rmse = float(np.sqrt(np.mean((y_clean - y_pred) ** 2)))
            mae  = float(np.mean(np.abs(y_clean - y_pred)))
            ss_tot = float(np.sum((y_clean - np.mean(y_clean)) ** 2))
            if ss_tot == 0.0:
                r2 = 1.0 if np.allclose(y_clean, y_pred) else 0.0
            else:
                ss_res = float(np.sum((y_clean - y_pred) ** 2))
                r2 = 1.0 - ss_res / ss_tot

            return LogisticFitResult(i, j, c, np.asarray(popt), rmse, mae, r2, True)

        except Exception as e:
            return LogisticFitResult(i, j, c, np.full(4, np.nan), np.nan, np.nan, np.nan, False, str(e))

    def fit_all(
        self,
        n_jobs: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit every ``(i, j, c)`` cell and return the populated ``(params, rmse, mae, r2)`` arrays.

        ``n_jobs == 1`` (or ``0``) runs serially; any other value uses a
        :class:`concurrent.futures.ThreadPoolExecutor` with ``max_workers=n_jobs`` (or
        ``None`` for the default when ``n_jobs < 0``). A :mod:`tqdm` progress bar tracks
        completion in both modes.
        """
        tasks = [(i, j, c)
                 for i in range(self.n_lon)
                 for j in range(self.n_lat)
                 for c in range(self.n_cells)]

        if n_jobs and n_jobs != 1:
            with ThreadPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as ex:
                futures = [ex.submit(self._fit_one, i, j, c) for (i, j, c) in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Fitting grid"):
                    res = fut.result()
                    self._store_result(res)
        else:
            for i, j, c in tqdm(tasks, desc="Fitting grid"):
                res = self._fit_one(i, j, c)
                self._store_result(res)

        return self.params, self.rmse, self.mae, self.r2

    def _store_result(self, res: LogisticFitResult) -> None:
        """Write a successful fit's params and error metrics into the pre-allocated output buffers.

        Failed fits (``res.ok is False``) are silently dropped so the corresponding cells
        keep their initial ``NaN`` values; the commented-out ``print`` is retained as a
        debugging hook.
        """
        if res.ok:
            self.params[:, res.i, res.j, res.c] = res.params
            self.rmse[res.i, res.j, res.c] = res.rmse
            self.mae[res.i, res.j, res.c]  = res.mae
            self.r2[res.i, res.j, res.c]   = res.r2
        # else:
        #     print(f"Error fitting grid {res.i}, {res.j}, {res.c}: {res.error}")
