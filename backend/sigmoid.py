import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.special import expit


def logistic(x, L, x0, k, b):
    # expit(z) = 1/(1+exp(-z)) is stable for large |z|
    return b + L * expit(-(x - x0) * k) # minus gives decreasing curve for k>0

@dataclass
class LogisticFitResult:
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
    def __init__(self, X, Y, dtype=np.float64):
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
    def _initial_guess(x, y):
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        b0 = y_min
        L0 = max(y_max - y_min, 1e-6)
        x00 = float(np.nanmedian(x))
        x_std = float(np.nanstd(x))
        k0 = 1.0 / (x_std + 1e-6)
        return [L0, x00, k0, b0]

    def _fit_one(self, i, j, c):
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

    def fit_all(self, n_jobs=0):
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

    def _store_result(self, res: LogisticFitResult):
        if res.ok:
            self.params[:, res.i, res.j, res.c] = res.params
            self.rmse[res.i, res.j, res.c] = res.rmse
            self.mae[res.i, res.j, res.c]  = res.mae
            self.r2[res.i, res.j, res.c]   = res.r2
        # else:
        #     print(f"Error fitting grid {res.i}, {res.j}, {res.c}: {res.error}")
