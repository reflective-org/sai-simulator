import numpy as np
from scipy.optimize import curve_fit
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.optimize import curve_fit

def stretched_sigmoid(x, lam, beta):
    """Vectorised stretched sigmoid (λ, β, x0 all scalars)."""
    z = -x/lam    # z ≥ 0 for valid domain
    z = np.maximum(z, 0.0) # clip to domain  z ≥ 0
    return 1.0 - np.exp(-(z ** beta))

def stretched_sigmoid_x0(x, lam, beta, x0):
    """Vectorised stretched sigmoid (λ, β, x0 all scalars)."""
    z = -(x - x0) / lam    # z ≥ 0 for valid domain
    z = np.maximum(z, 0.0) # clip to domain  z ≥ 0
    return 1.0 - np.exp(-(z ** beta))

class SigmoidRegression:
    """
    A class for fitting a sigmoid regression model.
    """
    def __init__(self):
        self.X_ = None
        self.y_ = None

    def fit(self, X, y):
        """
        Fit the sigmoid regression model.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples,)
        y : np.ndarray, shape (n_samples, n_features)
        
        Returns
        -------
        reg : np.ndarray, shape (2, n_features)  [row0=λ̂, row1=β̂]
        """
        self.X_ = X
        self.y_ = y
        return self.fit_sigmoid_columns_2par(y, X)

    def _fit_column_2par(self, y_col, x_data, p0, bounds, maxfev):
        if not np.isfinite(y_col).all() or np.all(y_col == 0):
            return np.inf, np.inf              # sentinel

        popt, _ = curve_fit(
            stretched_sigmoid,
            x_data, y_col,
            p0=p0, bounds=bounds,
            maxfev=maxfev
        )
        return popt # (λ̂, β̂)

    def fit_sigmoid_columns_2par(
            self,
            Y,
            X,
            maxfev=10_000,
            n_jobs=None,
            tiny_thr=1e-12
        ):
        """
        Parameters
        ----------
        Y : np.ndarray, shape (n_time, n_cols)
        X : np.ndarray, shape (n_time,)
        Returns
        -------
        reg : np.ndarray, shape (2, n_cols)  [row0=λ̂, row1=β̂]
        """
        x_data = X.ravel().astype(float)
        n_cols = Y.shape[1]
        reg    = np.empty((2, n_cols), dtype=float)

        # ---- skip perfectly-zero columns quickly ---------------------------
        zero_mask = (np.abs(Y).max(axis=0) < tiny_thr)
        reg[:, zero_mask] = np.inf

        cols = np.where(~zero_mask)[0]
        if cols.size == 0:
            return reg

        # ---- shared p0 / bounds -------------------------------------------
        p0     = (np.percentile(-x_data, 75), 0.9)
        bounds = ([1e-9, 0.3],   [np.inf, 1.0])

        worker = partial(self._fit_column_2par,
                        x_data=x_data, p0=p0,
                        bounds=bounds, maxfev=maxfev)

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            res = pool.map(worker, (Y[:, i] for i in cols))

        reg[:, cols] = np.array(list(res)).T
        return reg    

class SigmoidRegression_x0:
    """
    A class for fitting a sigmoid regression model with intercept (x0).
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.x0_ = None

    def _fit_one_column(self, y_col, x_data, p0, bounds, maxfev):
        """
        Returns (λ, β, x0) for a single 1-D y column.
        """
        popt, _ = curve_fit(
            stretched_sigmoid_x0,
            x_data, y_col,
            p0=p0, bounds=bounds,
            maxfev=maxfev
        )
        return popt              # tuple length 3

    def fit(self, X, y, tiny_thr=1e-3, maxfev=10_000, n_jobs=None):
        """
        Fit the sigmoid regression model.
        
        Parameters
        ----------
        X : ndarray, shape (n_time,) or broadcastable to (n_time, 1)
            Predictor array.
        y : ndarray, shape (n_time, n_cols)
            Target array - each column is one time-series to fit.
        tiny_thr : float
            Columns whose mean < tiny_thr are treated as "all zeros"
            and assigned (λ=∞, β=∞, x0=0).
        maxfev : int
            Max function evaluations passed to `curve_fit`.
        n_jobs : int or None
            Number of worker processes. None → os.cpu_count().

        Returns
        -------
        self : SigmoidRegression_x0
            The fitted model.
        """
        # --- 1 set up shared x-data, initial guess, bounds ---------------------
        x_data = np.ascontiguousarray(X.squeeze())
        x_min = x_data.min()
        x_max = x_data.max()
        p0 = (np.ptp(x_data) / 2.0, 1.0, x_min)
        bounds = ((1e-9, 0.01, x_min - 1),
                 (np.inf, 10.0, x_max + 2))

        n_cols = y.shape[1]
        reg = np.empty((3, n_cols), dtype=float)

        # --- 2 skip tiny columns in vectorised fashion -------------------------
        tiny_mask = y.mean(axis=0) < tiny_thr
        reg[:, tiny_mask] = np.array([np.inf, np.inf, 0.0]).reshape(3, 1)

        cols_to_fit = np.where(~tiny_mask)[0]
        if cols_to_fit.size == 0:        # nothing left
            self.coef_ = reg[0]  # lambda
            self.intercept_ = reg[1]  # beta
            self.x0_ = reg[2]  # x0
            return self

        # --- 3 run the expensive fits in parallel ------------------------------
        _worker = partial(
            self._fit_one_column,
            x_data=x_data,
            p0=p0,
            bounds=bounds,
            maxfev=maxfev
        )

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = pool.map(_worker, (y[:, idx] for idx in cols_to_fit))
        reg[:, cols_to_fit] = np.array(list(results)).T

        # Store the fitted parameters
        self.coef_ = reg[0]  # lambda
        self.intercept_ = reg[1]  # beta
        self.x0_ = reg[2]  # x0

        return self
    
    def predict(self, X):
        return stretched_sigmoid_x0(X, self.coef_, self.intercept_, self.x0_)
