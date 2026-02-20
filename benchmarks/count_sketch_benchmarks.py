import time
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse
from sketches.count_sketch import *


# Benchmark process:
# Iterate through a series of sketch dimensions using Ridge Regression
# (alpha ~ 0 for linear regression). For each dimension we record:
# 1. Speed up measured as (median time to fit no sketch) / (median time to fit with sketch dimension d)
# 2. RMSE ratio measured as (RMSE with sketch dimension d) / (RMSE no sketch)


def time_fit(estimator, X: sparse.csr_matrix, y: np.ndarray, repeats: int = 7):
    times = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        estimator.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))

def benchmark(
        X_train, y_train, X_test, y_test,
        sketch_dimension,
        alphas,
        repeats: int = 7,
        seed: int = None
):
    results = []

    for alpha in alphas:
        base_estimator = Ridge(alpha, fit_intercept=True)

        base_median_time = time_fit(base_estimator, X_train, y_train, repeats)
        base_rmse = mse(y_test, base_estimator.predict(X_test), squared=False)

        for d in sketch_dimension:
            SX_train = count_sketch_sparse(X_train, d, seed)
            SY_train = count_sketch_dense_vector(y_train, d, seed)

            est = Ridge(alpha, fit_intercept=True)
            median_time = time_fit(est, SX_train, SY_train, repeats)
            rmse = mse(y_test, est.predict(X_test), squared=False)

            results.append({
                "alpha": alpha,
                "d": d,
                "base_time": base_median_time,
                "sketched_time": median_time,
                "speedup": median_time / base_median_time,
                "base_rmse": base_rmse,
                "sketched_rmse": rmse,
                "rmse_ratio": base_rmse / rmse,
            })

    return results