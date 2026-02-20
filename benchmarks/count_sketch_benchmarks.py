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
        est = estimator()
        t0 = time.perf_counter()
        est.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), est

def benchmark(
        X_train, y_train, X_test, y_test,
        sketch_dimensions,
        alphas,
        repeats: int = 7,
        seed: int = None
):
    results = []
    rng = np.random.default_rng(seed)

    for alpha in alphas:
        lam_estimator = lambda: Ridge(alpha=alpha, fit_intercept=True, solver='lsqr')

        base_median_time, base_estimator = time_fit(lam_estimator, X_train, y_train, repeats)
        base_rmse = np.sqrt(mse(y_test, base_estimator.predict(X_test)))

        for d in sketch_dimensions:
            t0 = time.perf_counter()
            hash = make_hash_sign(rng, d, X_train.shape[0])
            SX_train = count_sketch_sparse(X_train, d, hash)
            SY_train = count_sketch_dense_vector(y_train, d, hash)
            t1 = time.perf_counter()
            sketch_time = t1 - t0

            median_time, est = time_fit(lam_estimator, SX_train, SY_train, repeats)
            rmse = np.sqrt(mse(y_test, est.predict(X_test)))

            results.append({
                "alpha": alpha,
                "d": d,
                "sketch_time": sketch_time,
                "base_fit_time": base_median_time,
                "sketched_fit_time": median_time,
                "fit_speedup": base_median_time / median_time,
                "total_speedup": base_median_time / (median_time + sketch_time),
                "base_rmse": base_rmse,
                "sketched_rmse": rmse,
                "rmse_ratio": rmse / base_rmse,
            })

    return results