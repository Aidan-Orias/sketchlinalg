import time
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse

from sketchlinalg.sketches.count_sketch import (
    count_sketch_dense_vector,
    count_sketch_sparse,
    make_hash_sign,
)


# Benchmark process:
# Iterate through a series of sketch dimensions using Ridge Regression
# (alpha ~ 0 for linear regression). For each dimension we record:
# 1. Speed up measured as (median time to fit no sketch) / (median time to fit with sketch dimension d)
# 2. RMSE ratio measured as (RMSE with sketch dimension d) / (RMSE no sketch)


def _validate_repeats(repeats: int) -> None:
    if not isinstance(repeats, int):
        raise TypeError("repeats must be an integer")
    if repeats <= 0:
        raise ValueError("repeats must be positive")


def _validate_alpha(alpha: float) -> None:
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")


def make_ridge_estimator(alpha: float) -> Ridge:
    _validate_alpha(alpha)
    return Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")


def time_fit(estimator_factory, X: sparse.csr_matrix, y: np.ndarray, repeats: int = 7):
    _validate_repeats(repeats)
    times = []

    for _ in range(repeats):
        est = estimator_factory()
        t0 = time.perf_counter()
        est.fit(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), est


def root_mean_squared_error(estimator, X_test, y_test) -> float:
    return float(np.sqrt(mse(y_test, estimator.predict(X_test))))


def fit_baseline(X_train, y_train, X_test, y_test, alpha: float, repeats: int = 7) -> tuple[float, float]:
    estimator_factory = lambda: make_ridge_estimator(alpha)
    fit_time, estimator = time_fit(estimator_factory, X_train, y_train, repeats)
    rmse = root_mean_squared_error(estimator, X_test, y_test)
    return fit_time, rmse


def sketch_training_data(X_train, y_train, d: int, rng) -> tuple[sparse.csr_matrix, np.ndarray, float]:
    t0 = time.perf_counter()
    hash_sign = make_hash_sign(rng, d, X_train.shape[0])
    SX_train = count_sketch_sparse(X_train, d, hash_sign)
    Sy_train = count_sketch_dense_vector(y_train, d, hash_sign)
    sketch_time = time.perf_counter() - t0
    return SX_train, Sy_train, sketch_time


def benchmark_one(
        X_train,
        y_train,
        X_test,
        y_test,
        d: int,
        alpha: float,
        rng,
        repeats: int = 7,
        baseline: tuple[float, float] | None = None,
) -> dict:
    base_fit_time, base_rmse = baseline or fit_baseline(
        X_train, y_train, X_test, y_test, alpha, repeats
    )
    SX_train, Sy_train, sketch_time = sketch_training_data(X_train, y_train, d, rng)

    estimator_factory = lambda: make_ridge_estimator(alpha)
    sketched_fit_time, estimator = time_fit(estimator_factory, SX_train, Sy_train, repeats)
    sketched_rmse = root_mean_squared_error(estimator, X_test, y_test)

    return {
        "alpha": alpha,
        "d": d,
        "sketch_time": sketch_time,
        "base_fit_time": base_fit_time,
        "sketched_fit_time": sketched_fit_time,
        "fit_speedup": base_fit_time / sketched_fit_time,
        "total_speedup": base_fit_time / (sketched_fit_time + sketch_time),
        "base_rmse": base_rmse,
        "sketched_rmse": sketched_rmse,
        "rmse_ratio": sketched_rmse / base_rmse,
    }


def benchmark(
        X_train, y_train, X_test, y_test,
        sketch_dimensions,
        alphas,
        repeats: int = 7,
        seed: int = None
):
    _validate_repeats(repeats)
    results = []
    rng = np.random.default_rng(seed)

    for alpha in alphas:
        _validate_alpha(alpha)
        baseline = fit_baseline(X_train, y_train, X_test, y_test, alpha, repeats)

        for d in sketch_dimensions:
            results.append(
                benchmark_one(
                    X_train, y_train, X_test, y_test, d, alpha, rng, repeats, baseline
                )
            )

    return results
