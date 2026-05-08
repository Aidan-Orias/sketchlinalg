from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from sketchlinalg.api.schemas import (
    BenchmarkMetrics,
    BenchmarkProgressEvent,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkTimings,
)
from sketchlinalg.benchmarks.count_sketch_benchmarks import (
    fit_baseline,
    make_ridge_estimator,
    root_mean_squared_error,
    sketch_training_data,
    time_fit,
)
from sketchlinalg.datasets import load_e2006


ProgressEmitter = Callable[[BenchmarkProgressEvent], None]


def _elapsed_since(start: float) -> float:
    return float(time.perf_counter() - start)


def run_count_sketch_benchmark(
        job_id: str,
        request: BenchmarkRequest,
        emit: ProgressEmitter,
) -> BenchmarkResult:
    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="load_dataset",
            progress=0,
            message="Loading cached E2006 matrices",
        )
    )
    dataset = load_e2006()
    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="load_dataset",
            progress=1,
            message="Dataset ready",
        )
    )

    X_train = dataset.train.X.tocsr(copy=False)
    X_test = dataset.test.X.tocsr(copy=False)
    y_train = dataset.train.y
    y_test = dataset.test.y

    regular_start = time.perf_counter()

    def emit_regular_progress(done: int, total: int, _: float) -> None:
        emit(
            BenchmarkProgressEvent(
                type="progress",
                job_id=job_id,
                phase="fit_regular",
                progress=done / total,
                elapsed_seconds=_elapsed_since(regular_start),
                message=f"Regular fit repeat {done}/{total}",
            )
        )

    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="fit_regular",
            progress=0,
            message="Fitting baseline ridge model",
        )
    )
    base_fit_time, base_rmse = fit_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        request.alpha,
        request.repeats,
        emit_regular_progress,
    )

    sketch_start = time.perf_counter()
    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="construct_sketch",
            progress=0,
            message="Constructing CountSketch training data",
        )
    )
    rng = np.random.default_rng(request.seed)
    SX_train, Sy_train, sketch_time = sketch_training_data(
        X_train,
        y_train,
        request.sketch_dimension,
        rng,
    )
    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="construct_sketch",
            progress=1,
            elapsed_seconds=_elapsed_since(sketch_start),
            message="Sketch construction complete",
        )
    )

    sketched_start = time.perf_counter()

    def emit_sketched_progress(done: int, total: int, _: float) -> None:
        emit(
            BenchmarkProgressEvent(
                type="progress",
                job_id=job_id,
                phase="fit_sketched",
                progress=done / total,
                elapsed_seconds=_elapsed_since(sketched_start),
                message=f"Sketched fit repeat {done}/{total}",
            )
        )

    emit(
        BenchmarkProgressEvent(
            type="progress",
            job_id=job_id,
            phase="fit_sketched",
            progress=0,
            message="Fitting ridge model on sketched data",
        )
    )
    estimator_factory = lambda: make_ridge_estimator(request.alpha)
    sketched_fit_time, estimator = time_fit(
        estimator_factory,
        SX_train,
        Sy_train,
        request.repeats,
        emit_sketched_progress,
    )
    sketched_rmse = root_mean_squared_error(estimator, X_test, y_test)

    total_sketched = sketch_time + sketched_fit_time
    result = BenchmarkResult(
        sketch_dimension=request.sketch_dimension,
        alpha=request.alpha,
        repeats=request.repeats,
        timings=BenchmarkTimings(
            sketch_construction=sketch_time,
            sketched_fit=sketched_fit_time,
            regular_fit=base_fit_time,
            total_sketched=total_sketched,
            total_regular=base_fit_time,
        ),
        metrics=BenchmarkMetrics(
            fit_speedup=base_fit_time / sketched_fit_time,
            total_speedup=base_fit_time / total_sketched,
            regular_rmse=base_rmse,
            sketched_rmse=sketched_rmse,
            rmse_ratio=sketched_rmse / base_rmse,
        ),
    )

    return result
