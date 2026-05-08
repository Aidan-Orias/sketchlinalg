from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


BenchmarkPhase = Literal[
    "queued",
    "load_dataset",
    "fit_regular",
    "construct_sketch",
    "fit_sketched",
    "complete",
    "error",
]


def model_to_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class DatasetSummary(BaseModel):
    name: str
    task: str
    train_rows: int
    test_rows: int
    features: int
    train_nnz: int
    test_nnz: int
    train_density: float
    test_density: float
    target_mean: float
    target_std: float
    target_min: float
    target_max: float


class BenchmarkOptions(BaseModel):
    sketch_dimension_min: int = 1000
    sketch_dimension_max: int = 50000
    sketch_dimension_step: int = 500
    default_sketch_dimension: int = 5000
    alpha_values: list[float] = Field(
        default_factory=lambda: [1e-6, 0.01, 0.1, 1.0, 5.0, 10.0]
    )
    default_alpha: float = 1.0
    default_repeats: int = 3
    default_seed: int = 127


class BenchmarkRequest(BaseModel):
    sketch_dimension: int = Field(default=5000, ge=1, le=250000)
    alpha: float = Field(default=1.0, ge=0)
    repeats: int = Field(default=3, ge=1, le=25)
    seed: int | None = 127


class BenchmarkTimings(BaseModel):
    sketch_construction: float
    sketched_fit: float
    regular_fit: float
    total_sketched: float
    total_regular: float


class BenchmarkMetrics(BaseModel):
    fit_speedup: float
    total_speedup: float
    regular_rmse: float
    sketched_rmse: float
    rmse_ratio: float


class BenchmarkResult(BaseModel):
    sketch_dimension: int
    alpha: float
    repeats: int
    timings: BenchmarkTimings
    metrics: BenchmarkMetrics


class BenchmarkJobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]


class BenchmarkJobSnapshot(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "error"]
    result: BenchmarkResult | None = None
    error: str | None = None


class BenchmarkProgressEvent(BaseModel):
    type: Literal["progress", "complete", "error"]
    job_id: str
    phase: BenchmarkPhase
    progress: float = Field(ge=0, le=1)
    elapsed_seconds: float | None = None
    message: str | None = None
    result: BenchmarkResult | None = None
    error: str | None = None
