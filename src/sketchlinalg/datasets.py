from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import bz2

import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class SparseDatasetSplit:
    X: sparse.spmatrix
    y: np.ndarray


@dataclass(frozen=True)
class E2006Dataset:
    train: SparseDatasetSplit
    test: SparseDatasetSplit


def _data_file(name: str) -> Path:
    return RAW_DIR / f"{name}.bz2"


def _cached_matrix_file(name: str) -> Path:
    return CACHE_DIR / f"{name}.X.npz"


def _cached_target_file(name: str) -> Path:
    return CACHE_DIR / f"{name}.y.npy"


def load_or_cache(
        name: str,
        *,
        n_features: int | None = None,
        force_recache: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bz2_path = _data_file(name)
    X_path = _cached_matrix_file(name)
    y_path = _cached_target_file(name)

    if force_recache:
        X_path.unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)

    if X_path.exists() and y_path.exists():
        X = sparse.load_npz(X_path)
        y = np.load(y_path)
        return X.tocsr(copy=False), y

    if not bz2_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file {bz2_path}. Expected E2006 data under data/raw."
        )

    with bz2.open(bz2_path, "rb") as f:
        if n_features is None:
            X, y = load_svmlight_file(f)
        else:
            X, y = load_svmlight_file(f, n_features=n_features)

    X = X.tocsr(copy=False)
    sparse.save_npz(X_path, X)
    np.save(y_path, y)
    return X, y


@lru_cache(maxsize=1)
def load_e2006() -> E2006Dataset:
    X_train, y_train = load_or_cache("E2006.train")
    X_test, y_test = load_or_cache("E2006.test", n_features=X_train.shape[1])
    return E2006Dataset(
        train=SparseDatasetSplit(X=X_train, y=y_train),
        test=SparseDatasetSplit(X=X_test, y=y_test),
    )


def recache_e2006() -> E2006Dataset:
    load_e2006.cache_clear()
    X_train, y_train = load_or_cache("E2006.train", force_recache=True)
    X_test, y_test = load_or_cache(
        "E2006.test", n_features=X_train.shape[1], force_recache=True
    )
    dataset = E2006Dataset(
        train=SparseDatasetSplit(X=X_train, y=y_train),
        test=SparseDatasetSplit(X=X_test, y=y_test),
    )
    load_e2006.cache_clear()
    return dataset


def sparse_density(X: sparse.spmatrix) -> float:
    rows, cols = X.shape
    return float(X.nnz / (rows * cols))


def e2006_summary() -> dict:
    dataset = load_e2006()
    y_train = dataset.train.y
    y_test = dataset.test.y

    return {
        "name": "E2006",
        "task": "Sparse ridge-regression benchmark",
        "train_rows": dataset.train.X.shape[0],
        "test_rows": dataset.test.X.shape[0],
        "features": dataset.train.X.shape[1],
        "train_nnz": int(dataset.train.X.nnz),
        "test_nnz": int(dataset.test.X.nnz),
        "train_density": sparse_density(dataset.train.X),
        "test_density": sparse_density(dataset.test.X),
        "target_mean": float(np.mean(y_train)),
        "target_std": float(np.std(y_train)),
        "target_min": float(min(np.min(y_train), np.min(y_test))),
        "target_max": float(max(np.max(y_train), np.max(y_test))),
    }
