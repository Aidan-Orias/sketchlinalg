from pathlib import Path
import bz2
import numpy as np
from sklearn.datasets import load_svmlight_file
from sketches.count_sketch import *
from benchmarks.count_sketch_benchmarks import *
from scipy import sparse

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_or_cache(name: str, *, n_features: int | None = None, force_recache: bool = False):
    bz2_path = RAW_DIR / f"{name}.bz2"
    X_path = CACHE_DIR / f"{name}.X.npz"
    y_path = CACHE_DIR / f"{name}.y.npy"

    if force_recache:
        X_path.unlink(missing_ok=True)
        y_path.unlink(missing_ok=True)

    if X_path.exists() and y_path.exists():
        X = sparse.load_npz(X_path)
        y = np.load(y_path)
        return X, y

    with bz2.open(bz2_path, "rb") as f:
        if n_features is None:
            X, y = load_svmlight_file(f)
        else:
            X, y = load_svmlight_file(f, n_features=n_features)

    sparse.save_npz(X_path, X)
    np.save(y_path, y)
    return X, y

# Force recache, and make test match train feature dimension
X_train, y_train = load_or_cache("E2006.train", force_recache=True)
X_test, y_test   = load_or_cache("E2006.test", n_features=X_train.shape[1], force_recache=True)

sketch_dims = [5000, 20000, 50000]
alphas = [1e-6, 1.0, 5.0]

X_train_coo = X_train.tocoo()

results = benchmark(X_train_coo, y_train, X_test, y_test, sketch_dims, alphas, repeats=3, seed=127)

for result in results:
    print(result)

