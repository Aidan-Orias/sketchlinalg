from pathlib import Path
import bz2
import numpy as np
from sklearn.datasets import load_svmlight_file
from sketches.count_sketch import *

from scipy import sparse

RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/processed")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_or_cache(name: str):
    # name like "E2006.train" or "E2006.test"
    bz2_path = RAW_DIR / f"{name}.bz2"
    X_path = CACHE_DIR / f"{name}.X.npz"
    y_path = CACHE_DIR / f"{name}.y.npy"

    if X_path.exists() and y_path.exists():
        X = sparse.load_npz(X_path)
        y = np.load(y_path)
        return X, y

    with bz2.open(bz2_path, "rb") as f:
        X, y = load_svmlight_file(f)

    sparse.save_npz(X_path, X)
    np.save(y_path, y)
    return X, y

X_train, y_train = load_or_cache("E2006.train")
X_test, y_test   = load_or_cache("E2006.test")

sketch_dims = [2000, 5000, 10000, 20000, 50000]
alphas = [1e-12, 1.0, 2.0, 5.0]



