# sketchlinalg

Randomized sketching experiments for fast linear algebra and linear models.

`sketchlinalg` currently focuses on two sketch families:

- **Gaussian sketches**, where a dense random projection matrix compresses the row dimension of an input matrix.
- **Count sketches**, where each original row is hashed into one sketched row with a random sign, giving a sparse projection that is useful for large sparse datasets.

The project includes small, testable implementations of the sketching primitives plus benchmark code for applying Count Sketch before ridge regression.

## What is here

```text
src/sketchlinalg/
  sketches/
    count_sketch.py       Count Sketch matrix construction and dense/sparse sketching helpers
    gaussian_sketch.py    Gaussian sketch matrix construction and matrix-product approximation

  benchmarks/
    count_sketch_benchmarks.py
                          Ridge regression benchmark utilities for sketched sparse data

  solvers/
    linear_ridge_regression.py
                          Example script for loading E2006 data and benchmarking ridge fits

tests/
  test_count_sketch.py
  test_gaussian_sketch.py
```

## Installation

Create an environment with Python 3.10 or newer, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

The repository uses a `src/` package layout, so examples import from `sketchlinalg`.

## Quick Start

Approximate a matrix product with a Gaussian sketch:

```python
import numpy as np

from sketchlinalg.sketches.gaussian_sketch import gaussian_sketch_multiplication

rng = np.random.default_rng(0)
A = rng.standard_normal((1000, 20))
B = rng.standard_normal((1000, 15))

approx = gaussian_sketch_multiplication(A, B, d=200, seed=0)
exact = A.T @ B

relative_error = np.linalg.norm(approx - exact, ord="fro") / np.linalg.norm(exact, ord="fro")
print(relative_error)
```

Apply a Count Sketch to a dense matrix without explicitly building the sketch matrix:

```python
import numpy as np

from sketchlinalg.sketches.count_sketch import count_sketch_apply

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 50))

SX = count_sketch_apply(X, d=200, seed=0)
print(SX.shape)  # (200, 50)
```

Apply the same Count Sketch hash to a sparse feature matrix and dense target vector:

```python
import numpy as np
from scipy import sparse

from sketchlinalg.sketches.count_sketch import (
    count_sketch_dense_vector,
    count_sketch_sparse,
    make_hash_sign,
)

rng = np.random.default_rng(0)
X = sparse.random(1000, 100, density=0.01, format="csr", random_state=0)
y = rng.standard_normal(1000)

hash_sign = make_hash_sign(rng, d=200, n=X.shape[0])
SX = count_sketch_sparse(X, d=200, hash=hash_sign)
Sy = count_sketch_dense_vector(y, d=200, hash=hash_sign)

print(SX.shape)  # (200, 100)
print(Sy.shape)  # (200,)
```

## API Overview

### Gaussian Sketch

`sketchlinalg.sketches.gaussian_sketch` provides:

- `gaussian_sketch_matrix(n, d, seed=None)`: returns a dense sketch matrix `S` with shape `(d, n)` and entries sampled from `N(0, 1/d)`.
- `gaussian_sketch_multiplication(A, B, d, seed=None)`: approximates `A.T @ B` with `(S @ A).T @ (S @ B)`.

### Count Sketch

`sketchlinalg.sketches.count_sketch` provides:

- `count_sketch_matrix(n, d, seed=None)`: returns an explicit Count Sketch matrix `S` with shape `(d, n)`.
- `count_sketch_multiplication(A, B, d, seed=None)`: approximates `A.T @ B` with Count Sketch.
- `count_sketch_apply(A, d, seed=None)`: applies Count Sketch to a dense matrix without explicitly constructing `S`.
- `make_hash_sign(rng, d, n)`: creates reusable row hashes and random signs.
- `count_sketch_sparse(X, d, hash)`: applies a reusable Count Sketch to a SciPy sparse matrix.
- `count_sketch_dense_vector(y, d, hash)`: applies the same sketch to a dense target vector.

## Running Tests

Run the test suite from the repository root:

```bash
python -m pytest
```

Some statistical accuracy checks are marked as slow:

```bash
python -m pytest -m slow
```

## Benchmarks

The benchmark utilities compare fitting ridge regression on the original sparse data against fitting ridge regression after Count Sketch compression.

The example solver script expects E2006 data files under `data/raw`:

```text
data/raw/E2006.train.bz2
data/raw/E2006.test.bz2
```

Run:

```bash
python src/sketchlinalg/solvers/linear_ridge_regression.py
```

The script caches parsed sparse matrices and targets under `data/processed`, then reports timing and RMSE ratios for several sketch dimensions and ridge penalties.

## Notes

- Sketch dimension `d` controls the speed-accuracy tradeoff. Larger sketches usually reduce approximation error, while smaller sketches give faster downstream fits.
- Count Sketch is especially useful when the original matrix is sparse because the implementation can sketch sparse data without constructing a dense projection matrix.
- The current code is a compact experimental implementation. It is intended for learning, testing sketch behavior, and benchmarking simple linear-model workflows.

## License

MIT
