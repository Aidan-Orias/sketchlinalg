import numpy as np
from scipy import sparse


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_2d_array(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if 0 in value.shape:
        raise ValueError(f"{name} must be nonempty")


def _validate_same_rows(A: np.ndarray, B: np.ndarray) -> None:
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows")


def _validate_hash_sign(hash_sign, d: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        h, s = hash_sign
    except (TypeError, ValueError) as exc:
        raise ValueError("hash must be a pair of row hashes and signs") from exc

    h = np.asarray(h)
    s = np.asarray(s)
    if h.shape != (n,) or s.shape != (n,):
        raise ValueError("hash and sign arrays must have length matching the input rows")
    if not np.issubdtype(h.dtype, np.integer):
        raise TypeError("hash row indices must be integers")
    if np.any((h < 0) | (h >= d)):
        raise ValueError("hash row indices must be in [0, d)")
    return h, s


def count_sketch_matrix(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """
    Returns count sketch matrix (d, n) where for each column a row i is randomly uniformly
    sampled such that S_ij = 1 with prob 1/2, -1 with prob 1/2 and S_kj = 0 for k ≠ i.
    Î := S.T @ S satisfies: E[Î_ij] = 1 if i=j, 0 otherwise, var(Î_ij) = 0 if i=j, 1/d otherwise

    n: Number of columns of S
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    _validate_positive_int("n", n)
    _validate_positive_int("d", d)
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, d, size=n)
    signs = rng.choice([-1, 1], size=n)
    S = np.zeros((d, n))
    S[rows, np.arange(n)] = signs
    return S


def count_sketch_multiplication(A: np.ndarray, B: np.ndarray, d: int, seed: int | None = None) -> np.ndarray:
    """
    Count Sketch approximate of A^T @ B

    A: (n, m)
    B: (n, p)
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    _validate_2d_array("A", A)
    _validate_2d_array("B", B)
    _validate_same_rows(A, B)
    _validate_positive_int("d", d)
    n = A.shape[0]
    S = count_sketch_matrix(n, d, seed)
    SA = S @ A  # (d, m)
    SB = S @ B  # (d, p)
    return SA.T @ SB


def count_sketch_apply(A: np.ndarray,  d: int, seed: int | None = None) -> np.ndarray:
    """
    Count sketch approximate of A without constructing S

    A: (n, m)
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    _validate_2d_array("A", A)
    _validate_positive_int("d", d)
    rng = np.random.default_rng(seed)
    n, m = A.shape

    SA = np.zeros((d, m))
    h, s = make_hash_sign(rng, d, n)
    np.add.at(SA, h, A * s[:, None])
    return SA


def count_sketch_sparse(X: sparse.spmatrix, d: int, hash) -> sparse.csr_matrix:
    """
        Sparse matrix count sketch approximate of A without constructing S. We start by changing X
        to COO format to more efficiently build the sketch. We then convert the sketched matrix
        into CSR format and return for faster multiplication and regression.

        X: Large sparse matrix
        d: Sketch dimension (d << n)
        seed: Random seed for reproducibility
    """
    if not sparse.issparse(X):
        raise TypeError("X must be a scipy sparse matrix")
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if 0 in X.shape:
        raise ValueError("X must be nonempty")
    _validate_positive_int("d", d)

    if not sparse.isspmatrix_coo(X):
        X = X.tocoo(copy=False)

    h, s = _validate_hash_sign(hash, d, X.shape[0])
    new_rows = h[X.row]
    new_data = X.data * s[X.row]

    SX = sparse.coo_matrix((new_data, (new_rows, X.col)), shape=(d, X.shape[1])).tocsr()
    return SX


def count_sketch_dense_vector(y: np.ndarray, d: int, hash) -> np.ndarray:
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y.shape[0] == 0:
        raise ValueError("y must be nonempty")
    _validate_positive_int("d", d)
    h, s = _validate_hash_sign(hash, d, y.shape[0])
    return np.bincount(h, weights=s * y, minlength=d)


def make_hash_sign(rng, d: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    _validate_positive_int("d", d)
    _validate_positive_int("n", n)
    h = rng.integers(0, d, n)
    s = rng.choice([-1.0, 1.0], n)
    return h, s
