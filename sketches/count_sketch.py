import numpy as np
from scipy import sparse


def count_sketch_matrix(n: int, d: int, seed: int | None = None) -> np.ndarray:
    """
    Returns count sketch matrix (d, n) where for each column a row i is randomly uniformly
    sampled such that S_ij = 1 with prob 1/2, -1 with prob 1/2 and S_kj = 0 for k ≠ i.
    Î := S.T @ S satisfies: E[Î_ij] = 1 if i=j, 0 otherwise, var(Î_ij) = 0 if i=j, 1/d otherwise

    n: Number of columns of S
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
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
    rng = np.random.default_rng(seed)
    n, m = A.shape

    SA = np.zeros((d ,m))
    h = rng.integers(0, d, size=n)
    s = rng.choice([-1, 1], size=n)
    np.add.at(SA, h, A * s[:, None])
    return SA


def count_sketch_sparse(X: sparse.spmatrix, d: int, seed: int | None = None) -> sparse.csr_matrix:
    """
        Sparse matrix count sketch approximate of A without constructing S. We start by changing X
        to COO format to more efficiently build the sketch. We then convert the sketched matrix
        into CSR format and return for faster multiplication and regression.

        X: Large sparse matrix
        d: Sketch dimension (d << n)
        seed: Random seed for reproducibility
    """
    X = X.tocoo(copy=False)
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    h = rng.integers(0, d, size=n)
    s = rng.choice(np.array([-1.0, 1.0]), size=n)

    new_rows = h[X.row]
    new_data = X.data * s[X.row]

    SX = sparse.coo_matrix((new_data, (new_rows, X.col)), shape=(d, X.shape[1])).tocsr()
    SX.sum_duplicates()
    return SX

def count_sketch_dense_vector(y: np.ndarray, d: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    h = rng.integers(0, d, size=n)
    s = rng.choice([-1.0, 1.0], size=n)
    return np.bincount(h, weights=s * y, minlength=d)