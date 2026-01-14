import numpy as np

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