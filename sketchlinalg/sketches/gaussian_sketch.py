from math import sqrt
import numpy as np

def gaussian_sketch_matrix(n: int, d: int, seed: int |None = None) -> np.ndarray:
    """
    Returns Gaussian sketch matrix (d, n) where S_ij ~ N(0, 1/d). Additionally, let
    Î := S.T @ S, then E[Î_ij] = 1 if i=j, 0 otherwise, and var(Î_ij) = 2/d if i=j, 1/d otherwise

    n: Number of columns of S
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, n)) / sqrt(d)


def gaussian_sketch_multiplication(A: np.ndarray, B: np.ndarray, d: int, seed: int | None = None) -> np.ndarray:
    """
    Gaussian Sketch approximate of A^T @ B

    A: (n, m)
    B: (n, p)
    d: Sketch dimension (d << n)
    seed: Random seed for reproducibility
    """
    n = A.shape[0]
    S = gaussian_sketch_matrix(n, d, seed)
    SA = S @ A  # (d, m)
    SB = S @ B  # (d, p)
    return SA.T @ SB
