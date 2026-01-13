import math
import numpy as np

def gaussian_sketch(A: np.ndarray, B: np.ndarray, d: int, seed: int | None = None) -> np.ndarray:
    """
    Gaussian Sketch approximate of A^T @ B

    A: (n, m)
    B: (n, p)
    d: sketch dimension (d << n)
    """
    n, m = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(f"Row mismatch: A is (n,m)=({n},{m}) but B is (n,p)=({n2},{p}).")

    S = np.random.randn(d, n) / math.sqrt(d)
    SA = S @ A  # (d, m)
    SB = S @ B  # (d, p)

    return SA.T @ SB
