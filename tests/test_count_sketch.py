import pytest
from sketchlinalg.sketchlinalg.sketches.count_sketch import *

# BASIC TESTS
def test_does_not_modify_inputs():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 4))
    B = rng.standard_normal((50, 3))
    A0, B0 = A.copy(), B.copy()
    _ = count_sketch_multiplication(A, B, d=10, seed=0)
    np.testing.assert_array_equal(A, A0)
    np.testing.assert_array_equal(B, B0)


def test_no_nan_inf_large_values():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 5)) * 1e8
    B = rng.standard_normal((200, 6)) * 1e8
    C = count_sketch_multiplication(A, B, d=50, seed=0)
    assert np.isfinite(C).all()


def test_shape():
    n, m, p, d = 100, 7, 9, 20
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((n, p))
    C = count_sketch_multiplication(A, B, d=d, seed=0)
    assert C.shape == (m, p)


def test_seed_determinism():
    n, m, p, d = 120, 8, 6, 30
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((n, p))
    C1 = count_sketch_multiplication(A, B, d=d, seed=123)
    C2 = count_sketch_multiplication(A, B, d=d, seed=123)
    np.testing.assert_array_equal(C1, C2)


def test_matches_explicit_S():
    n, m, p, d, seed = 60, 5, 7, 25, 42
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((n, p))

    rngS = np.random.default_rng(seed)
    rows = rngS.integers(0, d, size=n)
    signs = rngS.choice([-1, 1], size=n)
    S = np.zeros((d, n))
    S[rows, np.arange(n)] = signs
    expected = (S @ A).T @ (S @ B)

    got = count_sketch_multiplication(A, B, d=d, seed=seed)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


# MATHEMATICAL TESTS

# Expected Frobenius norm error and error variance should decrease as d increases
@pytest.mark.slow
def test_variance_accuracy():
    n, m, p = 300, 30, 20
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    B = rng.standard_normal((n, p))
    actual = A.T @ B

    seeds = range(100)
    def errs(d):
        return np.array([
            np.linalg.norm(actual - count_sketch_multiplication(A, B, d, seed), ord='fro')
            for seed in seeds
        ])

    err_small, err_large = errs(20), errs(120)

    assert np.var(err_large) < 0.9 * np.var(err_small)
    assert np.median(err_large) < 0.9 * np.median(err_small)


# Ensures linearity in B when S is fixed
def test_fixed_S_linearity_in_B():
    n, m, p, d = 80, 5, 6, 30
    rng = np.random.default_rng(0)
    S = count_sketch_matrix(n, d, seed=0)
    A = rng.standard_normal((n, m))
    B1 = rng.standard_normal((n, p))
    B2 = rng.standard_normal((n, p))

    left = (S @ A).T @ S @ (B1 + B2)
    right = (S @ A).T @ (S @ B1) + (S @ A).T @ (S @ B2)
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)