"""Unit tests for Löwdin symmetric orthogonalization."""

import numpy as np
import pandas as pd
import pytest

from orthoquant.orthogonal import (
    lowdin_orthogonalize,
    lowdin_correlation_transform,
    gram_schmidt_orthogonalize,
)
from orthoquant.metrics import (
    frobenius_distortion,
    condition_number,
    correlation_matrix,
)


@pytest.fixture
def sample_factor_matrix():
    rng = np.random.default_rng(42)
    N, K = 150, 4
    # Create collinear factors
    base = rng.normal(size=(N, 1))
    X = base + rng.normal(scale=0.3, size=(N, K))
    return X


@pytest.fixture
def sample_factor_df(sample_factor_matrix):
    cols = ["Mom_1M", "Rev_5D", "Vol_20D", "Turnover_10D"]
    idx = [f"SYM_{i:03d}" for i in range(sample_factor_matrix.shape[0])]
    return pd.DataFrame(sample_factor_matrix, index=idx, columns=cols)


def test_lowdin_orthonormality(sample_factor_matrix):
    """Verify that F_tilde^T F_tilde is exact identity matrix I_K."""
    X = sample_factor_matrix
    K = X.shape[1]

    # Test SVD method
    Q_svd = lowdin_orthogonalize(X, method="svd")
    gram_svd = Q_svd.T @ Q_svd
    np.testing.assert_allclose(gram_svd, np.eye(K), atol=1e-10)

    # Test Eig method
    Q_eig = lowdin_orthogonalize(X, method="eig")
    gram_eig = Q_eig.T @ Q_eig
    np.testing.assert_allclose(gram_eig, np.eye(K), atol=1e-10)

    # Both methods should give identical results
    np.testing.assert_allclose(Q_svd, Q_eig, atol=1e-10)


def test_lowdin_minimal_frobenius_distortion(sample_factor_matrix):
    """Verify that Löwdin achieves smaller Frobenius distortion than Gram-Schmidt."""
    X = sample_factor_matrix
    Q_lowdin = lowdin_orthogonalize(X, method="svd")
    Q_gs = gram_schmidt_orthogonalize(X)

    dist_lowdin = frobenius_distortion(X, Q_lowdin, relative=True)
    dist_gs = frobenius_distortion(X, Q_gs, relative=True)

    # Löwdin distortion must be strictly <= Gram-Schmidt distortion
    assert dist_lowdin <= dist_gs + 1e-10


def test_lowdin_permutation_symmetry():
    """Verify that Löwdin is permutation symmetric.

    If input columns are permuted: F_perm = F P,
    then the orthogonalized columns are permuted by the exact same permutation:
    lowdin(F P) = lowdin(F) P.
    """
    rng = np.random.default_rng(123)
    N, K = 100, 3
    X = rng.normal(size=(N, K))
    # Apply permutation [2, 0, 1]
    perm = [2, 0, 1]
    X_perm = X[:, perm]

    Q_orig = lowdin_orthogonalize(X)
    Q_perm = lowdin_orthogonalize(X_perm)

    np.testing.assert_allclose(Q_perm, Q_orig[:, perm], atol=1e-10)


def test_lowdin_dataframe_preservation(sample_factor_df):
    """Verify index, columns, and types are preserved when passing a DataFrame."""
    df_ortho = lowdin_orthogonalize(sample_factor_df, method="svd")
    assert isinstance(df_ortho, pd.DataFrame)
    assert list(df_ortho.columns) == list(sample_factor_df.columns)
    assert list(df_ortho.index) == list(sample_factor_df.index)

    # Check condition number drops to 1.0
    cond = condition_number(df_ortho)
    np.testing.assert_allclose(cond, 1.0, atol=1e-10)


def test_lowdin_correlation_transform(sample_factor_matrix):
    """Verify that lowdin_correlation_transform produces zero pairwise cross-correlations."""
    Z_ortho = lowdin_correlation_transform(sample_factor_matrix)
    K = sample_factor_matrix.shape[1]
    C = np.corrcoef(Z_ortho, rowvar=False)
    np.testing.assert_allclose(C, np.eye(K), atol=1e-10)


def test_lowdin_standardize_option(sample_factor_matrix):
    """Verify standardize=True yields zero-mean, unit-variance columns."""
    Q = lowdin_orthogonalize(sample_factor_matrix, standardize=True)
    means = np.mean(Q, axis=0)
    stds = np.std(Q, axis=0, ddof=1)
    np.testing.assert_allclose(means, 0.0, atol=1e-10)
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_lowdin_error_handling():
    """Verify validation and error raising for invalid inputs."""
    # N < K
    with pytest.raises(ValueError, match="must be >= number of factors"):
        lowdin_orthogonalize(np.ones((3, 5)))

    # NaNs
    X_nan = np.ones((10, 3))
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        lowdin_orthogonalize(X_nan)

    # 1D array
    with pytest.raises(ValueError, match="Expected 2D"):
        lowdin_orthogonalize(np.ones(10))

    # Unknown method
    with pytest.raises(ValueError, match="Unknown orthogonalization method"):
        lowdin_orthogonalize(np.ones((10, 2)), method="invalid")

    # Collinear rank-deficient matrix
    X_collinear = np.ones((10, 3))
    with pytest.raises(ValueError, match="rank-deficient"):
        lowdin_orthogonalize(X_collinear, rcond=1e-6)
