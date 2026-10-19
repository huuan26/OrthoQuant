"""Unit tests for Gram-Schmidt sequential orthogonalization."""

import numpy as np
import pandas as pd
import pytest

from orthoquant.orthogonal import gram_schmidt_orthogonalize


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(99)
    N, K = 120, 3
    X = rng.normal(size=(N, K))
    df = pd.DataFrame(X, columns=["CoreAlpha", "SecondaryAlpha", "TertiaryAlpha"])
    return X, df


def test_gram_schmidt_orthonormality(sample_data):
    """Verify Q^T Q is identity and F = Q R."""
    X, _ = sample_data
    Q, R = gram_schmidt_orthogonalize(X, return_r=True)

    K = X.shape[1]
    np.testing.assert_allclose(Q.T @ Q, np.eye(K), atol=1e-10)
    # Check reconstruction F = Q R
    np.testing.assert_allclose(Q @ R, X, atol=1e-10)
    # R must be upper triangular
    assert np.allclose(R, np.triu(R), atol=1e-10)


def test_gram_schmidt_first_column_direction(sample_data):
    """Verify first orthogonal column is in the exact positive direction of first original column."""
    X, _ = sample_data
    Q = gram_schmidt_orthogonalize(X)

    # Correlation between first original and first orthogonal must be +1.0
    corr = np.corrcoef(X[:, 0], Q[:, 0])[0, 1]
    np.testing.assert_allclose(corr, 1.0, atol=1e-10)


def test_gram_schmidt_priority_reordering(sample_data):
    """Verify specifying priority_order reorders and preserves the designated primary factor."""
    _, df = sample_data
    # Prioritize TertiaryAlpha first
    df_ortho = gram_schmidt_orthogonalize(
        df, priority_order=["TertiaryAlpha", "CoreAlpha", "SecondaryAlpha"]
    )

    assert list(df_ortho.columns) == ["TertiaryAlpha", "CoreAlpha", "SecondaryAlpha"]
    # TertiaryAlpha must have correlation 1.0 with original TertiaryAlpha
    corr = np.corrcoef(df["TertiaryAlpha"], df_ortho["TertiaryAlpha"])[0, 1]
    np.testing.assert_allclose(corr, 1.0, atol=1e-10)
