"""Unit tests for cross-sectional risk neutralization."""

import numpy as np
import pandas as pd
import pytest

from alpha_orthogonal.neutralization import (
    neutralize_factors,
    build_sector_exposure_matrix,
)


@pytest.fixture
def risk_neutral_data():
    rng = np.random.default_rng(77)
    n_assets, n_factors = 100, 4
    F = rng.normal(size=(n_assets, n_factors))

    sectors = [f"Sector_{i % 3}" for i in range(n_assets)]
    X_sectors = build_sector_exposure_matrix(sectors)
    market_beta = rng.normal(loc=1.0, scale=0.2, size=(n_assets, 1))

    X = np.hstack([X_sectors.values, market_beta])
    weights = rng.uniform(low=1.0, high=10.0, size=n_assets)

    return F, X, weights


def test_ols_neutralization_orthogonality(risk_neutral_data):
    """Verify X^T F_neutral is numerically zero (exact orthogonality to risk exposures)."""
    F, X, _ = risk_neutral_data
    F_neutral = neutralize_factors(F, X, standardize=False)

    # Matrix product X^T F_neutral must be zero
    proj = X.T @ F_neutral
    np.testing.assert_allclose(proj, 0.0, atol=1e-10)


def test_wls_neutralization_orthogonality(risk_neutral_data):
    """Verify X^T W F_neutral is numerically zero under weighted least squares."""
    F, X, weights = risk_neutral_data
    F_neutral_w = neutralize_factors(F, X, weights=weights, standardize=False)

    W = np.diag(weights / np.sum(weights) * len(weights))
    proj_w = X.T @ (W @ F_neutral_w)
    np.testing.assert_allclose(proj_w, 0.0, atol=1e-10)


def test_sector_dummy_construction():
    """Verify build_sector_exposure_matrix creates correct one-hot shape and sum."""
    sectors = ["Tech", "Fin", "Tech", "Energy", "Fin"]
    dummies = build_sector_exposure_matrix(sectors, add_intercept=False)

    assert dummies.shape == (5, 3)
    # Row sum must be 1.0 (each asset belongs to exactly 1 sector)
    np.testing.assert_allclose(dummies.sum(axis=1), 1.0)


def test_neutralization_dataframe_support(risk_neutral_data):
    """Verify DataFrame inputs preserve labels and index."""
    F, X, _ = risk_neutral_data
    df_F = pd.DataFrame(F, columns=[f"Alpha_{i}" for i in range(F.shape[1])])
    df_X = pd.DataFrame(X, columns=[f"Risk_{i}" for i in range(X.shape[1])])

    df_neutral = neutralize_factors(df_F, df_X)
    assert isinstance(df_neutral, pd.DataFrame)
    assert list(df_neutral.columns) == list(df_F.columns)
    assert list(df_neutral.index) == list(df_F.index)


def test_neutralization_dimension_mismatch():
    """Verify error when number of assets differs."""
    F = np.ones((50, 3))
    X = np.ones((60, 2))
    with pytest.raises(ValueError, match="Asset count mismatch"):
        neutralize_factors(F, X)
