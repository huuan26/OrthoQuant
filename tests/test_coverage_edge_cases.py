"""Edge cases and branch coverage tests."""

import numpy as np
import pandas as pd
import pytest

from alpha_orthogonal.orthogonal import (
    lowdin_orthogonalize,
    lowdin_correlation_transform,
    gram_schmidt_orthogonalize,
)
from alpha_orthogonal.neutralization import (
    neutralize_factors,
    build_sector_exposure_matrix,
)
from alpha_orthogonal.metrics import (
    correlation_matrix,
    condition_number,
    variance_inflation_factors,
    frobenius_distortion,
    effective_number_of_factors,
    information_coefficient,
    marginal_r_squared,
)


def test_metrics_edge_cases():
    # Correlation matrix array vs df
    arr = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    c_arr = correlation_matrix(arr)
    assert c_arr.shape == (2, 2)
    c_df = correlation_matrix(pd.DataFrame(arr, columns=["A", "B"]))
    assert isinstance(c_df, pd.DataFrame)

    # Condition number with zero singular value
    singular_mat = np.array([[1.0, 1.0], [1.0, 1.0]])
    cond = condition_number(singular_mat)
    assert cond == float("inf")

    # VIF with singular matrix
    vifs = variance_inflation_factors(singular_mat)
    assert np.all(np.isinf(vifs))

    # Frobenius distortion relative vs absolute
    X1 = np.ones((5, 2))
    X2 = np.zeros((5, 2))
    dist_abs = frobenius_distortion(X1, X2, relative=False)
    assert dist_abs > 0.0

    # Frobenius distortion with zero original
    dist_zero = frobenius_distortion(np.zeros((5, 2)), X2, relative=True)
    assert dist_zero == 0.0

    # IC error handling
    with pytest.raises(ValueError, match="Asset count mismatch"):
        information_coefficient(np.ones((10, 2)), np.ones(5))

    with pytest.raises(ValueError, match="Unknown IC method"):
        information_coefficient(np.ones((10, 2)), np.ones(10), method="invalid")

    # IC with NaNs and too few valid points
    F_nan = np.full((10, 1), np.nan)
    ic_nan = information_coefficient(F_nan, np.ones(10))
    assert np.isnan(ic_nan[0])

    # Marginal R2 with constant returns (ss_tot == 0)
    r2_zero = marginal_r_squared(np.ones((10, 2)), np.ones(10))
    np.testing.assert_allclose(r2_zero, 0.0)


def test_neutralization_edge_cases():
    F = np.ones((10, 2))
    X = np.ones((10, 2))

    # Non-2D inputs
    with pytest.raises(ValueError, match="must be 2D"):
        neutralize_factors(np.ones(10), X)
    with pytest.raises(ValueError, match="must be 2D"):
        neutralize_factors(F, np.ones(10))

    # Non-finite values
    F_bad = F.copy()
    F_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        neutralize_factors(F_bad, X)

    # Invalid weights length
    with pytest.raises(ValueError, match="Weight length"):
        neutralize_factors(F, X, weights=np.ones(5))

    # Non-positive weights
    with pytest.raises(ValueError, match="strictly positive"):
        neutralize_factors(F, X, weights=np.array([1.0] * 9 + [-1.0]))

    # Sector matrix with add_intercept=True
    sectors = ["A", "B", "A", "C"]
    dummies_intercept = build_sector_exposure_matrix(sectors, add_intercept=True)
    assert "market_intercept" in dummies_intercept.columns


def test_gram_schmidt_edge_cases():
    # 1D array
    with pytest.raises(ValueError, match="Expected 2D"):
        gram_schmidt_orthogonalize(np.ones(10))

    # N < K
    with pytest.raises(ValueError, match="must be >= number of factors"):
        gram_schmidt_orthogonalize(np.ones((3, 5)))

    # Non-finite
    X_nan = np.ones((10, 2))
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gram_schmidt_orthogonalize(X_nan)

    # Integer priority order on array
    X = np.random.randn(20, 3)
    Q = gram_schmidt_orthogonalize(X, priority_order=[2, 0, 1])
    assert Q.shape == (20, 3)


def test_lowdin_correlation_transform_edge_cases():
    # Constant factor column
    X_const = np.ones((20, 3))
    with pytest.raises(ValueError, match="Constant factor column"):
        lowdin_correlation_transform(X_const)

    # Singular correlation matrix
    X_collinear = np.ones((20, 2))
    X_collinear[:, 0] = np.linspace(1, 20, 20)
    X_collinear[:, 1] = X_collinear[:, 0] * 2.0  # perfectly collinear
    with pytest.raises(ValueError, match="singular"):
        lowdin_correlation_transform(X_collinear, rcond=1e-5)
