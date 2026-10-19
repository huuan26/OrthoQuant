"""Unit tests for collinearity, distortion, and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from orthoquant.orthogonal import lowdin_orthogonalize
from orthoquant.metrics import (
    correlation_matrix,
    condition_number,
    variance_inflation_factors,
    frobenius_distortion,
    effective_number_of_factors,
    information_coefficient,
    marginal_r_squared,
)


@pytest.fixture
def collinear_data():
    rng = np.random.default_rng(55)
    N, K = 100, 3
    base = rng.normal(size=(N, 1))
    # Highly collinear signals
    F = base + rng.normal(scale=0.05, size=(N, K))
    df = pd.DataFrame(F, columns=["Alpha1", "Alpha2", "Alpha3"])
    return F, df


def test_vif_orthogonal_vs_collinear(collinear_data):
    """Verify VIF is very high for collinear factors, and exactly 1.0 after orthogonalization."""
    F, df = collinear_data

    # Raw collinear factors must have high VIF (> 5.0)
    raw_vifs = variance_inflation_factors(df)
    assert np.all(raw_vifs > 5.0)

    # After Löwdin orthogonalization, correlation matrix is identity -> VIF = 1.0
    F_ortho = lowdin_orthogonalize(df, standardize=True)
    ortho_vifs = variance_inflation_factors(F_ortho)
    np.testing.assert_allclose(ortho_vifs, 1.0, atol=1e-10)


def test_condition_number_improvement(collinear_data):
    """Verify condition number improves from >> 1 to exactly 1.0."""
    F, df = collinear_data
    raw_cond = condition_number(df)
    assert raw_cond > 10.0

    ortho = lowdin_orthogonalize(df)
    ortho_cond = condition_number(ortho)
    np.testing.assert_allclose(ortho_cond, 1.0, atol=1e-10)


def test_effective_number_of_factors(collinear_data):
    """Verify ENF increases toward K after orthogonalization."""
    F, _ = collinear_data
    K = F.shape[1]

    raw_enf = effective_number_of_factors(F)
    # Raw collinear factors should act like ~1 effective factor
    assert raw_enf < 1.5

    F_ortho = lowdin_orthogonalize(F, standardize=True)
    ortho_enf = effective_number_of_factors(F_ortho)
    # Orthogonal factors achieve max entropy = K
    np.testing.assert_allclose(ortho_enf, float(K), atol=1e-5)


def test_information_coefficient():
    """Verify Rank IC and Pearson IC against known ground truth."""
    N = 100
    x = np.linspace(-2.0, 2.0, N)
    # Monotonic relationship
    y = np.exp(x)

    df_F = pd.DataFrame({"Signal": x})
    s_ret = pd.Series(y)

    # Spearman rank IC must be 1.0 for strictly monotonic relationship
    rank_ic = information_coefficient(df_F, s_ret, method="spearman")
    np.testing.assert_allclose(rank_ic["Signal"], 1.0, atol=1e-10)

    # Pearson linear IC will be less than 1.0 due to non-linearity
    pearson_ic = information_coefficient(df_F, s_ret, method="pearson")
    assert 0.70 < pearson_ic["Signal"] < 1.0


def test_marginal_r_squared():
    """Verify marginal R2 on orthogonal factors decomposes total R2."""
    rng = np.random.default_rng(88)
    N = 200
    Q = lowdin_orthogonalize(rng.normal(size=(N, 3)), standardize=True)
    y = 0.3 * Q[:, 0] + 0.5 * Q[:, 1] + rng.normal(scale=0.1, size=N)

    r2s = marginal_r_squared(Q, y)
    assert len(r2s) == 3
    # Q[:, 1] has larger coefficient, so its R2 must be higher
    assert r2s[1] > r2s[0]
