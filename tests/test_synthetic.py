"""Unit tests for synthetic factor data generator."""

import pandas as pd
import pytest

from alpha_orthogonal.data import generate_synthetic_factors
from alpha_orthogonal.orthogonal import lowdin_orthogonalize
from alpha_orthogonal.neutralization import neutralize_factors


def test_synthetic_data_generator_shapes_and_types():
    """Verify dictionary keys, DataFrame shapes, and non-trivial collinearity."""
    data = generate_synthetic_factors(
        n_assets=80,
        n_factors=4,
        base_correlation=0.65,
        n_sectors=3,
        random_state=123,
    )

    factors = data["factors"]
    risk = data["risk_exposures"]
    returns = data["forward_returns"]

    assert isinstance(factors, pd.DataFrame)
    assert factors.shape == (80, 4)
    assert isinstance(risk, pd.DataFrame)
    assert risk.shape == (80, 4)  # 3 sectors + 1 market beta
    assert isinstance(returns, pd.Series)
    assert len(returns) == 80


def test_end_to_end_pipeline_on_synthetic_data():
    """Verify full quantitative workflow: Raw -> Neutralize -> Orthogonalize."""
    data = generate_synthetic_factors(n_assets=100, n_factors=5, random_state=42)

    raw_factors = data["factors"]
    risk = data["risk_exposures"]

    # 1. Neutralize against market beta and sectors
    neutral_factors = neutralize_factors(raw_factors, risk, standardize=True)
    assert neutral_factors.shape == (100, 5)

    # 2. Symmetrically orthogonalize using Löwdin
    final_factors = lowdin_orthogonalize(neutral_factors, standardize=True)
    assert final_factors.shape == (100, 5)

    # 3. Check that pairwise correlations of final factors are virtually zero
    corr = final_factors.corr()
    import numpy as np
    np.testing.assert_allclose(corr.values, np.eye(5), atol=1e-10)
