"""Synthetic Factor & Market Data Generator with Ground-Truth Properties."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def generate_synthetic_factors(
    n_assets: int = 100,
    n_factors: int = 5,
    base_correlation: float = 0.70,
    n_sectors: int = 4,
    random_state: Optional[int] = 42,
) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Generate synthetic cross-sectional alpha factors with controlled collinearity and risk exposures.

    Parameters
    ----------
    n_assets : int, default 100
        Number of assets in the cross section.
    n_factors : int, default 5
        Number of raw factor signals.
    base_correlation : float, default 0.70
        Average off-diagonal correlation among raw factors to simulate collinearity.
    n_sectors : int, default 4
        Number of distinct economic sectors.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        - 'factors': pd.DataFrame of shape (n_assets, n_factors)
        - 'risk_exposures': pd.DataFrame of shape (n_assets, n_sectors + 1) [sectors + market beta]
        - 'forward_returns': pd.Series of shape (n_assets,)
        - 'sectors': pd.Series of categorical sector assignments
        - 'market_caps': pd.Series of asset market capitalizations
    """
    rng = np.random.default_rng(random_state)
    asset_ids = [f"ASSET_{i:03d}" for i in range(n_assets)]
    factor_names = [f"Alpha_{k+1}" for k in range(n_factors)]

    # 1. Build a target correlation matrix with base_correlation
    # C = (1 - rho) * I + rho * 1 1^T
    C = (1.0 - base_correlation) * np.eye(n_factors) + base_correlation * np.ones(
        (n_factors, n_factors)
    )
    # Add slight noise to make eigenvalues non-degenerate
    noise = rng.normal(scale=0.05, size=(n_factors, n_factors))
    C_noisy = C + (noise + noise.T) / 2.0
    # Project to nearest positive definite correlation matrix
    eigvals, eigvecs = np.linalg.eigh(C_noisy)
    eigvals = np.maximum(eigvals, 0.05)
    C_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(C_pd))
    C_final = C_pd / np.outer(d, d)

    # Cholesky factor to induce correlation
    L = np.linalg.cholesky(C_final)

    # Independent Gaussian draws
    Z = rng.normal(size=(n_assets, n_factors))
    # Correlated factors
    F_raw = Z @ L.T

    # 2. Assign sectors and market capitalization
    sectors = [f"Sector_{i % n_sectors + 1}" for i in range(n_assets)]
    log_caps = rng.normal(loc=10.0, scale=1.5, size=n_assets)
    market_caps = np.exp(log_caps)

    # Sector dummy matrix
    sector_dummies = pd.get_dummies(pd.Series(sectors, index=asset_ids), prefix="Sec", dtype=float)

    # Market beta
    market_betas = rng.normal(loc=1.0, scale=0.3, size=n_assets)

    # Combine risk exposures
    risk_df = sector_dummies.copy()
    risk_df["Market_Beta"] = market_betas

    # Inject sector and beta bias into the raw factors so neutralization has a clear effect
    for k in range(n_factors):
        sector_loading = 0.5 * (sector_dummies.iloc[:, k % n_sectors].values)
        beta_loading = 0.3 * market_betas
        F_raw[:, k] += sector_loading + beta_loading

    factors_df = pd.DataFrame(F_raw, index=asset_ids, columns=factor_names)

    # 3. Generate forward returns: linear combination of true orthogonal signal + noise
    # True alpha vector
    true_weights = rng.uniform(low=0.01, high=0.05, size=n_factors)
    systematic_return = factors_df.values @ true_weights
    idiosyncratic_noise = rng.normal(scale=0.03, size=n_assets)
    forward_returns = pd.Series(
        systematic_return + idiosyncratic_noise,
        index=asset_ids,
        name="Forward_Return",
    )

    return {
        "factors": factors_df,
        "risk_exposures": risk_df,
        "forward_returns": forward_returns,
        "sectors": pd.Series(sectors, index=asset_ids, name="Sector"),
        "market_caps": pd.Series(market_caps, index=asset_ids, name="Market_Cap"),
    }
