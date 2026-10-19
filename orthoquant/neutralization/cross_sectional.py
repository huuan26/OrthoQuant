"""Cross-Sectional Factor Risk Neutralization via Projection.

Notation:
    - N: Number of assets (rows).
    - K: Number of factors (columns).
    - M: Number of risk exposures (columns), M < N.
    - F in R^{N x K}: Factor signal matrix.
    - X in R^{N x M}: Risk exposure matrix with rank(X) = M.
    - P_X = X (X^T X)^{-1} X^T: Projection matrix onto Col(X).
    - M_X = I_N - P_X: Annihilator matrix projecting onto Col(X)^perp.

Formulation:
    Given factor signals F in R^{N x K} and a risk exposure matrix X in R^{N x M}
    (e.g., sector dummy one-hots, Market Beta, Log-Market Cap, Style factors),
    cross-sectional neutralization computes the orthogonal residuals from regressing F on X:

        F = X B + E

    where B = (X^T X)^{-1} X^T F is the OLS coefficient matrix.
    The neutralized residual signals E = F_neutral satisfy the Frisch-Waugh-Lovell projection:

        F_neutral = (I_N - X (X^T X)^{-1} X^T) F = M_X F

    By construction, the neutralized factors are strictly orthogonal to all risk exposures:
        X^T F_neutral = 0_{M x K}

    Numerical Stability:
        Rather than explicitly forming and inverting X^T X (which may be ill-conditioned),
        we compute the thin QR decomposition of X:
            X = Q_X R_X
        Then the projection simplifies to:
            F_neutral = F - Q_X (Q_X^T F)
        which avoids matrix inversion entirely and guarantees machine-precision stability.

References:
    - Frisch, R., & Waugh, F. V. (1933). "Partial time regressions as compared with individual trends".
      Econometrica, 1(4), 387–401.
    - Lovell, M. C. (1963). "Seasonal adjustment of economic time series and multiple regression analysis".
      Journal of the American Statistical Association, 58(304), 993–1010.
    - Kakushadze, Z. (2015). "Factor models and alpha streams".
      The Journal of Investment Strategies, 5(1), 1–45.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def neutralize_factors(
    factors: Union[np.ndarray, pd.DataFrame],
    risk_exposures: Union[np.ndarray, pd.DataFrame],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    standardize: bool = True,
) -> Union[np.ndarray, pd.DataFrame]:
    """Neutralize factors against common risk exposures (Market, Sector, Style).

    Parameters
    ----------
    factors : np.ndarray or pd.DataFrame, shape (N, K)
        Raw factor matrix for N assets and K factors.
    risk_exposures : np.ndarray or pd.DataFrame, shape (N, M)
        Risk exposure matrix containing M risk factors (e.g. market intercept, sector
        dummies, beta, size). Must have the same number of assets N.
    weights : np.ndarray or pd.Series, shape (N,), optional
        Asset weighting vector for Weighted Least Squares (WLS) neutralization
        (e.g., Market Capitalization weights). If None, standard OLS is applied.
    standardize : bool, default True
        If True, z-score standardizes each residual factor column to have zero mean
        and unit variance cross-sectionally.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Neutralized factor matrix F_neutral of shape (N, K), with zero exposure to X.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        index = factors.index
        columns = factors.columns
        F = factors.to_numpy(dtype=float, copy=True)
    else:
        F = np.asarray(factors, dtype=float)
        index = None
        columns = None

    if isinstance(risk_exposures, pd.DataFrame):
        X = risk_exposures.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(risk_exposures, dtype=float)

    if F.ndim != 2 or X.ndim != 2:
        raise ValueError("Both factors and risk_exposures must be 2D matrices.")

    n_assets, n_factors = F.shape
    n_risk_assets, n_risk_factors = X.shape

    if n_assets != n_risk_assets:
        raise ValueError(
            f"Asset count mismatch: factors have {n_assets} assets, "
            f"risk_exposures have {n_risk_assets} assets."
        )

    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(X)):
        raise ValueError("Input matrices contain non-finite values (NaN or Inf).")

    if weights is not None:
        if isinstance(weights, pd.Series):
            w = weights.to_numpy(dtype=float)
        else:
            w = np.asarray(weights, dtype=float)

        if len(w) != n_assets:
            raise ValueError(
                f"Weight length {len(w)} does not match asset count {n_assets}."
            )
        if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
            raise ValueError("Asset weights must be strictly positive and finite.")

        # Normalize weights to sum to N
        w = w / np.sum(w) * n_assets
        sqrt_w = np.sqrt(w)[:, np.newaxis]

        # Weighted QR decomposition: X_tilde = sqrt(W) X, F_tilde = sqrt(W) F
        X_tilde = X * sqrt_w
        F_tilde = F * sqrt_w

        Q_x, _ = np.linalg.qr(X_tilde, mode="reduced")
        F_neutral_tilde = F_tilde - Q_x @ (Q_x.T @ F_tilde)
        F_neutral = F_neutral_tilde / sqrt_w

    else:
        # Standard OLS via Thin QR
        Q_x, _ = np.linalg.qr(X, mode="reduced")
        F_neutral = F - Q_x @ (Q_x.T @ F)

    if standardize:
        means = np.mean(F_neutral, axis=0, keepdims=True)
        stds = np.std(F_neutral, axis=0, keepdims=True, ddof=1)
        stds = np.where(stds == 0.0, 1.0, stds)
        F_neutral = (F_neutral - means) / stds

    if is_df:
        return pd.DataFrame(F_neutral, index=index, columns=columns)
    return F_neutral


def build_sector_exposure_matrix(
    sectors: Union[np.ndarray, pd.Series, list],
    add_intercept: bool = False,
) -> pd.DataFrame:
    """Construct a one-hot encoded sector dummy risk matrix.

    Parameters
    ----------
    sectors : array-like of shape (N,)
        Categorical sector or industry identifiers for N assets.
    add_intercept : bool, default False
        If True, includes a column of ones for the market intercept.
        (Note: if all sector dummies are included, adding an intercept creates collinearity;
        QR decomposition safely handles it, but typically one dummy is dropped or intercept omitted).

    Returns
    -------
    pd.DataFrame
        One-hot encoded sector matrix of shape (N, S).
    """
    s_series = pd.Series(sectors)
    dummies = pd.get_dummies(s_series, prefix="sector", dtype=float)

    if add_intercept:
        dummies.insert(0, "market_intercept", 1.0)

    return dummies
