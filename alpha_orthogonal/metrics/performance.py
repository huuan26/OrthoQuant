"""Factor Performance and Information Coefficient (IC) Analysis."""

from typing import Union
import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    factors: Union[np.ndarray, pd.DataFrame],
    forward_returns: Union[np.ndarray, pd.Series],
    method: str = "spearman",
) -> Union[pd.Series, np.ndarray]:
    """Compute cross-sectional Information Coefficient (Rank IC or Pearson IC).

    Parameters
    ----------
    factors : np.ndarray or pd.DataFrame, shape (N, K)
        Factor signal values for N assets across K factors.
    forward_returns : np.ndarray or pd.Series, shape (N,)
        Forward realized returns for the N assets.
    method : str, default 'spearman'
        'spearman' for Rank IC, 'pearson' for Linear IC.

    Returns
    -------
    pd.Series or np.ndarray
        IC values for each factor column.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        cols = factors.columns
        F = factors.to_numpy(dtype=float)
    else:
        cols = None
        F = np.asarray(factors, dtype=float)

    if isinstance(forward_returns, pd.Series):
        r = forward_returns.to_numpy(dtype=float)
    else:
        r = np.asarray(forward_returns, dtype=float).squeeze()

    if F.shape[0] != len(r):
        raise ValueError(
            f"Asset count mismatch: factors have {F.shape[0]} rows, returns have {len(r)}."
        )

    n_factors = F.shape[1]
    ics = np.zeros(n_factors)

    for k in range(n_factors):
        fk = F[:, k]
        # Mask out NaNs
        valid = np.isfinite(fk) & np.isfinite(r)
        if np.sum(valid) < 3:
            ics[k] = np.nan
            continue

        if method == "spearman":
            res = stats.spearmanr(fk[valid], r[valid])
            ics[k] = res.statistic
        elif method == "pearson":
            res = stats.pearsonr(fk[valid], r[valid])
            ics[k] = res.statistic
        else:
            raise ValueError(f"Unknown IC method: '{method}'. Choose 'spearman' or 'pearson'.")

    if is_df:
        return pd.Series(ics, index=cols, name=f"{method.capitalize()}_IC")
    return ics


def marginal_r_squared(
    factors: Union[np.ndarray, pd.DataFrame],
    forward_returns: Union[np.ndarray, pd.Series],
) -> Union[pd.Series, np.ndarray]:
    """Compute individual R-squared contribution of each factor in OLS.

    For an orthonormal factor set Q (where Q^T Q = I), the multi-factor R^2
    is exactly the sum of individual factor R^2 values:
        R^2_total = sum_{k=1}^K R^2_k
    This provides an exact decomposition of predictive power without multicollinearity distortion.
    """
    is_df = isinstance(factors, pd.DataFrame)
    cols = factors.columns if is_df else None
    F = factors.to_numpy(dtype=float) if is_df else np.asarray(factors, dtype=float)
    y = (
        forward_returns.to_numpy(dtype=float)
        if isinstance(forward_returns, pd.Series)
        else np.asarray(forward_returns, dtype=float).squeeze()
    )

    # Standardize y
    y_centered = y - np.mean(y)
    ss_tot = np.sum(y_centered ** 2)
    if ss_tot == 0.0:
        r2s = np.zeros(F.shape[1])
    else:
        # For each column: projection and R^2
        r2s = np.zeros(F.shape[1])
        for k in range(F.shape[1]):
            fk = F[:, k]
            fk_norm_sq = np.sum(fk ** 2)
            if fk_norm_sq > 0.0:
                beta = np.dot(fk, y_centered) / fk_norm_sq
                y_pred = beta * fk
                ss_reg = np.sum(y_pred ** 2)
                r2s[k] = ss_reg / ss_tot

    if is_df:
        return pd.Series(r2s, index=cols, name="Marginal_R2")
    return r2s
