"""Collinearity, Condition Number, and Distortion Metrics for Factor Sets.

References:
    - Belsley, D. A., Kuh, E., & Welsch, R. E. (2005). Regression diagnostics: Identifying
      influential data and sources of collinearity. John Wiley & Sons.
    - Meucci, A. (2009). "Managing diversification". Risk, 22(5), 74–79.
"""

from typing import Union
import numpy as np
import pandas as pd


def correlation_matrix(
    factors: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute cross-sectional Pearson correlation matrix of factors."""
    if isinstance(factors, pd.DataFrame):
        return factors.corr(method="pearson")
    X = np.asarray(factors, dtype=float)
    return np.corrcoef(X, rowvar=False)


def condition_number(
    factors: Union[np.ndarray, pd.DataFrame], rcond: float = 1e-14
) -> float:
    """Compute the 2-norm condition number of the factor matrix.

    kappa(F) = sigma_max / sigma_min

    An orthogonal set achieves the optimal condition number of 1.0.
    A collinear factor set has kappa >> 1 (often > 100 or 1000).
    """
    X = factors.to_numpy() if isinstance(factors, pd.DataFrame) else np.asarray(factors)
    s = np.linalg.svd(X, compute_uv=False)
    if len(s) == 0 or s[-1] <= rcond * s[0]:
        return float("inf")
    return float(s[0] / s[-1])


def variance_inflation_factors(
    factors: Union[np.ndarray, pd.DataFrame]
) -> Union[pd.Series, np.ndarray]:
    """Calculate the Variance Inflation Factor (VIF) for each factor.

    Formulation:
        For standardized variables, VIF_k is the k-th diagonal element of the
        inverse correlation matrix:
            VIF_k = (C^{-1})_{kk} = 1 / (1 - R_k^2)

        - VIF = 1.0: Zero collinearity (orthogonal).
        - VIF > 5.0 or 10.0: Severe multicollinearity requiring orthogonalization.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        cols = factors.columns
        X = factors.to_numpy(dtype=float)
    else:
        cols = None
        X = np.asarray(factors, dtype=float)

    n_samples, n_factors = X.shape
    # Center and scale
    means = np.mean(X, axis=0, keepdims=True)
    stds = np.std(X, axis=0, keepdims=True, ddof=1)
    stds = np.where(stds == 0.0, 1.0, stds)
    Z = (X - means) / stds

    # Correlation matrix C
    C = (Z.T @ Z) / (n_samples - 1)

    try:
        inv_C = np.linalg.inv(C)
        vifs = np.diag(inv_C)
    except np.linalg.LinAlgError:
        # Singular correlation matrix -> VIF is infinite
        vifs = np.full(n_factors, float("inf"))

    if is_df:
        return pd.Series(vifs, index=cols, name="VIF")
    return vifs


def frobenius_distortion(
    factors_original: Union[np.ndarray, pd.DataFrame],
    factors_orthogonal: Union[np.ndarray, pd.DataFrame],
    relative: bool = True,
) -> float:
    """Compute the Frobenius signal distortion distance between original and orthogonal factors.

    Distortion = ||F_orig - F_ortho||_F / ||F_orig||_F

    Löwdin symmetric orthogonalization is mathematically proven to minimize this metric
    among all possible orthonormal transformations.
    """
    F1 = (
        factors_original.to_numpy(dtype=float)
        if isinstance(factors_original, pd.DataFrame)
        else np.asarray(factors_original, dtype=float)
    )
    F2 = (
        factors_orthogonal.to_numpy(dtype=float)
        if isinstance(factors_orthogonal, pd.DataFrame)
        else np.asarray(factors_orthogonal, dtype=float)
    )

    diff = F1 - F2
    frob_diff = float(np.linalg.norm(diff, ord="fro"))

    if relative:
        frob_orig = float(np.linalg.norm(F1, ord="fro"))
        if frob_orig == 0.0:
            return frob_diff
        return frob_diff / frob_orig
    return frob_diff


def effective_number_of_factors(
    factors: Union[np.ndarray, pd.DataFrame]
) -> float:
    """Compute the Effective Number of Factors (ENF) using spectral entropy.

    Formulation:
        Let p_i = lambda_i / sum(lambda_j) be the normalized eigenvalues of the
        correlation matrix. The Shannon entropy is:
            H = - sum_{i=1}^K p_i ln(p_i)
        The effective number of factors is:
            ENF = exp(H)

        - ENF = 1: All factors are perfectly collinear (rank 1).
        - ENF = K: All K factors are completely orthogonal.
    """
    X = factors.to_numpy() if isinstance(factors, pd.DataFrame) else np.asarray(factors)
    # Correlation matrix
    C = np.corrcoef(X, rowvar=False)
    eigenvals = np.linalg.eigvalsh(C)
    eigenvals = np.maximum(eigenvals, 1e-15)
    p = eigenvals / np.sum(eigenvals)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))
