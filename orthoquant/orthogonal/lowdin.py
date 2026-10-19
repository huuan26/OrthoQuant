"""Löwdin Symmetric Factor Orthogonalization.

Notation:
    - N: Number of assets (rows).
    - K: Number of factors (columns), N >= K.
    - F in R^{N x K}: Factor matrix with rank(F) = K.
    - I_K in R^{K x K}: Identity matrix.
    - ||.||_F: Frobenius matrix norm, ||A||_F = sqrt(Tr(A^T A)).

Formulation:
    Given a factor matrix F in R^{N x K} representing K alpha signals on N assets,
    Löwdin symmetric orthogonalization finds the orthogonal matrix F_tilde in R^{N x K}
    that is closest to the original factors F in the Frobenius norm metric:

        min_{F_tilde} ||F - F_tilde||_F^2
        subject to F_tilde^T F_tilde = I_K

    The unique analytical solution is given by:
        F_tilde = F (F^T F)^{-1/2}

    Using Thin Singular Value Decomposition (SVD):
        F = U Sigma V^T
        F_tilde = U V^T

    which represents the unitary polar factor in the polar decomposition of F.
    Unlike Gram-Schmidt, Löwdin is permutation-symmetric and treats all factors
    symmetrically without preferential bias toward any specific input order.

References:
    - Löwdin, P.-O. (1950). "On the non-orthogonality problem connected with the use
      of atomic wave functions in the theory of molecules and crystals".
      The Journal of Chemical Physics, 18(3), 365–375.
    - Higham, N. J. (1988). "Computing a nearest symmetric positive semidefinite matrix".
      Linear Algebra and its Applications, 103, 103–118.
    - Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.), Section 8.4.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def lowdin_orthogonalize(
    factors: Union[np.ndarray, pd.DataFrame],
    method: str = "svd",
    rcond: float = 1e-12,
    standardize: bool = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Perform Löwdin symmetric orthogonalization on a factor matrix.

    Parameters
    ----------
    factors : np.ndarray or pd.DataFrame, shape (N, K)
        Factor matrix where N is the number of assets and K is the number of factors.
    method : str, default 'svd'
        Algorithm to compute the orthogonal basis:
        - 'svd': Computes thin SVD F = U Sigma V^T and returns U V^T.
                 Numerically superior and avoids explicitly forming F^T F.
        - 'eig': Computes eigendecomposition of the Gram matrix F^T F = V Lambda V^T
                 and computes F V Lambda^{-1/2} V^T.
    rcond : float, default 1e-12
        Cutoff for small singular values or eigenvalues to detect rank deficiency.
    standardize : bool, default False
        If True, z-score standardizes the resulting orthogonal columns to have zero mean
        and unit variance.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Orthonormal factor matrix F_tilde of shape (N, K). Preserves DataFrame index
        and column names if input is a DataFrame.

    Raises
    ------
    ValueError
        If factors have fewer assets than factors (N < K), contain NaNs/Infs,
        or are rank-deficient beyond rcond threshold.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        index = factors.index
        columns = factors.columns
        X = factors.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(factors, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D factor matrix, got {X.ndim}D array.")

    n_samples, n_factors = X.shape
    if n_samples < n_factors:
        raise ValueError(
            f"Number of assets N ({n_samples}) must be >= number of factors K ({n_factors})."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError("Factor matrix contains NaN, Inf, or non-finite values.")

    if standardize:
        X = X - np.mean(X, axis=0, keepdims=True)

    if method == "svd":
        # SVD: X = U Sigma V^T
        # full_matrices=False computes thin SVD where U is (N, K)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        max_s = s[0] if len(s) > 0 else 1.0
        if np.any(s < rcond * max_s):
            raise ValueError(
                f"Factor matrix is rank-deficient: smallest singular value {s[-1]:.2e} "
                f"< threshold {rcond * max_s:.2e}."
            )
        # Polar unitary factor: F_tilde = U V^T
        X_tilde = U @ Vt

    elif method == "eig":
        # Gram matrix: S = X^T X
        S = X.T @ X
        eigenvals, V = np.linalg.eigh(S)
        max_eig = eigenvals[-1] if len(eigenvals) > 0 else 1.0
        if np.any(eigenvals < rcond * max_eig):
            raise ValueError(
                f"Gram matrix is rank-deficient: smallest eigenvalue {eigenvals[0]:.2e} "
                f"< threshold {rcond * max_eig:.2e}."
            )
        # S^{-1/2} = V diag(lambda^{-1/2}) V^T
        inv_sqrt_diag = 1.0 / np.sqrt(eigenvals)
        S_inv_sqrt = (V * inv_sqrt_diag) @ V.T
        X_tilde = X @ S_inv_sqrt

    else:
        raise ValueError(f"Unknown orthogonalization method: '{method}'. Choose 'svd' or 'eig'.")

    if standardize:
        # Scale each orthonormal column to exact unit sample variance (ddof=1)
        X_tilde = X_tilde * np.sqrt(n_samples - 1)

    if is_df:
        return pd.DataFrame(X_tilde, index=index, columns=columns)
    return X_tilde


def lowdin_correlation_transform(
    factors: Union[np.ndarray, pd.DataFrame],
    rcond: float = 1e-12,
) -> Union[np.ndarray, pd.DataFrame]:
    """Compute Löwdin symmetric orthogonalization in the correlation space.

    First centers and scales factors to zero-mean and unit-variance, then transforms
    them by C^{-1/2} where C is the cross-sectional correlation matrix.

    Parameters
    ----------
    factors : np.ndarray or pd.DataFrame, shape (N, K)
        Factor matrix of N assets and K signals.
    rcond : float, default 1e-12
        Cutoff for eigenvalue positivity.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Correlation-orthogonalized factor matrix with zero pairwise cross-correlations.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        index = factors.index
        columns = factors.columns
        X = factors.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(factors, dtype=float).copy()

    # Center and scale
    means = np.mean(X, axis=0, keepdims=True)
    stds = np.std(X, axis=0, keepdims=True, ddof=1)
    if np.any(stds == 0.0):
        raise ValueError("Constant factor column detected with zero variance.")
    Z = (X - means) / stds

    # Correlation matrix C = (Z^T Z) / (N - 1)
    N = Z.shape[0]
    C = (Z.T @ Z) / (N - 1)

    eigenvals, V = np.linalg.eigh(C)
    max_eig = eigenvals[-1]
    if np.any(eigenvals < rcond * max_eig):
        raise ValueError(
            f"Correlation matrix is singular: smallest eigenvalue {eigenvals[0]:.2e}."
        )

    C_inv_sqrt = (V * (1.0 / np.sqrt(eigenvals))) @ V.T
    Z_ortho = Z @ C_inv_sqrt

    if is_df:
        return pd.DataFrame(Z_ortho, index=index, columns=columns)
    return Z_ortho
