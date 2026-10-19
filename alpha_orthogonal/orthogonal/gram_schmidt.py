"""Sequential / Hierarchical Gram-Schmidt Factor Orthogonalization.

Formulation:
    Given factor matrix F = [f_1, f_2, ..., f_K], sequential orthogonalization creates
    an orthonormal basis Q = [q_1, q_2, ..., q_K] such that:
        span(q_1, ..., q_k) = span(f_1, ..., f_k)  for all k in {1, ..., K}

    Using the QR decomposition:
        F = Q R
    where Q in R^{N x K} has orthonormal columns (Q^T Q = I_K) and R in R^{K x K}
    is upper triangular.

    This method is appropriate when there is an explicit hierarchy or precedence among
    signals (e.g. Core proprietary alpha preserved first, followed by incremental
    residualization of secondary candidate signals).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def gram_schmidt_orthogonalize(
    factors: Union[np.ndarray, pd.DataFrame],
    priority_order: Optional[Union[List[str], List[int]]] = None,
    return_r: bool = False,
    standardize: bool = False,
) -> Union[
    np.ndarray,
    pd.DataFrame,
    Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray],
]:
    """Perform sequential Gram-Schmidt orthogonalization via QR decomposition.

    Parameters
    ----------
    factors : np.ndarray or pd.DataFrame, shape (N, K)
        Factor matrix with N assets and K factors.
    priority_order : list of str or int, optional
        Explicit priority order for orthogonalization.
        If DataFrame is provided, column names can be specified.
        If array is provided, column indices should be specified.
        If None, the original column ordering is preserved.
    return_r : bool, default False
        If True, also returns the upper-triangular transition matrix R from F = Q R.
    standardize : bool, default False
        If True, z-score standardizes the resulting orthogonal columns to unit variance.

    Returns
    -------
    Q : np.ndarray or pd.DataFrame
        Orthonormal factor matrix of shape (N, K).
    R : np.ndarray (optional)
        Upper triangular transition matrix of shape (K, K), returned if return_r=True.
    """
    is_df = isinstance(factors, pd.DataFrame)
    if is_df:
        index = factors.index
        orig_columns = list(factors.columns)
        if priority_order is not None:
            ordered_columns = [
                c if isinstance(c, str) else orig_columns[c] for c in priority_order
            ]
            df_reordered = factors[ordered_columns]
            columns = ordered_columns
        else:
            df_reordered = factors
            columns = orig_columns
        X = df_reordered.to_numpy(dtype=float, copy=True)
    else:
        X = np.asarray(factors, dtype=float)
        if priority_order is not None:
            order_indices = [int(i) for i in priority_order]
            X = X[:, order_indices]
        columns = None

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

    # Economic QR decomposition: Q is (N, K), R is (K, K)
    Q, R = np.linalg.qr(X, mode="reduced")

    # Enforce positive diagonal elements on R for uniqueness and positive correlation
    # with the original signals: diag(R) > 0
    diag_signs = np.sign(np.diag(R))
    diag_signs[diag_signs == 0.0] = 1.0
    Q = Q * diag_signs
    R = diag_signs[:, np.newaxis] * R

    if standardize:
        Q = Q * np.sqrt(n_samples - 1)

    if is_df:
        res_q = pd.DataFrame(Q, index=index, columns=columns)
    else:
        res_q = Q

    if return_r:
        return res_q, R
    return res_q
