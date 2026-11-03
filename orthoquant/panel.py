"""Polars & polars-anta Panel Integration for Cross-Sectional Factor Orthogonalization.

Bridges `polars-anta` (Rust-accelerated factor generation, microstructure extraction,
cross-sectional winsorization/z-score, and Newey-West IC evaluation) with `OrthoQuant`
(Frisch-Waugh-Lovell Thin-QR risk neutralization and Lowdin SVD symmetric orthogonalization)
across multi-period panel datasets `(timestamp, symbol, ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import polars as pl
import polars_anta as pa  # noqa: F401 - registers .anta and .ta namespaces

from orthoquant.neutralization.cross_sectional import (
    build_sector_exposure_matrix,
    neutralize_factors,
)
from orthoquant.orthogonal.gram_schmidt import gram_schmidt_orthogonalize
from orthoquant.orthogonal.lowdin import (
    lowdin_correlation_transform,
    lowdin_orthogonalize,
)
from orthoquant.metrics.collinearity import (
    condition_number,
    effective_number_of_factors,
    frobenius_distortion,
    variance_inflation_factors,
)


@dataclass
class PanelOrthogonalityReport:
    """Summary diagnostics for multi-period panel orthogonalization."""

    n_timestamps: int
    n_factors: int
    mean_raw_max_vif: float
    mean_ortho_max_vif: float
    mean_raw_condition_number: float
    mean_ortho_condition_number: float
    mean_raw_enf: float
    mean_ortho_enf: float
    mean_frobenius_distortion: float
    ic_summary_table: Optional[pl.DataFrame] = None

    def to_dict(self) -> Dict[str, float]:
        """Return scalar diagnostic metrics as a dictionary."""
        return {
            "n_timestamps": self.n_timestamps,
            "n_factors": self.n_factors,
            "mean_raw_max_vif": round(self.mean_raw_max_vif, 6),
            "mean_ortho_max_vif": round(self.mean_ortho_max_vif, 6),
            "mean_raw_condition_number": round(self.mean_raw_condition_number, 6),
            "mean_ortho_condition_number": round(self.mean_ortho_condition_number, 6),
            "mean_raw_enf": round(self.mean_raw_enf, 6),
            "mean_ortho_enf": round(self.mean_ortho_enf, 6),
            "mean_frobenius_distortion": round(self.mean_frobenius_distortion, 6),
        }


def extract_polars_anta_factors(
    df: pl.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    period: int = 10,
    winsorize: bool = True,
    zscore: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """Generate a multi-factor cross-sectional panel from OHLCV data using `polars-anta`.

    Extracts correlated momentum, microstructure, and WorldQuant formulaic alpha features
    per symbol and applies cross-sectional MAD winsorization and z-score standardization
    within each timestamp slice via `polars-anta`.

    Parameters
    ----------
    df : pl.DataFrame
        Input OHLCV panel sorted by `(symbol_col, time_col)` or `(time_col, symbol_col)`.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'timestamp'
        Timestamp/date grouping column.
    open_col, high_col, low_col, close_col, volume_col : str
        OHLCV column names.
    period : int, default 10
        Lookback window for rolling technical and microstructure primitives.
    winsorize : bool, default True
        Whether to apply `pa.cs_winsorize(..., method="mad")` per `time_col`.
    zscore : bool, default True
        Whether to apply `pa.cs_zscore(...)` per `time_col`.

    Returns
    -------
    Tuple[pl.DataFrame, List[str]]
        (Polars DataFrame enriched with factor columns, list of generated factor column names).
    """
    required = {symbol_col, time_col, open_col, high_col, low_col, close_col, volume_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    sorted_df = df.sort([symbol_col, time_col])
    factor_cols = [
        "f_rsi",
        "f_roc",
        "f_vpin",
        "f_alpha101",
        "f_alpha12",
    ]

    enriched = sorted_df.with_columns(
        pl.col(close_col).anta.rsi(period=period).over(symbol_col).alias("f_rsi"),
        pl.col(close_col).anta.roc(period=period).over(symbol_col).alias("f_roc"),
        pa.vpin(pl.col(close_col), pl.col(volume_col), period=period)
        .over(symbol_col)
        .alias("f_vpin"),
        pa.alpha_101(
            pl.col(open_col),
            pl.col(high_col),
            pl.col(low_col),
            pl.col(close_col),
        )
        .over(symbol_col)
        .alias("f_alpha101"),
        pa.alpha_012(pl.col(close_col), pl.col(volume_col))
        .over(symbol_col)
        .alias("f_alpha12"),
    )

    if winsorize:
        enriched = enriched.with_columns(
            [
                pa.cs_winsorize(pl.col(c), over=time_col, method="mad", threshold=3.0).alias(c)
                for c in factor_cols
            ]
        )

    if zscore:
        enriched = enriched.with_columns(
            [pa.cs_zscore(pl.col(c), over=time_col).alias(c) for c in factor_cols]
        )

    return enriched.sort([time_col, symbol_col]), factor_cols


def panel_neutralize_and_orthogonalize(
    df: pl.DataFrame,
    factor_cols: Sequence[str],
    time_col: str = "timestamp",
    symbol_col: str = "symbol",
    risk_cols: Optional[Sequence[str]] = None,
    sector_col: Optional[str] = None,
    weight_col: Optional[str] = None,
    method: Literal["lowdin", "lowdin_corr", "gram_schmidt"] = "lowdin",
    standardize: bool = True,
    rcond: float = 1e-10,
    suffix: str = "_ortho",
) -> pl.DataFrame:
    """Neutralize and symmetrically orthogonalize factors across each cross-section of a Polars panel.

    For each timestamp $t$, extracts the cross-sectional factor matrix $\\mathbf{F}_t \\in \\mathbb{R}^{N_t \\times K}$,
    optionally projects out continuous/sector risk exposures $\\mathbf{X}_t$ via Frisch-Waugh-Lovell Thin-QR,
    and applies Lowdin symmetric SVD orthogonalization ($\\widetilde{\\mathbf{F}}_t = \\mathbf{U}_t \\mathbf{V}_t^T$).

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars panel DataFrame.
    factor_cols : Sequence[str]
        Column names of the $K$ candidate alpha factors.
    time_col : str, default 'timestamp'
        Cross-sectional time grouping column.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    risk_cols : Sequence[str], optional
        Continuous risk exposure columns (e.g., market beta, log size, volatility).
    sector_col : str, optional
        Categorical sector column to one-hot encode for sector neutralization.
    weight_col : str, optional
        Asset weight column for WLS risk neutralization.
    method : {'lowdin', 'lowdin_corr', 'gram_schmidt'}, default 'lowdin'
        Orthogonalization method applied at each cross-section.
    standardize : bool, default True
        Whether to standardize orthogonalized columns to zero mean and unit variance per timestamp.
    rcond : float, default 1e-10
        Singular value cutoff threshold.
    suffix : str, default '_ortho'
        Suffix appended to the orthogonalized output factor column names.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing the original columns plus the orthogonalized factor columns
        (for valid cross-sections where $N_t > K$ and values are finite).
    """
    if not factor_cols:
        raise ValueError("factor_cols must contain at least one factor column name.")
    if method not in ("lowdin", "lowdin_corr", "gram_schmidt"):
        raise ValueError(
            f"Unknown method '{method}'. Choose 'lowdin', 'lowdin_corr', or 'gram_schmidt'."
        )

    factor_list = list(factor_cols)
    risk_list = list(risk_cols) if risk_cols is not None else []
    ortho_cols = [f"{c}{suffix}" for c in factor_list]
    K = len(factor_list)

    sorted_df = df.sort([time_col, symbol_col])
    partitions = sorted_df.partition_by(time_col, maintain_order=True)
    processed_parts: List[pl.DataFrame] = []

    for part in partitions:
        n_rows = part.height
        F_raw = part.select(factor_list).to_numpy()
        valid_mask = np.isfinite(F_raw).all(axis=1)

        if risk_list:
            X_cont = part.select(risk_list).to_numpy()
            valid_mask &= np.isfinite(X_cont).all(axis=1)
        else:
            X_cont = None

        if weight_col is not None:
            w_raw = part.get_column(weight_col).to_numpy()
            valid_mask &= np.isfinite(w_raw) & (w_raw > 0.0)
        else:
            w_raw = None

        out_mat = np.full((n_rows, K), np.nan, dtype=np.float64)
        n_valid = int(np.sum(valid_mask))

        if n_valid > K:
            F_sub = F_raw[valid_mask].astype(np.float64, copy=True)

            # Build risk matrix if requested
            risk_blocks = []
            if X_cont is not None:
                risk_blocks.append(X_cont[valid_mask].astype(np.float64, copy=False))
            if sector_col is not None:
                sec_vals = part.get_column(sector_col).to_list()
                sec_valid = [sec_vals[i] for i in range(n_rows) if valid_mask[i]]
                sec_dummies = build_sector_exposure_matrix(sec_valid, add_intercept=False)
                risk_blocks.append(sec_dummies.to_numpy(dtype=np.float64))

            if risk_blocks:
                X_sub = np.hstack(risk_blocks)
                if n_valid > X_sub.shape[1]:
                    w_sub = w_raw[valid_mask] if w_raw is not None else None
                    F_sub = neutralize_factors(
                        F_sub,
                        X_sub,
                        weights=w_sub,
                        standardize=True,
                    )

            try:
                if method == "lowdin":
                    F_ortho = lowdin_orthogonalize(
                        F_sub,
                        method="svd",
                        rcond=rcond,
                        standardize=standardize,
                    )
                elif method == "lowdin_corr":
                    F_ortho = lowdin_correlation_transform(F_sub, rcond=rcond)
                else:
                    F_ortho = gram_schmidt_orthogonalize(F_sub, standardize=standardize)
                out_mat[valid_mask] = F_ortho
            except ValueError:
                # If a particular timestamp slice is singular/constant, leave NaNs
                pass

        new_cols = [
            pl.Series(name=ortho_cols[k], values=out_mat[:, k], dtype=pl.Float64).fill_nan(None)
            for k in range(K)
        ]
        processed_parts.append(part.with_columns(new_cols))

    return pl.concat(processed_parts, how="vertical")


def evaluate_panel_orthogonality(
    df: pl.DataFrame,
    raw_factor_cols: Sequence[str],
    ortho_factor_cols: Sequence[str],
    time_col: str = "timestamp",
    forward_return_col: Optional[str] = None,
    nw_lags: int = 5,
) -> PanelOrthogonalityReport:
    """Evaluate collinearity reduction and `polars-anta` Newey-West IC across a Polars panel.

    Parameters
    ----------
    df : pl.DataFrame
        Polars DataFrame containing both `raw_factor_cols` and `ortho_factor_cols`.
    raw_factor_cols : Sequence[str]
        Names of the original correlated factor columns.
    ortho_factor_cols : Sequence[str]
        Names of the orthogonalized factor columns.
    time_col : str, default 'timestamp'
        Timestamp grouping column.
    forward_return_col : str, optional
        Forward asset return column; if provided, computes `polars_anta.ic_summary`
        with Newey-West HAC t-statistics for all raw and orthogonalized factors.
    nw_lags : int, default 5
        Newey-West autocorrelation lags passed to `polars_anta.ic_summary`.

    Returns
    -------
    PanelOrthogonalityReport
        Structured summary of panel VIF, condition number, ENF, Frobenius distortion,
        and optional `polars-anta` IC summary table.
    """
    raw_list = list(raw_factor_cols)
    ortho_list = list(ortho_factor_cols)
    if len(raw_list) != len(ortho_list):
        raise ValueError("raw_factor_cols and ortho_factor_cols must have the same length.")

    K = len(raw_list)
    partitions = df.partition_by(time_col, maintain_order=True)

    raw_vifs: List[float] = []
    ortho_vifs: List[float] = []
    raw_kappas: List[float] = []
    ortho_kappas: List[float] = []
    raw_enfs: List[float] = []
    ortho_enfs: List[float] = []
    distortions: List[float] = []

    for part in partitions:
        F_raw = part.select(raw_list).to_numpy()
        F_ort = part.select(ortho_list).to_numpy()
        mask = np.isfinite(F_raw).all(axis=1) & np.isfinite(F_ort).all(axis=1)
        if int(np.sum(mask)) <= K:
            continue

        R_sub = F_raw[mask].astype(np.float64)
        O_sub = F_ort[mask].astype(np.float64)
        if np.any(np.std(R_sub, axis=0, ddof=1) == 0.0) or np.any(
            np.std(O_sub, axis=0, ddof=1) == 0.0
        ):
            continue

        raw_vifs.append(float(np.max(variance_inflation_factors(R_sub))))
        ortho_vifs.append(float(np.max(variance_inflation_factors(O_sub))))
        raw_kappas.append(condition_number(R_sub))
        ortho_kappas.append(condition_number(O_sub))
        raw_enfs.append(effective_number_of_factors(R_sub))
        ortho_enfs.append(effective_number_of_factors(O_sub))

        # Standardize R_sub to unit variance for fair Frobenius comparison
        R_z = (R_sub - np.mean(R_sub, axis=0, keepdims=True)) / np.std(
            R_sub, axis=0, keepdims=True, ddof=1
        )
        O_z = (O_sub - np.mean(O_sub, axis=0, keepdims=True)) / np.std(
            O_sub, axis=0, keepdims=True, ddof=1
        )
        distortions.append(frobenius_distortion(R_z, O_z))

    ic_table: Optional[pl.DataFrame] = None
    if forward_return_col is not None and forward_return_col in df.columns:
        valid_df = df.drop_nulls(subset=raw_list + ortho_list + [forward_return_col])
        rows = []
        for col_name, group_label in [(c, "raw") for c in raw_list] + [
            (c, "ortho") for c in ortho_list
        ]:
            stats = pa.ic_summary(
                valid_df,
                factor_col=col_name,
                return_col=forward_return_col,
                group_by=time_col,
                method="spearman",
                nw_lags=nw_lags,
            )
            rows.append(
                {
                    "factor": col_name,
                    "stage": group_label,
                    **stats,
                }
            )
        if rows:
            ic_table = pl.DataFrame(rows)

    return PanelOrthogonalityReport(
        n_timestamps=len(raw_vifs),
        n_factors=K,
        mean_raw_max_vif=float(np.mean(raw_vifs)) if raw_vifs else float("nan"),
        mean_ortho_max_vif=float(np.mean(ortho_vifs)) if ortho_vifs else float("nan"),
        mean_raw_condition_number=float(np.mean(raw_kappas)) if raw_kappas else float("nan"),
        mean_ortho_condition_number=float(np.mean(ortho_kappas)) if ortho_kappas else float("nan"),
        mean_raw_enf=float(np.mean(raw_enfs)) if raw_enfs else float("nan"),
        mean_ortho_enf=float(np.mean(ortho_enfs)) if ortho_enfs else float("nan"),
        mean_frobenius_distortion=float(np.mean(distortions)) if distortions else float("nan"),
        ic_summary_table=ic_table,
    )
