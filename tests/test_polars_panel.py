"""Tests for Polars & polars-anta panel integration in OrthoQuant."""

import numpy as np
import polars as pl
import pytest

from orthoquant import (
    evaluate_panel_orthogonality,
    extract_polars_anta_factors,
    lowdin_correlation_transform,
    lowdin_orthogonalize,
    panel_neutralize_and_orthogonalize,
)


def _make_ohlcv_panel(n_symbols: int = 30, n_bars: int = 25, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sectors = ["Tech", "Finance", "Energy"]
    for s_idx in range(n_symbols):
        sym = f"SYM_{s_idx:02d}"
        sec = sectors[s_idx % len(sectors)]
        beta = 0.8 + 0.02 * s_idx
        price = 100.0 + s_idx
        for t in range(n_bars):
            ret = rng.normal(0.0005, 0.02)
            open_p = price
            close_p = price * (1.0 + ret)
            high_p = max(open_p, close_p) * (1.0 + abs(rng.normal(0.0, 0.005)))
            low_p = min(open_p, close_p) * (1.0 - abs(rng.normal(0.0, 0.005)))
            vol = float(rng.integers(10_000, 100_000))
            fwd_ret = rng.normal(0.0, 0.02)
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "sector": sec,
                    "beta": beta,
                    "weight": float(s_idx + 1),
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": vol,
                    "fwd_ret": fwd_ret,
                }
            )
            price = close_p
    return pl.DataFrame(rows)


def test_polars_dataframe_direct_lowdin():
    """Verify lowdin_orthogonalize and lowdin_correlation_transform accept pl.DataFrame."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((50, 1))
    mat = np.hstack([base + 0.3 * rng.standard_normal((50, 1)) for _ in range(3)])
    pl_df = pl.DataFrame(mat, schema=["f1", "f2", "f3"])

    ortho_df = lowdin_orthogonalize(pl_df, standardize=True)
    assert isinstance(ortho_df, pl.DataFrame)
    assert ortho_df.columns == ["f1", "f2", "f3"]
    arr = ortho_df.to_numpy()
    corr = np.corrcoef(arr, rowvar=False)
    np.testing.assert_allclose(corr, np.eye(3), atol=1e-10)

    corr_df = lowdin_correlation_transform(pl_df)
    assert isinstance(corr_df, pl.DataFrame)
    arr_c = corr_df.to_numpy()
    np.testing.assert_allclose(np.corrcoef(arr_c, rowvar=False), np.eye(3), atol=1e-10)


def test_end_to_end_polars_anta_panel_orthogonalization():
    """Verify end-to-end factor extraction via polars-anta, FWL neutralization, and Lowdin SVD."""
    panel = _make_ohlcv_panel(n_symbols=30, n_bars=25)
    enriched, factor_cols = extract_polars_anta_factors(panel, period=6)

    assert len(factor_cols) == 5
    for col in factor_cols:
        assert col in enriched.columns

    ortho_panel = panel_neutralize_and_orthogonalize(
        enriched,
        factor_cols=factor_cols,
        risk_cols=["beta"],
        sector_col="sector",
        weight_col="weight",
        method="lowdin",
        standardize=True,
    )

    ortho_cols = [f"{c}_ortho" for c in factor_cols]
    for col in ortho_cols:
        assert col in ortho_panel.columns

    report = evaluate_panel_orthogonality(
        ortho_panel,
        raw_factor_cols=factor_cols,
        ortho_factor_cols=ortho_cols,
        forward_return_col="fwd_ret",
        nw_lags=3,
    )

    assert report.n_timestamps > 0
    assert report.mean_ortho_max_vif == pytest.approx(1.0, abs=1e-5)
    assert report.mean_ortho_condition_number == pytest.approx(1.0, abs=1e-5)
    assert report.mean_ortho_enf == pytest.approx(float(len(factor_cols)), abs=1e-5)
    assert report.ic_summary_table is not None
    assert "ic_mean" in report.ic_summary_table.columns
    diag_dict = report.to_dict()
    assert diag_dict["mean_ortho_max_vif"] == pytest.approx(1.0, abs=1e-5)


def test_panel_methods_and_validation_errors():
    """Verify alternative panel orthogonalization methods and input validation."""
    panel = _make_ohlcv_panel(n_symbols=20, n_bars=15)
    enriched, factor_cols = extract_polars_anta_factors(panel, period=5)

    for method in ("lowdin_corr", "gram_schmidt"):
        res = panel_neutralize_and_orthogonalize(
            enriched,
            factor_cols=factor_cols[:3],
            method=method,
        )
        assert f"{factor_cols[0]}_ortho" in res.columns

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        extract_polars_anta_factors(pl.DataFrame({"symbol": ["A"], "timestamp": [1]}))

    with pytest.raises(ValueError, match="factor_cols must contain"):
        panel_neutralize_and_orthogonalize(enriched, factor_cols=[])

    with pytest.raises(ValueError, match="Unknown method"):
        panel_neutralize_and_orthogonalize(enriched, factor_cols=factor_cols, method="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="same length"):
        evaluate_panel_orthogonality(enriched, raw_factor_cols=["a", "b"], ortho_factor_cols=["a"])
