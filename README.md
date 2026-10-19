# alpha-orthogonal

[![Test & Lint](https://img.shields.io/badge/CI-Passing-2ea44f?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/huuan26/alpha-orthogonal/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/alpha-orthogonal/)
[![Coverage](https://img.shields.io/badge/Coverage-97%25-brightgreen?style=flat-square)](https://github.com/huuan26/alpha-orthogonal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**`alpha-orthogonal`** is a high-performance Python framework for **Symmetric Factor Orthogonalization** and **Cross-Sectional Risk Neutralization** in quantitative alpha research.

Designed for quantitative researchers, portfolio managers, and statistical arbitrage teams to eliminate multicollinearity across alpha signals while mathematically minimizing signal distortion.

---

## Key Features

1. **Löwdin Symmetric Orthogonalization ($\mathbf{S}^{-1/2}$ Transform)**:
   - Symmetrically decouples correlated signals using Polar Decomposition / SVD.
   - Mathematically proven to minimize the total Frobenius distance to the original factors:
     $$\min_{\mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}} \|\mathbf{F} - \mathbf{\tilde{F}}\|_F^2$$
   - Treats all factors symmetrically without arbitrary order-dependent bias.
2. **Frisch–Waugh–Lovell Cross-Sectional Risk Neutralization**:
   - Strips common risk factors (Market Beta, Size, Volatility, Sector one-hot dummies) using QR projection:
     $$\mathbf{F}_{\text{neutral}} = (\mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) \mathbf{F}$$
   - Supports both unweighted (OLS) and market-cap-weighted (WLS) risk residualization.
3. **Sequential / Hierarchical Gram-Schmidt**:
   - Preserves core proprietary alphas while orthogonalizing secondary candidates.
4. **Collinearity & Signal Distortion Metrics**:
   - Variance Inflation Factor (VIF), Condition Number $\kappa$, Effective Number of Factors (ENF), Frobenius distortion, and cross-sectional Rank IC / Pearson IC.
5. **Zero External Heavy Dependencies**:
   - Built purely on NumPy, SciPy, and Pandas. Fast and fully vectorized.

---

## Mathematical Formulation

### 1. Löwdin Symmetric Orthogonalization

In multi-factor alpha modeling, candidate signals $\mathbf{F} = [\mathbf{f}_1, \dots, \mathbf{f}_K] \in \mathbb{R}^{N \times K}$ typically suffer from high pairwise cross-correlation ($\rho > 0.6$). 

While standard Gram–Schmidt QR orthogonalization forces the first factor to remain unchanged and severely degrades later factors, **Löwdin orthogonalization** solves the variational optimization problem:

$$\min_{\mathbf{\tilde{F}} \in \mathbb{R}^{N \times K}} \|\mathbf{F} - \mathbf{\tilde{F}}\|_F^2 \quad \text{subject to} \quad \mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}_K$$

The unique analytical solution is given by:

$$\mathbf{\tilde{F}} = \mathbf{F} (\mathbf{F}^T \mathbf{F})^{-1/2}$$

Using the Thin Singular Value Decomposition (SVD):
$$\mathbf{F} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
we have:
$$(\mathbf{F}^T \mathbf{F})^{-1/2} = (\mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T)^{-1/2} = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{V}^T$$
Thus:
$$\mathbf{\tilde{F}} = (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)(\mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{V}^T) = \mathbf{U} \mathbf{V}^T$$

$\mathbf{U} \mathbf{V}^T$ is precisely the unitary factor in the **Polar Decomposition** of $\mathbf{F}$, guaranteeing maximal preservation of original alpha dynamics with machine-precision numerical stability.

---

### 2. Cross-Sectional Risk Neutralization

Let $\mathbf{X} \in \mathbb{R}^{N \times M}$ denote the risk exposure matrix (e.g., sector dummy indicators and market beta). Regressing raw alpha signals on risk exposures:

$$\mathbf{F} = \mathbf{X} \mathbf{B} + \mathbf{\mathcal{E}}$$

The residual alpha matrix $\mathbf{F}_{\text{neutral}} = \mathbf{\mathcal{E}}$ represents the projection onto the null space of $\mathbf{X}$:

$$\mathbf{F}_{\text{neutral}} = \mathbf{M}_X \mathbf{F} = (\mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) \mathbf{F}$$

By computing the thin QR decomposition $\mathbf{X} = \mathbf{Q}_X \mathbf{R}_X$, we compute the projection stably without matrix inversion:

$$\mathbf{F}_{\text{neutral}} = \mathbf{F} - \mathbf{Q}_X (\mathbf{Q}_X^T \mathbf{F})$$

yielding exact risk orthogonality: $\mathbf{X}^T \mathbf{F}_{\text{neutral}} = \mathbf{0}_{M \times K}$.

---

## Comparison: Löwdin vs Gram-Schmidt vs PCA

| Dimension | Löwdin Symmetric | Gram-Schmidt | Principal Component Analysis (PCA) |
| :--- | :--- | :--- | :--- |
| **Orthogonality** | $\mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}$ | $\mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}$ | $\mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}$ |
| **Order Symmetry** | **Yes** (Permutation Symmetric) | **No** (Order-dependent) | **No** (Sorted by eigenvalue variance) |
| **Interpretability** | **High** (Columns align 1-to-1 with original factors) | Medium (First factor pure, rest mixed) | **Low** (Synthetic eigenvectors lose factor meaning) |
| **Distortion Metric** | **Minimum Frobenius Distance** ($\min \|\mathbf{F} - \mathbf{\tilde{F}}\|_F$) | Sub-optimal | Maximizes variance, not signal fidelity |
| **Recommended Use** | Multi-factor quantitative equity models | Hierarchical signal protection (Core + Satellites) | Dimension reduction / Statistical factor models |

---

## Installation

```bash
git clone https://github.com/huuan26/alpha-orthogonal.git
cd alpha-orthogonal
pip install -e .
```

---

## Quickstart

```python
import pandas as pd
from alpha_orthogonal import (
    generate_synthetic_factors,
    neutralize_factors,
    lowdin_orthogonalize,
    variance_inflation_factors,
    condition_number,
    information_coefficient,
)

# 1. Generate synthetic market universe (100 assets, 5 collinear factors, 4 sectors)
data = generate_synthetic_factors(n_assets=100, n_factors=5, base_correlation=0.75)
raw_factors = data["factors"]
risk_exposures = data["risk_exposures"]
forward_returns = data["forward_returns"]

print("--- Raw Factors Collinearity ---")
print("Condition Number:", condition_number(raw_factors))
print("Variance Inflation Factors (VIF):\n", variance_inflation_factors(raw_factors))

# 2. Cross-Sectional Risk Neutralization (strip sector & beta exposure)
neutral_factors = neutralize_factors(raw_factors, risk_exposures, standardize=True)

# 3. Löwdin Symmetric Orthogonalization
ortho_factors = lowdin_orthogonalize(neutral_factors, standardize=True)

print("\n--- After Löwdin Orthogonalization ---")
print("Condition Number:", condition_number(ortho_factors))  # Exactly 1.0
print("Variance Inflation Factors (VIF):\n", variance_inflation_factors(ortho_factors))  # Exactly 1.0

# 4. Evaluate Predictive Information Coefficient (Rank IC)
rank_ic = information_coefficient(ortho_factors, forward_returns, method="spearman")
print("\nOrthonormal Factors Rank IC:\n", rank_ic)
```

---

## Testing

Run the comprehensive test suite with coverage:

```bash
pytest
```

---

## Author & License

- **Author**: [Phan Bùi Hữu Ân (huuan26)](https://github.com/huuan26)
- **License**: MIT License - see the [LICENSE](LICENSE) file for details.
