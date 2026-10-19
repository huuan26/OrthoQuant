# OrthoQuant

[![Test & Lint](https://img.shields.io/badge/CI-Passing-2ea44f?style=flat-square&logo=githubactions&logoColor=white)](https://github.com/huuan26/OrthoQuant/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/orthoquant/)
[![Coverage](https://img.shields.io/badge/Coverage-97%25-brightgreen?style=flat-square)](https://github.com/huuan26/OrthoQuant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**`OrthoQuant`** is a high-performance Python framework for **Symmetric Factor Orthogonalization** and **Cross-Sectional Risk Neutralization** in quantitative alpha research.

Designed for quantitative researchers, portfolio managers, and statistical arbitrage teams to eliminate multicollinearity across alpha signals while mathematically minimizing signal distortion.

> 📖 **Full Mathematical Derivations**: Complete proofs, lemmas, and spectral analyses are documented in [docs/mathematical_formulation.md](docs/mathematical_formulation.md).

---

## Mathematical Notation & Definitions

Throughout this framework, all mathematical objects are strictly defined as follows:

| Symbol | Mathematical Domain | Description |
| :--- | :--- | :--- |
| $`N \in \mathbb{N}`$ | Dimension | Number of assets in the cross-sectional universe ($`N \ge K`$) |
| $`K \in \mathbb{N}`$ | Dimension | Number of candidate alpha factors |
| $`M \in \mathbb{N}`$ | Dimension | Number of common risk exposures ($`M < N`$) |
| $`\mathbf{F} \in \mathbb{R}^{N \times K}`$ | Matrix | Factor matrix, $`\mathbf{F} = [\mathbf{f}_1, \dots, \mathbf{f}_K]`$, where $`\mathbf{f}_k \in \mathbb{R}^N`$ is factor $`k`$ |
| $`\mathbf{X} \in \mathbb{R}^{N \times M}`$ | Matrix | Risk exposure matrix (e.g., sector dummy one-hots, Market Beta, Size) |
| $`\mathbf{I}_K \in \mathbb{R}^{K \times K}`$ | Matrix | Identity matrix of size $`K \times K`$ |
| $`\mathcal{V}_K(\mathbb{R}^N)`$ | Stiefel Manifold | Set of $`N \times K`$ orthonormal matrices: $`\{ \mathbf{Q} \in \mathbb{R}^{N \times K} : \mathbf{Q}^T \mathbf{Q} = \mathbf{I}_K \}`$ |
| $`\|\cdot\|_F`$ | Matrix Norm | Frobenius norm: $`\lVert \mathbf{A} \rVert_F = \sqrt{\mathrm{Tr}(\mathbf{A}^T \mathbf{A})} = \sqrt{\sum_{i,j} a_{ij}^2}`$ |
| $`\mathrm{Tr}(\cdot)`$ | Operator | Matrix trace: $`\mathrm{Tr}(\mathbf{A}) = \sum_i a_{ii}`$ |
| $`\kappa_2(\cdot)`$ | Metric | Matrix condition number: $`\kappa_2(\mathbf{A}) = \sigma_{\max}(\mathbf{A}) / \sigma_{\min}(\mathbf{A})`$ |

---

## Mathematical Formulation & Theorems

### 1. Löwdin Symmetric Orthogonalization

#### Problem Statement
Given candidate alpha signals $`\mathbf{F} \in \mathbb{R}^{N \times K}`$ with $`\mathrm{rank}(\mathbf{F}) = K`$, we seek an orthonormal factor matrix $`\widetilde{\mathbf{F}} \in \mathcal{V}_K(\mathbb{R}^N)`$ that minimizes the total Frobenius distortion from the original signals:

\[\min_{\widetilde{\mathbf{F}} \in \mathbb{R}^{N \times K}} \lVert \mathbf{F} - \widetilde{\mathbf{F}} \rVert_F^2 \quad \text{subject to} \quad \widetilde{\mathbf{F}}^T \widetilde{\mathbf{F}} = \mathbf{I}_K \tag{1}\]

#### Theorem 1 (Optimality & Polar Decomposition)

The unique global minimizer to (1) is:

$$\widetilde{\mathbf{F}} = \mathbf{F} (\mathbf{F}^T \mathbf{F})^{-1/2} = \mathbf{U} \mathbf{V}^T \tag{2}$$

where $`\mathbf{F} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T`$ is the thin Singular Value Decomposition (SVD) of $`\mathbf{F}`$ ($`\mathbf{U} \in \mathbb{R}^{N \times K}`$, $`\mathbf{\Sigma} = \mathrm{diag}(\sigma_1, \dots, \sigma_K) \succ 0`$, $`\mathbf{V} \in \mathbb{R}^{K \times K}`$).

*Proof Sketch:* Expanding $`\lVert \mathbf{F} - \widetilde{\mathbf{F}} \rVert_F^2 = \mathrm{Tr}(\mathbf{F}^T \mathbf{F}) - 2 \mathrm{Tr}(\widetilde{\mathbf{F}}^T \mathbf{F}) + \mathrm{Tr}(\mathbf{I}_K)`$, minimizing distance is equivalent to maximizing trace overlap $`\max_{\widetilde{\mathbf{F}}^T \widetilde{\mathbf{F}} = \mathbf{I}} \mathrm{Tr}(\widetilde{\mathbf{F}}^T \mathbf{F})`$. Substituting SVD $`\mathbf{F} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T`$, we have $`\mathrm{Tr}(\widetilde{\mathbf{F}}^T \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T) = \mathrm{Tr}(\mathbf{Z}^T \mathbf{\Sigma})`$ where $`\mathbf{Z} \triangleq \mathbf{U}^T \widetilde{\mathbf{F}} \mathbf{V}`$. Since $`\lVert \mathbf{Z} \rVert_2 \le 1`$, $`\lvert z_{kk} \rvert \le 1`$, the upper bound $`\sum_k \sigma_k`$ is attained if and only if $`\mathbf{Z} = \mathbf{I}_K \iff \widetilde{\mathbf{F}} = \mathbf{U} \mathbf{V}^T`$. $`\blacksquare`$ (See [docs/mathematical_formulation.md](docs/mathematical_formulation.md) for full proof; cf. Löwdin (1950), Higham (1988)).

#### Theorem 2 (Permutation Symmetry / Equivariance)

Let $`\mathbf{P} \in \mathbb{R}^{K \times K}`$ be any permutation matrix ($`\mathbf{P}^T \mathbf{P} = \mathbf{I}_K`$). If factors are permuted as $`\mathbf{F}_{\pi} = \mathbf{F} \mathbf{P}`$, then:

$$\widetilde{\mathbf{F}}_{\pi} = \widetilde{\mathbf{F}} \mathbf{P} \tag{3}$$

*Proof Sketch:* $`(\mathbf{F} \mathbf{P})((\mathbf{F} \mathbf{P})^T (\mathbf{F} \mathbf{P}))^{-1/2} = \mathbf{F} \mathbf{P} (\mathbf{P}^T \mathbf{S} \mathbf{P})^{-1/2} = \mathbf{F} \mathbf{P} (\mathbf{P}^T \mathbf{S}^{-1/2} \mathbf{P}) = \mathbf{F} \mathbf{S}^{-1/2} \mathbf{P} = \widetilde{\mathbf{F}} \mathbf{P}`$. $`\blacksquare`$

---

### 2. Frisch–Waugh–Lovell Cross-Sectional Risk Neutralization

Let $`\mathbf{X} \in \mathbb{R}^{N \times M}`$ be the risk exposure matrix ($`M < N`$). Regressing factors on risk exposures:

$$\mathbf{F} = \mathbf{X} \mathbf{B} + \mathbf{\mathcal{E}}, \quad \text{where } \mathbf{B} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{F}$$

The residual factor matrix $`\mathbf{F}_{\text{neutral}} = \mathbf{\mathcal{E}}`$ is the orthogonal projection onto the null space of $`\mathbf{X}^T`$:

$$\mathbf{F}_{\text{neutral}} = \mathbf{M}_X \mathbf{F} = (\mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) \mathbf{F} \tag{4}$$

#### Theorem 3 (Exact Risk Orthogonality)

$$\mathbf{X}^T \mathbf{F}_{\text{neutral}} = \mathbf{0}_{M \times K} \tag{5}$$

*Proof:* $`\mathbf{X}^T \mathbf{M}_X = \mathbf{X}^T (\mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) = \mathbf{X}^T - (\mathbf{X}^T \mathbf{X})(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T = \mathbf{0}`$. $`\blacksquare`$ (Frisch & Waugh 1933; Lovell 1963).

#### Theorem 4 (Numerical Stability via QR Decomposition)

Let $`\mathbf{X} = \mathbf{Q}_X \mathbf{R}_X`$ be the thin QR decomposition of $`\mathbf{X}`$ ($`\mathbf{Q}_X^T \mathbf{Q}_X = \mathbf{I}_M`$, $`\mathbf{R}_X`$ invertible). Then:

$$\mathbf{M}_X \mathbf{F} = \mathbf{F} - \mathbf{Q}_X (\mathbf{Q}_X^T \mathbf{F}) \tag{6}$$

*Proof:* $`\mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T = \mathbf{Q}_X \mathbf{R}_X (\mathbf{R}_X^T \mathbf{Q}_X^T \mathbf{Q}_X \mathbf{R}_X)^{-1} \mathbf{R}_X^T \mathbf{Q}_X^T = \mathbf{Q}_X \mathbf{R}_X (\mathbf{R}_X^T \mathbf{R}_X)^{-1} \mathbf{R}_X^T \mathbf{Q}_X^T = \mathbf{Q}_X \mathbf{Q}_X^T`$. $`\blacksquare`$  
This formulation eliminates explicit inversion of $`\mathbf{X}^T \mathbf{X}`$, reducing the condition number from $`\kappa(\mathbf{X})^2`$ to $`\kappa(\mathbf{X})`$.

---

### 3. Collinearity & Performance Metrics

#### Variance Inflation Factor (VIF)

For standardized factors $`\mathbf{Z} \in \mathbb{R}^{N \times K}`$ with correlation matrix $`\mathbf{C} = \frac{1}{N-1} \mathbf{Z}^T \mathbf{Z}`$:

$$\mathrm{VIF}_k = (\mathbf{C}^{-1})_{kk} = \frac{1}{1 - R_k^2} \tag{7}$$

where $`R_k^2`$ is the coefficient of determination from regressing factor $`k`$ on all other $`K-1`$ factors (Belsley et al., 2005).  
For orthogonal factors, $`\mathbf{C} = \mathbf{I}_K \implies \mathrm{VIF}_k = 1.0`$ for all $`k`$.

#### Effective Number of Factors (ENF)

Using the spectral entropy of the eigenvalue distribution $`p_k = \lambda_k / K`$ of $`\mathbf{C}`$ (Meucci, 2009):

$$\mathrm{ENF} \triangleq \exp\left(-\sum_{k=1}^K p_k \ln(p_k)\right) \tag{8}$$

- $`\mathrm{ENF} = 1.0`$: Completely collinear factors (rank 1).
- $`\mathrm{ENF} = K`$: Completely orthogonal factors (maximum spectral entropy).

#### Theorem 5 (R² Additivity Under Orthogonality)

If factors $`\mathbf{Q} \in \mathcal{V}_K(\mathbb{R}^N)`$ are orthonormal ($`\mathbf{Q}^T \mathbf{Q} = \mathbf{I}_K`$), then in the multiple regression $`\mathbf{y} = \sum_{k=1}^K \beta_k \mathbf{q}_k + \boldsymbol{\varepsilon}`$, the overall $`R^2`$ decomposes strictly additively:

$$R^2_{\text{total}} = \sum_{k=1}^K R^2_k \tag{9}$$

*Proof:* Since $`\mathbf{q}_i^T \mathbf{q}_j = 0`$ for $`i \ne j`$, cross-terms $`\beta_i \beta_j \mathbf{q}_i^T \mathbf{q}_j`$ vanish identically from the regression sum of squares $`\lVert \sum_k \beta_k \mathbf{q}_k \rVert^2 = \sum_k \beta_k^2 \lVert \mathbf{q}_k \rVert^2`$. $`\blacksquare`$

---

## Comparison: Löwdin vs Gram-Schmidt vs PCA

| Dimension | Löwdin Symmetric | Gram-Schmidt | Principal Component Analysis (PCA) |
| :--- | :--- | :--- | :--- |
| **Orthogonality** | $`\widetilde{\mathbf{F}}^T \widetilde{\mathbf{F}} = \mathbf{I}`$ | $`\widetilde{\mathbf{F}}^T \widetilde{\mathbf{F}} = \mathbf{I}`$ | $`\widetilde{\mathbf{F}}^T \widetilde{\mathbf{F}} = \mathbf{I}`$ |
| **Order Symmetry** | **Yes** (Permutation Symmetric) | **No** (Order-dependent) | **No** (Sorted by eigenvalue variance) |
| **Interpretability** | **High** (Columns align 1-to-1 with original factors) | Medium (First factor pure, rest mixed) | **Low** (Synthetic eigenvectors lose factor meaning) |
| **Distortion Metric** | **Minimum Frobenius Distance** ($`\min \|\mathbf{F} - \widetilde{\mathbf{F}}\|_F`$) | Sub-optimal | Maximizes variance, not signal fidelity |
| **Recommended Use** | Multi-factor quantitative equity models | Hierarchical signal protection (Core + Satellites) | Dimension reduction / Statistical factor models |

---

## Installation

```bash
git clone https://github.com/huuan26/OrthoQuant.git
cd OrthoQuant
pip install -e .
```

---

## Quickstart

```python
import pandas as pd
from orthoquant import (
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

## Testing & Verification

Run the comprehensive test suite with coverage:

```bash
pytest
```

---

## Academic References & Bibliography

1. **Löwdin, P.-O. (1950).** On the non-orthogonality problem connected with the use of atomic wave functions in the theory of molecules and crystals. *The Journal of Chemical Physics*, 18(3), 365–375.
2. **Frisch, R., & Waugh, F. V. (1933).** Partial time regressions as compared with individual trends. *Econometrica*, 1(4), 387–401.
3. **Lovell, M. C. (1963).** Seasonal adjustment of economic time series and multiple regression analysis. *Journal of the American Statistical Association*, 58(304), 993–1010.
4. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.). Johns Hopkins University Press.
5. **Higham, N. J. (1988).** Computing a nearest symmetric positive semidefinite matrix. *Linear Algebra and its Applications*, 103, 103–118.
6. **Belsley, D. A., Kuh, E., & Welsch, R. E. (2005).** *Regression diagnostics: Identifying influential data and sources of collinearity*. John Wiley & Sons.
7. **Meucci, A. (2009).** Managing diversification. *Risk*, 22(5), 74–79.
8. **Kakushadze, Z. (2015).** Factor models and alpha streams. *The Journal of Investment Strategies*, 5(1), 1–45.

---

## Author & License

- **Author**: [Phan Bùi Hữu Ân (huuan26)](https://github.com/huuan26)
- **License**: MIT License - see the [LICENSE](LICENSE) file for details.
