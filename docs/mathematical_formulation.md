# Mathematical Formulation & Theoretical Foundations

This document provides rigorous mathematical formulations, proofs, and academic citations for the methods implemented in **`alpha-orthogonal`**.

---

## 1. Notation & Matrix Conventions

Throughout this document and codebase, we adhere to the following mathematical conventions:

| Symbol | Mathematical Domain | Description |
| :--- | :--- | :--- |
| $N \in \mathbb{N}$ | Dimension | Number of assets in the cross-sectional universe ($N \ge K$) |
| $K \in \mathbb{N}$ | Dimension | Number of alpha factors / signals |
| $M \in \mathbb{N}$ | Dimension | Number of common risk exposures / constraints ($M < N$) |
| $\mathbf{F} \in \mathbb{R}^{N \times K}$ | Matrix | Raw factor signal matrix, $\mathbf{F} = [\mathbf{f}_1, \mathbf{f}_2, \dots, \mathbf{f}_K]$ |
| $\mathbf{f}_k \in \mathbb{R}^N$ | Vector | Column vector representing the $k$-th factor score across all $N$ assets |
| $\mathbf{X} \in \mathbb{R}^{N \times M}$ | Matrix | Risk exposure matrix (e.g., sector dummy one-hots, Market Beta, Size) |
| $\mathbf{I}_K \in \mathbb{R}^{K \times K}$ | Matrix | $K \times K$ identity matrix |
| $\mathcal{V}_K(\mathbb{R}^N)$ | Stiefel Manifold | Set of $N \times K$ orthonormal matrices: $\{ \mathbf{Q} \in \mathbb{R}^{N \times K} : \mathbf{Q}^T \mathbf{Q} = \mathbf{I}_K \}$ |
| $\lVert \mathbf{A} \rVert_F$ | Norm | Frobenius norm: $\lVert \mathbf{A} \rVert_F = \sqrt{\mathrm{Tr}(\mathbf{A}^T \mathbf{A})} = \sqrt{\sum_{i,j} a_{ij}^2}$ |
| $\lVert \mathbf{A} \rVert_2$ | Norm | Spectral / operator norm: $\lVert \mathbf{A} \rVert_2 = \sigma_{\max}(\mathbf{A})$ |
| $\mathrm{Tr}(\mathbf{A})$ | Operator | Trace operator: $\sum_i a_{ii}$ |
| $\mathrm{Col}(\mathbf{A})$ | Subspace | Column space (range) of matrix $\mathbf{A}$ |
| $\mathrm{Null}(\mathbf{A})$ | Subspace | Null space (kernel) of matrix $\mathbf{A}$ |

---

## 2. Löwdin Symmetric Orthogonalization

### 2.1 Optimization Problem Formulation

Given a factor matrix $\mathbf{F} \in \mathbb{R}^{N \times K}$ with $\mathrm{rank}(\mathbf{F}) = K$, we seek an orthonormal factor matrix $\mathbf{\tilde{F}} \in \mathcal{V}_K(\mathbb{R}^N)$ that is as close as possible to the original factor matrix $\mathbf{F}$ in the Frobenius metric:

$$\min_{\mathbf{\tilde{F}} \in \mathbb{R}^{N \times K}} \lVert \mathbf{F} - \mathbf{\tilde{F}} \rVert_F^2 \quad \text{subject to} \quad \mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}_K \tag{1}$$

### 2.2 Theorem: Closed-Form Solution via Polar Decomposition

**Theorem 1.** *The unique global minimizer to (1) is given by:*

$$\mathbf{\tilde{F}} = \mathbf{F} (\mathbf{F}^T \mathbf{F})^{-1/2} = \mathbf{U} \mathbf{V}^T \tag{2}$$

*where $\mathbf{F} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ is the thin Singular Value Decomposition (SVD) of $\mathbf{F}$ ($\mathbf{U} \in \mathbb{R}^{N \times K}$, $\mathbf{\Sigma} = \mathrm{diag}(\sigma_1, \dots, \sigma_K) \succ 0$, $\mathbf{V} \in \mathbb{R}^{K \times K}$).*

#### Proof:

Expand the squared Frobenius norm objective in (1):

$$\lVert \mathbf{F} - \mathbf{\tilde{F}} \rVert_F^2 = \mathrm{Tr}\left((\mathbf{F} - \mathbf{\tilde{F}})^T (\mathbf{F} - \mathbf{\tilde{F}})\right) = \mathrm{Tr}(\mathbf{F}^T \mathbf{F}) - 2 \mathrm{Tr}(\mathbf{\tilde{F}}^T \mathbf{F}) + \mathrm{Tr}(\mathbf{\tilde{F}}^T \mathbf{\tilde{F}})$$

Because $\mathbf{\tilde{F}}^T \mathbf{\tilde{F}} = \mathbf{I}_K$, the last term is $\mathrm{Tr}(\mathbf{I}_K) = K$ (constant). Similarly, $\mathrm{Tr}(\mathbf{F}^T \mathbf{F}) = \sum_{k=1}^K \sigma_k^2$ is constant with respect to $\mathbf{\tilde{F}}$. Therefore, minimizing the objective is strictly equivalent to maximizing the trace overlap:

$$\max_{\mathbf{\tilde{F}} \in \mathcal{V}_K(\mathbb{R}^N)} \mathrm{Tr}(\mathbf{\tilde{F}}^T \mathbf{F}) \tag{3}$$

Substitute the thin SVD $\mathbf{F} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ into (3):

$$\mathrm{Tr}(\mathbf{\tilde{F}}^T \mathbf{F}) = \mathrm{Tr}(\mathbf{\tilde{F}}^T \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T) = \mathrm{Tr}(\mathbf{V}^T \mathbf{\tilde{F}}^T \mathbf{U} \mathbf{\Sigma})$$

Define the matrix $\mathbf{Z} \triangleq \mathbf{U}^T \mathbf{\tilde{F}} \mathbf{V} \in \mathbb{R}^{K \times K}$. Using the sub-multiplicative property of the operator norm:

$$\lVert \mathbf{Z} \rVert_2 \le \lVert \mathbf{U}^T \rVert_2 \lVert \mathbf{\tilde{F}} \rVert_2 \lVert \mathbf{V} \rVert_2 = 1 \cdot 1 \cdot 1 = 1$$

Since $\lVert \mathbf{Z} \rVert_2 \le 1$, each diagonal entry of $\mathbf{Z}$ satisfies $\lvert z_{kk} \rvert \le 1$. Therefore:

$$\mathrm{Tr}(\mathbf{Z}^T \mathbf{\Sigma}) = \sum_{k=1}^K z_{kk} \sigma_k \le \sum_{k=1}^K \sigma_k$$

Because all singular values $\sigma_k > 0$, equality is attained if and only if $z_{kk} = 1$ for all $k \in \{1, \dots, K\}$, which requires $\mathbf{Z} = \mathbf{I}_K$. Thus:

$$\mathbf{U}^T \mathbf{\tilde{F}} \mathbf{V} = \mathbf{I}_K \implies \mathbf{\tilde{F}} = \mathbf{U} \mathbf{V}^T$$

To verify the inverse square root representation:

$$(\mathbf{F}^T \mathbf{F})^{-1/2} = \left((\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)^T (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)\right)^{-1/2} = (\mathbf{V} \mathbf{\Sigma}^2 \mathbf{V}^T)^{-1/2} = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{V}^T$$

Multiplying by $\mathbf{F}$:

$$\mathbf{F} (\mathbf{F}^T \mathbf{F})^{-1/2} = (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)(\mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{V}^T) = \mathbf{U} \mathbf{\Sigma} \mathbf{\Sigma}^{-1} \mathbf{V}^T = \mathbf{U} \mathbf{V}^T = \mathbf{\tilde{F}}$$

This completes the proof. $\blacksquare$

### 2.3 Theorem: Permutation Symmetry (Equivariance)

**Theorem 2.** *Löwdin orthogonalization is permutation equivariant. Let $\mathbf{P} \in \mathbb{R}^{K \times K}$ be any $K \times K$ permutation matrix ($\mathbf{P}^T \mathbf{P} = \mathbf{P} \mathbf{P}^T = \mathbf{I}_K$). If factors are permuted as $\mathbf{F}_{\pi} = \mathbf{F} \mathbf{P}$, then the orthogonalized factors satisfy:*

$$\mathbf{\tilde{F}}_{\pi} = \mathbf{\tilde{F}} \mathbf{P} \tag{4}$$

#### Proof:

By definition of the Löwdin transform on $\mathbf{F}_{\pi}$:

$$\mathbf{\tilde{F}}_{\pi} = (\mathbf{F} \mathbf{P}) \left((\mathbf{F} \mathbf{P})^T (\mathbf{F} \mathbf{P})\right)^{-1/2} = \mathbf{F} \mathbf{P} (\mathbf{P}^T \mathbf{F}^T \mathbf{F} \mathbf{P})^{-1/2}$$

Let $\mathbf{S} = \mathbf{F}^T \mathbf{F}$. For any symmetric positive definite matrix $\mathbf{S}$ and orthogonal matrix $\mathbf{P}$, the spectral decomposition yields:

$$(\mathbf{P}^T \mathbf{S} \mathbf{P})^{-1/2} = \mathbf{P}^T \mathbf{S}^{-1/2} \mathbf{P}$$

Substituting this identity:

$$\mathbf{\tilde{F}}_{\pi} = \mathbf{F} \mathbf{P} (\mathbf{P}^T \mathbf{S}^{-1/2} \mathbf{P}) = \mathbf{F} (\mathbf{P} \mathbf{P}^T) \mathbf{S}^{-1/2} \mathbf{P} = \mathbf{F} \mathbf{S}^{-1/2} \mathbf{P} = \mathbf{\tilde{F}} \mathbf{P}$$

Thus, permuting the input factors merely permutes the output factors identically, with zero preferential bias. $\blacksquare$

---

## 3. Frisch–Waugh–Lovell Cross-Sectional Risk Neutralization

### 3.1 Projection Operators

Let $\mathbf{X} \in \mathbb{R}^{N \times M}$ be the risk exposure matrix ($M < N$), with $\mathrm{rank}(\mathbf{X}) = M$.
We define:

**1. Hat / Projection Matrix** onto $\mathrm{Col}(\mathbf{X})$:

$$\mathbf{P}_X \triangleq \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \in \mathbb{R}^{N \times N}$$

**2. Annihilator / Residual-Maker Matrix** onto $\mathrm{Col}(\mathbf{X})^\perp = \mathrm{Null}(\mathbf{X}^T)$:

$$\mathbf{M}_X \triangleq \mathbf{I}_N - \mathbf{P}_X = \mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$$

### 3.2 Theorem: Exact Risk Orthogonality

**Theorem 3.** *Both $\mathbf{P}_X$ and $\mathbf{M}_X$ are symmetric and idempotent projection operators ($\mathbf{M}_X^T = \mathbf{M}_X$, $\mathbf{M}_X^2 = \mathbf{M}_X$). The residual factor matrix $\mathbf{F}_{\text{neutral}} = \mathbf{M}_X \mathbf{F}$ satisfies exact risk orthogonality:*

$$\mathbf{X}^T \mathbf{F}_{\text{neutral}} = \mathbf{0}_{M \times K} \tag{5}$$

#### Proof:

Direct computation of $\mathbf{X}^T \mathbf{M}_X$:

$$\mathbf{X}^T \mathbf{M}_X = \mathbf{X}^T (\mathbf{I}_N - \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T) = \mathbf{X}^T - (\mathbf{X}^T \mathbf{X}) (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T = \mathbf{X}^T - \mathbf{X}^T = \mathbf{0}_{M \times N}$$

Multiplying by $\mathbf{F}$:

$$\mathbf{X}^T \mathbf{F}_{\text{neutral}} = \mathbf{X}^T \mathbf{M}_X \mathbf{F} = \mathbf{0}_{M \times N} \mathbf{F} = \mathbf{0}_{M \times K}$$

$\blacksquare$

### 3.3 Theorem: QR Decomposition Computational Stability

**Theorem 4.** *Let $\mathbf{X} = \mathbf{Q}_X \mathbf{R}_X$ be the thin QR decomposition of $\mathbf{X}$, where $\mathbf{Q}_X \in \mathbb{R}^{N \times M}$ satisfies $\mathbf{Q}_X^T \mathbf{Q}_X = \mathbf{I}_M$ and $\mathbf{R}_X \in \mathbb{R}^{M \times M}$ is non-singular upper triangular. Then:*

$$\mathbf{P}_X = \mathbf{Q}_X \mathbf{Q}_X^T \quad \text{and} \quad \mathbf{M}_X \mathbf{F} = \mathbf{F} - \mathbf{Q}_X (\mathbf{Q}_X^T \mathbf{F}) \tag{6}$$

#### Proof:

Substitute $\mathbf{X} = \mathbf{Q}_X \mathbf{R}_X$ into $\mathbf{P}_X$:

$$\mathbf{P}_X = (\mathbf{Q}_X \mathbf{R}_X) \left((\mathbf{Q}_X \mathbf{R}_X)^T (\mathbf{Q}_X \mathbf{R}_X)\right)^{-1} (\mathbf{Q}_X \mathbf{R}_X)^T = \mathbf{Q}_X \mathbf{R}_X (\mathbf{R}_X^T \mathbf{Q}_X^T \mathbf{Q}_X \mathbf{R}_X)^{-1} \mathbf{R}_X^T \mathbf{Q}_X^T$$

Since $\mathbf{Q}_X^T \mathbf{Q}_X = \mathbf{I}_M$:

$$= \mathbf{Q}_X \mathbf{R}_X (\mathbf{R}_X^T \mathbf{R}_X)^{-1} \mathbf{R}_X^T \mathbf{Q}_X^T = \mathbf{Q}_X \mathbf{R}_X \mathbf{R}_X^{-1} (\mathbf{R}_X^T)^{-1} \mathbf{R}_X^T \mathbf{Q}_X^T = \mathbf{Q}_X \mathbf{Q}_X^T$$

Consequently:

$$\mathbf{M}_X \mathbf{F} = (\mathbf{I}_N - \mathbf{P}_X) \mathbf{F} = \mathbf{F} - \mathbf{Q}_X (\mathbf{Q}_X^T \mathbf{F})$$

This formulation avoids forming $\mathbf{X}^T \mathbf{X}$ and inverting it, reducing the condition number from $\kappa(\mathbf{X})^2$ to $\kappa(\mathbf{X})$ and eliminating matrix inversion failure. $\blacksquare$

### 3.4 Weighted Least Squares (WLS) Neutralization

When assets have heterogeneous capitalization weights $\mathbf{w} = [w_1, \dots, w_N]^T > 0$, let $\mathbf{W} = \mathrm{diag}(\mathbf{w})$. The WLS objective minimizes $\sum_i w_i e_{ik}^2 = \mathbf{e}_k^T \mathbf{W} \mathbf{e}_k$.

Defining transformed variables $\mathbf{\tilde{X}} = \mathbf{W}^{1/2} \mathbf{X}$ and $\mathbf{\tilde{F}} = \mathbf{W}^{1/2} \mathbf{F}$, computing the thin QR decomposition $\mathbf{\tilde{X}} = \mathbf{Q}_{\tilde{X}} \mathbf{R}_{\tilde{X}}$, the weighted residual is:

$$\mathbf{F}_{\text{neutral}} = \mathbf{W}^{-1/2} \left(\mathbf{\tilde{F}} - \mathbf{Q}_{\tilde{X}} (\mathbf{Q}_{\tilde{X}}^T \mathbf{\tilde{F}})\right)$$

which satisfies exact weighted risk orthogonality: $\mathbf{X}^T \mathbf{W} \mathbf{F}_{\text{neutral}} = \mathbf{0}_{M \times K}$.

---

## 4. Diagnostics & Metric Formulations

### 4.1 Variance Inflation Factor (VIF)

Let $\mathbf{Z} \in \mathbb{R}^{N \times K}$ be the cross-sectionally standardized factor matrix ($\bar{\mathbf{z}}_k = 0$, $s_k = 1$). The sample correlation matrix is:

$$\mathbf{C} = \frac{1}{N - 1} \mathbf{Z}^T \mathbf{Z} \in \mathbb{R}^{K \times K}$$

The Variance Inflation Factor for factor $k$ is given by:

$$\mathrm{VIF}_k = (\mathbf{C}^{-1})_{kk} = \frac{1}{1 - R_k^2} \tag{7}$$

where $R_k^2$ is the coefficient of determination obtained from regressing $\mathbf{z}_k$ on all remaining $K - 1$ factors.
- **For collinear factors:** $R_k^2 \to 1 \implies \mathrm{VIF}_k \to \infty$.
- **For orthonormal factors:** $\mathbf{C} = \mathbf{I}_K \implies \mathbf{C}^{-1} = \mathbf{I}_K \implies \mathrm{VIF}_k = 1.0$ for all $k \in \{1, \dots, K\}$.

### 4.2 Matrix Condition Number ($\kappa$)

$$\kappa_2(\mathbf{F}) = \frac{\sigma_{\max}(\mathbf{F})}{\sigma_{\min}(\mathbf{F})} \tag{8}$$

For any orthogonal matrix $\mathbf{Q} \in \mathcal{V}_K(\mathbb{R}^N)$, all singular values $\sigma_k = 1$, thus $\kappa_2(\mathbf{Q}) = 1.0$, which represents the theoretical minimum possible condition number.

### 4.3 Effective Number of Factors (ENF) via Spectral Entropy

Let $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_K \ge 0$ be the eigenvalues of the correlation matrix $\mathbf{C}$. Because $\mathrm{Tr}(\mathbf{C}) = K$, we have $\sum_{k=1}^K \lambda_k = K$.

Define the spectral probability distribution:

$$p_k \triangleq \frac{\lambda_k}{\sum_{j=1}^K \lambda_j} = \frac{\lambda_k}{K}, \quad \sum_{k=1}^K p_k = 1$$

The Shannon entropy of the spectrum is:

$$H(\mathbf{C}) \triangleq -\sum_{k=1}^K p_k \ln(p_k)$$

The Effective Number of Factors is defined as the exponential of entropy:

$$\mathrm{ENF} \triangleq \exp\left(H(\mathbf{C})\right) \tag{9}$$

- **Extreme Case 1 (Perfect Collinearity):** $\lambda_1 = K, \lambda_2 = \dots = \lambda_K = 0 \implies p_1 = 1 \implies H = 0 \implies \mathrm{ENF} = 1.0$.
- **Extreme Case 2 (Complete Orthogonality):** $\lambda_k = 1 \implies p_k = 1/K \implies H = \ln K \implies \mathrm{ENF} = K$.

### 4.4 Additive Decomposition of Multiple $R^2$ Under Orthogonality

**Theorem 5.** *Let $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_K] \in \mathbb{R}^{N \times K}$ be an orthonormal factor matrix ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}_K$). In the linear model $\mathbf{y} = \mathbf{Q} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$, the total regression coefficient of determination $R^2_{\text{total}}$ decomposes additively without interaction terms:*

$$R^2_{\text{total}} = \sum_{k=1}^K R^2_k \tag{10}$$

#### Proof:

The OLS estimator is $\hat{\boldsymbol{\beta}} = (\mathbf{Q}^T \mathbf{Q})^{-1} \mathbf{Q}^T \mathbf{y} = \mathbf{I}_K \mathbf{Q}^T \mathbf{y} = \mathbf{Q}^T \mathbf{y}$. Thus $\hat{\beta}_k = \mathbf{q}_k^T \mathbf{y}$.
The fitted values are $\hat{\mathbf{y}} = \mathbf{Q} \hat{\boldsymbol{\beta}} = \sum_{k=1}^K \hat{\beta}_k \mathbf{q}_k$.

The regression sum of squares (SSR) is:

$$\mathrm{SSR}_{\text{total}} = \lVert \hat{\mathbf{y}} \rVert^2 = \left\lVert \sum_{k=1}^K \hat{\beta}_k \mathbf{q}_k \right\rVert^2 = \sum_{j=1}^K \sum_{k=1}^K \hat{\beta}_j \hat{\beta}_k (\mathbf{q}_j^T \mathbf{q}_k)$$

Because $\mathbf{q}_j^T \mathbf{q}_k = \delta_{jk}$ (Kronecker delta):

$$\mathrm{SSR}_{\text{total}} = \sum_{k=1}^K \hat{\beta}_k^2 \lVert \mathbf{q}_k \rVert^2 = \sum_{k=1}^K \mathrm{SSR}_k$$

Dividing by the total sum of squares $\mathrm{SST} = \lVert \mathbf{y} - \bar{y} \rVert^2$ yields $R^2_{\text{total}} = \sum_{k=1}^K R^2_k$. $\blacksquare$

---

## 5. Academic References & Bibliography

1. **Löwdin, P.-O. (1950).** On the non-orthogonality problem connected with the use of atomic wave functions in the theory of molecules and crystals. *The Journal of Chemical Physics*, 18(3), 365–375.
2. **Frisch, R., & Waugh, F. V. (1933).** Partial time regressions as compared with individual trends. *Econometrica*, 1(4), 387–401.
3. **Lovell, M. C. (1963).** Seasonal adjustment of economic time series and multiple regression analysis. *Journal of the American Statistical Association*, 58(304), 993–1010.
4. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.). Johns Hopkins University Press.
5. **Higham, N. J. (1988).** Computing a nearest symmetric positive semidefinite matrix. *Linear Algebra and its Applications*, 103, 103–118.
6. **Belsley, D. A., Kuh, E., & Welsch, R. E. (2005).** *Regression diagnostics: Identifying influential data and sources of collinearity*. John Wiley & Sons.
7. **Meucci, A. (2009).** Managing diversification. *Risk*, 22(5), 74–79.
8. **Kakushadze, Z. (2015).** Factor models and alpha streams. *The Journal of Investment Strategies*, 5(1), 1–45.
