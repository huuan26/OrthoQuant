"""OrthoQuant: Symmetric Factor Orthogonalization and Cross-Sectional Risk Neutralization."""

__version__ = "0.2.0"
__author__ = "Phan Bùi Hữu Ân"

from .orthogonal import (
    lowdin_orthogonalize,
    lowdin_correlation_transform,
    gram_schmidt_orthogonalize,
)
from .neutralization import (
    neutralize_factors,
    build_sector_exposure_matrix,
)
from .metrics import (
    correlation_matrix,
    condition_number,
    variance_inflation_factors,
    frobenius_distortion,
    effective_number_of_factors,
    information_coefficient,
    marginal_r_squared,
)
from .data import generate_synthetic_factors
from .panel import (
    PanelOrthogonalityReport,
    extract_polars_anta_factors,
    panel_neutralize_and_orthogonalize,
    evaluate_panel_orthogonality,
)

__all__ = [
    "__version__",
    "__author__",
    "lowdin_orthogonalize",
    "lowdin_correlation_transform",
    "gram_schmidt_orthogonalize",
    "neutralize_factors",
    "build_sector_exposure_matrix",
    "correlation_matrix",
    "condition_number",
    "variance_inflation_factors",
    "frobenius_distortion",
    "effective_number_of_factors",
    "information_coefficient",
    "marginal_r_squared",
    "generate_synthetic_factors",
    "PanelOrthogonalityReport",
    "extract_polars_anta_factors",
    "panel_neutralize_and_orthogonalize",
    "evaluate_panel_orthogonality",
]
