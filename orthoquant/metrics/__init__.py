"""Metrics for collinearity, distortion, and factor performance."""

from .collinearity import (
    correlation_matrix,
    condition_number,
    variance_inflation_factors,
    frobenius_distortion,
    effective_number_of_factors,
)
from .performance import (
    information_coefficient,
    marginal_r_squared,
)

__all__ = [
    "correlation_matrix",
    "condition_number",
    "variance_inflation_factors",
    "frobenius_distortion",
    "effective_number_of_factors",
    "information_coefficient",
    "marginal_r_squared",
]
