"""Risk neutralization and factor exposure stripping."""

from .cross_sectional import neutralize_factors, build_sector_exposure_matrix

__all__ = [
    "neutralize_factors",
    "build_sector_exposure_matrix",
]
