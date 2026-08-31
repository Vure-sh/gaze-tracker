"""Backward-compatibility module wrapper for GazeRegressionModel and regressors.

Directs callers to modular implementations in `src.models.regressor`.
"""

from src.models.regressor import (
    BaseGazeRegressor,
    PolynomialRidgeRegressor,
    RobustHuberRegressor,
    SVRGazeRegressor,
    GazeRegressionModel,
)
from src.models.serializer import ModelProfileSerializer

__all__ = [
    "BaseGazeRegressor",
    "PolynomialRidgeRegressor",
    "RobustHuberRegressor",
    "SVRGazeRegressor",
    "GazeRegressionModel",
    "ModelProfileSerializer",
]
