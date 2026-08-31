"""Gaze regression models mapping eye and head features to screen coordinates.

Implements BaseGazeRegressor and specialized backends:
- PolynomialRidgeRegressor: Degree-2 polynomial with cross-validated L2 Ridge regularization.
- RobustHuberRegressor: Outlier-robust Huber regression loss.
- SVRGazeRegressor: Support Vector Regression with RBF / Polynomial kernel.
Includes Leave-One-Point-Out (LOPO) group cross-validation and display coordinate clamping.
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV, HuberRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import GazeConfig
from src.models.serializer import ModelProfileSerializer


class BaseGazeRegressor(ABC):
    """Abstract base class for all gaze regression backends."""

    def __init__(self, config: GazeConfig):
        self.config = config
        self.is_trained: bool = False
        self.metrics: Dict[str, float] = {}
        self.pipeline: Optional[Pipeline] = self._build_pipeline()

    @abstractmethod
    def _build_pipeline(self) -> Pipeline:
        """Constructs and returns the scikit-learn Pipeline for this regressor."""
        pass

    def compute_lopo_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        point_ids: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Performs Leave-One-Point-Out (LOPO) Group Cross-Validation.

        Args:
            X: Feature matrix of shape (N, D).
            y: Target coordinates of shape (N, 2).
            point_ids: Optional group index for each sample. If None, target coordinates are clustered.

        Returns:
            (lopo_mae_px, lopo_rmse_px) or (None, None) if insufficient groups.
        """
        if point_ids is None:
            # Group by unique target coordinate rows
            _, point_ids = np.unique(y, axis=0, return_inverse=True)

        unique_groups = np.unique(point_ids)
        if len(unique_groups) < 3:
            return None, None

        holdout_errors: List[float] = []

        for grp in unique_groups:
            test_mask = (point_ids == grp)
            train_mask = ~test_mask

            if np.sum(train_mask) < 6 or np.sum(test_mask) < 1:
                continue

            fold_pipeline = self._build_pipeline()
            fold_pipeline.fit(X[train_mask], y[train_mask])
            preds = fold_pipeline.predict(X[test_mask])

            # Compute Euclidean error per sample
            errors = np.linalg.norm(preds - y[test_mask], axis=1)
            holdout_errors.extend(errors.tolist())

        if not holdout_errors:
            return None, None

        lopo_mae = float(np.mean(holdout_errors))
        lopo_rmse = float(np.sqrt(np.mean(np.array(holdout_errors) ** 2)))
        return lopo_mae, lopo_rmse

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        point_ids: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Trains the regression pipeline and calculates resubstitution + LOPO CV metrics.

        Args:
            X: Calibration feature matrix (N, D).
            y: Target screen pixel coordinates (N, 2).
            point_ids: Optional point index array for LOPO CV grouping.

        Returns:
            Dict containing mae_px, rmse_px, lopo_mae_px, lopo_rmse_px, etc.
        """
        if len(X) < 6:
            raise ValueError(f"Insufficient training samples: {len(X)}. Need at least 6 points.")

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X, y)
        self.is_trained = True

        y_pred = self.pipeline.predict(X)
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        lopo_mae, lopo_rmse = self.compute_lopo_cv(X, y, point_ids)

        self.metrics = {
            "mae_px": mae,
            "rmse_px": rmse,
            "samples_count": float(len(X)),
        }

        if lopo_mae is not None and lopo_rmse is not None:
            self.metrics["lopo_mae_px"] = lopo_mae
            self.metrics["lopo_rmse_px"] = lopo_rmse

        return self.metrics

    def predict(self, feature_vector: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Predicts screen (X, Y) pixel coordinates, clamped to screen boundaries.

        Returns:
            (screen_x, screen_y) in pixels, or None if model is untrained.
        """
        if not self.is_trained or feature_vector is None or self.pipeline is None:
            return None

        feat = feature_vector.reshape(1, -1)
        pred = self.pipeline.predict(feat)[0]

        x = float(np.clip(pred[0], 0.0, float(self.config.screen_width)))
        y = float(np.clip(pred[1], 0.0, float(self.config.screen_height)))
        return (x, y)

    def save_profile(self, filepath: str) -> None:
        """Serializes model profile using Schema 2.0."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Cannot save an untrained model.")

        ModelProfileSerializer.serialize_profile(
            model_type=self.__class__.__name__,
            pipeline=self.pipeline,
            screen_width=self.config.screen_width,
            screen_height=self.config.screen_height,
            feature_dimension=self.config.feature_dimension,
            poly_degree=self.config.poly_degree,
            metrics=self.metrics,
            hyperparameters={"alpha": getattr(self.config, "ridge_alpha", 1.0)},
            filepath=filepath
        )

    def load_profile(self, filepath: str) -> bool:
        """Loads and verifies a Schema 2.0 or legacy model profile."""
        profile = ModelProfileSerializer.deserialize_profile(filepath)
        if profile is None or "pipeline" not in profile:
            return False

        self.pipeline = profile["pipeline"]
        self.metrics = profile.get("metrics", {})
        self.is_trained = True
        return True

    def save(self, filepath: Optional[str] = None) -> None:
        """Backward-compatible alias for save_profile()."""
        path = filepath or self.config.calibration_file
        self.save_profile(path)

    def load(self, filepath: Optional[str] = None) -> bool:
        """Backward-compatible alias for load_profile()."""
        path = filepath or self.config.calibration_file
        return self.load_profile(path)


class PolynomialRidgeRegressor(BaseGazeRegressor):
    """
    Degree-2 Polynomial feature expansion with L2 Ridge cross-validation.
    """

    def _build_pipeline(self) -> Pipeline:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        # If user explicitly set a specific ridge_alpha, ensure it is evaluated in alphas
        if hasattr(self.config, "ridge_alpha") and self.config.ridge_alpha is not None:
            if self.config.ridge_alpha not in alphas:
                alphas = sorted(alphas + [self.config.ridge_alpha])

        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=self.config.poly_degree, include_bias=True)),
            ("ridge", RidgeCV(alphas=alphas))
        ])


class RobustHuberRegressor(BaseGazeRegressor):
    """
    Degree-2 Polynomial feature expansion with Huber loss for outlier-robust fitting.
    """

    def _build_pipeline(self) -> Pipeline:
        alpha_val = getattr(self.config, "ridge_alpha", 1.0)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=self.config.poly_degree, include_bias=True)),
            ("huber", MultiOutputRegressor(HuberRegressor(alpha=alpha_val, epsilon=1.35, max_iter=300)))
        ])


class SVRGazeRegressor(BaseGazeRegressor):
    """
    Support Vector Regression (SVR) gaze estimator backend.
    """

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=self.config.poly_degree, include_bias=True)),
            ("svr", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.01)))
        ])


class GazeRegressionModel(PolynomialRidgeRegressor):
    """
    Default gaze regression model maintaining complete backward compatibility.
    """
    pass
