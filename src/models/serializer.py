"""Schema-versioned model profile serialization, verification, and backward compatibility loading."""

from __future__ import annotations
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple
import numpy as np


CURRENT_SCHEMA_VERSION = "2.0"


class ModelProfileSerializer:
    """Handles schema-versioned serialization and deserialization of gaze regression profiles."""

    @staticmethod
    def serialize_profile(
        model_type: str,
        pipeline: Any,
        screen_width: int,
        screen_height: int,
        feature_dimension: int,
        poly_degree: int,
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a schema 2.0 profile dictionary and optionally writes it to disk.

        Args:
            model_type: Name of regressor class (e.g. 'PolynomialRidgeRegressor').
            pipeline: Fitted scikit-learn Pipeline.
            screen_width: Display width in pixels.
            screen_height: Display height in pixels.
            feature_dimension: Number of input features (8, 10, 14).
            poly_degree: Polynomial degree expansion.
            metrics: Fitting and validation metrics (train_mae, lopo_mae, etc.).
            hyperparameters: Hyperparameters dict.
            filepath: Optional destination path to write serialized binary.

        Returns:
            Dictionary containing the full schema 2.0 payload.
        """
        payload: Dict[str, Any] = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "model_type": model_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
            "feature_dimension": int(feature_dimension),
            "poly_degree": int(poly_degree),
            "metrics": dict(metrics),
            "hyperparameters": hyperparameters or {},
            "pipeline": pipeline,
        }

        if filepath is not None:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return payload

    @staticmethod
    def deserialize_profile(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Loads and verifies a calibration profile from disk.
        Supports Schema 2.0 and legacy Schema 1.0 payloads.

        Args:
            filepath: Path to serialized profile file.

        Returns:
            Verified profile dictionary, or None if file does not exist or is corrupted.
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            if not isinstance(data, dict):
                return None

            # Legacy Schema 1.0 detection and upgrade
            if "schema_version" not in data:
                if "pipeline" in data:
                    # Upgrade legacy dictionary to schema 2.0 structure
                    upgraded: Dict[str, Any] = {
                        "schema_version": "1.0",
                        "model_type": "GazeRegressionModel_Legacy",
                        "created_at": "legacy",
                        "screen_width": data.get("screen_width", 1920),
                        "screen_height": data.get("screen_height", 1080),
                        "feature_dimension": data.get("feature_dimension", 14),
                        "poly_degree": data.get("poly_degree", 2),
                        "metrics": data.get("metrics", {}),
                        "hyperparameters": {},
                        "pipeline": data["pipeline"]
                    }
                    return upgraded
                return None

            # Schema 2.0 verification
            if "pipeline" not in data:
                return None

            return data

        except Exception:
            return None

    @staticmethod
    def verify_profile_compatibility(
        profile: Dict[str, Any],
        expected_features: Optional[int] = None,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates profile parameters against current runtime requirements.

        Returns:
            (is_compatible, optional_warning_or_error_message)
        """
        if "pipeline" not in profile:
            return False, "Missing fitted pipeline in profile."

        prof_w = profile.get("screen_width", 1920)
        prof_h = profile.get("screen_height", 1080)

        if screen_w and screen_h and (prof_w != screen_w or prof_h != screen_h):
            return True, f"Display resolution mismatch: saved {prof_w}x{prof_h}, current {screen_w}x{screen_h}."

        prof_dim = profile.get("feature_dimension")
        if expected_features and prof_dim and prof_dim != expected_features:
            return False, f"Feature dimension mismatch: profile has {prof_dim}D, runtime expects {expected_features}D."

        return True, None
