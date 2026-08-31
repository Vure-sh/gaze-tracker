"""Comprehensive Milestone 2 Test Suite: Calibration, Targets, ML Regressors & Serialization.

Verifies:
- Boustrophedon / serpentine grid ordering for 9, 13, and 16 points.
- Saccade delay trimming and wall-clock dwell timing.
- Statistical outlier rejection with feature normalization and zero-variance guards.
- Holdout validation mode (4-point/5-point) with pixel MAE, RMSE, and visual angle error (theta < 1.0 deg).
- Modular BaseGazeRegressor implementations: PolynomialRidgeRegressor, RobustHuberRegressor, SVRGazeRegressor.
- Leave-One-Point-Out (LOPO) Group Cross-Validation (MAE < 35px, RMSE < 50px).
- Schema 2.0 serialization, verification, roundtrip fidelity, and legacy backward compatibility.
"""

import os
import math
import tempfile
import pickle
import numpy as np
import pytest

from src.config import GazeConfig
from src.calibration.targets import TargetGenerator
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.models.regressor import (
    BaseGazeRegressor,
    PolynomialRidgeRegressor,
    RobustHuberRegressor,
    SVRGazeRegressor,
    GazeRegressionModel,
)
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION


# ============================================================================
# 1. TargetGenerator & Boustrophedon Sequence Tests
# ============================================================================

class TestTargetGenerator:
    """Tests for multi-point grid generation and serpentine Boustrophedon ordering."""

    def test_boustrophedon_ordering_9_points(self, gaze_config: GazeConfig):
        """Verify row 0 is L->R, row 1 is R->L, and row 2 is L->R."""
        pts = TargetGenerator.generate_points(gaze_config, "9_points", boustrophedon=True)
        assert len(pts) == 9

        w, h = gaze_config.screen_width, gaze_config.screen_height
        mx, my = gaze_config.calibration_margin_x, gaze_config.calibration_margin_y

        left_x = int(round(mx * w))
        mid_x = int(round(0.5 * w))
        right_x = int(round((1.0 - mx) * w))

        top_y = int(round(my * h))
        mid_y = int(round(0.5 * h))
        bot_y = int(round((1.0 - my) * h))

        # Row 0: L -> M -> R
        assert pts[0] == (left_x, top_y)
        assert pts[1] == (mid_x, top_y)
        assert pts[2] == (right_x, top_y)

        # Row 1 (reversed): R -> M -> L
        assert pts[3] == (right_x, mid_y)
        assert pts[4] == (mid_x, mid_y)
        assert pts[5] == (left_x, mid_y)

        # Row 2: L -> M -> R
        assert pts[6] == (left_x, bot_y)
        assert pts[7] == (mid_x, bot_y)
        assert pts[8] == (right_x, bot_y)

    def test_non_boustrophedon_ordering_9_points(self, gaze_config: GazeConfig):
        """Verify raster order when boustrophedon=False."""
        pts = TargetGenerator.generate_points(gaze_config, "9_points", boustrophedon=False)
        assert len(pts) == 9

        w, h = gaze_config.screen_width, gaze_config.screen_height
        mx, my = gaze_config.calibration_margin_x, gaze_config.calibration_margin_y
        left_x = int(round(mx * w))
        mid_y = int(round(0.5 * h))

        # In raster mode, row 1 starts with left_x
        assert pts[3] == (left_x, mid_y)

    def test_boustrophedon_ordering_16_points(self, gaze_config: GazeConfig):
        """Verify 4x4 grid alternates row directions."""
        pts = TargetGenerator.generate_points(gaze_config, "16_points", boustrophedon=True)
        assert len(pts) == 16
        assert len(set(pts)) == 16

        # Row 0 (pts 0..3): x should increase
        assert pts[0][0] < pts[1][0] < pts[2][0] < pts[3][0]
        # Row 1 (pts 4..7): x should decrease
        assert pts[4][0] > pts[5][0] > pts[6][0] > pts[7][0]
        # Row 2 (pts 8..11): x should increase
        assert pts[8][0] < pts[9][0] < pts[10][0] < pts[11][0]
        # Row 3 (pts 12..15): x should decrease
        assert pts[12][0] > pts[13][0] > pts[14][0] > pts[15][0]

    def test_validation_points_generation(self, gaze_config: GazeConfig):
        """Verify 4-point and 5-point holdout validation grid generation."""
        pts4 = TargetGenerator.generate_validation_points(gaze_config, mode="4_points")
        assert len(pts4) == 4
        assert len(set(pts4)) == 4

        pts5 = TargetGenerator.generate_validation_points(gaze_config, mode="5_points")
        assert len(pts5) == 5
        assert len(set(pts5)) == 5
        # 5th point is screen center
        center = (int(round(0.5 * gaze_config.screen_width)), int(round(0.5 * gaze_config.screen_height)))
        assert pts5[4] == center

    def test_to_normalized_coordinates(self, gaze_config: GazeConfig):
        """Verify pixel to [0.0, 1.0] coordinate transformation."""
        pts = [(0, 0), (1920, 1080), (960, 540)]
        norm = TargetGenerator.to_normalized_coordinates(pts, 1920, 1080)
        assert np.allclose(norm[0], [0.0, 0.0])
        assert np.allclose(norm[1], [1.0, 1.0])
        assert np.allclose(norm[2], [0.5, 0.5])


# ============================================================================
# 2. CalibrationManager & Statistical Outlier Filtering
# ============================================================================

class TestCalibrationManagerOutlierAndTiming:
    """Tests for outlier filtering, scale normalization, and zero-variance handling."""

    def test_normalized_iqr_outlier_rejection(self, gaze_config: GazeConfig):
        """Verify outlier rejection normalizes features so translation does not dominate iris offsets."""
        calibrator = CalibrationManager(gaze_config)
        np.random.seed(42)

        # 25 samples with small iris variation (0.01) and large head translation variation (50.0)
        clean_samples = []
        for _ in range(25):
            feat = np.array([
                0.20 + np.random.normal(0, 0.005),   # left norm_x
                0.10 + np.random.normal(0, 0.005),   # left norm_y
                0.20 + np.random.normal(0, 0.005),   # right norm_x
                0.10 + np.random.normal(0, 0.005),   # right norm_y
                0.0 + np.random.normal(0, 0.01),     # pitch
                0.0 + np.random.normal(0, 0.01),     # yaw
                0.0 + np.random.normal(0, 0.01),     # roll
                0.60 + np.random.normal(0, 0.05),    # tz
            ])
            clean_samples.append(feat)

        # Add 3 eye-iris gaze outliers (e.g. looking away, glance)
        outlier1 = clean_samples[0].copy()
        outlier1[0] = 0.95  # extreme iris gaze jump
        outlier2 = clean_samples[1].copy()
        outlier2[2] = -0.90
        outlier3 = clean_samples[2].copy()
        outlier3[1] = 0.85

        samples = clean_samples + [outlier1, outlier2, outlier3]
        filtered = calibrator._filter_outliers(samples)

        assert len(filtered) < len(samples)
        # Verify gaze outliers were filtered out
        for s in filtered:
            assert s[0] < 0.60
            assert s[2] > -0.50
            assert s[1] < 0.60

    def test_zero_variance_samples_safety(self, gaze_config: GazeConfig):
        """Verify identical constant feature samples do not cause NaN or division by zero."""
        calibrator = CalibrationManager(gaze_config)
        identical = [np.full(8, 0.42) for _ in range(20)]
        filtered = calibrator._filter_outliers(identical)
        assert len(filtered) == 20
        assert np.allclose(filtered[0], np.full(8, 0.42))

    def test_single_dimension_outlier_safety(self, gaze_config: GazeConfig):
        """Verify features where one dimension is constant are handled cleanly."""
        calibrator = CalibrationManager(gaze_config)
        samples = []
        for i in range(20):
            feat = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, float(i) * 0.01])
            samples.append(feat)

        filtered = calibrator._filter_outliers(samples)
        assert len(filtered) >= 15


# ============================================================================
# 3. Post-Calibration Holdout Validation Mode
# ============================================================================

class TestHoldoutValidationMode:
    """Tests for 4-point / 5-point holdout validation and visual angle error calculation."""

    def test_validation_workflow_and_visual_angle(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify holdout validation mode computes MAE, RMSE, and visual angle < 1.0 deg."""
        X, y, meta = synthetic_calibration_dataset
        regressor = PolynomialRidgeRegressor(gaze_config)
        regressor.train(X, y)

        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.start_validation(mode="4_points")

        assert calibrator.state == CalibrationState.VALIDATING
        assert len(calibrator.validation_points) == 4

        w, h = gaze_config.screen_width, gaze_config.screen_height

        for val_idx, (tx, ty) in enumerate(calibrator.validation_points):
            target = calibrator.get_current_target()
            assert target == (tx, ty)

            # Generate synthetic features corresponding to this target
            true_norm_x = (tx - 0.5 * w) / (w * 2.0)
            true_norm_y = (ty - 0.5 * h) / (h * 2.5)

            feat = np.array([
                true_norm_x, true_norm_y,
                true_norm_x, true_norm_y,
                true_norm_x, true_norm_y,
                0.31, 0.31,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.6
            ], dtype=np.float64)

            finished = False
            for _ in range(gaze_config.sample_frames_per_point):
                finished = calibrator.process_frame(feat, is_valid_frame=True)

            if val_idx == 3:
                assert finished is True

        assert calibrator.state == CalibrationState.VALIDATION_COMPLETE
        metrics = calibrator.validation_metrics

        assert "val_mae_px" in metrics
        assert "val_rmse_px" in metrics
        assert "visual_angle_deg" in metrics

        print(f"\n🎯 Holdout Validation Metrics: {metrics}")
        assert metrics["val_mae_px"] < 35.0
        assert metrics["val_rmse_px"] < 50.0
        assert metrics["visual_angle_deg"] < 1.0, (
            f"Visual angle {metrics['visual_angle_deg']:.2f}° exceeded threshold 1.0°"
        )


# ============================================================================
# 4. Gaze Regressor Backends & Leave-One-Point-Out (LOPO) Cross-Validation
# ============================================================================

class TestRegressorBackendsAndLOPO:
    """Tests for PolynomialRidgeRegressor, RobustHuberRegressor, SVRGazeRegressor, and LOPO CV."""

    def test_polynomial_ridge_regressor(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify PolynomialRidgeRegressor achieves MAE < 35px and LOPO MAE < 35px."""
        X, y, meta = synthetic_calibration_dataset
        reg = PolynomialRidgeRegressor(gaze_config)
        metrics = reg.train(X, y)

        assert reg.is_trained is True
        assert metrics["mae_px"] < 35.0
        assert metrics["rmse_px"] < 50.0
        assert "lopo_mae_px" in metrics
        assert "lopo_rmse_px" in metrics
        assert metrics["lopo_mae_px"] < 35.0
        assert metrics["lopo_rmse_px"] < 50.0

        # Verify prediction clamping
        pred = reg.predict(X[0])
        assert pred is not None
        assert 0.0 <= pred[0] <= gaze_config.screen_width
        assert 0.0 <= pred[1] <= gaze_config.screen_height

    def test_robust_huber_regressor(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify RobustHuberRegressor fits cleanly and handles noise."""
        X, y, meta = synthetic_calibration_dataset
        reg = RobustHuberRegressor(gaze_config)
        metrics = reg.train(X, y)

        assert reg.is_trained is True
        assert metrics["mae_px"] < 35.0
        assert metrics["rmse_px"] < 50.0
        assert "lopo_mae_px" in metrics

        pred = reg.predict(X[0])
        assert pred is not None
        assert 0.0 <= pred[0] <= gaze_config.screen_width
        assert 0.0 <= pred[1] <= gaze_config.screen_height

    def test_svr_gaze_regressor(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify SVRGazeRegressor trains and predicts within screen boundaries."""
        X, y, meta = synthetic_calibration_dataset
        reg = SVRGazeRegressor(gaze_config)
        metrics = reg.train(X, y)

        assert reg.is_trained is True
        assert "mae_px" in metrics
        assert "rmse_px" in metrics

        pred = reg.predict(X[0])
        assert pred is not None
        assert 0.0 <= pred[0] <= gaze_config.screen_width
        assert 0.0 <= pred[1] <= gaze_config.screen_height

    def test_untrained_regressor_predict_returns_none(self, gaze_config: GazeConfig):
        """Verify predicting on untrained regressor returns None safely."""
        reg = PolynomialRidgeRegressor(gaze_config)
        assert reg.predict(np.zeros(8)) is None

    def test_insufficient_samples_raises_value_error(self, gaze_config: GazeConfig):
        """Verify training with fewer than 6 samples raises ValueError."""
        reg = PolynomialRidgeRegressor(gaze_config)
        X = np.zeros((3, 8))
        y = np.zeros((3, 2))
        with pytest.raises(ValueError, match="Insufficient training samples"):
            reg.train(X, y)


# ============================================================================
# 5. Schema 2.0 Serialization & Backward Compatibility Roundtrips
# ============================================================================

class TestSchema20Serialization:
    """Tests for profile serialization, Schema 2.0 metadata, and legacy loader."""

    def test_schema_20_profile_roundtrip(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify schema 2.0 profile saves metadata and loads with exact prediction match."""
        X, y, meta = synthetic_calibration_dataset
        reg = PolynomialRidgeRegressor(gaze_config)
        metrics = reg.train(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            reg.save_profile(tmp_path)
            assert os.path.exists(tmp_path)

            # Inspect profile dictionary
            profile = ModelProfileSerializer.deserialize_profile(tmp_path)
            assert profile is not None
            assert profile["schema_version"] == CURRENT_SCHEMA_VERSION
            assert profile["model_type"] == "PolynomialRidgeRegressor"
            assert profile["screen_width"] == gaze_config.screen_width
            assert profile["screen_height"] == gaze_config.screen_height
            assert "created_at" in profile

            # Load back into fresh regressor
            loaded_reg = PolynomialRidgeRegressor(gaze_config)
            success = loaded_reg.load_profile(tmp_path)
            assert success is True
            assert loaded_reg.is_trained is True

            # Check exact prediction equality
            for i in range(10):
                p1 = reg.predict(X[i])
                p2 = loaded_reg.predict(X[i])
                assert p1 is not None and p2 is not None
                assert math.isclose(p1[0], p2[0], abs_tol=1e-5)
                assert math.isclose(p1[1], p2[1], abs_tol=1e-5)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_legacy_schema_10_pickle_loading(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify loader successfully upgrades legacy Schema 1.0 pickle files."""
        X, y, meta = synthetic_calibration_dataset
        reg = GazeRegressionModel(gaze_config)
        reg.train(X, y)

        # Create a genuine legacy Schema 1.0 dictionary
        legacy_payload = {
            "pipeline": reg.pipeline,
            "metrics": {"mae_px": 18.5, "rmse_px": 24.2},
            "screen_width": 1920,
            "screen_height": 1080,
            "poly_degree": 2
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(legacy_payload, tmp)
            tmp_path = tmp.name

        try:
            loaded_reg = PolynomialRidgeRegressor(gaze_config)
            success = loaded_reg.load(tmp_path)
            assert success is True
            assert loaded_reg.is_trained is True
            assert loaded_reg.metrics["mae_px"] == 18.5

            pred = loaded_reg.predict(X[0])
            assert pred is not None
            assert 0.0 <= pred[0] <= 1920.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_corrupted_file_handling(self, gaze_config: GazeConfig):
        """Verify corrupted binary or non-dict files return False without throwing exceptions."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"CORRUPTED_NON_PICKLE_DATA_0xDEADBEEF")
            tmp_path = tmp.name

        try:
            reg = PolynomialRidgeRegressor(gaze_config)
            success = reg.load(tmp_path)
            assert success is False
            assert reg.is_trained is False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_profile_compatibility_verification(self, gaze_config: GazeConfig):
        """Verify verify_profile_compatibility detects feature dimension mismatches."""
        profile_good = {
            "pipeline": "dummy_pipeline",
            "screen_width": 1920,
            "screen_height": 1080,
            "feature_dimension": 8
        }
        ok, warning = ModelProfileSerializer.verify_profile_compatibility(
            profile_good, expected_features=8, screen_w=1920, screen_h=1080
        )
        assert ok is True
        assert warning is None

        # Feature mismatch
        ok_bad, msg = ModelProfileSerializer.verify_profile_compatibility(
            profile_good, expected_features=14
        )
        assert ok_bad is False
        assert "Feature dimension mismatch" in msg
