"""Tier 3: Calibration & Regression Accuracy Tests for Gaze Tracker.

Covers: 9/13/16-point grid generators, saccade delay filtering, outlier rejection (IQR),
RidgeCV / polynomial fitting, Leave-One-Point-Out (LOPO) CV (MAE < 35px, RMSE < 50px),
and profile save/load serialization round-trips.
"""

import os
import tempfile
import math
import numpy as np
import pytest

from src.config import GazeConfig
from src.calibrator import CalibrationManager, CalibrationState
from src.gaze_regressor import GazeRegressionModel


# ============================================================================
# 1. Multi-Point Grid Generators (9, 13, 16 Points)
# ============================================================================

class TestMultiPointGridGenerators:
    """Verifies screen grid coordinate generation, margins, and fallback behaviors."""

    def test_9_point_grid_layout(self, gaze_config: GazeConfig):
        """Verify 9-point grid generates 9 distinct points with correct margin bounds."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        pts = calibrator.generate_points("9_points")
        assert len(pts) == 9
        assert len(set(pts)) == 9

        w, h = gaze_config.screen_width, gaze_config.screen_height
        mx, my = gaze_config.calibration_margin_x, gaze_config.calibration_margin_y

        for x, y in pts:
            assert int(mx * w) <= x <= int(math.ceil((1.0 - mx) * w))
            assert int(my * h) <= y <= int(math.ceil((1.0 - my) * h))

    def test_13_point_grid_layout(self, gaze_config: GazeConfig):
        """Verify 13-point grid includes 9 perimeter/center points plus 4 inner quadrant points."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        pts = calibrator.generate_points("13_points")
        assert len(pts) == 13
        assert len(set(pts)) == 13

        # Check inner points are present
        w, h = gaze_config.screen_width, gaze_config.screen_height
        inner_x1 = int(0.35 * w)
        inner_y1 = int(0.35 * h)
        assert (inner_x1, inner_y1) in pts

    def test_16_point_grid_layout(self, gaze_config: GazeConfig):
        """Verify 16-point grid generates a 4x4 equidistant grid of 16 distinct points."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        pts = calibrator.generate_points("16_points")
        assert len(pts) == 16
        assert len(set(pts)) == 16

    def test_invalid_grid_type_fallback_to_9_points(self, gaze_config: GazeConfig):
        """Verify unsupported grid type string falls back to 9-point grid."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        pts = calibrator.generate_points("unknown_invalid_grid")
        assert len(pts) == 9


# ============================================================================
# 2. Saccade Delay & Calibration Sequence State Machine
# ============================================================================

class TestCalibrationStateMachine:
    """Verifies saccade trimming, valid frame ingestion, and calibration progression."""

    def test_saccade_delay_skips_initial_frames(self, gaze_config: GazeConfig):
        """Verify the first saccade_delay_frames are discarded before collecting samples."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.start_calibration("9_points")

        dummy_feat = np.zeros(14, dtype=np.float64)

        # Feed frames during saccade window
        for _ in range(gaze_config.saccade_delay_frames):
            calibrator.process_frame(dummy_feat, is_valid_frame=True)

        assert len(calibrator.current_point_samples) == 0

        # Feed one frame after saccade window
        calibrator.process_frame(dummy_feat, is_valid_frame=True)
        assert len(calibrator.current_point_samples) == 1

    def test_invalid_frames_dropped_during_collection(self, gaze_config: GazeConfig):
        """Verify invalid frames (e.g. blinking) are not appended to sample buffer."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.start_calibration("9_points")

        dummy_feat = np.zeros(14, dtype=np.float64)

        # Pass saccade delay
        for _ in range(gaze_config.saccade_delay_frames):
            calibrator.process_frame(dummy_feat, is_valid_frame=True)

        # Feed invalid frames (blink)
        for _ in range(5):
            calibrator.process_frame(dummy_feat, is_valid_frame=False)

        assert len(calibrator.current_point_samples) == 0

    def test_full_calibration_session_progression(self, gaze_config: GazeConfig):
        """Verify full calibration cycle transitions through all 9 points and trains model."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.start_calibration("9_points")

        assert calibrator.state == CalibrationState.COLLECTING
        total_pts = len(calibrator.points)

        for pt_idx in range(total_pts):
            target = calibrator.get_current_target()
            assert target == calibrator.points[pt_idx]

            idx, total, prog = calibrator.get_progress()
            assert idx == pt_idx + 1
            assert total == total_pts

            feat = np.array([
                target[0] / 1920.0, target[1] / 1080.0,
                target[0] / 1920.0, target[1] / 1080.0,
                target[0] / 1920.0, target[1] / 1080.0,
                0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            ], dtype=np.float64)

            finished = False
            for _ in range(gaze_config.sample_frames_per_point):
                finished = calibrator.process_frame(feat, is_valid_frame=True)

            if pt_idx == total_pts - 1:
                assert finished is True

        assert calibrator.state == CalibrationState.FINISHED
        assert regressor.is_trained is True


# ============================================================================
# 3. Statistical Outlier Rejection (IQR)
# ============================================================================

class TestOutlierRejectionIQR:
    """Verifies IQR-based outlier filtering on calibration feature samples."""

    def test_outlier_filtering_removes_extreme_points(self, gaze_config: GazeConfig):
        """Verify extreme outlier feature vectors are filtered out."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)

        np.random.seed(42)
        # 20 clean samples clustered around 0.5
        clean_samples = [np.full(14, 0.5) + np.random.normal(0, 0.01, 14) for _ in range(20)]
        # 3 extreme outlier samples
        outliers = [np.full(14, 50.0), np.full(14, -50.0), np.full(14, 100.0)]
        all_samples = clean_samples + outliers

        filtered = calibrator._filter_outliers(all_samples)
        assert len(filtered) < len(all_samples)
        # Verify outliers are removed
        for s in filtered:
            assert np.all(s < 10.0)

    def test_outlier_filtering_preserves_clean_samples(self, gaze_config: GazeConfig):
        """Verify tightly clustered clean samples are all preserved."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)

        np.random.seed(42)
        clean_samples = [np.full(14, 0.5) + np.random.normal(0, 0.001, 14) for _ in range(15)]
        filtered = calibrator._filter_outliers(clean_samples)
        assert len(filtered) >= 12

    def test_outlier_filtering_zero_variance_safety(self, gaze_config: GazeConfig):
        """Verify identical samples with zero variance are safely handled without crash."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)

        identical_samples = [np.full(14, 0.5) for _ in range(10)]
        filtered = calibrator._filter_outliers(identical_samples)
        assert len(filtered) == 10


# ============================================================================
# 4. Polynomial Ridge Gaze Regression Fitting & Prediction
# ============================================================================

class TestPolynomialRidgeRegression:
    """Verifies regression pipeline fitting, evaluation metrics, and prediction bounds."""

    def test_regression_train_and_metrics(self, gaze_config: GazeConfig, synthetic_calibration_dataset):
        """Verify train() fits pipeline and returns MAE and RMSE metrics."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        metrics = regressor.train(X, y)

        assert regressor.is_trained is True
        assert "mae_px" in metrics
        assert "rmse_px" in metrics
        assert metrics["mae_px"] < 35.0
        assert metrics["rmse_px"] < 50.0

    def test_prediction_output_format_and_clamping(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify predict() produces 2D screen coordinate tuples clamped to display boundaries."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        regressor.train(X, y)

        # Test valid prediction
        pred = regressor.predict(X[0])
        assert isinstance(pred, tuple)
        assert len(pred) == 2
        assert 0 <= pred[0] <= gaze_config.screen_width
        assert 0 <= pred[1] <= gaze_config.screen_height

        # Test extreme input clamping
        extreme_feat = np.full(14, 1000.0)
        pred_clamped = regressor.predict(extreme_feat)
        assert pred_clamped[0] <= gaze_config.screen_width
        assert pred_clamped[1] <= gaze_config.screen_height

    @pytest.mark.parametrize("alpha", [0.01, 1.0, 10.0])
    def test_regularization_parameter_variation(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, alpha: float
    ):
        """Verify regression pipeline converges under various L2 ridge regularization alphas."""
        X, y, meta = synthetic_calibration_dataset
        cfg = GazeConfig(ridge_alpha=alpha)
        regressor = GazeRegressionModel(cfg)
        metrics = regressor.train(X, y)
        assert metrics["mae_px"] < 40.0


# ============================================================================
# 5. Leave-One-Point-Out (LOPO) Cross-Validation & MAE Verification
# ============================================================================

class TestLOPOAccuracyVerification:
    """Verifies Leave-One-Point-Out cross-validation meets accuracy requirements (MAE < 35px, RMSE < 50px)."""

    def test_leave_one_point_out_cv_accuracy(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """
        Performs Leave-One-Point-Out cross-validation across all 9 calibration targets.
        Verifies overall holdout MAE < 35px and RMSE < 50px.
        """
        X, y, meta = synthetic_calibration_dataset
        targets = meta["screen_targets"]
        samples_per_pt = meta["samples_per_pt"]

        holdout_errors = []

        for holdout_idx in range(len(targets)):
            # Split into train (8 points) and test (1 holdout point)
            test_mask = np.zeros(len(X), dtype=bool)
            start_idx = holdout_idx * samples_per_pt
            end_idx = start_idx + samples_per_pt
            test_mask[start_idx:end_idx] = True
            train_mask = ~test_mask

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            model = GazeRegressionModel(gaze_config)
            model.train(X_train, y_train)

            for i in range(len(X_test)):
                pred = model.predict(X_test[i])
                err = np.linalg.norm(np.array(pred) - y_test[i])
                holdout_errors.append(err)

        lopo_mae = float(np.mean(holdout_errors))
        lopo_rmse = float(np.sqrt(np.mean(np.array(holdout_errors) ** 2)))

        print(f"\n🎯 LOPO Validation Results: MAE = {lopo_mae:.2f}px, RMSE = {lopo_rmse:.2f}px")
        assert lopo_mae < 35.0, f"LOPO MAE {lopo_mae:.2f}px exceeded target threshold of 35px"
        assert lopo_rmse < 50.0, f"LOPO RMSE {lopo_rmse:.2f}px exceeded target threshold of 50px"


# ============================================================================
# 6. Model Profile Serialization & Deserialization
# ============================================================================

class TestModelSerializationRoundtrip:
    """Verifies calibration model save/load fidelity, metadata preservation, and error recovery."""

    def test_save_and_load_profile_roundtrip(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify model saved to disk reloads with identical predictions and metadata."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        regressor.train(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            regressor.save(tmp_path)
            assert os.path.exists(tmp_path)

            loaded_model = GazeRegressionModel(gaze_config)
            success = loaded_model.load(tmp_path)
            assert success is True
            assert loaded_model.is_trained is True
            assert "mae_px" in loaded_model.metrics

            # Verify predictions match bit-for-bit
            for i in range(5):
                orig_pred = regressor.predict(X[i])
                load_pred = loaded_model.predict(X[i])
                assert orig_pred == load_pred

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_corrupted_pickle_file_returns_false(self, gaze_config: GazeConfig):
        """Verify attempting to load a corrupted/non-pickle file returns False without raising exception."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(b"CORRUPTED_NON_PICKLE_BINARY_DATA_12345")
            tmp_path = tmp.name

        try:
            regressor = GazeRegressionModel(gaze_config)
            success = regressor.load(tmp_path)
            assert success is False
            assert regressor.is_trained is False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
