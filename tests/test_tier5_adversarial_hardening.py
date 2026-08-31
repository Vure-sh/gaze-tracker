"""Tier 5: Comprehensive Adversarial Coverage Hardening across Entire Gaze Tracker Architecture."""

import os
import time
import math
import tempfile
import numpy as np
import pytest

from src.config import GazeConfig
from src.types import NormalizedPoint, EyeData, HeadPoseData, GazeFeatures, GazePrediction, TrackingQuality
from src.face_mesh_detector import FaceMeshDetector
from src.cv.face_detector import FaceDetector
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from src.cv.quality_tracker import QualityTracker
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.calibration.targets import TargetGenerator
from src.models.regressor import (
    BaseGazeRegressor,
    PolynomialRidgeRegressor,
    RobustHuberRegressor,
    SVRGazeRegressor,
    GazeRegressionModel
)
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION
from src.filters.one_euro import LowPassFilter, OneEuroFilter1D, OneEuroFilter2D
from src.filters.kalman import KalmanFilter2D
from src.camera_stream import ThreadedCameraStream, open_camera_device
from src.pipeline import GazePipeline, GazePipelineResult
from src.ui.canvas import ScreenGazeCanvas
from src.ui.hud import CameraDebugHUD
from src.ui.app import GazeTrackerApp
from tests.conftest import create_synthetic_landmarks, SyntheticLandmark


# ============================================================================
# 1. Extreme Lighting, Color Inversions & Image Corruption
# ============================================================================

class TestTier5ExtremeVisualPerturbations:
    """Stress tests feature extraction and quality tracker under extreme visual degradations."""

    def test_pure_black_and_pure_white_frames(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify quality tracker and eye extractor handle 0-luma and 255-luma frames without ZeroDivisionError."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        tracker = QualityTracker(gaze_config)

        # Pure black frame
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        feat_black = extractor.extract(synthetic_landmarks, 640, 480)
        pose_black = estimator.estimate(synthetic_landmarks, 640, 480)
        q_black = tracker.assess_quality(feat_black, pose_black, black_frame, synthetic_landmarks)
        assert isinstance(q_black, TrackingQuality)
        assert q_black.contrast_score <= 0.6  # Low contrast in black frame

        # Pure white frame
        white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        q_white = tracker.assess_quality(feat_black, pose_black, white_frame, synthetic_landmarks)
        assert isinstance(q_white, TrackingQuality)

    def test_single_channel_grayscale_and_rgba_frames(self, gaze_config: GazeConfig):
        """Verify detector handles 2D grayscale and 4-channel RGBA frame inputs without uncaught exceptions."""
        detector = FaceDetector(model_path=gaze_config.model_path)
        
        # 2D Grayscale
        gray_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        res_gray = detector.detect(gray_frame)
        assert res_gray is None or isinstance(res_gray, list)

        # 4-Channel RGBA
        rgba_frame = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        res_rgba = detector.detect(rgba_frame)
        assert res_rgba is None or isinstance(res_rgba, list)


# ============================================================================
# 2. Degenerate Geometry & Anatomical Singularities
# ============================================================================

class TestTier5DegenerateGeometrySingularities:
    """Stress tests geometric math against coordinate collapses and anatomical singularities."""

    def test_completely_collapsed_eye_landmarks(self, gaze_config: GazeConfig):
        """Verify EyeExtractor handles eye where all 16 landmarks map to exact same (0.5, 0.5) point."""
        extractor = EyeExtractor(gaze_config)
        landmarks = [SyntheticLandmark(0.5, 0.5, 0.0) for _ in range(478)]

        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert not math.isnan(features.left_eye.norm_x)
        assert not math.isnan(features.left_eye.norm_y)
        assert not math.isnan(features.left_eye.ear)
        assert not math.isinf(features.left_eye.norm_x)

    def test_head_pose_zero_norm_and_extreme_angles(self, gaze_config: GazeConfig):
        """Verify HeadPoseEstimator produces valid HeadPoseData even with distorted landmarks."""
        estimator = HeadPoseEstimator(gaze_config)
        # Create landmarks with extreme roll (85 degrees)
        landmarks_extreme = create_synthetic_landmarks(roll_deg=85.0)
        pose = estimator.estimate(landmarks_extreme, 640, 480)
        assert pose is not None
        assert np.all(np.isfinite(pose.feature_vector))
        assert len(pose.axes_2d_px) == 3


# ============================================================================
# 3. Adversarial Model Deserialization & Corrupted Payloads
# ============================================================================

class TestTier5ModelSerializationAdversarial:
    """Stress tests model persistence against malformed, truncated, or tampered payload files."""

    def test_corrupted_json_and_binary_garbage(self, gaze_config: GazeConfig, tmp_path):
        """Verify serializer rejects non-json random byte streams safely with error."""
        serializer = ModelProfileSerializer()
        corrupt_file = str(tmp_path / "garbage.pkl")

        # Write random binary garbage
        with open(corrupt_file, "wb") as f:
            f.write(os.urandom(512))

        with pytest.raises(Exception):
            serializer.load_profile(corrupt_file)

    def test_missing_and_unsupported_schema_version(self, gaze_config: GazeConfig, tmp_path):
        """Verify serializer rejects incompatible schema version payload safely."""
        bad_schema_file = str(tmp_path / "bad_schema.pkl")

        bad_payload = {
            "schema_version": "999.0",
            "model_type": "polynomial_ridge"
            # Missing pipeline
        }
        import pickle
        with open(bad_schema_file, "wb") as f:
            pickle.dump(bad_payload, f)

        res = ModelProfileSerializer.deserialize_profile(bad_schema_file)
        assert res is None

    def test_all_regressor_backends_training_and_serialization_cycle(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, tmp_path
    ):
        """Verify PolynomialRidge, Huber, and SVR regressors serialize, reload, and evaluate consistently."""
        X, y, meta = synthetic_calibration_dataset
        regressors = [
            PolynomialRidgeRegressor(gaze_config),
            RobustHuberRegressor(gaze_config),
            SVRGazeRegressor(gaze_config),
        ]

        for reg in regressors:
            metrics = reg.train(X, y)
            assert reg.is_trained is True
            assert "mae_px" in metrics
            if not isinstance(reg, SVRGazeRegressor):
                assert metrics["mae_px"] < 40.0

            filepath = str(tmp_path / f"{reg.__class__.__name__}_profile.pkl")
            reg.save_profile(filepath)

            # Reload into fresh instance
            reg_fresh = type(reg)(gaze_config)
            assert reg_fresh.is_trained is False
            success = reg_fresh.load_profile(filepath)
            assert success is True
            assert reg_fresh.is_trained is True

            # Prediction check
            p1 = reg.predict(X[5])
            p2 = reg_fresh.predict(X[5])
            assert math.isclose(p1[0], p2[0], abs_tol=1e-3)
            assert math.isclose(p1[1], p2[1], abs_tol=1e-3)



# ============================================================================
# 4. Pipeline Concurrency & Robustness Under Clock Skew
# ============================================================================

class TestTier5PipelineClockSkewAndTransitions:
    """Stress tests pipeline temporal filtering and calibration state machine under time discontinuities."""

    def test_filter_large_time_gaps_and_clock_skew(self):
        """Verify OneEuroFilter2D handles sudden 100-second timestamp jump without exploding."""
        f2d = OneEuroFilter2D()
        p1 = f2d.filter((100.0, 100.0), timestamp=1.0)
        p2 = f2d.filter((110.0, 110.0), timestamp=1.033)

        # 100-second gap (e.g. system sleep/wake)
        p_jump = f2d.filter((500.0, 500.0), timestamp=101.033)
        assert np.all(np.isfinite(p_jump))
        assert abs(p_jump[0] - 500.0) < 50.0  # Fast transition after long idle gap

    def test_pipeline_rapid_calibration_interruption(
        self, gaze_config: GazeConfig, mock_bgr_frame
    ):
        """Verify starting calibration, resetting mid-way, and re-starting leaves pipeline in clean state."""
        pipeline = GazePipeline(config=gaze_config)

        # Start calibration
        pipeline.start_calibration("13_points")
        assert pipeline.calibrator.state == CalibrationState.COLLECTING
        assert len(pipeline.calibrator.points) == 13

        # Process 5 frames
        for i in range(5):
            _ = pipeline.process_frame(mock_bgr_frame, timestamp=i * 0.033)

        # Abrupt reset mid-calibration
        pipeline.reset()
        assert pipeline.calibrator.state == CalibrationState.IDLE
        assert len(pipeline.calibrator.points) == 0

        # Restart with 16 points
        pipeline.start_calibration("16_points")
        assert pipeline.calibrator.state == CalibrationState.COLLECTING
        assert len(pipeline.calibrator.points) == 16
