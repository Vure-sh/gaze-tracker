"""Tier 1: Unit & Component Integrity Tests for Gaze Tracker.

Covers: Parameter validation, landmark geometry, EAR calculation, solvePnP rotation matrices,
filter step response, serialization guards, and UI rendering integrity (>= 5 tests per feature).
"""

import math
import os
import tempfile
import numpy as np
import pytest

from src.config import GazeConfig
from src.face_mesh_detector import FaceMeshDetector
from src.eye_extractor import EyeExtractor, EyeData, GazeFeatures
from src.head_pose import HeadPoseEstimator, HeadPoseData
from src.calibrator import CalibrationManager, CalibrationState
from src.gaze_regressor import GazeRegressionModel
from src.filters import LowPassFilter, OneEuroFilter1D, OneEuroFilter2D, KalmanFilter2D
from src.visualizer import GazeVisualizer
from tests.conftest import SyntheticLandmark, create_synthetic_landmarks


# ============================================================================
# F01: MediaPipe FaceLandmarker Initialization & Inference
# ============================================================================

class TestF01FaceMeshDetector:
    """Unit tests for FaceMeshDetector initialization, guarding, and inference."""

    def test_detector_init_with_valid_model(self, gaze_config: GazeConfig):
        """Verify detector initializes with the configured model path."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path, num_faces=1)
        assert detector.model_path == gaze_config.model_path
        assert detector.detector is not None

    def test_detector_detect_none_frame(self, gaze_config: GazeConfig):
        """Verify detect() gracefully returns None when passed None."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path)
        result = detector.detect(None)
        assert result is None

    def test_detector_detect_empty_frame(self, gaze_config: GazeConfig):
        """Verify detect() gracefully returns None when passed a zero-sized array."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path)
        empty_frame = np.zeros((0, 0), dtype=np.uint8)
        result = detector.detect(empty_frame)
        assert result is None

    def test_detector_detect_1d_corrupted_frame(self, gaze_config: GazeConfig):
        """Verify detect() gracefully handles 1D corrupted arrays without crashing."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path)
        corrupted_frame = np.zeros(100, dtype=np.uint8)
        result = detector.detect(corrupted_frame)
        assert result is None

    def test_detector_model_url_constant(self):
        """Verify the public MediaPipe task bundle download URL is properly specified."""
        assert "face_landmarker" in FaceMeshDetector.MODEL_URL
        assert FaceMeshDetector.MODEL_URL.startswith("https://")


# ============================================================================
# F02: Dual-Eye Normalized Iris Projection
# ============================================================================

class TestF02EyeExtractorNormalization:
    """Unit tests for orthonormal scale/roll-invariant eye and iris normalization."""

    def test_extractor_short_landmarks_guard(self, gaze_config: GazeConfig):
        """Verify extractor returns None if landmark list has fewer than 478 points."""
        extractor = EyeExtractor(gaze_config)
        assert extractor.extract(None, 640, 480) is None
        short_landmarks = [SyntheticLandmark(0.5, 0.5) for _ in range(200)]
        assert extractor.extract(short_landmarks, 640, 480) is None

    def test_extractor_extracts_left_and_right_eye_data(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify both eyes are extracted with valid EyeData instances."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        assert isinstance(features, GazeFeatures)
        assert isinstance(features.left_eye, EyeData)
        assert isinstance(features.right_eye, EyeData)

    def test_extractor_neutral_gaze_normalized_coords(self, gaze_config: GazeConfig):
        """Verify centered synthetic gaze yields zero-centered norm_x ~ 0.0 and norm_y ~ 0.0."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(
            left_iris_offset_x=0.0, left_iris_offset_y=0.0,
            right_iris_offset_x=0.0, right_iris_offset_y=0.0
        )
        features = extractor.extract(landmarks, 640, 480)
        assert math.isclose(features.left_eye.norm_x, 0.0, abs_tol=1e-3)
        assert math.isclose(features.left_eye.norm_y, 0.0, abs_tol=1e-3)
        assert math.isclose(features.right_eye.norm_x, 0.0, abs_tol=1e-3)
        assert math.isclose(features.right_eye.norm_y, 0.0, abs_tol=1e-3)

    def test_extractor_horizontal_gaze_shifts(self, gaze_config: GazeConfig):
        """Verify horizontal iris movement shifts norm_x toward positive (right) and negative (left)."""
        extractor = EyeExtractor(gaze_config)
        # Gaze right -> positive norm_x for both eyes
        landmarks_right = create_synthetic_landmarks(left_iris_offset_x=0.20, right_iris_offset_x=0.20)
        feat_right = extractor.extract(landmarks_right, 640, 480)
        assert feat_right.left_eye.norm_x > 0.15
        assert feat_right.right_eye.norm_x > 0.15

        # Gaze left -> negative norm_x for both eyes
        landmarks_left = create_synthetic_landmarks(left_iris_offset_x=-0.20, right_iris_offset_x=-0.20)
        feat_left = extractor.extract(landmarks_left, 640, 480)
        assert feat_left.left_eye.norm_x < -0.15
        assert feat_left.right_eye.norm_x < -0.15

    def test_extractor_vertical_gaze_shifts(self, gaze_config: GazeConfig):
        """Verify vertical iris movement shifts norm_y toward positive (down) and negative (up)."""
        extractor = EyeExtractor(gaze_config)
        # Gaze down -> positive norm_y
        landmarks_down = create_synthetic_landmarks(left_iris_offset_y=0.20, right_iris_offset_y=0.20)
        feat_down = extractor.extract(landmarks_down, 640, 480)
        assert feat_down.left_eye.norm_y > 0.15
        assert feat_down.right_eye.norm_y > 0.15

        # Gaze up -> negative norm_y
        landmarks_up = create_synthetic_landmarks(left_iris_offset_y=-0.20, right_iris_offset_y=-0.20)
        feat_up = extractor.extract(landmarks_up, 640, 480)
        assert feat_up.left_eye.norm_y < -0.15
        assert feat_up.right_eye.norm_y < -0.15

    def test_extractor_zero_width_eye_corner_guard(self, gaze_config: GazeConfig):
        """Verify collapsed eye box (outer == inner) clamps without ZeroDivisionError."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks()
        # Collapse left eye corners to same point
        landmarks[gaze_config.left_eye_outer] = landmarks[gaze_config.left_eye_inner]
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert isinstance(features.left_eye.norm_x, float)


# ============================================================================
# F03: 6-Point EAR & Dynamic Blink Detection
# ============================================================================

class TestF03BlinkDetectionEAR:
    """Unit tests for Eye Aspect Ratio (EAR) computation and eye closure classification."""

    def test_ear_calculation_open_eye(self, gaze_config: GazeConfig):
        """Verify open eye has EAR above standard threshold."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(left_eye_closed=False, right_eye_closed=False)
        features = extractor.extract(landmarks, 640, 480)
        assert features.left_eye.ear >= gaze_config.ear_blink_threshold
        assert features.left_eye.is_open is True
        assert features.right_eye.is_open is True
        assert features.is_valid is True

    def test_ear_calculation_closed_eye(self, gaze_config: GazeConfig):
        """Verify closed eye (collapsed eyelids) yields EAR < threshold and is_open=False."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(left_eye_closed=True, right_eye_closed=True)
        features = extractor.extract(landmarks, 640, 480)
        assert features.left_eye.ear < gaze_config.ear_blink_threshold
        assert features.left_eye.is_open is False
        assert features.right_eye.is_open is False
        assert features.is_valid is False

    def test_gaze_features_validity_requires_both_eyes_open(self, gaze_config: GazeConfig):
        """Verify is_valid is False when only one eye is closed."""
        extractor = EyeExtractor(gaze_config)
        # Left closed, right open
        lm_left_closed = create_synthetic_landmarks(left_eye_closed=True, right_eye_closed=False)
        feat_1 = extractor.extract(lm_left_closed, 640, 480)
        assert feat_1.left_eye.is_open is False
        assert feat_1.right_eye.is_open is True
        assert feat_1.is_valid is False

        # Left open, right closed
        lm_right_closed = create_synthetic_landmarks(left_eye_closed=False, right_eye_closed=True)
        feat_2 = extractor.extract(lm_right_closed, 640, 480)
        assert feat_2.left_eye.is_open is True
        assert feat_2.right_eye.is_open is False
        assert feat_2.is_valid is False

    def test_dynamic_ear_threshold_configuration(self):
        """Verify changing ear_blink_threshold alters the open/closed decision boundary."""
        cfg_high = GazeConfig(ear_blink_threshold=0.45)
        extractor_high = EyeExtractor(cfg_high)
        landmarks = create_synthetic_landmarks(left_eye_closed=False)
        feat = extractor_high.extract(landmarks, 640, 480)
        # Standard synthetic eye has EAR ~ 0.31, so with threshold 0.45 it should be classified as closed
        assert feat.left_eye.ear < 0.45
        assert feat.left_eye.is_open is False

    def test_ear_zero_height_eyelid_guard(self, gaze_config: GazeConfig):
        """Verify completely collapsed eyelid landmarks result in EAR=0.0."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(left_eye_closed=True)
        feat = extractor.extract(landmarks, 640, 480)
        assert feat.left_eye.ear == 0.0
        assert feat.left_eye.is_open is False


# ============================================================================
# F04: Eye Contour & Metric Geometry
# ============================================================================

class TestF04EyeContourGeometry:
    """Unit tests for 16-point eyelid perimeter contour extraction and pixel formatting."""

    def test_contour_16_points_extracted(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify left and right eye contours contain exactly 16 vertex tuples."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        assert len(features.left_eye.contour_px) == 16
        assert len(features.right_eye.contour_px) == 16

    def test_contour_coordinates_bounded_within_image(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify all contour coordinates are integer tuples within [0, img_w] and [0, img_h]."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        for eye in (features.left_eye, features.right_eye):
            for pt in eye.contour_px:
                assert isinstance(pt, tuple)
                assert len(pt) == 2
                assert 0 <= pt[0] <= 640
                assert 0 <= pt[1] <= 480

    def test_keypoint_pixel_tuples_validity(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify inner corner, outer corner, top eyelid, bottom eyelid, and iris center are integer tuples."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        for eye in (features.left_eye, features.right_eye):
            for kpt in (eye.inner_corner_px, eye.outer_corner_px, eye.top_eyelid_px, eye.bottom_eyelid_px, eye.iris_center_px):
                assert isinstance(kpt, tuple)
                assert len(kpt) == 2
                assert isinstance(kpt[0], int)
                assert isinstance(kpt[1], int)

    def test_landmark_index_configuration_lists(self, gaze_config: GazeConfig):
        """Verify landmark index lists in GazeConfig match canonical MediaPipe 478 specifications."""
        assert len(gaze_config.left_eye_contour) == 16
        assert len(gaze_config.right_eye_contour) == 16
        assert len(gaze_config.left_iris_points) == 5
        assert len(gaze_config.right_iris_points) == 5
        assert gaze_config.left_iris_center == 468
        assert gaze_config.right_iris_center == 473

    def test_iris_center_pixel_matches_landmark_coordinate(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify extracted iris center pixel equals int(round(lm.x * img_w)), int(round(lm.y * img_h))."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        lm_left = synthetic_landmarks[gaze_config.left_iris_center]
        expected_px = (int(round(lm_left.x * 640)), int(round(lm_left.y * 480)))
        assert features.left_eye.iris_center_px == expected_px


# ============================================================================
# F05: 3D Head Pose solvePnP & Angle Extraction
# ============================================================================

class TestF05HeadPoseEstimation:
    """Unit tests for Perspective-n-Point 3D head pose and Euler angle extraction."""

    def test_head_pose_short_landmarks_guard(self, gaze_config: GazeConfig):
        """Verify estimator returns None if landmark list has fewer than 468 points."""
        estimator = HeadPoseEstimator(gaze_config)
        assert estimator.estimate(None, 640, 480) is None
        short_lm = [SyntheticLandmark(0.5, 0.5) for _ in range(100)]
        assert estimator.estimate(short_lm, 640, 480) is None

    def test_head_pose_camera_matrix_calculation(self, gaze_config: GazeConfig):
        """Verify camera intrinsic matrix has positive focal lengths and image center."""
        estimator = HeadPoseEstimator(gaze_config)
        cam_mat = estimator._get_camera_matrix(640, 480)
        assert cam_mat.shape == (3, 3)
        assert cam_mat[0, 0] > 0.0
        assert cam_mat[1, 1] > 0.0
        assert cam_mat[0, 2] == 320.0
        assert cam_mat[1, 2] == 240.0

    def test_head_pose_returns_valid_data_structure(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify estimate() produces a complete HeadPoseData dataclass."""
        estimator = HeadPoseEstimator(gaze_config)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        assert isinstance(pose, HeadPoseData)
        assert isinstance(pose.pitch, float)
        assert isinstance(pose.yaw, float)
        assert isinstance(pose.roll, float)
        assert pose.rvec.shape == (3, 1)
        assert pose.tvec.shape == (3, 1)
        assert len(pose.axes_2d_px) == 3

    def test_head_pose_axes_endpoints_count_and_types(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify 3D orientation axis projection returns 3 pixel coordinate tuples."""
        estimator = HeadPoseEstimator(gaze_config)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        assert len(pose.axes_2d_px) == 3
        for pt in pose.axes_2d_px:
            assert isinstance(pt, tuple)
            assert len(pt) == 2
            assert isinstance(pt[0], int)
            assert isinstance(pt[1], int)

    def test_head_pose_anthropometric_model_points(self):
        """Verify MODEL_POINTS contains 6 keypoints matching anthropometric face dimensions."""
        assert HeadPoseEstimator.MODEL_POINTS.shape == (6, 3)
        # Nose tip at origin
        assert np.array_equal(HeadPoseEstimator.MODEL_POINTS[0], [0.0, 0.0, 0.0])


# ============================================================================
# F06: Multi-Dimensional Tracking Quality & Combined Feature Vector
# ============================================================================

class TestF06FeatureVectorAggregation:
    """Unit tests for normalized feature vector construction and dimension verification."""

    def test_eye_feature_vector_dimension_8d(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify EyeExtractor returns an 8D feature vector."""
        extractor = EyeExtractor(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        assert features.feature_vector.shape == (8,)

    def test_head_pose_feature_vector_dimension_6d(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify HeadPoseEstimator returns a 6D normalized feature vector."""
        estimator = HeadPoseEstimator(gaze_config)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        assert pose.feature_vector.shape == (6,)

    def test_combined_14d_feature_vector_aggregation(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify legacy vector_14d property produces shape (14,) without NaN or Inf."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        features = extractor.extract(synthetic_landmarks, 640, 480, head_pose=pose)

        assert features.vector_14d.shape == (14,)
        assert np.all(np.isfinite(features.vector_14d))

    def test_avg_norm_coordinates_computation(self, gaze_config: GazeConfig):
        """Verify avg_norm_x and avg_norm_y correctly average the left and right eyes."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(
            left_iris_offset_x=0.10, left_iris_offset_y=0.15,
            right_iris_offset_x=0.20, right_iris_offset_y=0.25
        )
        features = extractor.extract(landmarks, 640, 480)
        expected_avg_x = (features.left_eye.norm_x + features.right_eye.norm_x) / 2.0
        expected_avg_y = (features.left_eye.norm_y + features.right_eye.norm_y) / 2.0
        assert math.isclose(features.avg_norm_x, expected_avg_x, abs_tol=1e-5)
        assert math.isclose(features.avg_norm_y, expected_avg_y, abs_tol=1e-5)

    def test_feature_vector_has_no_nan_or_inf(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify feature vector values are strictly finite without NaN or Inf."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        features = extractor.extract(synthetic_landmarks, 640, 480, head_pose=pose)
        assert not np.isnan(features.feature_vector).any()
        assert not np.isinf(features.feature_vector).any()


# ============================================================================
# F07-F13: Calibration & Regressor Units
# ============================================================================

class TestF07ToF13CalibrationAndRegressorUnits:
    """Unit tests for calibration manager, outlier filter, regressor, and serialization."""

    def test_calibrator_init_state_idle(self, gaze_config: GazeConfig):
        """Verify CalibrationManager initializes in IDLE state."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        assert calibrator.state == CalibrationState.IDLE
        assert len(calibrator.points) == 0

    def test_calibrator_generate_points_default_9_points(self, gaze_config: GazeConfig):
        """Verify generate_points() generates 9 distinct screen target points."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        points = calibrator.generate_points("9_points")
        assert len(points) == 9
        assert len(set(points)) == 9

    def test_calibrator_outlier_filter_with_small_sample(self, gaze_config: GazeConfig):
        """Verify outlier filter retains all samples when count < 5."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        samples = [np.random.randn(14) for _ in range(4)]
        clean = calibrator._filter_outliers(samples)
        assert len(clean) == 4

    def test_regressor_insufficient_samples_raises_value_error(self, gaze_config: GazeConfig):
        """Verify GazeRegressionModel.train() raises ValueError if samples < 6."""
        regressor = GazeRegressionModel(gaze_config)
        X = np.random.randn(5, 14)
        y = np.random.uniform(0, 1000, (5, 2))
        with pytest.raises(ValueError, match="Insufficient training samples"):
            regressor.train(X, y)

    def test_regressor_untrained_predict_returns_none(self, gaze_config: GazeConfig):
        """Verify predict() returns None when model is untrained."""
        regressor = GazeRegressionModel(gaze_config)
        assert regressor.is_trained is False
        pred = regressor.predict(np.random.randn(14))
        assert pred is None

    def test_regressor_untrained_save_raises_runtime_error(self, gaze_config: GazeConfig):
        """Verify save() raises RuntimeError when called on an untrained model."""
        regressor = GazeRegressionModel(gaze_config)
        with pytest.raises(RuntimeError, match="Cannot save an untrained model"):
            regressor.save()

    def test_regressor_load_missing_file_returns_false(self, gaze_config: GazeConfig):
        """Verify load() returns False safely when the file does not exist."""
        regressor = GazeRegressionModel(gaze_config)
        result = regressor.load("/tmp/non_existent_calibration_model_987654321.pkl")
        assert result is False


# ============================================================================
# F14-F15: Temporal Smoothing Filters Units
# ============================================================================

class TestF14ToF15TemporalFiltersUnits:
    """Unit tests for LowPassFilter, OneEuroFilter1D/2D, and KalmanFilter2D."""

    def test_low_pass_filter_initialization_and_step(self):
        """Verify LowPassFilter first value pass-through and exponential weighting."""
        lpf = LowPassFilter(alpha=0.5)
        assert lpf.filter(100.0) == 100.0
        assert lpf.filter(200.0) == 150.0  # 0.5*200 + 0.5*100
        lpf.reset()
        assert lpf.hat_x_prev is None

    def test_one_euro_1d_zero_delta_time_guard(self):
        """Verify OneEuroFilter1D returns previous filtered value without crash when dt <= 1e-5."""
        f = OneEuroFilter1D()
        val1 = f.filter(100.0, timestamp=1.0)
        val2 = f.filter(200.0, timestamp=1.0)  # Same timestamp, dt = 0
        assert val2 == val1

    def test_one_euro_2d_filter_and_reset(self):
        """Verify OneEuroFilter2D filters 2D point tuples and resets cleanly."""
        f2d = OneEuroFilter2D()
        p1 = f2d.filter((100.0, 200.0), timestamp=0.0)
        assert p1 == (100.0, 200.0)
        p2 = f2d.filter((150.0, 250.0), timestamp=0.033)
        assert isinstance(p2, tuple)
        assert len(p2) == 2
        f2d.reset()
        assert f2d.fx.t_prev is None

    def test_kalman_filter_2d_initialization(self):
        """Verify KalmanFilter2D initializes internal state vector on first point."""
        kf = KalmanFilter2D()
        assert kf.is_initialized is False
        pt = kf.filter((500.0, 300.0), timestamp=1.0)
        assert pt == (500.0, 300.0)
        assert kf.is_initialized is True
        assert kf.state[0, 0] == 500.0
        assert kf.state[1, 0] == 300.0

    def test_kalman_filter_2d_reset(self):
        """Verify KalmanFilter2D reset clears initialization state."""
        kf = KalmanFilter2D()
        kf.filter((100.0, 100.0), timestamp=1.0)
        assert kf.is_initialized is True
        kf.reset()
        assert kf.is_initialized is False


# ============================================================================
# F17-F24: UI Visualizer & Config Units
# ============================================================================

class TestF17ToF24VisualizerAndConfigUnits:
    """Unit tests for screen canvas rendering, HUD drawing, FPS calculation, and config."""

    def test_canvas_dimensions_and_type(self, gaze_config: GazeConfig):
        """Verify create_screen_canvas generates an image with correct dimensions and uint8 dtype."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)
        canvas = viz.create_screen_canvas(None, calibrator, False, {})
        assert canvas.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)
        assert canvas.dtype == np.uint8

    def test_canvas_uncalibrated_rendering(self, gaze_config: GazeConfig):
        """Verify uncalibrated state produces canvas without crashing."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)
        canvas = viz.create_screen_canvas(None, calibrator, False, {})
        assert canvas.sum() > 0  # Contains rendered text pixels

    def test_canvas_calibrated_tracking_rendering(self, gaze_config: GazeConfig):
        """Verify tracking state with gaze cursor renders and updates trail history."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)
        canvas = viz.create_screen_canvas((960.0, 540.0), calibrator, True, {"mae_px": 25.0})
        assert len(viz.trail_history) == 1
        assert canvas.sum() > 0

    def test_debug_hud_drawing_with_none_inputs(self, gaze_config: GazeConfig, mock_bgr_frame):
        """Verify draw_debug_hud does not crash when gaze features and head pose are None."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)
        hud = viz.draw_debug_hud(mock_bgr_frame, None, None, None, 30.0, False, calibrator)
        assert hud.shape == mock_bgr_frame.shape

    def test_debug_hud_drawing_with_valid_features(
        self, gaze_config: GazeConfig, mock_bgr_frame, synthetic_landmarks
    ):
        """Verify draw_debug_hud successfully renders all overlays with valid features."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)

        features = extractor.extract(synthetic_landmarks, 640, 480)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        hud = viz.draw_debug_hud(mock_bgr_frame, features, pose, (500.0, 400.0), 45.0, True, calibrator)
        assert hud.shape == mock_bgr_frame.shape

    def test_visualizer_compute_fps(self, gaze_config: GazeConfig):
        """Verify compute_fps() calculates a positive finite framerate."""
        viz = GazeVisualizer(gaze_config)
        fps1 = viz.compute_fps()
        assert fps1 > 0.0

    def test_config_screen_resolution_defaults(self, gaze_config: GazeConfig):
        """Verify GazeConfig defaults to standard 1080p resolution."""
        assert gaze_config.screen_width == 1920
        assert gaze_config.screen_height == 1080
        assert gaze_config.camera_width == 640
        assert gaze_config.camera_height == 480
