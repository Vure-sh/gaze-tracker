"""Milestone 1 Unit and Invariance Test Suite for CV & Robust Feature Engineering."""

import math
import numpy as np
import pytest
import cv2

from src.types import (
    NormalizedPoint,
    EyeData,
    HeadPoseData,
    GazeFeatures,
    GazePrediction,
    TrackingQuality,
    FaceDetectionResult
)
from src.config import GazeConfig, CameraConfig, QualityConfig
from src.cv.face_detector import FaceDetector
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from src.cv.quality_tracker import QualityTracker
from src.face_mesh_detector import FaceMeshDetector


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def gaze_config():
    return GazeConfig()


def build_synthetic_landmarks(
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    ear_height: float = 18.0,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    img_w: int = 640,
    img_h: int = 480
):
    """Builds a geometrically calibrated synthetic 478-landmark face."""
    lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    pts = {
        # Head pose landmarks
        1: np.array([320.0, 240.0]),     # Nose tip
        152: np.array([320.0, 340.0]),   # Chin
        33: np.array([240.0, 200.0]),    # Left outer canthus
        263: np.array([400.0, 200.0]),   # Right outer canthus
        61: np.array([280.0, 280.0]),    # Left mouth corner
        291: np.array([360.0, 280.0]),   # Right mouth corner

        # Left eye landmarks
        133: np.array([280.0, 200.0]),   # Left inner canthus
        159: np.array([260.0, 200.0 - ear_height / 2.0]),
        145: np.array([260.0, 200.0 + ear_height / 2.0]),
        160: np.array([250.0, 200.0 - ear_height / 2.2]),
        144: np.array([250.0, 200.0 + ear_height / 2.2]),
        158: np.array([270.0, 200.0 - ear_height / 2.2]),
        153: np.array([270.0, 200.0 + ear_height / 2.2]),
        468: np.array([260.0 + shift_x, 200.0 + shift_y]),
        469: np.array([260.0 + shift_x, 194.0 + shift_y]),
        470: np.array([266.0 + shift_x, 200.0 + shift_y]),
        471: np.array([260.0 + shift_x, 206.0 + shift_y]),
        472: np.array([254.0 + shift_x, 200.0 + shift_y]),

        # Right eye landmarks
        362: np.array([360.0, 200.0]),   # Right inner canthus
        386: np.array([380.0, 200.0 - ear_height / 2.0]),
        374: np.array([380.0, 200.0 + ear_height / 2.0]),
        385: np.array([370.0, 200.0 - ear_height / 2.2]),
        380: np.array([370.0, 200.0 + ear_height / 2.2]),
        387: np.array([390.0, 200.0 - ear_height / 2.2]),
        373: np.array([390.0, 200.0 + ear_height / 2.2]),
        473: np.array([380.0 + shift_x, 200.0 + shift_y]),
        474: np.array([380.0 + shift_x, 194.0 + shift_y]),
        475: np.array([386.0 + shift_x, 200.0 + shift_y]),
        476: np.array([380.0 + shift_x, 206.0 + shift_y]),
        477: np.array([374.0 + shift_x, 200.0 + shift_y]),
    }

    config = GazeConfig()
    for idx in config.left_eye_contour:
        if idx not in pts:
            pts[idx] = np.array([260.0, 200.0])
    for idx in config.right_eye_contour:
        if idx not in pts:
            pts[idx] = np.array([380.0, 200.0])

    center = np.array([320.0, 240.0])
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    for idx, pt in pts.items():
        transformed = center + scale * (rot_mat @ (pt - center))
        lms[idx] = NormalizedPoint(x=transformed[0] / img_w, y=transformed[1] / img_h, z=0.0)

    return lms


# ============================================================================
# 1. src/types.py Data Contracts Tests
# ============================================================================

class TestCoreTypes:
    def test_normalized_point_conversion(self):
        p = NormalizedPoint(x=0.25, y=0.50, z=0.10)
        px, py = p.to_pixel(640, 480)
        assert px == 160
        assert py == 240
        np_arr = p.to_numpy(640, 480)
        assert np.allclose(np_arr, [160.0, 240.0, 64.0])

    def test_normalized_point_from_landmark(self):
        class DummyLM:
            x = 0.3
            y = 0.7
            z = -0.05
        p = NormalizedPoint.from_landmark(DummyLM())
        assert math.isclose(p.x, 0.3)
        assert math.isclose(p.y, 0.7)
        assert math.isclose(p.z, -0.05)

    def test_gaze_features_vectors(self):
        eye = EyeData(
            norm_x=0.1, norm_y=-0.2, ear=0.35, is_open=True,
            iris_center_px=(260, 200), inner_corner_px=(280, 200),
            outer_corner_px=(240, 200), top_eyelid_px=(260, 190),
            bottom_eyelid_px=(260, 210), contour_px=[(240, 200), (280, 200)],
            iris_points_px=[(260, 200)], iris_diameter_px=12.0, circularity=0.98,
            iris_depth_mm=550.0
        )
        hp = HeadPoseData(
            pitch=10.0, yaw=-15.0, roll=5.0,
            rvec=np.zeros((3, 1)), tvec=np.array([[0.0], [0.0], [600.0]]),
            nose_2d_px=(320, 240), axes_2d_px=[(350, 240), (320, 270), (320, 240)],
            feature_vector=np.zeros(6)
        )
        gf = GazeFeatures(
            left_eye=eye, right_eye=eye, avg_norm_x=0.1, avg_norm_y=-0.2,
            head_pose=hp, confidence=0.95, is_valid=True
        )

        v8 = gf.vector_8d
        assert v8.shape == (8,)
        assert math.isclose(v8[0], 0.1)
        assert math.isclose(v8[4], 10.0 / 45.0)
        assert math.isclose(v8[5], -15.0 / 45.0)
        assert math.isclose(v8[7], 0.6)

        v10 = gf.vector_10d
        assert v10.shape == (10,)
        assert math.isclose(v10[4], 0.1)  # avg_norm_x

        v14 = gf.vector_14d
        assert v14.shape == (14,)


# ============================================================================
# 2. src/config.py Tests
# ============================================================================

class TestConfig:
    def test_camera_config_fov_matrix(self):
        cam = CameraConfig(width=640, height=480, fov_h_deg=65.0, fov_v_deg=48.75)
        mat = cam.get_camera_matrix()
        assert mat.shape == (3, 3)
        assert mat[0, 2] == 320.0
        assert mat[1, 2] == 240.0
        assert 450.0 < mat[0, 0] < 550.0

    def test_gaze_config_landmarks_and_thresholds(self, gaze_config: GazeConfig):
        assert len(gaze_config.left_eye_ear_indices) == 6
        assert len(gaze_config.right_eye_ear_indices) == 6
        assert len(gaze_config.left_iris_points) == 5
        assert len(gaze_config.right_iris_points) == 5
        assert gaze_config.iris_metric_diameter_mm == 11.7
        assert gaze_config.ear_adaptive_ratio == 0.60


# ============================================================================
# 3. src/cv/face_detector.py & Wrapper Tests
# ============================================================================

class TestFaceDetector:
    def test_detector_initialization_and_wrapper(self, gaze_config: GazeConfig):
        det = FaceDetector(model_path=gaze_config.model_path)
        assert det.detector is not None
        wrapper = FaceMeshDetector(model_path=gaze_config.model_path)
        assert isinstance(wrapper, FaceDetector)

    def test_detector_none_and_invalid_inputs(self, gaze_config: GazeConfig):
        det = FaceDetector(model_path=gaze_config.model_path)
        assert det.detect(None) is None
        assert det.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None
        assert det.detect_full(None) is None


# ============================================================================
# 4. src/cv/eye_extractor.py Tests
# ============================================================================

class TestEyeExtractor:
    def test_eye_extractor_directional_sensitivity(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        neutral = build_synthetic_landmarks(shift_x=0.0)
        right = build_synthetic_landmarks(shift_x=6.0)
        left = build_synthetic_landmarks(shift_x=-6.0)

        f_neutral = ee.extract(neutral, 640, 480)
        f_right = ee.extract(right, 640, 480)
        f_left = ee.extract(left, 640, 480)

        # Both eyes must increase norm_x when looking right (+X)
        assert f_right.left_eye.norm_x > f_neutral.left_eye.norm_x
        assert f_right.right_eye.norm_x > f_neutral.right_eye.norm_x
        assert f_right.avg_norm_x > f_neutral.avg_norm_x

        # Both eyes must decrease norm_x when looking left (-X)
        assert f_left.left_eye.norm_x < f_neutral.left_eye.norm_x
        assert f_left.right_eye.norm_x < f_neutral.right_eye.norm_x
        assert f_left.avg_norm_x < f_neutral.avg_norm_x

    def test_eye_extractor_roll_invariance(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        base = build_synthetic_landmarks(shift_x=5.0, shift_y=-2.0, angle_deg=0.0)
        f_base = ee.extract(base, 640, 480)

        for angle in [-30.0, -15.0, 15.0, 30.0]:
            rotated = build_synthetic_landmarks(shift_x=5.0, shift_y=-2.0, angle_deg=angle)
            f_rot = ee.extract(rotated, 640, 480)
            assert math.isclose(f_rot.avg_norm_x, f_base.avg_norm_x, abs_tol=1e-3)
            assert math.isclose(f_rot.avg_norm_y, f_base.avg_norm_y, abs_tol=1e-3)

    def test_eye_extractor_scale_invariance(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        base = build_synthetic_landmarks(shift_x=5.0, shift_y=3.0, scale=1.0)
        f_base = ee.extract(base, 640, 480)

        for scale in [0.6, 0.8, 1.2, 1.6, 2.0]:
            scaled = build_synthetic_landmarks(shift_x=5.0, shift_y=3.0, scale=scale)
            f_scaled = ee.extract(scaled, 640, 480)
            assert math.isclose(f_scaled.avg_norm_x, f_base.avg_norm_x, abs_tol=1e-3)
            assert math.isclose(f_scaled.avg_norm_y, f_base.avg_norm_y, abs_tol=1e-3)

    def test_6point_ear_and_blink_detection(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        open_lms = build_synthetic_landmarks(ear_height=18.0)
        closed_lms = build_synthetic_landmarks(ear_height=2.0)

        f_open = ee.extract(open_lms, 640, 480)
        f_closed = ee.extract(closed_lms, 640, 480)

        assert f_open.left_eye.is_open is True
        assert f_open.left_eye.ear > 0.25
        assert f_open.is_valid is True

        assert f_closed.left_eye.is_open is False
        assert f_closed.left_eye.ear < 0.12
        assert f_closed.is_valid is False

    def test_5point_iris_circularity_and_metric_depth(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        lms = build_synthetic_landmarks(scale=1.0)
        feat = ee.extract(lms, 640, 480)
        assert feat.left_eye.circularity > 0.95
        assert 400.0 < feat.left_eye.iris_depth_mm < 700.0


# ============================================================================
# 5. src/cv/head_pose.py Tests
# ============================================================================

class TestHeadPoseEstimator:
    def test_neutral_pose_euler_angles(self, gaze_config: GazeConfig):
        hpe = HeadPoseEstimator(gaze_config)
        cam_mat = gaze_config.get_camera_matrix(640, 480)
        rvec_zero = np.zeros((3, 1), dtype=np.float64)
        tvec_zero = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        proj_pts, _ = cv2.projectPoints(hpe.MODEL_POINTS_CORRECTED, rvec_zero, tvec_zero, cam_mat, dist_coeffs)

        lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        for i, idx in enumerate(gaze_config.head_pose_mesh_indices):
            px, py = proj_pts[i, 0]
            lms[idx] = NormalizedPoint(x=px / 640.0, y=py / 480.0, z=0.0)

        data = hpe.estimate(lms, 640, 480)
        assert data is not None
        assert abs(data.pitch) < 0.1
        assert abs(data.yaw) < 0.1
        assert abs(data.roll) < 0.1

    def test_degenerate_landmarks_return_none(self, gaze_config: GazeConfig):
        hpe = HeadPoseEstimator(gaze_config)
        lms_degenerate = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        # All points identical -> solvePnP fails gracefully
        assert hpe.estimate(lms_degenerate, 640, 480) is None

    def test_short_landmark_list_guard(self, gaze_config: GazeConfig):
        hpe = HeadPoseEstimator(gaze_config)
        assert hpe.estimate([], 640, 480) is None
        assert hpe.estimate(None, 640, 480) is None


# ============================================================================
# 6. src/cv/quality_tracker.py Tests
# ============================================================================

class TestQualityTracker:
    def test_quality_tracker_valid_tracking(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        qt = QualityTracker(gaze_config)
        lms = build_synthetic_landmarks(ear_height=18.0)
        feat = ee.extract(lms, 640, 480)

        quality = qt.evaluate(feat, lms, img_w=640, img_h=480)
        assert quality.is_valid is True
        assert quality.confidence >= 0.70
        assert quality.ear_score > 0.80
        assert quality.circularity_score > 0.80

    def test_quality_tracker_blink_detection(self, gaze_config: GazeConfig):
        ee = EyeExtractor(gaze_config)
        qt = QualityTracker(gaze_config)
        lms = build_synthetic_landmarks(ear_height=2.0)
        feat = ee.extract(lms, 640, 480)

        quality = qt.evaluate(feat, lms, img_w=640, img_h=480)
        assert quality.is_valid is False
        assert quality.confidence <= 0.25
        assert any("blink" in reason.lower() for reason in quality.failure_reasons)

    def test_quality_tracker_none_guard(self, gaze_config: GazeConfig):
        qt = QualityTracker(gaze_config)
        quality = qt.evaluate(None, None)
        assert quality.is_valid is False
        assert quality.confidence == 0.0
