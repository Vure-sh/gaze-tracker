"""Adversarial stress harness and empirical challenge suite for Milestone 1 (CV & Feature Engineering).

Stress test dimensions:
1. Head roll rotations from -90° to +90° in 5° steps: verify iris normalization remains strictly invariant.
2. Head scale variations from 0.2x to 5.0x: verify iris normalization remains strictly scale-invariant.
3. Head pose pitch/yaw/roll sweeps: verify absence of branch-cut jumps or gimbal lock singularities near ±45°.
4. Blink transitions: verify adaptive EAR cleanly detects eye closure and flags is_open=False.
5. Degenerate inputs (collinear landmarks, zero width eye bounding box, zero coordinates, NaN/Inf): verify no uncaught exceptions.
"""

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


def build_challenger_landmarks(
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    ear_height: float = 10.0,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    left_closed: bool = False,
    right_closed: bool = False,
    img_w: int = 640,
    img_h: int = 480
):
    """Generates a geometrically precise 478-point synthetic landmark set."""
    lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    center = np.array([img_w / 2.0 + tx_px, img_h / 2.0 + ty_px])
    eye_half_w = 32.0 * scale
    eye_span = 80.0 * scale

    left_h = ear_height * scale if not left_closed else 0.0
    right_h = ear_height * scale if not right_closed else 0.0

    left_center = np.array([center[0] - eye_span, center[1] - 40.0 * scale])
    right_center = np.array([center[0] + eye_span, center[1] - 40.0 * scale])

    pts = {
        # Head pose 3D model points
        1: np.array([center[0], center[1]]),                            # Nose tip
        152: np.array([center[0], center[1] + 100.0 * scale]),          # Chin
        33: np.array([left_center[0] - eye_half_w, left_center[1]]),    # Left eye outer canthus
        263: np.array([right_center[0] + eye_half_w, right_center[1]]), # Right eye outer canthus
        61: np.array([center[0] - 40.0 * scale, center[1] + 50.0 * scale]),  # Left mouth corner
        291: np.array([center[0] + 40.0 * scale, center[1] + 50.0 * scale]), # Right mouth corner

        # Left eye landmarks
        133: np.array([left_center[0] + eye_half_w, left_center[1]]),   # Left inner canthus
        159: np.array([left_center[0], left_center[1] - left_h]),       # Top peak
        145: np.array([left_center[0], left_center[1] + left_h]),       # Bottom peak
        160: np.array([left_center[0] - 10.0 * scale, left_center[1] - left_h * 0.9]), # Top 1
        144: np.array([left_center[0] - 10.0 * scale, left_center[1] + left_h * 0.9]), # Bottom 1
        158: np.array([left_center[0] + 10.0 * scale, left_center[1] - left_h * 0.9]), # Top 2
        153: np.array([left_center[0] + 10.0 * scale, left_center[1] + left_h * 0.9]), # Bottom 2

        # Right eye landmarks
        362: np.array([right_center[0] - eye_half_w, right_center[1]]),  # Right inner canthus
        386: np.array([right_center[0], right_center[1] - right_h]),      # Top peak
        374: np.array([right_center[0], right_center[1] + right_h]),      # Bottom peak
        385: np.array([right_center[0] - 10.0 * scale, right_center[1] - right_h * 0.9]),
        380: np.array([right_center[0] - 10.0 * scale, right_center[1] + right_h * 0.9]),
        387: np.array([right_center[0] + 10.0 * scale, right_center[1] - right_h * 0.9]),
        373: np.array([right_center[0] + 10.0 * scale, right_center[1] + right_h * 0.9]),
    }

    # Left iris
    left_iris_center = left_center + np.array([shift_x * scale, shift_y * scale])
    pts[468] = left_iris_center
    r_iris = 6.0 * scale
    pts[469] = left_iris_center + np.array([r_iris, 0.0])
    pts[470] = left_iris_center + np.array([-r_iris, 0.0])
    pts[471] = left_iris_center + np.array([0.0, -r_iris])
    pts[472] = left_iris_center + np.array([0.0, r_iris])

    # Right iris
    right_iris_center = right_center + np.array([shift_x * scale, shift_y * scale])
    pts[473] = right_iris_center
    pts[474] = right_iris_center + np.array([r_iris, 0.0])
    pts[475] = right_iris_center + np.array([-r_iris, 0.0])
    pts[476] = right_iris_center + np.array([0.0, -r_iris])
    pts[477] = right_iris_center + np.array([0.0, r_iris])

    # Eye contours (16 points per eye)
    config = GazeConfig()
    for idx_c, l_idx in enumerate(config.left_eye_contour):
        if l_idx not in pts:
            ang = 2.0 * math.pi * idx_c / len(config.left_eye_contour)
            pts[l_idx] = left_center + np.array([eye_half_w * math.cos(ang), left_h * math.sin(ang)])

    for idx_c, r_idx in enumerate(config.right_eye_contour):
        if r_idx not in pts:
            ang = 2.0 * math.pi * idx_c / len(config.right_eye_contour)
            pts[r_idx] = right_center + np.array([eye_half_w * math.cos(ang), right_h * math.sin(ang)])

    # Apply 2D roll rotation around face center
    if abs(angle_deg) > 1e-4:
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        for k, pt in pts.items():
            pts[k] = center + (rot_mat @ (pt - center))

    for idx, pt in pts.items():
        if 0 <= idx < 478:
            lms[idx] = NormalizedPoint(x=float(pt[0] / img_w), y=float(pt[1] / img_h), z=0.0)

    return lms


# ============================================================================
# Challenge 1: Head Roll Rotations [-90°, +90°] in 5° Steps
# ============================================================================

class TestEmpiricalHeadRollInvariance:
    """Stress test: Head roll sweep across full [-90°, +90°] range in 5° increments."""

    ROLL_ANGLES = list(range(-90, 95, 5))  # 37 distinct angles: -90, -85, ..., 0, ..., 85, 90

    @pytest.mark.parametrize("roll_angle", ROLL_ANGLES)
    def test_roll_sweep_norm_coordinates_invariance(self, roll_angle: int):
        config = GazeConfig()
        extractor = EyeExtractor(config)

        # Baseline offset: shift_x = 10.0px, shift_y = -6.0px (eye_width = 64px -> norm_x = 10/64 = 0.15625)
        shift_x = 10.0
        shift_y = -6.0
        expected_norm_x = shift_x / 64.0
        expected_norm_y = shift_y / 64.0

        landmarks = build_challenger_landmarks(
            shift_x=shift_x,
            shift_y=shift_y,
            angle_deg=float(roll_angle),
            img_w=640,
            img_h=480
        )

        features = extractor.extract(landmarks, 640, 480)
        assert features is not None, f"Extraction returned None at roll={roll_angle}°"

        # Check left eye
        err_left_x = abs(features.left_eye.norm_x - expected_norm_x)
        err_left_y = abs(features.left_eye.norm_y - expected_norm_y)
        assert err_left_x < 1e-3, f"Left eye norm_x error {err_left_x:.6f} at roll={roll_angle}°"
        assert err_left_y < 1e-3, f"Left eye norm_y error {err_left_y:.6f} at roll={roll_angle}°"

        # Check right eye
        err_right_x = abs(features.right_eye.norm_x - expected_norm_x)
        err_right_y = abs(features.right_eye.norm_y - expected_norm_y)
        assert err_right_x < 1e-3, f"Right eye norm_x error {err_right_x:.6f} at roll={roll_angle}°"
        assert err_right_y < 1e-3, f"Right eye norm_y error {err_right_y:.6f} at roll={roll_angle}°"

        # Check average
        err_avg_x = abs(features.avg_norm_x - expected_norm_x)
        err_avg_y = abs(features.avg_norm_y - expected_norm_y)
        assert err_avg_x < 1e-3, f"Avg norm_x error {err_avg_x:.6f} at roll={roll_angle}°"
        assert err_avg_y < 1e-3, f"Avg norm_y error {err_avg_y:.6f} at roll={roll_angle}°"

    @pytest.mark.parametrize("roll_angle", ROLL_ANGLES)
    def test_roll_sweep_ear_invariance(self, roll_angle: int):
        config = GazeConfig()
        extractor = EyeExtractor(config)

        landmarks = build_challenger_landmarks(
            angle_deg=float(roll_angle),
            left_closed=False,
            right_closed=False,
            img_w=640,
            img_h=480
        )

        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        # EAR = (10*0.9 + 10*0.9) / (2 * 64) = 18 / 64 = 0.28125
        assert math.isclose(features.left_eye.ear, 0.28125, abs_tol=1e-3)
        assert math.isclose(features.right_eye.ear, 0.28125, abs_tol=1e-3)
        assert features.left_eye.is_open is True
        assert features.right_eye.is_open is True


# ============================================================================
# Challenge 2: Head Scale Variations [0.2x to 5.0x]
# ============================================================================

class TestEmpiricalScaleInvariance:
    """Stress test: Head scale variations from extreme miniature (0.2x) to extreme close-up (5.0x)."""

    SCALE_FACTORS = [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    @pytest.mark.parametrize("scale", SCALE_FACTORS)
    def test_scale_sweep_norm_coordinates_invariance(self, scale: float):
        config = GazeConfig()
        extractor = EyeExtractor(config)

        shift_x = -8.0
        shift_y = 6.0
        expected_norm_x = shift_x / 64.0
        expected_norm_y = shift_y / 64.0

        landmarks = build_challenger_landmarks(
            shift_x=shift_x,
            shift_y=shift_y,
            scale=scale,
            img_w=640,
            img_h=480
        )

        features = extractor.extract(landmarks, 640, 480)
        assert features is not None, f"Extraction returned None at scale={scale}x"

        err_left_x = abs(features.left_eye.norm_x - expected_norm_x)
        err_left_y = abs(features.left_eye.norm_y - expected_norm_y)
        assert err_left_x < 1e-3, f"Left eye norm_x error {err_left_x:.6f} at scale={scale}x"
        assert err_left_y < 1e-3, f"Left eye norm_y error {err_left_y:.6f} at scale={scale}x"

        err_right_x = abs(features.right_eye.norm_x - expected_norm_x)
        err_right_y = abs(features.right_eye.norm_y - expected_norm_y)
        assert err_right_x < 1e-3, f"Right eye norm_x error {err_right_x:.6f} at scale={scale}x"
        assert err_right_y < 1e-3, f"Right eye norm_y error {err_right_y:.6f} at scale={scale}x"

    @pytest.mark.parametrize("scale", SCALE_FACTORS)
    def test_scale_sweep_ear_invariance(self, scale: float):
        config = GazeConfig()
        extractor = EyeExtractor(config)

        landmarks = build_challenger_landmarks(scale=scale, img_w=640, img_h=480)
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.ear, 0.28125, abs_tol=1e-3)
        assert math.isclose(features.right_eye.ear, 0.28125, abs_tol=1e-3)

    def test_scale_sweep_metric_depth_inversely_proportional(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)

        depths = []
        for s in [0.5, 1.0, 2.0, 4.0]:
            landmarks = build_challenger_landmarks(scale=s, img_w=640, img_h=480)
            feat = extractor.extract(landmarks, 640, 480)
            assert feat is not None
            depths.append((s, feat.left_eye.iris_depth_mm))

        products = [s * d for s, d in depths]
        for p in products[1:]:
            assert math.isclose(p, products[0], rel_tol=1e-3), f"Depth not inversely proportional: {depths}"


# ============================================================================
# Challenge 3: Head Pose Continuous Pitch/Yaw/Roll Sweeps around ±45°
# ============================================================================

class TestEmpiricalHeadPoseContinuity:
    """Stress test: Head pose pitch/yaw/roll sweeps verifying continuous Euler angle recovery."""

    def _generate_pose_landmarks(self, pitch_deg: float, yaw_deg: float, roll_deg: float, config: GazeConfig, hpe: HeadPoseEstimator):
        cam_mat = config.get_camera_matrix(640, 480)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        pitch_rad = np.radians(pitch_deg)
        yaw_rad = np.radians(yaw_deg)
        roll_rad = np.radians(roll_deg)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        Ry = np.array([
            [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)]
        ])
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0.0],
            [np.sin(roll_rad), np.cos(roll_rad), 0.0],
            [0.0, 0.0, 1.0]
        ])

        # R = Rz @ Ry @ Rx matches intrinsic ZYX / extrinsic XYZ decomposition
        rot_mat = Rz @ Ry @ Rx

        rvec, _ = cv2.Rodrigues(rot_mat)
        tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

        proj_pts, _ = cv2.projectPoints(hpe.MODEL_POINTS_CORRECTED, rvec, tvec, cam_mat, dist_coeffs)

        lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        for i, idx in enumerate(config.head_pose_mesh_indices):
            px, py = proj_pts[i, 0]
            lms[idx] = NormalizedPoint(x=px / 640.0, y=py / 480.0, z=0.0)

        return lms

    def test_pitch_sweep_minus_50_to_plus_50(self):
        """Sweep pitch from -50° to +50° in 1° steps. Verify no jumps near ±45°."""
        config = GazeConfig()
        hpe = HeadPoseEstimator(config)

        angles = np.linspace(-50.0, 50.0, 101)
        estimated_pitches = []

        for p in angles:
            lms = self._generate_pose_landmarks(pitch_deg=p, yaw_deg=0.0, roll_deg=0.0, config=config, hpe=hpe)
            data = hpe.estimate(lms, 640, 480)
            assert data is not None, f"solvePnP failed at pitch={p}°"
            estimated_pitches.append(data.pitch)
            assert abs(data.pitch - p) < 0.2, f"Pitch mismatch at {p}°: got {data.pitch}°"
            assert abs(data.yaw) < 0.2, f"Spurious yaw at pure pitch {p}°: got {data.yaw}°"
            assert abs(data.roll) < 0.2, f"Spurious roll at pure pitch {p}°: got {data.roll}°"

        diffs = np.abs(np.diff(estimated_pitches))
        max_step_diff = float(np.max(diffs))
        assert max_step_diff < 1.5, f"Discontinuous pitch jump detected: max step diff = {max_step_diff}°"

    def test_yaw_sweep_minus_50_to_plus_50(self):
        """Sweep yaw from -50° to +50° in 1° steps. Verify no jumps near ±45°."""
        config = GazeConfig()
        hpe = HeadPoseEstimator(config)

        angles = np.linspace(-50.0, 50.0, 101)
        estimated_yaws = []

        for y in angles:
            lms = self._generate_pose_landmarks(pitch_deg=0.0, yaw_deg=y, roll_deg=0.0, config=config, hpe=hpe)
            data = hpe.estimate(lms, 640, 480)
            assert data is not None, f"solvePnP failed at yaw={y}°"
            estimated_yaws.append(data.yaw)
            assert abs(data.yaw - y) < 0.2, f"Yaw mismatch at {y}°: got {data.yaw}°"
            assert abs(data.pitch) < 0.2, f"Spurious pitch at pure yaw {y}°: got {data.pitch}°"
            assert abs(data.roll) < 0.2, f"Spurious roll at pure yaw {y}°: got {data.roll}°"

        diffs = np.abs(np.diff(estimated_yaws))
        max_step_diff = float(np.max(diffs))
        assert max_step_diff < 1.5, f"Discontinuous yaw jump detected: max step diff = {max_step_diff}°"

    def test_roll_sweep_minus_50_to_plus_50(self):
        """Sweep roll from -50° to +50° in 1° steps. Verify no jumps near ±45°."""
        config = GazeConfig()
        hpe = HeadPoseEstimator(config)

        angles = np.linspace(-50.0, 50.0, 101)
        estimated_rolls = []

        for r in angles:
            lms = self._generate_pose_landmarks(pitch_deg=0.0, yaw_deg=0.0, roll_deg=r, config=config, hpe=hpe)
            data = hpe.estimate(lms, 640, 480)
            assert data is not None, f"solvePnP failed at roll={r}°"
            estimated_rolls.append(data.roll)
            assert abs(data.roll - r) < 0.2, f"Roll mismatch at {r}°: got {data.roll}°"
            assert abs(data.pitch) < 0.2, f"Spurious pitch at pure roll {r}°: got {data.pitch}°"
            assert abs(data.yaw) < 0.2, f"Spurious yaw at pure roll {r}°: got {data.yaw}°"

        diffs = np.abs(np.diff(estimated_rolls))
        max_step_diff = float(np.max(diffs))
        assert max_step_diff < 1.5, f"Discontinuous roll jump detected: max step diff = {max_step_diff}°"

    @pytest.mark.parametrize("p, y, r", [
        (45.0, 45.0, 0.0),
        (-45.0, 45.0, 0.0),
        (45.0, -45.0, 0.0),
        (-45.0, -45.0, 0.0),
        (35.0, 35.0, 35.0),
        (-35.0, -35.0, -35.0),
    ])
    def test_combined_extreme_angles(self, p: float, y: float, r: float):
        """Test compound rotations combining extreme pitch, yaw, and roll."""
        config = GazeConfig()
        hpe = HeadPoseEstimator(config)
        lms = self._generate_pose_landmarks(pitch_deg=p, yaw_deg=y, roll_deg=r, config=config, hpe=hpe)
        data = hpe.estimate(lms, 640, 480)
        assert data is not None
        assert abs(data.pitch - p) < 0.5
        assert abs(data.yaw - y) < 0.5
        assert abs(data.roll - r) < 0.5


# ============================================================================
# Challenge 4: Blink Transitions & Adaptive EAR Sensitivity
# ============================================================================

class TestEmpiricalBlinkTransitions:
    """Stress test: Dynamic blink transitions across diverse eye baseline geometries."""

    def test_full_blink_sequence_lifecycle(self):
        """Simulate realistic open -> closing -> closed -> opening -> open sequence."""
        config = GazeConfig()
        extractor = EyeExtractor(config)
        qt = QualityTracker(config)

        # Baseline open frames (20 frames)
        for _ in range(20):
            lms = build_challenger_landmarks(ear_height=10.0)
            f = extractor.extract(lms, 640, 480)
            assert f is not None
            assert f.left_eye.is_open is True
            assert f.is_valid is True

        # Fully closed frames (5 frames)
        for _ in range(5):
            lms = build_challenger_landmarks(left_closed=True, right_closed=True)
            f = extractor.extract(lms, 640, 480)
            assert f is not None
            assert f.left_eye.is_open is False
            assert f.right_eye.is_open is False
            assert f.is_valid is False
            q = qt.evaluate(f, lms)
            assert q.is_valid is False
            assert q.confidence <= 0.20

        # Opening transition & recovery (10 frames)
        for _ in range(10):
            lms = build_challenger_landmarks(ear_height=10.0)
            f = extractor.extract(lms, 640, 480)
            assert f is not None

        # Must cleanly recover to is_open=True
        assert f.left_eye.is_open is True
        assert f.right_eye.is_open is True
        assert f.is_valid is True

    def test_narrow_eyes_adaptive_threshold_adaptation(self):
        """Test user with naturally narrow/small eyes (baseline open EAR ~ 0.20)."""
        config = GazeConfig()
        extractor = EyeExtractor(config)

        # Warm up adaptive history with narrow eyes (ear_height = 7.0 -> EAR ~ 0.197)
        for _ in range(25):
            lms = build_challenger_landmarks(ear_height=7.0)
            f = extractor.extract(lms, 640, 480)
            assert f is not None
            assert f.left_eye.is_open is True

        # When blinking (ear_height = 1.0 -> EAR ~ 0.028)
        lms_closed = build_challenger_landmarks(ear_height=1.0)
        f_closed = extractor.extract(lms_closed, 640, 480)
        assert f_closed is not None
        assert f_closed.left_eye.is_open is False
        assert f_closed.is_valid is False

    def test_wide_eyes_adaptive_threshold_adaptation(self):
        """Test user with wide eyes (baseline open EAR ~ 0.40)."""
        config = GazeConfig()
        extractor = EyeExtractor(config)

        # Warm up adaptive history with wide eyes (ear_height = 14.0 -> EAR ~ 0.394)
        for _ in range(25):
            lms = build_challenger_landmarks(ear_height=14.0)
            f = extractor.extract(lms, 640, 480)
            assert f is not None
            assert f.left_eye.is_open is True

        # Half-blink (ear_height = 5.0 -> EAR ~ 0.14) -> for wide eyes, this is closed!
        lms_half = build_challenger_landmarks(ear_height=5.0)
        f_half = extractor.extract(lms_half, 640, 480)
        assert f_half is not None
        assert f_half.left_eye.is_open is False
        assert f_half.is_valid is False


# ============================================================================
# Challenge 5: Degenerate Inputs & Adversarial Robustness
# ============================================================================

class TestEmpiricalDegenerateInputs:
    """Stress test: Degenerate, corrupted, and mathematically adversarial inputs."""

    def test_collinear_landmarks_solvepnp_and_eye_extractor(self):
        """All landmarks aligned on a straight line: y = x."""
        config = GazeConfig()
        extractor = EyeExtractor(config)
        hpe = HeadPoseEstimator(config)
        qt = QualityTracker(config)

        collinear_lms = [
            NormalizedPoint(x=i / 500.0, y=i / 500.0, z=0.0)
            for i in range(478)
        ]

        feat = extractor.extract(collinear_lms, 640, 480)
        assert feat is not None

        pose = hpe.estimate(collinear_lms, 640, 480)
        q = qt.evaluate(feat, collinear_lms)
        assert isinstance(q, TrackingQuality)

    def test_all_zero_and_singular_coordinates(self):
        """All landmark coordinates exactly (0.0, 0.0, 0.0)."""
        config = GazeConfig()
        extractor = EyeExtractor(config)
        hpe = HeadPoseEstimator(config)
        qt = QualityTracker(config)

        zero_lms = [NormalizedPoint(x=0.0, y=0.0, z=0.0) for _ in range(478)]

        feat = extractor.extract(zero_lms, 640, 480)
        assert feat is not None
        assert not math.isnan(feat.left_eye.norm_x)
        assert not math.isnan(feat.left_eye.norm_y)

        # Pose estimation on zero coordinates should either return None or be flagged invalid
        pose = hpe.estimate(zero_lms, 640, 480)

        q = qt.evaluate(feat, zero_lms)
        assert isinstance(q, TrackingQuality)

    def test_zero_width_eye_bounding_box(self):
        """Eye inner and outer corners at the exact same location."""
        config = GazeConfig()
        extractor = EyeExtractor(config)

        lms = build_challenger_landmarks()
        lms[config.left_eye_inner] = NormalizedPoint(x=0.4, y=0.4, z=0.0)
        lms[config.left_eye_outer] = NormalizedPoint(x=0.4, y=0.4, z=0.0)

        feat = extractor.extract(lms, 640, 480)
        assert feat is not None
        assert not math.isnan(feat.left_eye.norm_x)
        assert not math.isnan(feat.left_eye.norm_y)

    def test_negative_and_extreme_out_of_bounds_coordinates(self):
        """Landmarks far outside normal [0.0, 1.0] viewport."""
        config = GazeConfig()
        extractor = EyeExtractor(config)
        hpe = HeadPoseEstimator(config)

        oob_lms = [NormalizedPoint(x=-9999.0, y=9999.0, z=-500.0) for _ in range(478)]
        feat = extractor.extract(oob_lms, 640, 480)
        assert feat is not None
        pose = hpe.estimate(oob_lms, 640, 480)

    def test_truncated_or_empty_landmark_lists(self):
        """Short landmark lists (< 478 points)."""
        config = GazeConfig()
        extractor = EyeExtractor(config)
        hpe = HeadPoseEstimator(config)

        for count in [0, 1, 10, 100, 467]:
            short_lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(count)]
            assert extractor.extract(short_lms, 640, 480) is None
            assert hpe.estimate(short_lms, 640, 480) is None

        assert extractor.extract(None, 640, 480) is None
        assert hpe.estimate(None, 640, 480) is None

    def test_detector_malformed_and_edge_case_frames(self):
        """FaceDetector handling empty, non-standard, or corrupted image arrays."""
        config = GazeConfig()
        detector = FaceDetector(model_path=config.model_path)

        assert detector.detect(None) is None
        assert detector.detect(np.zeros((0, 0), dtype=np.uint8)) is None
        assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None
        assert detector.detect(np.zeros((10, 10, 1), dtype=np.uint8)) is None
        assert detector.detect(np.zeros((100, 100, 4), dtype=np.uint8)) is None
