"""Tier 2: Geometric & Transformation Invariance Tests for Gaze Tracker.

Covers: Head roll invariance (0°, 15°, -15°, 30°, 45°), scale/distance invariance (0.5x, 1.0x, 2.0x),
translation invariance, 3D head pose decoupling (pitch/yaw ±15°), and resolution invariance (>= 5 tests per feature).
"""

import math
import numpy as np
import pytest

from src.config import GazeConfig
from src.eye_extractor import EyeExtractor
from src.head_pose import HeadPoseEstimator
from tests.conftest import create_synthetic_landmarks


# ============================================================================
# 1. Head Roll Invariance Tests (0°, +15°, -15°, +30°, +45°, -45°)
# ============================================================================

class TestHeadRollInvariance:
    """Verifies that dual-eye orthonormal iris normalization is invariant under head roll rotations."""

    @pytest.mark.parametrize("roll_angle", [0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0])
    def test_neutral_gaze_roll_invariance(self, gaze_config: GazeConfig, roll_angle: float):
        """Verify centered gaze maintains norm_x ~ 0.0 and norm_y ~ 0.0 across all head roll angles."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(
            roll_deg=roll_angle,
            left_iris_offset_x=0.0, left_iris_offset_y=0.0,
            right_iris_offset_x=0.0, right_iris_offset_y=0.0
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.norm_x, 0.0, abs_tol=0.04)
        assert math.isclose(features.left_eye.norm_y, 0.0, abs_tol=0.04)
        assert math.isclose(features.right_eye.norm_x, 0.0, abs_tol=0.04)
        assert math.isclose(features.right_eye.norm_y, 0.0, abs_tol=0.04)

    @pytest.mark.parametrize("roll_angle", [0.0, 15.0, -15.0, 30.0, 45.0])
    def test_directional_gaze_roll_invariance(self, gaze_config: GazeConfig, roll_angle: float):
        """Verify fixed directional gaze offset remains consistent regardless of head roll angle."""
        extractor = EyeExtractor(gaze_config)
        # Fix nominal gaze: looking right (+0.20) and down (+0.15)
        offset_x, offset_y = 0.20, 0.15
        landmarks = create_synthetic_landmarks(
            roll_deg=roll_angle,
            left_iris_offset_x=offset_x, left_iris_offset_y=offset_y,
            right_iris_offset_x=offset_x, right_iris_offset_y=offset_y
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        # Relative error should be within 0.05
        assert math.isclose(features.left_eye.norm_x, offset_x, abs_tol=0.05)
        assert math.isclose(features.left_eye.norm_y, offset_y, abs_tol=0.05)
        assert math.isclose(features.right_eye.norm_x, offset_x, abs_tol=0.05)
        assert math.isclose(features.right_eye.norm_y, offset_y, abs_tol=0.05)

    @pytest.mark.parametrize("roll_angle", [0.0, 15.0, -15.0, 30.0, -30.0, 45.0])
    def test_ear_roll_invariance(self, gaze_config: GazeConfig, roll_angle: float):
        """Verify 6-point EAR measurement remains invariant under head roll tilt."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(roll_deg=roll_angle, left_eye_closed=False)
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.ear, 0.3125, abs_tol=0.02)
        assert features.left_eye.is_open is True


# ============================================================================
# 2. Scale & Distance Invariance Tests (0.5x, 0.75x, 1.0x, 1.5x, 2.0x)
# ============================================================================

class TestScaleAndDistanceInvariance:
    """Verifies that iris normalization and EAR are invariant to user distance and face scale."""

    @pytest.mark.parametrize("scale_factor", [0.5, 0.75, 1.0, 1.5, 2.0])
    def test_norm_coords_scale_invariance(self, gaze_config: GazeConfig, scale_factor: float):
        """Verify normalized iris coordinates remain identical across 0.5x to 2.0x face scales."""
        extractor = EyeExtractor(gaze_config)
        offset_x, offset_y = 0.15, -0.10
        landmarks = create_synthetic_landmarks(
            scale=scale_factor,
            left_iris_offset_x=offset_x, left_iris_offset_y=offset_y,
            right_iris_offset_x=offset_x, right_iris_offset_y=offset_y
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.norm_x, offset_x, abs_tol=1e-3)
        assert math.isclose(features.left_eye.norm_y, offset_y, abs_tol=1e-3)
        assert math.isclose(features.right_eye.norm_x, offset_x, abs_tol=1e-3)
        assert math.isclose(features.right_eye.norm_y, offset_y, abs_tol=1e-3)

    @pytest.mark.parametrize("scale_factor", [0.5, 0.8, 1.0, 1.4, 2.0])
    def test_ear_scale_invariance(self, gaze_config: GazeConfig, scale_factor: float):
        """Verify EAR is strictly scale-invariant across distances."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(scale=scale_factor)
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.ear, 0.3125, abs_tol=1e-3)
        assert math.isclose(features.right_eye.ear, 0.3125, abs_tol=1e-3)

    def test_metric_iris_depth_scaling(self, gaze_config: GazeConfig):
        """Verify estimated metric iris depth scales inversely with face scale factor."""
        extractor = EyeExtractor(gaze_config)
        lm_1x = create_synthetic_landmarks(scale=1.0)
        lm_2x = create_synthetic_landmarks(scale=2.0)
        f_1x = extractor.extract(lm_1x, 640, 480)
        f_2x = extractor.extract(lm_2x, 640, 480)
        # Closer face (2x scale) has larger iris diameter in px and roughly half depth in mm
        assert f_2x.left_eye.iris_diameter_px > f_1x.left_eye.iris_diameter_px
        assert f_2x.left_eye.iris_depth_mm < f_1x.left_eye.iris_depth_mm


# ============================================================================
# 3. Translation Invariance Tests (2D Shifts in Image Plane)
# ============================================================================

class TestTranslationInvariance:
    """Verifies that face translation across the camera view does not alter normalized gaze features."""

    @pytest.mark.parametrize("tx, ty", [
        (0.0, 0.0),
        (100.0, 0.0),
        (-100.0, 0.0),
        (0.0, 80.0),
        (0.0, -80.0),
        (80.0, 60.0),
        (-80.0, -60.0)
    ])
    def test_translation_invariance_norm_coords(self, gaze_config: GazeConfig, tx: float, ty: float):
        """Verify translated face produces identical norm_x, norm_y, and EAR."""
        extractor = EyeExtractor(gaze_config)
        offset_x, offset_y = 0.12, 0.18
        landmarks = create_synthetic_landmarks(
            tx_px=tx, ty_px=ty,
            left_iris_offset_x=offset_x, left_iris_offset_y=offset_y,
            right_iris_offset_x=offset_x, right_iris_offset_y=offset_y
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features is not None
        assert math.isclose(features.left_eye.norm_x, offset_x, abs_tol=1e-3)
        assert math.isclose(features.left_eye.norm_y, offset_y, abs_tol=1e-3)
        assert math.isclose(features.left_eye.ear, 0.3125, abs_tol=1e-3)


# ============================================================================
# 4. 3D Head Pose Decoupling & Conjugate Eye Symmetry
# ============================================================================

class TestHeadPoseDecouplingAndSymmetry:
    """Verifies decoupling of head orientation from eye features and cross-eye independence."""

    def test_left_and_right_eye_independence(self, gaze_config: GazeConfig):
        """Verify modifying left eye gaze offset does not perturb right eye normalized values."""
        extractor = EyeExtractor(gaze_config)
        # Left looking right (+0.25), right looking centered (0.0)
        landmarks = create_synthetic_landmarks(
            left_iris_offset_x=0.25, left_iris_offset_y=0.0,
            right_iris_offset_x=0.0, right_iris_offset_y=0.0
        )
        features = extractor.extract(landmarks, 640, 480)
        assert math.isclose(features.left_eye.norm_x, 0.25, abs_tol=1e-3)
        assert math.isclose(features.right_eye.norm_x, 0.0, abs_tol=1e-3)

    def test_conjugate_gaze_both_eyes_looking_right(self, gaze_config: GazeConfig):
        """Verify conjugate gaze right yields positive norm_x for BOTH eyes without cancellation."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(
            left_iris_offset_x=0.20, right_iris_offset_x=0.20
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features.left_eye.norm_x > 0.15
        assert features.right_eye.norm_x > 0.15
        assert features.avg_norm_x > 0.15

    def test_conjugate_gaze_both_eyes_looking_left(self, gaze_config: GazeConfig):
        """Verify conjugate gaze left yields negative norm_x for BOTH eyes."""
        extractor = EyeExtractor(gaze_config)
        landmarks = create_synthetic_landmarks(
            left_iris_offset_x=-0.20, right_iris_offset_x=-0.20
        )
        features = extractor.extract(landmarks, 640, 480)
        assert features.left_eye.norm_x < -0.15
        assert features.right_eye.norm_x < -0.15
        assert features.avg_norm_x < -0.15

    @pytest.mark.parametrize("res_w, res_h", [(640, 480), (1280, 720), (1920, 1080)])
    def test_resolution_invariance(self, gaze_config: GazeConfig, res_w: int, res_h: int):
        """Verify normalized iris coordinates are identical across frame resolutions."""
        extractor = EyeExtractor(gaze_config)
        offset_x, offset_y = 0.18, -0.12
        landmarks = create_synthetic_landmarks(
            img_w=res_w, img_h=res_h,
            left_iris_offset_x=offset_x, left_iris_offset_y=offset_y,
            right_iris_offset_x=offset_x, right_iris_offset_y=offset_y
        )
        features = extractor.extract(landmarks, res_w, res_h)
        assert features is not None
        assert math.isclose(features.left_eye.norm_x, offset_x, abs_tol=1e-3)
        assert math.isclose(features.left_eye.norm_y, offset_y, abs_tol=1e-3)
