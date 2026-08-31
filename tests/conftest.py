"""Shared test fixtures, synthetic landmark generators, and mock datasets for Gaze Tracker test suite."""

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pytest

from src.config import GazeConfig


@dataclass
class SyntheticLandmark:
    """Mock MediaPipe NormalizedLandmark object with (x, y, z) coordinates in [0.0, 1.0]."""
    x: float
    y: float
    z: float = 0.0


def create_synthetic_landmarks(
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    roll_deg: float = 0.0,
    scale: float = 1.0,
    tx_px: float = 0.0,
    ty_px: float = 0.0,
    left_iris_offset_x: float = 0.0,
    left_iris_offset_y: float = 0.0,
    right_iris_offset_x: float = 0.0,
    right_iris_offset_y: float = 0.0,
    left_eye_closed: bool = False,
    right_eye_closed: bool = False,
    img_w: int = 640,
    img_h: int = 480,
) -> List[SyntheticLandmark]:
    """
    Generates a high-fidelity 478-point synthetic MediaPipe facial landmark list
    with controllable 3D head pose, face scaling, translation, zero-centered iris gaze offsets,
    and eye open/closed status.
    """
    center_x = img_w / 2.0 + tx_px
    center_y = img_h / 2.0 + ty_px

    eye_half_w = 32.0 * scale
    left_eye_h = 10.0 * scale if not left_eye_closed else 0.0
    right_eye_h = 10.0 * scale if not right_eye_closed else 0.0
    eye_span = 80.0 * scale

    # Left eye center in pixel space (Observer left)
    left_center_x = center_x - eye_span
    left_center_y = center_y - 40.0 * scale

    # Right eye center in pixel space (Observer right)
    right_center_x = center_x + eye_span
    right_center_y = center_y - 40.0 * scale

    pts_px: Dict[int, Tuple[float, float, float]] = {
        # Nose tip (1)
        1: (center_x, center_y, 0.0),
        # Chin (152)
        152: (center_x, center_y + 110.0 * scale, 0.0),
        # Left mouth corner (61)
        61: (center_x - 45.0 * scale, center_y + 60.0 * scale, 0.0),
        # Right mouth corner (291)
        291: (center_x + 45.0 * scale, center_y + 60.0 * scale, 0.0),
        # Left eye corners & 6-point EAR landmarks
        33: (left_center_x - eye_half_w, left_center_y, 0.0),   # Outer corner (temporal)
        133: (left_center_x + eye_half_w, left_center_y, 0.0),  # Inner corner (nasal)
        159: (left_center_x, left_center_y - left_eye_h, 0.0),  # Top eyelid peak
        145: (left_center_x, left_center_y + left_eye_h, 0.0),  # Bottom eyelid peak
        160: (left_center_x - 10.0 * scale, left_center_y - left_eye_h, 0.0), # Top 1
        144: (left_center_x - 10.0 * scale, left_center_y + left_eye_h, 0.0), # Bottom 1
        158: (left_center_x + 10.0 * scale, left_center_y - left_eye_h, 0.0), # Top 2
        153: (left_center_x + 10.0 * scale, left_center_y + left_eye_h, 0.0), # Bottom 2

        # Right eye corners & 6-point EAR landmarks
        362: (right_center_x - eye_half_w, right_center_y, 0.0), # Inner corner (nasal)
        263: (right_center_x + eye_half_w, right_center_y, 0.0), # Outer corner (temporal)
        386: (right_center_x, right_center_y - right_eye_h, 0.0), # Top eyelid peak
        374: (right_center_x, right_center_y + right_eye_h, 0.0), # Bottom eyelid peak
        385: (right_center_x - 10.0 * scale, right_center_y - right_eye_h, 0.0), # Top 1
        380: (right_center_x - 10.0 * scale, right_center_y + right_eye_h, 0.0), # Bottom 1
        387: (right_center_x + 10.0 * scale, right_center_y - right_eye_h, 0.0), # Top 2
        373: (right_center_x + 10.0 * scale, right_center_y + right_eye_h, 0.0), # Bottom 2
    }

    # Derive Left iris center from zero-centered offset:
    # norm_x = offset_x, norm_y = offset_y
    left_iris_px_x = left_center_x + left_iris_offset_x * (2.0 * eye_half_w)
    left_iris_px_y = left_center_y + left_iris_offset_y * (2.0 * eye_half_w)
    pts_px[468] = (left_iris_px_x, left_iris_px_y, 0.0)

    r_iris = 6.0 * scale
    pts_px[469] = (left_iris_px_x + r_iris, left_iris_px_y, 0.0)
    pts_px[470] = (left_iris_px_x - r_iris, left_iris_px_y, 0.0)
    pts_px[471] = (left_iris_px_x, left_iris_px_y - r_iris, 0.0)
    pts_px[472] = (left_iris_px_x, left_iris_px_y + r_iris, 0.0)

    # Derive Right iris center from zero-centered offset:
    right_iris_px_x = right_center_x + right_iris_offset_x * (2.0 * eye_half_w)
    right_iris_px_y = right_center_y + right_iris_offset_y * (2.0 * eye_half_w)
    pts_px[473] = (right_iris_px_x, right_iris_px_y, 0.0)

    pts_px[474] = (right_iris_px_x + r_iris, right_iris_px_y, 0.0)
    pts_px[475] = (right_iris_px_x - r_iris, right_iris_px_y, 0.0)
    pts_px[476] = (right_iris_px_x, right_iris_px_y - r_iris, 0.0)
    pts_px[477] = (right_iris_px_x, right_iris_px_y + r_iris, 0.0)

    # Left eye contour (16 points)
    left_contour_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    for idx_c, l_idx in enumerate(left_contour_indices):
        if l_idx not in pts_px:
            ang = 2.0 * math.pi * idx_c / len(left_contour_indices)
            pts_px[l_idx] = (
                left_center_x + eye_half_w * math.cos(ang),
                left_center_y + left_eye_h * math.sin(ang),
                0.0
            )

    # Right eye contour (16 points)
    right_contour_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    for idx_c, r_idx in enumerate(right_contour_indices):
        if r_idx not in pts_px:
            ang = 2.0 * math.pi * idx_c / len(right_contour_indices)
            pts_px[r_idx] = (
                right_center_x + eye_half_w * math.cos(ang),
                right_center_y + right_eye_h * math.sin(ang),
                0.0
            )

    # Apply 2D roll rotation around face center if roll != 0
    if abs(roll_deg) > 1e-4:
        rad = math.radians(roll_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        for k, (px, py, pz) in pts_px.items():
            dx = px - center_x
            dy = py - center_y
            rx = center_x + (dx * cos_r - dy * sin_r)
            ry = center_y + (dx * sin_r + dy * cos_r)
            pts_px[k] = (rx, ry, pz)

    # Build full 478 landmark list
    landmarks = [SyntheticLandmark(x=center_x / img_w, y=center_y / img_h, z=0.0) for _ in range(478)]
    for idx, (px, py, pz) in pts_px.items():
        if 0 <= idx < 478:
            landmarks[idx] = SyntheticLandmark(
                x=float(px / img_w),
                y=float(py / img_h),
                z=float(pz / img_w)
            )

    return landmarks


@pytest.fixture
def gaze_config() -> GazeConfig:
    """Provides a clean default GazeConfig instance configured for 1920x1080 screen and 640x480 camera."""
    cfg = GazeConfig()
    cfg.screen_width = 1920
    cfg.screen_height = 1080
    cfg.camera_width = 640
    cfg.camera_height = 480
    return cfg


@pytest.fixture
def synthetic_landmarks() -> List[SyntheticLandmark]:
    """Provides standard canonical neutral face landmarks (478 points)."""
    return create_synthetic_landmarks()


@pytest.fixture
def mock_bgr_frame() -> np.ndarray:
    """Provides a synthetic 480x640x3 BGR frame representing a standard webcam image."""
    return np.full((480, 640, 3), 40, dtype=np.uint8)


@pytest.fixture
def mock_corrupted_frames() -> Dict[str, Any]:
    """Provides a suite of malformed, zero-size, corrupted, or non-standard frame inputs."""
    return {
        "none": None,
        "empty_0x0": np.zeros((0, 0), dtype=np.uint8),
        "empty_0x0x3": np.zeros((0, 0, 3), dtype=np.uint8),
        "all_zeros_3d": np.zeros((480, 640, 3), dtype=np.uint8),
        "all_ones_3d": np.full((480, 640, 3), 255, dtype=np.uint8),
        "random_noise": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
    }


@pytest.fixture
def synthetic_calibration_dataset(gaze_config: GazeConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a realistic synthetic calibration dataset with 9 ground-truth screen points,
    realistic 14D feature vectors, and controlled Gaussian noise.
    Ideal for testing regression fitting, LOPO cross-validation, and accuracy thresholds.
    """
    np.random.seed(42)
    w, h = gaze_config.screen_width, gaze_config.screen_height
    mx, my = gaze_config.calibration_margin_x, gaze_config.calibration_margin_y

    xs = [mx * w, 0.5 * w, (1.0 - mx) * w]
    ys = [my * h, 0.5 * h, (1.0 - my) * h]
    screen_targets = [(int(x), int(y)) for y in ys for x in xs]

    samples_per_pt = 25
    X_list = []
    y_list = []

    for tx, ty in screen_targets:
        # Zero-centered gaze mapping:
        # Screen center (w/2, h/2) maps to norm_x = 0.0, norm_y = 0.0
        # Screen bounds map to norm_x in [-0.25, +0.25], norm_y in [-0.20, +0.20]
        true_norm_x = (tx - 0.5 * w) / (w * 2.0)
        true_norm_y = (ty - 0.5 * h) / (h * 2.5)

        for _ in range(samples_per_pt):
            noise_x = np.random.normal(0, 0.003)
            noise_y = np.random.normal(0, 0.003)

            norm_x_L = true_norm_x + noise_x
            norm_y_L = true_norm_y + noise_y
            norm_x_R = true_norm_x + np.random.normal(0, 0.003)
            norm_y_R = true_norm_y + np.random.normal(0, 0.003)
            avg_norm_x = (norm_x_L + norm_x_R) / 2.0
            avg_norm_y = (norm_y_L + norm_y_R) / 2.0

            ear_L = 0.31 + np.random.normal(0, 0.005)
            ear_R = 0.31 + np.random.normal(0, 0.005)

            pitch_norm = np.random.normal(0, 0.01)
            yaw_norm = np.random.normal(0, 0.01)
            roll_norm = np.random.normal(0, 0.01)
            tx_norm = np.random.normal(0, 0.005)
            ty_norm = np.random.normal(0, 0.005)
            tz_norm = 0.6 + np.random.normal(0, 0.005)

            feat = np.array([
                norm_x_L, norm_y_L,
                norm_x_R, norm_y_R,
                avg_norm_x, avg_norm_y,
                ear_L, ear_R,
                pitch_norm, yaw_norm, roll_norm,
                tx_norm, ty_norm, tz_norm
            ], dtype=np.float64)

            X_list.append(feat)
            y_list.append([tx, ty])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    metadata = {
        "screen_targets": screen_targets,
        "samples_per_pt": samples_per_pt,
        "num_targets": len(screen_targets),
        "total_samples": len(X_list)
    }

    return X, y, metadata
