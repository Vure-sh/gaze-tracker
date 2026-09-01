"""Configuration settings, landmark indices, and hyperparameters for the gaze tracker."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import screeninfo


@dataclass
class CameraConfig:
    """Camera capture and intrinsic optical parameters."""
    device_index: int = 0
    device_path: str = "/dev/video9"
    width: int = 640
    height: int = 480
    fps: int = 30
    fov_h_deg: float = 65.0              # Horizontal field of view in degrees
    fov_v_deg: float = 48.75             # Vertical field of view in degrees (standard 4:3)
    use_fov_matrix: bool = True          # Use FOV trigonometry for intrinsic focal length

    def get_camera_matrix(self, img_w: Optional[int] = None, img_h: Optional[int] = None) -> np.ndarray:
        """Computes the 3x3 camera intrinsic matrix using FOV geometry."""
        w = img_w or self.width
        h = img_h or self.height
        if self.use_fov_matrix:
            # f = (W/2) / tan(FOV_h / 2)
            fov_rad_h = math.radians(self.fov_h_deg)
            fx = (w / 2.0) / math.tan(fov_rad_h / 2.0)
            fov_rad_v = math.radians(self.fov_v_deg)
            fy = (h / 2.0) / math.tan(fov_rad_v / 2.0)
        else:
            fx = float(w)
            fy = float(w)

        cx = w / 2.0
        cy = h / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass
class QualityConfig:
    """Tracking quality evaluation and confidence thresholds."""
    min_confidence: float = 0.50         # Minimum overall confidence for valid tracking
    min_contrast_stddev: float = 12.0    # Minimum grayscale stddev in periocular region
    max_landmark_jitter_px: float = 8.0  # Landmark displacement threshold for jitter detection
    ear_weight: float = 0.35             # Composite weight for EAR score
    circularity_weight: float = 0.25     # Composite weight for iris circularity score
    contrast_weight: float = 0.20        # Composite weight for lighting contrast
    stability_weight: float = 0.20       # Composite weight for landmark temporal stability


@dataclass
class GazeConfig:
    """Unified configuration dataclass maintaining full backward compatibility."""

    # Model Asset Paths
    model_path: str = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")
    calibration_file: str = str(Path(__file__).parent.parent / "models" / "calibration_model.pkl")

    # Camera & Capture settings
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    target_fps: int = 30
    camera_fov_h_deg: float = 65.0
    camera_fov_v_deg: float = 48.75
    use_fov_camera_matrix: bool = True

    # Screen dimensions (auto-detected or default fallback)
    screen_width: int = 1920
    screen_height: int = 1080

    # Blink & Eye Aspect Ratio (EAR) Thresholds
    ear_blink_threshold: float = 0.18    # Fixed fallback threshold
    ear_adaptive_ratio: float = 0.60     # Adaptive threshold ratio (threshold = ratio * open_baseline)
    ear_history_length: int = 150        # History window length (samples) for baseline estimation
    ear_open_percentile: float = 90.0    # Percentile representing open-eye baseline
    ear_min_threshold: float = 0.12      # Hard floor for adaptive threshold
    ear_max_threshold: float = 0.28      # Hard ceiling for adaptive threshold
    ear_min_valid_samples: int = 15      # Minimum clean frames per calibration point

    # Iris Metric & Geometry Constants
    iris_metric_diameter_mm: float = 11.7 # Mean human corneal diameter in millimeters
    iris_circularity_sigma: float = 2.0  # Gaussian variance scale for circularity score
    iris_min_circularity: float = 0.50   # Minimum acceptable circularity score

    # Calibration settings
    calibration_grid_type: str = "9_points"  # "9_points", "13_points", or "16_points"
    calibration_margin_x: float = 0.12       # 12% margin from screen edges
    calibration_margin_y: float = 0.12
    saccade_delay_frames: int = 12           # Frames to ignore at target appearance (reaction time)
    saccade_delay_seconds: float = 0.35      # Wall-clock saccade latency delay in seconds
    sample_frames_per_point: int = 35        # Target frames collected per point
    collect_duration_seconds: float = 0.85   # Wall-clock fixation collection duration in seconds

    # MediaPipe Face Landmark Indices
    # Left Eye (Anatomical right / observer left in mirrored frame)
    # Inner canthus (nasal): 133, Outer canthus (temporal): 33
    left_eye_inner: int = 133
    left_eye_outer: int = 33
    left_eye_top: int = 159
    left_eye_bottom: int = 145
    left_eye_corners: List[int] = field(default_factory=lambda: [33, 133])
    left_eye_eyelids: List[int] = field(default_factory=lambda: [159, 145])
    left_eye_contour: List[int] = field(default_factory=lambda: [
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
    ])
    left_iris_center: int = 468
    left_iris_points: List[int] = field(default_factory=lambda: [468, 469, 470, 471, 472])
    # 6-point EAR landmark indices: [outer, inner, top1, bottom1, top2, bottom2]
    left_eye_ear_indices: List[int] = field(default_factory=lambda: [33, 133, 160, 144, 158, 153])

    # Right Eye (Anatomical left / observer right in mirrored frame)
    # Inner canthus (nasal): 362, Outer canthus (temporal): 263
    right_eye_inner: int = 362
    right_eye_outer: int = 263
    right_eye_top: int = 386
    right_eye_bottom: int = 374
    right_eye_corners: List[int] = field(default_factory=lambda: [362, 263])
    right_eye_eyelids: List[int] = field(default_factory=lambda: [386, 374])
    right_eye_contour: List[int] = field(default_factory=lambda: [
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ])
    right_iris_center: int = 473
    right_iris_points: List[int] = field(default_factory=lambda: [473, 474, 475, 476, 477])
    # 6-point EAR landmark indices: [inner, outer, top1, bottom1, top2, bottom2]
    right_eye_ear_indices: List[int] = field(default_factory=lambda: [362, 263, 385, 380, 387, 373])

    # Head Pose 3D Model Keypoints (solvePnP)
    head_pose_mesh_indices: List[int] = field(default_factory=lambda: [
        1,    # Nose tip
        152,  # Chin
        33,   # Left eye outer corner
        263,  # Right eye outer corner
        61,   # Left mouth corner
        291   # Right mouth corner
    ])
    head_pose_use_corrected_model: bool = True
    head_pose_pitch_limit_deg: float = 45.0
    head_pose_yaw_limit_deg: float = 45.0
    head_pose_roll_limit_deg: float = 45.0

    # Quality and Confidence Config
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Feature Representation settings
    feature_dimension: int = 8           # 8 (clean 8D) or 10 or 14 (legacy)

    # Smoothing Filter Settings (One-Euro Filter)
    filter_type: str = "one_euro"        # "one_euro", "kalman", "ema"
    one_euro_min_cutoff: float = 0.20    # Adjusted baseline min cutoff
    one_euro_beta: float = 0.02          # Saccade speed coefficient
    one_euro_d_cutoff: float = 1.0       # Derivative cutoff
    one_euro_velocity_threshold: float = 20.0  # Pixel speed deadband for velocity gating

    # Regression Parameters
    poly_degree: int = 2
    ridge_alpha: float = 1.0

    def get_camera_matrix(self, img_w: Optional[int] = None, img_h: Optional[int] = None) -> np.ndarray:
        """Computes camera intrinsic matrix for solvePnP and depth projection."""
        w = img_w or self.camera_width
        h = img_h or self.camera_height
        if self.use_fov_camera_matrix:
            fov_rad_h = math.radians(self.camera_fov_h_deg)
            fx = (w / 2.0) / math.tan(fov_rad_h / 2.0)
            fov_rad_v = math.radians(self.camera_fov_v_deg)
            fy = (h / 2.0) / math.tan(fov_rad_v / 2.0)
        else:
            fx = float(w)
            fy = float(w)
        cx = w / 2.0
        cy = h / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def auto_detect_screen(self):
        """Auto-detect primary monitor resolution."""
        try:
            monitors = screeninfo.get_monitors()
            primary = next((m for m in monitors if m.is_primary), monitors[0] if monitors else None)
            if primary:
                self.screen_width = primary.width
                self.screen_height = primary.height
        except Exception:
            pass

    def validate(self) -> list:
        """Validate configuration values and return a list of warning strings.

        Returns:
            List of human-readable warning strings for any out-of-range settings.
            An empty list means the configuration is fully valid.
        """
        warnings: list = []
        if not 0.0 < self.quality.min_confidence < 1.0:
            warnings.append(f"quality.min_confidence={self.quality.min_confidence} should be in (0, 1)")
        if self.poly_degree < 1 or self.poly_degree > 4:
            warnings.append(f"poly_degree={self.poly_degree} unusual; expected 1-4")
        if self.ridge_alpha <= 0:
            warnings.append(f"ridge_alpha={self.ridge_alpha} must be positive")
        if self.one_euro_min_cutoff <= 0:
            warnings.append(f"one_euro_min_cutoff={self.one_euro_min_cutoff} must be positive")
        if self.screen_width <= 0 or self.screen_height <= 0:
            warnings.append(f"screen_resolution={self.screen_width}x{self.screen_height} invalid")
        if self.camera_width <= 0 or self.camera_height <= 0:
            warnings.append(f"camera_resolution={self.camera_width}x{self.camera_height} invalid")
        return warnings
