"""Core data contracts and typed representations for the gaze tracking pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np


@dataclass
class NormalizedPoint:
    """Represents a 3D landmark point normalized to [0.0, 1.0] image dimensions."""
    x: float
    y: float
    z: float = 0.0

    def to_pixel(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """Convert normalized (x, y) coordinates to integer pixel coordinates."""
        return int(round(self.x * img_w)), int(round(self.y * img_h))

    def to_numpy(self, img_w: Optional[int] = None, img_h: Optional[int] = None) -> np.ndarray:
        """Convert to numpy array, optionally scaled by image dimensions."""
        if img_w is not None and img_h is not None:
            return np.array([self.x * img_w, self.y * img_h, self.z * img_w], dtype=np.float64)
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_landmark(cls, lm: Any) -> NormalizedPoint:
        """Construct from MediaPipe NormalizedLandmark or any object with x, y, (z)."""
        z_val = getattr(lm, "z", 0.0)
        return cls(x=float(lm.x), y=float(lm.y), z=float(z_val))


@dataclass
class EyeData:
    """Geometric and normalized features for a single eye."""
    norm_x: float                        # Orthonormal horizontal position (scale & roll invariant)
    norm_y: float                        # Orthonormal vertical position (scale & roll invariant)
    ear: float                           # 6-point Eye Aspect Ratio
    is_open: bool                        # True if eye aperture exceeds adaptive blink threshold
    iris_center_px: Tuple[int, int]      # Pixel coordinates of iris center
    inner_corner_px: Tuple[int, int]     # Pixel coordinates of inner canthus
    outer_corner_px: Tuple[int, int]     # Pixel coordinates of outer canthus
    top_eyelid_px: Tuple[int, int]       # Pixel coordinates of top eyelid peak
    bottom_eyelid_px: Tuple[int, int]    # Pixel coordinates of bottom eyelid peak
    contour_px: List[Tuple[int, int]]    # 16-point eyelid contour pixel polygon
    iris_points_px: List[Tuple[int, int]] = field(default_factory=list)  # 5-point iris pixel coordinates
    iris_diameter_px: float = 0.0        # Estimated iris diameter in pixels
    circularity: float = 1.0             # Iris circularity symmetry score in [0.0, 1.0]
    iris_depth_mm: float = 0.0           # Metric depth from camera based on 11.7mm iris baseline


@dataclass
class HeadPoseData:
    """3D Head pose representation derived from facial solvePnP."""
    pitch: float                         # Up (-) / Down (+) angle in degrees
    yaw: float                           # Left (-) / Right (+) angle in degrees
    roll: float                          # Tilt Left (-) / Tilt Right (+) angle in degrees
    rvec: np.ndarray                     # 3x1 Rodrigues rotation vector
    tvec: np.ndarray                     # 3x1 Translation vector in camera space (mm)
    nose_2d_px: Tuple[int, int]          # 2D projection of nose tip
    axes_2d_px: List[Tuple[int, int]]    # 2D projections of RGB orientation axes (X, Y, Z endpoints)
    feature_vector: np.ndarray           # Normalized pose vector: [pitch/45, yaw/45, roll/45, tx/500, ty/500, tz/1000]
    rot_mat: Optional[np.ndarray] = None # 3x3 Rotation matrix in camera coordinates


@dataclass
class TrackingQuality:
    """Multi-dimensional tracking quality and signal integrity assessment."""
    confidence: float                    # Composite tracking confidence in [0.0, 1.0]
    ear_score: float                     # Aperture / blink quality score in [0.0, 1.0]
    circularity_score: float             # Iris geometry / visibility score in [0.0, 1.0]
    contrast_score: float                # Periocular lighting & contrast score in [0.0, 1.0]
    stability_score: float               # Landmark temporal stability score in [0.0, 1.0]
    is_valid: bool                       # True if tracking meets all minimum quality gates
    failure_reasons: List[str] = field(default_factory=list)  # Diagnostic failure descriptions

    def __repr__(self) -> str:
        return (
            f"TrackingQuality(conf={self.confidence:.2f}, ear={self.ear_score:.2f}, "
            f"circ={self.circularity_score:.2f}, valid={self.is_valid}, "
            f"failures={self.failure_reasons})"
        )


@dataclass
class GazeFeatures:
    """Aggregated eye, head pose, and quality features for gaze estimation."""
    left_eye: EyeData
    right_eye: EyeData
    avg_norm_x: float
    avg_norm_y: float
    head_pose: Optional[HeadPoseData]
    confidence: float = 1.0
    is_valid: bool = True
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))
    quality: Optional[TrackingQuality] = None
    timestamp: float = 0.0

    @property
    def vector_8d(self) -> np.ndarray:
        """Clean 8D feature vector: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, pitch/45, yaw/45, roll/45, tz/1000]."""
        p = self.head_pose.pitch / 45.0 if self.head_pose else 0.0
        y = self.head_pose.yaw / 45.0 if self.head_pose else 0.0
        r = self.head_pose.roll / 45.0 if self.head_pose else 0.0
        tz = float(self.head_pose.tvec[2, 0]) / 1000.0 if (self.head_pose is not None and self.head_pose.tvec is not None) else 0.6
        return np.array([
            self.left_eye.norm_x, self.left_eye.norm_y,
            self.right_eye.norm_x, self.right_eye.norm_y,
            p, y, r, tz
        ], dtype=np.float64)

    @property
    def vector_10d(self) -> np.ndarray:
        """10D feature vector including averaged eye coordinates."""
        p = self.head_pose.pitch / 45.0 if self.head_pose else 0.0
        y = self.head_pose.yaw / 45.0 if self.head_pose else 0.0
        r = self.head_pose.roll / 45.0 if self.head_pose else 0.0
        tz = float(self.head_pose.tvec[2, 0]) / 1000.0 if (self.head_pose is not None and self.head_pose.tvec is not None) else 0.6
        return np.array([
            self.left_eye.norm_x, self.left_eye.norm_y,
            self.right_eye.norm_x, self.right_eye.norm_y,
            self.avg_norm_x, self.avg_norm_y,
            p, y, r, tz
        ], dtype=np.float64)

    @property
    def vector_14d(self) -> np.ndarray:
        """Legacy 14D feature vector for backward compatibility."""
        hp_vec = self.head_pose.feature_vector if self.head_pose is not None else np.zeros(6, dtype=np.float64)
        eye_vec = np.array([
            self.left_eye.norm_x, self.left_eye.norm_y,
            self.right_eye.norm_x, self.right_eye.norm_y,
            self.avg_norm_x, self.avg_norm_y,
            self.left_eye.ear, self.right_eye.ear
        ], dtype=np.float64)
        return np.concatenate([eye_vec, hp_vec])


@dataclass
class GazePrediction:
    """Predicted screen gaze coordinate and confidence."""
    screen_x: float
    screen_y: float
    norm_x: float                        # Screen coordinate normalized to [0.0, 1.0]
    norm_y: float                        # Screen coordinate normalized to [0.0, 1.0]
    confidence: float = 1.0
    is_valid: bool = True
    timestamp: float = 0.0

    def __repr__(self) -> str:
        return (
            f"GazePrediction(screen=({self.screen_x:.1f}, {self.screen_y:.1f}), "
            f"norm=({self.norm_x:.3f}, {self.norm_y:.3f}), "
            f"conf={self.confidence:.2f}, valid={self.is_valid})"
        )


@dataclass
class FaceDetectionResult:
    """Complete output of the Face Detector."""
    landmarks: List[NormalizedPoint]
    blendshapes: Optional[Dict[str, float]] = None
    facial_transformation_matrix: Optional[np.ndarray] = None
    timestamp_ms: int = 0
