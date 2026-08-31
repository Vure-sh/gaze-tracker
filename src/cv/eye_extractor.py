"""Orthonormal scale- and roll-invariant eye & iris feature extraction,
6-point adaptive EAR blink detection, and 5-point iris circularity geometry.
"""

from __future__ import annotations
import collections
from typing import List, Tuple, Optional, Any
import numpy as np

from src.config import GazeConfig
from src.types import EyeData, GazeFeatures, NormalizedPoint, HeadPoseData


class EyeExtractor:
    """Extracts scale- and rotation-invariant normalized eye & iris features from facial landmarks."""

    def __init__(self, config: Optional[GazeConfig] = None):
        self.config = config or GazeConfig()
        self.left_ear_history = collections.deque(maxlen=self.config.ear_history_length)
        self.right_ear_history = collections.deque(maxlen=self.config.ear_history_length)

    def _compute_6point_ear(
        self,
        p_outer: np.ndarray,
        p_inner: np.ndarray,
        p_top1: np.ndarray,
        p_bottom1: np.ndarray,
        p_top2: np.ndarray,
        p_bottom2: np.ndarray
    ) -> float:
        """
        Computes 6-point Eye Aspect Ratio (EAR) according to Soukupova & Cech (2016):
        EAR = (||p_top1 - p_bottom1|| + ||p_top2 - p_bottom2||) / (2 * ||p_outer - p_inner||)
        """
        width = np.linalg.norm(p_outer - p_inner)
        if width < 1e-6:
            return 0.0
        h1 = np.linalg.norm(p_top1 - p_bottom1)
        h2 = np.linalg.norm(p_top2 - p_bottom2)
        return float((h1 + h2) / (2.0 * width))

    def _compute_iris_geometry(
        self,
        p_iris_center: np.ndarray,
        iris_pts_px: List[np.ndarray],
        focal_length_px: float
    ) -> Tuple[float, float, float]:
        """
        Calculates mean iris diameter (px), circularity symmetry score [0.0, 1.0],
        and metric camera depth estimate (mm).
        """
        if len(iris_pts_px) >= 4:
            radii = [np.linalg.norm(pt - p_iris_center) for pt in iris_pts_px]
            mean_radius = float(np.mean(radii))
            diameter_px = max(1.0, 2.0 * mean_radius)
            var_r = float(np.var(radii))
            sigma = max(0.1, self.config.iris_circularity_sigma)
            circularity = float(np.exp(-var_r / (sigma ** 2)))
        else:
            diameter_px = 12.0
            circularity = 1.0

        # Metric depth: Z = (f * D_metric) / D_px
        iris_depth_mm = float(
            (focal_length_px * self.config.iris_metric_diameter_mm) / diameter_px
        )

        return diameter_px, circularity, iris_depth_mm

    def _update_adaptive_ear(self, ear: float, history: collections.deque) -> Tuple[float, bool]:
        """
        Updates running EAR history and computes dynamic blink threshold.
        Returns: (adaptive_threshold, is_open)
        """
        if 0.05 < ear < 0.60:
            history.append(ear)

        if len(history) >= 15:
            baseline_open = float(np.percentile(list(history), self.config.ear_open_percentile))
            threshold = np.clip(
                baseline_open * self.config.ear_adaptive_ratio,
                self.config.ear_min_threshold,
                self.config.ear_max_threshold
            )
        else:
            threshold = self.config.ear_blink_threshold

        is_open = ear >= threshold
        return float(threshold), bool(is_open)

    def _extract_eye_data(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int,
        is_left: bool,
        focal_length_px: float
    ) -> EyeData:
        """
        Extracts orthonormal scale- and roll-invariant iris metrics, 6-point EAR,
        and 5-point iris geometry for a single eye.
        """
        def to_pt(idx: int) -> np.ndarray:
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h], dtype=np.float64)

        if is_left:
            # Left Eye in mirrored image (Observer Left / Anatomical Right)
            # Outer canthus (temporal) is Landmark 33 (smaller X)
            # Inner canthus (nasal) is Landmark 133 (larger X)
            p_outer = to_pt(self.config.left_eye_outer)
            p_inner = to_pt(self.config.left_eye_inner)
            p_top = to_pt(self.config.left_eye_top)
            p_bottom = to_pt(self.config.left_eye_bottom)
            p_iris = to_pt(self.config.left_iris_center)

            ear_indices = self.config.left_eye_ear_indices
            contour_indices = self.config.left_eye_contour
            iris_indices = self.config.left_iris_points
            history = self.left_ear_history

            # Direction vector pointing LEFT-TO-RIGHT in screen/image space (+X)
            # From outer (33) to inner (133)
            canthal_vec = p_inner - p_outer

        else:
            # Right Eye in mirrored image (Observer Right / Anatomical Left)
            # Inner canthus (nasal) is Landmark 362 (smaller X)
            # Outer canthus (temporal) is Landmark 263 (larger X)
            p_inner = to_pt(self.config.right_eye_inner)
            p_outer = to_pt(self.config.right_eye_outer)
            p_top = to_pt(self.config.right_eye_top)
            p_bottom = to_pt(self.config.right_eye_bottom)
            p_iris = to_pt(self.config.right_iris_center)

            ear_indices = self.config.right_eye_ear_indices
            contour_indices = self.config.right_eye_contour
            iris_indices = self.config.right_iris_points
            history = self.right_ear_history

            # Direction vector pointing LEFT-TO-RIGHT in screen/image space (+X)
            # From inner (362) to outer (263)
            canthal_vec = p_outer - p_inner

        eye_width = float(np.linalg.norm(canthal_vec))
        if eye_width < 1e-6:
            eye_width = 1e-6

        # 1. Orthonormal Basis Construction
        # u: unit horizontal axis along eye fissure pointing RIGHT
        u = canthal_vec / eye_width
        # u_perp: unit vertical axis orthogonal to u, pointing DOWNWARDS (+Y)
        u_perp = np.array([-u[1], u[0]], dtype=np.float64)

        # Eye fissure midpoint
        p_mid = (p_inner + p_outer) / 2.0

        # 2. Orthonormal Zero-Centered Scale- and Roll-Invariant Iris Normalization
        # Gaze right -> positive norm_x for BOTH eyes
        # Gaze down -> positive norm_y for BOTH eyes
        norm_x = float(np.dot(p_iris - p_mid, u) / eye_width)
        norm_y = float(np.dot(p_iris - p_mid, u_perp) / eye_width)

        # 3. 6-Point EAR Computation
        p_ear_pts = [to_pt(idx) for idx in ear_indices]
        if len(p_ear_pts) == 6:
            ear = self._compute_6point_ear(
                p_ear_pts[0], p_ear_pts[1],
                p_ear_pts[2], p_ear_pts[3],
                p_ear_pts[4], p_ear_pts[5]
            )
        else:
            ear = float(np.linalg.norm(p_top - p_bottom) / eye_width)

        # Adaptive Blink Thresholding
        _, is_open = self._update_adaptive_ear(ear, history)

        # 4. 5-Point Iris Geometry and Circularity
        perimeter_pts = [to_pt(idx) for idx in iris_indices if idx != iris_indices[0]]
        diameter_px, circularity, iris_depth_mm = self._compute_iris_geometry(
            p_iris, perimeter_pts, focal_length_px
        )

        contour_px = [
            (int(round(landmarks[idx].x * img_w)), int(round(landmarks[idx].y * img_h)))
            for idx in contour_indices
        ]
        iris_pts_px = [
            (int(round(landmarks[idx].x * img_w)), int(round(landmarks[idx].y * img_h)))
            for idx in iris_indices
        ]

        return EyeData(
            norm_x=norm_x,
            norm_y=norm_y,
            ear=ear,
            is_open=is_open,
            iris_center_px=(int(round(p_iris[0])), int(round(p_iris[1]))),
            inner_corner_px=(int(round(p_inner[0])), int(round(p_inner[1]))),
            outer_corner_px=(int(round(p_outer[0])), int(round(p_outer[1]))),
            top_eyelid_px=(int(round(p_top[0])), int(round(p_top[1]))),
            bottom_eyelid_px=(int(round(p_bottom[0])), int(round(p_bottom[1]))),
            contour_px=contour_px,
            iris_points_px=iris_pts_px,
            iris_diameter_px=diameter_px,
            circularity=circularity,
            iris_depth_mm=iris_depth_mm
        )

    def extract(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int,
        head_pose: Optional[HeadPoseData] = None
    ) -> Optional[GazeFeatures]:
        """
        Processes both eyes and returns the complete normalized gaze feature bundle.

        Args:
            landmarks: List of at least 478 NormalizedPoint/Landmark objects.
            img_w: Image width in pixels.
            img_h: Image height in pixels.
            head_pose: Optional estimated HeadPoseData.

        Returns:
            GazeFeatures instance or None.
        """
        if landmarks is None or len(landmarks) < 478:
            return None

        camera_mat = self.config.get_camera_matrix(img_w, img_h)
        focal_length_px = float(camera_mat[0, 0])

        left = self._extract_eye_data(landmarks, img_w, img_h, is_left=True, focal_length_px=focal_length_px)
        right = self._extract_eye_data(landmarks, img_w, img_h, is_left=False, focal_length_px=focal_length_px)

        avg_norm_x = (left.norm_x + right.norm_x) / 2.0
        avg_norm_y = (left.norm_y + right.norm_y) / 2.0
        is_valid = bool(left.is_open and right.is_open)

        # Baseline composite confidence
        mean_circularity = (left.circularity + right.circularity) / 2.0
        confidence = float(np.clip(mean_circularity if is_valid else 0.0, 0.0, 1.0))

        features = GazeFeatures(
            left_eye=left,
            right_eye=right,
            avg_norm_x=avg_norm_x,
            avg_norm_y=avg_norm_y,
            head_pose=head_pose,
            confidence=confidence,
            is_valid=is_valid
        )

        # Populate feature_vector property default to 8D clean vector
        features.feature_vector = features.vector_8d

        return features
