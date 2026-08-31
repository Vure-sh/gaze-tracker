"""Multi-dimensional tracking quality and confidence evaluation combining EAR,
iris circularity geometry, periocular lighting contrast, and landmark stability.
"""

from __future__ import annotations
import collections
from typing import Optional, List, Tuple, Any
import numpy as np
import cv2

from src.config import GazeConfig, QualityConfig
from src.types import TrackingQuality, GazeFeatures, NormalizedPoint, HeadPoseData


class QualityTracker:
    """Evaluates real-time tracking confidence across multiple geometric and optical dimensions."""

    def __init__(self, config: Optional[GazeConfig] = None):
        self.config = config or GazeConfig()
        self.quality_config = getattr(self.config, "quality", QualityConfig())
        self._prev_landmarks: Optional[np.ndarray] = None
        self._jitter_history = collections.deque(maxlen=10)

    def evaluate(
        self,
        gaze_features: Optional[GazeFeatures],
        landmarks: Optional[List[Any]] = None,
        frame: Optional[np.ndarray] = None,
        img_w: int = 640,
        img_h: int = 480
    ) -> TrackingQuality:
        """
        Computes composite tracking confidence and identifies potential degradation causes.

        Args:
            gaze_features: Extracted GazeFeatures for the current frame.
            landmarks: Raw normalized landmark list.
            frame: Optional BGR input frame for photometric contrast analysis.
            img_w: Frame width in pixels.
            img_h: Frame height in pixels.

        Returns:
            TrackingQuality dataclass with composite confidence and diagnostic details.
        """
        failure_reasons: List[str] = []

        if gaze_features is None or landmarks is None or len(landmarks) < 468:
            return TrackingQuality(
                confidence=0.0,
                ear_score=0.0,
                circularity_score=0.0,
                contrast_score=0.0,
                stability_score=0.0,
                is_valid=False,
                failure_reasons=["No face or landmarks detected"]
            )

        left = gaze_features.left_eye
        right = gaze_features.right_eye
        head_pose = gaze_features.head_pose

        # 1. EAR / Aperture Quality Score
        if not left.is_open or not right.is_open:
            ear_score = 0.0
            failure_reasons.append("Eye blink or eye closure detected")
        else:
            mean_ear = (left.ear + right.ear) / 2.0
            # Normalize EAR: 0.18 is threshold, 0.30+ is fully open
            ear_score = float(np.clip((mean_ear - 0.10) / 0.20, 0.0, 1.0))

        # 2. Iris Circularity & Symmetry Score
        circularity_score = float(np.clip((left.circularity + right.circularity) / 2.0, 0.0, 1.0))
        if circularity_score < self.config.iris_min_circularity:
            failure_reasons.append("Iris contour deformation or partial occlusion")

        # 3. Periocular Lighting & Contrast Score
        contrast_score = 1.0
        if frame is not None and frame.size > 0:
            contrast_score = self._compute_contrast_score(frame, left, right)
            if contrast_score < 0.40:
                failure_reasons.append("Low periocular lighting contrast")

        # 4. Landmark Temporal Stability / Jitter Score
        stability_score = self._compute_stability_score(landmarks, img_w, img_h)
        if stability_score < 0.40:
            failure_reasons.append("Landmark high-frequency tracking jitter")

        # 5. Head Pose Angle Limit Check
        if head_pose is not None:
            if (
                abs(head_pose.pitch) > self.config.head_pose_pitch_limit_deg
                or abs(head_pose.yaw) > self.config.head_pose_yaw_limit_deg
                or abs(head_pose.roll) > self.config.head_pose_roll_limit_deg
            ):
                failure_reasons.append("Extreme head pose rotation")

        # 6. Composite Weighted Confidence
        w_ear = self.quality_config.ear_weight
        w_circ = self.quality_config.circularity_weight
        w_cont = self.quality_config.contrast_weight
        w_stab = self.quality_config.stability_weight

        composite_score = (
            w_ear * ear_score
            + w_circ * circularity_score
            + w_cont * contrast_score
            + w_stab * stability_score
        )

        # Gate composite score if eyes are closed or extreme failure occurred
        if not left.is_open or not right.is_open:
            composite_score = min(composite_score, 0.20)

        composite_score = float(np.clip(composite_score, 0.0, 1.0))
        is_valid = bool(
            composite_score >= self.quality_config.min_confidence
            and left.is_open
            and right.is_open
            and "Extreme head pose rotation" not in failure_reasons
        )

        return TrackingQuality(
            confidence=composite_score,
            ear_score=ear_score,
            circularity_score=circularity_score,
            contrast_score=contrast_score,
            stability_score=stability_score,
            is_valid=is_valid,
            failure_reasons=failure_reasons
        )

    def _compute_contrast_score(self, frame: np.ndarray, left: Any, right: Any) -> float:
        """Computes photometric standard deviation in periocular regions."""
        try:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Get bounding boxes for left and right eyes
            stddevs = []
            for eye in (left, right):
                if len(eye.contour_px) > 0:
                    pts = np.array(eye.contour_px)
                    x_min, y_min = np.clip(np.min(pts, axis=0) - 5, [0, 0], [w - 1, h - 1])
                    x_max, y_max = np.clip(np.max(pts, axis=0) + 5, [0, 0], [w - 1, h - 1])
                    if x_max > x_min and y_max > y_min:
                        roi = gray[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if roi.size > 0:
                            stddevs.append(float(np.std(roi)))

            if stddevs:
                mean_std = float(np.mean(stddevs))
                # Map stddev [0, 40] to [0.0, 1.0]
                return float(np.clip(mean_std / 35.0, 0.0, 1.0))
        except Exception:
            pass
        return 1.0

    def _compute_stability_score(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int
    ) -> float:
        """Computes temporal displacement jitter across consecutive frames."""
        try:
            key_indices = self.config.head_pose_mesh_indices + [
                self.config.left_iris_center,
                self.config.right_iris_center
            ]
            current_pts = np.array([
                [landmarks[i].x * img_w, landmarks[i].y * img_h]
                for i in key_indices if i < len(landmarks)
            ], dtype=np.float64)

            if self._prev_landmarks is not None and self._prev_landmarks.shape == current_pts.shape:
                diffs = np.linalg.norm(current_pts - self._prev_landmarks, axis=1)
                mean_diff = float(np.mean(diffs))
                self._jitter_history.append(mean_diff)
            else:
                self._jitter_history.append(0.0)

            self._prev_landmarks = current_pts

            if len(self._jitter_history) > 0:
                avg_jitter = float(np.mean(self._jitter_history))
                # Jitter <= 2.0px -> score ~ 1.0, Jitter >= 8.0px -> score drops
                stability = float(np.exp(-max(0.0, avg_jitter - 2.0) / 4.0))
                return float(np.clip(stability, 0.0, 1.0))
        except Exception:
            pass
        return 1.0

    def reset(self):
        """Resets temporal tracking history."""
        self._prev_landmarks = None
        self._jitter_history.clear()

    def assess_quality(
        self,
        gaze_features: Optional[GazeFeatures],
        head_pose: Optional[HeadPoseData] = None,
        frame: Optional[np.ndarray] = None,
        landmarks: Optional[List[Any]] = None,
        img_w: int = 640,
        img_h: int = 480
    ) -> TrackingQuality:
        """Alias for evaluate() with flexible argument ordering."""
        if gaze_features is not None and head_pose is not None and gaze_features.head_pose is None:
            gaze_features.head_pose = head_pose
        return self.evaluate(
            gaze_features=gaze_features,
            landmarks=landmarks,
            frame=frame,
            img_w=img_w,
            img_h=img_h
        )

