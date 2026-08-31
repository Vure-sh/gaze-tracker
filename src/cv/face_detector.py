"""MediaPipe Face Landmarker wrapper supporting IMAGE and VIDEO tracking modes,
blendshapes extraction, and 4x4 facial transformation matrices.
"""

from __future__ import annotations
import os
import time
import urllib.request
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.types import NormalizedPoint, FaceDetectionResult


class FaceDetector:
    """Detects 478 dense 3D facial and iris landmarks with MediaPipe FaceLandmarker."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.model_path = model_path
        self.num_faces = num_faces
        self.running_mode = running_mode
        self._last_timestamp_ms = 0
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=running_mode
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Downloads the MediaPipe task bundle if missing from local filesystem."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Downloading face landmarker model to {self.model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.model_path)

    def detect(
        self,
        bgr_image: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Optional[List[NormalizedPoint]]:
        """
        Runs face detection on a BGR OpenCV frame and returns 478 normalized landmarks.

        Args:
            bgr_image: BGR numpy image array.
            timestamp_ms: Monotonic millisecond timestamp (required if running in VIDEO mode).

        Returns:
            List of 478 NormalizedPoint objects, or None if no face is detected.
        """
        result = self.detect_full(bgr_image, timestamp_ms)
        if result is not None and result.landmarks:
            return result.landmarks
        return None

    def detect_full(
        self,
        bgr_image: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Optional[FaceDetectionResult]:
        """
        Runs detection returning landmarks, 52 blendshape channels, and 4x4 matrix.

        Returns:
            FaceDetectionResult instance or None.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        if timestamp_ms is None:
            now_ms = int(time.time() * 1000)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            timestamp_ms = now_ms

        self._last_timestamp_ms = timestamp_ms

        try:
            if self.running_mode == vision.RunningMode.VIDEO:
                detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
            else:
                detection_result = self.detector.detect(mp_image)

            if not detection_result or not detection_result.face_landmarks:
                return None

            raw_landmarks = detection_result.face_landmarks[0]
            landmarks = [NormalizedPoint.from_landmark(lm) for lm in raw_landmarks]

            # Extract blendshapes if present
            blendshapes: Optional[Dict[str, float]] = None
            if (
                detection_result.face_blendshapes
                and len(detection_result.face_blendshapes) > 0
            ):
                blendshapes = {
                    b.category_name: float(b.score)
                    for b in detection_result.face_blendshapes[0]
                }

            # Extract 4x4 facial transformation matrix if present
            matrix: Optional[np.ndarray] = None
            if (
                detection_result.facial_transformation_matrixes
                and len(detection_result.facial_transformation_matrixes) > 0
            ):
                matrix = np.array(
                    detection_result.facial_transformation_matrixes[0],
                    dtype=np.float64
                )

            return FaceDetectionResult(
                landmarks=landmarks,
                blendshapes=blendshapes,
                facial_transformation_matrix=matrix,
                timestamp_ms=timestamp_ms
            )

        except Exception as e:
            # Handle transient runtime exceptions gracefully
            return None

    def close(self):
        """Closes the MediaPipe detector instance."""
        if hasattr(self, "detector") and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
