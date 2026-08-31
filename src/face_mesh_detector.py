"""Backwards-compatible wrapper for MediaPipe Face Landmarker detection."""

from typing import Optional, List, Any
import numpy as np
from src.cv.face_detector import FaceDetector
from src.types import NormalizedPoint, FaceDetectionResult


class FaceMeshDetector(FaceDetector):
    """Backwards-compatible wrapper inheriting from FaceDetector."""

    def __init__(self, model_path: str, num_faces: int = 1):
        super().__init__(model_path=model_path, num_faces=num_faces)

    def detect(self, bgr_image: np.ndarray) -> Optional[List[Any]]:
        """
        Runs face and iris detection on a BGR OpenCV frame.
        
        Returns:
            List of 478 NormalizedPoint/Landmark objects for the primary face, or None.
        """
        return super().detect(bgr_image)
