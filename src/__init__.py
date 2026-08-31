"""Real-time gaze tracker package."""

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

__all__ = [
    "NormalizedPoint",
    "EyeData",
    "HeadPoseData",
    "GazeFeatures",
    "GazePrediction",
    "TrackingQuality",
    "FaceDetectionResult",
    "GazeConfig",
    "CameraConfig",
    "QualityConfig",
    "FaceDetector",
    "EyeExtractor",
    "HeadPoseEstimator",
    "QualityTracker"
]
