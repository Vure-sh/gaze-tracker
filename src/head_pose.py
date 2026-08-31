"""Backwards-compatible wrapper module for head pose estimation."""

from src.types import HeadPoseData
from src.cv.head_pose import HeadPoseEstimator

__all__ = ["HeadPoseData", "HeadPoseEstimator"]
