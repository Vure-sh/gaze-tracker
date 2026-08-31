"""Backwards-compatible wrapper module for eye extraction."""

from src.types import EyeData, GazeFeatures
from src.cv.eye_extractor import EyeExtractor

__all__ = ["EyeData", "GazeFeatures", "EyeExtractor"]
