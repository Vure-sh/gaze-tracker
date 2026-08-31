"""Calibration and validation module for screen gaze tracking."""

from .calibrator import CalibrationManager, CalibrationState
from .targets import TargetGenerator

__all__ = [
    "CalibrationManager",
    "CalibrationState",
    "TargetGenerator",
]
