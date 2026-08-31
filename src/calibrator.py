"""Backward-compatibility module wrapper for CalibrationManager and calibration targets.

Directs callers to modular implementations in `src.calibration`.
"""

from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.calibration.targets import TargetGenerator

__all__ = [
    "CalibrationManager",
    "CalibrationState",
    "TargetGenerator",
]
