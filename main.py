#!/usr/bin/env python3
"""
Real-Time Webcam Eye & Gaze Tracker
Pipeline: Webcam -> Face & Iris Landmarks -> Eye Feature Extraction -> Head Pose -> Calibration Model -> Temporal Filter -> Real-Time Visualization
"""

import sys
import os
import argparse
import time
import subprocess
import cv2
import numpy as np

from src.config import GazeConfig
from src.face_mesh_detector import FaceMeshDetector
from src.eye_extractor import EyeExtractor
from src.head_pose import HeadPoseEstimator
from src.gaze_regressor import GazeRegressionModel
from src.calibrator import CalibrationManager, CalibrationState
from src.filters import OneEuroFilter2D, KalmanFilter2D
from src.visualizer import GazeVisualizer
from src.camera_stream import ensure_tablet_stream, open_camera_device
from src.ui.app import GazeTrackerApp

scrcpy_process = None


def open_camera(camera_arg: str) -> cv2.VideoCapture:
    """Attempts to open the specified camera index or device path with fallback."""
    cap, actual = open_camera_device(camera_arg)
    if cap is not None and cap.isOpened():
        print(f"✅ Connected to camera device: {actual}")
        return cap
    raise RuntimeError(f"Could not open any webcam or video capture device from '{camera_arg}'.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Eye/Gaze Tracker")
    parser.add_argument("--camera", type=str, default="/dev/video9", help="Camera index (e.g. 0, 1) or device path (e.g. /dev/video9)")
    parser.add_argument("--points", type=str, default="9_points", choices=["9_points", "13_points", "16_points"], help="Calibration grid density")
    parser.add_argument("--filter", type=str, default="one_euro", choices=["one_euro", "kalman"], help="Temporal smoothing filter")
    parser.add_argument("--regressor", type=str, default="ridge", choices=["ridge", "huber", "svr"], help="Regression model backend")
    parser.add_argument("--load", type=str, default=None, help="Path to existing calibration file to load at startup")
    parser.add_argument("--fullscreen", action="store_true", help="Launch screen gaze canvas in fullscreen mode")
    parser.add_argument("--no-hud", action="store_true", help="Hide webcam debug HUD by default")
    parser.add_argument("--width", type=int, default=None, help="Custom screen width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Custom screen height in pixels")
    args = parser.parse_args()

    # 1. Initialize Configuration
    config = GazeConfig()
    config.auto_detect_screen()
    if args.width and args.height:
        config.screen_width = args.width
        config.screen_height = args.height
    config.calibration_grid_type = args.points
    config.filter_type = args.filter

    print("=" * 60)
    print("🎯 REAL-TIME WEBCAM EYE & GAZE TRACKER")
    print(f"• Screen Target Resolution: {config.screen_width}x{config.screen_height}")
    print(f"• Calibration Grid: {config.calibration_grid_type}")
    print(f"• Smoothing Filter: {config.filter_type}")
    print(f"• Regression Model: {args.regressor}")
    print("=" * 60)

    # Launch Desktop Application
    app = GazeTrackerApp(
        config=config,
        camera_src=args.camera,
        fullscreen=args.fullscreen,
        show_hud=not args.no_hud,
        grid_type=args.points,
        filter_type=args.filter,
        load_profile=args.load
    )
    app.run()


if __name__ == "__main__":
    main()
