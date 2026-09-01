# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `GazeConfig.validate()` — runtime sanity check for all configuration parameters
- `GazePipeline.get_stats()` — live snapshot of pipeline performance metrics (FPS, latency, calibration state)
- `resolution` property on `ThreadedCameraStream` returning actual capture dimensions
- `__repr__` on `GazePrediction` and `TrackingQuality` for readable debug output
- Contributing guidelines in README
- Project description, classifiers, keywords, and MIT license in `pyproject.toml`

## [0.1.0] — 2026-06-01

### Added
- Real-time webcam gaze tracking with MediaPipe FaceLandmarker
- Orthonormal dual-eye normalization and 3D head pose compensation via solvePnP
- Polynomial Ridge regression gaze model with interactive 9-point calibration UI
- Velocity-gated One-Euro filter and constant-velocity 2D Kalman filter
- Multi-dimensional tracking quality assessment (EAR, iris circularity, contrast, stability)
- Threaded camera capture stream with automatic device fallback
- Full-screen gaze canvas and webcam debug HUD visualizer
- Comprehensive 5-tier automated test suite (393 tests)
