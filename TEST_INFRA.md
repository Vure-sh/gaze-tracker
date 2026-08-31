# E2E Test Infra: Real-Time Gaze Tracker

## Test Philosophy
- Requirement-driven, opaque-box testing derived directly from `ORIGINAL_REQUEST.md`.
- No reliance on internal module private methods; test interfaces and data contracts.
- Systematic 4-tier test architecture: Category-Partition, Boundary Value Analysis (BVA), Pairwise Combinatorial, and Real-World Workload Testing.

## Feature Inventory
| # | Feature | Source (Requirement) | Tier 1 (>=5) | Tier 2 (>=5) | Tier 3 | Tier 4 |
|---|---|---|:---:|:---:|:---:|:---:|
| F01 | MediaPipe FaceLandmarker Initialization & Inference | R1, R2 | 5 | 5 | ✓ | ✓ |
| F02 | Dual-Eye Orthonormal Iris Normalization | R2 | 5 | 5 | ✓ | ✓ |
| F03 | 6-Point EAR & Dynamic Blink Detection | R2 | 5 | 5 | ✓ | ✓ |
| F04 | Eye Contour & 5-Point Iris Metric Geometry | R2 | 5 | 5 | ✓ | ✓ |
| F05 | 3D Head Pose solvePnP & Angle Extraction | R2 | 5 | 5 | ✓ | ✓ |
| F06 | Multi-Dimensional Tracking Quality & Confidence | R2 | 5 | 5 | ✓ | ✓ |
| F07 | Multi-Point Grid Generation (9, 13, 16 points) | R3 | 5 | 5 | ✓ | ✓ |
| F08 | Saccade Latency Trimming & Sample Filtering | R3 | 5 | 5 | ✓ | ✓ |
| F09 | Statistical Outlier Rejection (IQR) | R3 | 5 | 5 | ✓ | ✓ |
| F10 | Polynomial Ridge Gaze Regression (RidgeCV) | R3 | 5 | 5 | ✓ | ✓ |
| F11 | Alternative Regressors (Huber, SVR) | R3 | 5 | 5 | ✓ | ✓ |
| F12 | LOPO Cross-Validation & Holdout Validation | R3 | 5 | 5 | ✓ | ✓ |
| F13 | Model Serialization & Deserialization | R3 | 5 | 5 | ✓ | ✓ |
| F14 | Velocity-Gated One-Euro Temporal Filter | R4 | 5 | 5 | ✓ | ✓ |
| F15 | 2D Constant Velocity Kalman Filter | R4 | 5 | 5 | ✓ | ✓ |
| F16 | Asynchronous Threaded Camera Stream & Fallback | R4 | 5 | 5 | ✓ | ✓ |
| F17 | Screen Gaze Canvas Rendering | R5 | 5 | 5 | ✓ | ✓ |
| F18 | Animated Pulsing Calibration Target | R5 | 5 | 5 | ✓ | ✓ |
| F19 | Gaze Cursor & Glowing Heat Trail | R5 | 5 | 5 | ✓ | ✓ |
| F20 | Camera Debug HUD Window & Overlays | R5 | 5 | 5 | ✓ | ✓ |
| F21 | Translucent Telemetry Card Rendering | R5 | 5 | 5 | ✓ | ✓ |
| F22 | 3D Head Pose Orientation Axes Projection | R5 | 5 | 5 | ✓ | ✓ |
| F23 | Interactive Keyboard Controls & Actions | R5 | 5 | 5 | ✓ | ✓ |
| F24 | CLI Argument Parser & Configuration | R5 | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: `pytest -v tests/` or `python3 -m unittest discover tests`
- **Pass/Fail Semantics**: All tests must exit code 0 with 0 failures, 0 errors.
- **Fixture Strategy**: Synthetic landmark points, synthetic frame arrays, simulated gaze calibration paths with known ground truth mapping.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|---|---|---|
| 1 | Full Calibration Flow (9-point) -> Validation (4-point holdout) | F07, F08, F09, F10, F12, F13 | High |
| 2 | Continuous Gaze Tracking under Dynamic Saccades & Fixations | F02, F05, F10, F14, F19 | High |
| 3 | Rapid Blink & Extreme Head Pitch/Yaw Recovery | F03, F05, F06, F14, F21 | High |
| 4 | Offline Calibration Profile Save -> Clean Restart -> Profile Load & Verification | F10, F13, F24 | Medium |
| 5 | Video Stream Corruption & Missing Landmark Graceful Handling | F01, F06, F16, F20 | Medium |

## Coverage Thresholds
- Tier 1: >= 5 tests per feature (Component integrity, data structures, error guards)
- Tier 2: >= 5 tests per feature (Rotation invariance [±15°, ±30°], scale invariance [0.5x, 2.0x], translation)
- Tier 3: Pairwise cross-feature interactions & calibration accuracy (MAE < 35px, RMSE < 50px)
- Tier 4: >= 5 realistic end-to-end workload application scenarios (FPS >= 30-60, Latency < 35ms, zero crash)
- Minimum total test count: > 100 tests across the test suite.
