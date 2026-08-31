# Project: Real-Time Gaze Tracker

## Architecture
The gaze tracking system processes webcam video feeds to predict real-time 2D screen gaze coordinates with head pose compensation and temporal filtering.

```
[Camera Frame / Threaded Stream]
               │
               ▼
   [Face Landmark Detector] (MediaPipe 478 3D mesh + iris)
         │                   │
         ▼                   ▼
[Eye Feature Extractor]   [3D Head Pose Estimator]
(Orthonormal Norm, EAR,   (solvePnP, corrected 3D model,
 Iris circularity metric)  Euler angles, translation)
         │                   │
         └─────────┬─────────┘
                   ▼
       [Gaze Features Aggregator]
       (Quality check & tracking state)
                   │
                   ▼
     [Calibration / Gaze Regressor]
     (RidgeCV / Huber / Polynomial model)
                   │
                   ▼
       [Temporal Filter] (Velocity-Gated One-Euro / Kalman)
                   │
                   ▼
       [Screen Gaze Coordinate]
         │                   │
         ▼                   ▼
[Screen Gaze Canvas]    [Camera Debug HUD & Telemetry]
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F01 | MediaPipe 478 FaceLandmarker | 478 3D dense facial and dual-iris landmark detector with video tracking | M1 | Survey (F01) [VERIFIED] |
| F02 | Orthonormal Dual-Eye Normalization | Scale- and roll-invariant iris coordinates on canthal axes (no left/right cancellation) | M1 | Survey (F02) [VERIFIED] |
| F03 | 6-Point EAR & Dynamic Blink Detection | Dynamic adaptive EAR thresholding and eye closure detection | M1 | Survey (F03) [VERIFIED] |
| F04 | Eye Contour & Iris Geometry | 16-point eyelid perimeter and 5-point metric iris circularity/depth | M1 | Survey (F04) [VERIFIED] |
| F05 | 3D Head Pose Estimation (solvePnP) | Corrected 3D anthropometric face model, camera FOV matrix, pitch branch-cut fix | M1 | Survey (F05) [VERIFIED] |
| F06 | Multi-Dimensional Tracking Quality | Tracking confidence score combining EAR, iris circularity, lighting, pose limits | M1 | Survey (F06) [VERIFIED] |
| F07 | Multi-Point Grid Generator | 9, 13, 16-point screen target grid generator with Boustrophedon ordering | M2 | Survey (F07) [VERIFIED] |
| F08 | Saccade Latency & Wall-Clock Dwell | Wall-clock 350ms saccade delay + collection duration with valid frame count | M2 | Survey (F08) [VERIFIED] |
| F09 | Statistical Outlier Filtering | Normalized feature space median Euclidean distance with 1.5*IQR threshold | M2 | Survey (F09) [VERIFIED] |
| F10 | Polynomial Ridge Gaze Regressor | Degree-2 Polynomial RidgeCV pipeline mapping features to screen coordinates | M2 | Survey (F10) [VERIFIED] |
| F11 | Alternative Estimator Backends | Modular BaseGazeRegressor supporting HuberRegressor and SVR | M2 | Survey (F11) [VERIFIED] |
| F12 | Validation & Live Accuracy Metrics | Leave-One-Point-Out (LOPO) CV and 4-point interactive holdout validation (MAE/RMSE/angle) | M2 | Survey (F12) [VERIFIED] |
| F13 | Model Serialization & Deserialization | Schema-versioned model profile save/load (JSON/NPZ or safe pickle) | M2 | Survey (F13) [VERIFIED] |
| F14 | Velocity-Gated One-Euro Filter | Adaptive low-pass filter with velocity deadband (<1.1px jitter, 0ms saccade lag) | M3 | Survey (F14) [VERIFIED] |
| F15 | 2D Kalman Filter Option | Linear constant-velocity Kalman filter with state covariance reset | M3 | Survey (F15) [VERIFIED] |
| F16 | Asynchronous Threaded Camera Stream | Dedicated capture thread with automatic fallback (/dev/video9 -> 0, 1, 2) | M3 | Survey (F16) [VERIFIED] |
| F17 | Screen Gaze Canvas Renderer | High-visibility dark slate canvas with gaze cursor and coordinate tags | M4 | Survey (F17) [VERIFIED] |
| F18 | Animated Pulsing Target | Pulsing concentric circles with 360° circular progress arc | M4 | Survey (F18) [VERIFIED] |
| F19 | Gaze Cursor & Glowing Heat Trail | Multi-ring glowing gaze dot with alpha-faded 20-frame decaying history trail | M4 | Survey (F19) [VERIFIED] |
| F20 | Camera Debug HUD Window | Live camera overlay with eye contours, iris centers, 3D pose axes, and mesh | M4 | Survey (F20) [VERIFIED] |
| F21 | Translucent Telemetry Card | Alpha-blended HUD card with FPS, Euler angles, EAR, norm coordinates, status | M4 | Survey (F21) [VERIFIED] |
| F22 | 3D Head Pose Orientation Axes | Projected RGB orthogonal orientation vector arrows from nose tip | M4 | Survey (F22) [VERIFIED] |
| F23 | Interactive Keyboard Controls | Hotkeys: C (Calibrate), R (Reset), S (Save), L (Load), D (HUD), F (Fullscreen), Q/ESC | M4 | Survey (F23) [VERIFIED] |
| F24 | CLI Configuration & Options | CLI arguments: --camera, --points, --filter, --load, --fullscreen, --no-hud, etc. | M4 | Survey (F24) [VERIFIED] |
| F25 | 4-Tier Automated Test Suite | Comprehensive unit, invariance, accuracy, and stress test harness | E2E | Survey (F25) [VERIFIED] |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| E2E | E2E Testing Track | Requirement-driven 4-tier opaque-box test suite (`tests/`, `TEST_READY.md`) | none | DONE |
| M1 | CV & Robust Feature Engineering | MediaPipe tracking, orthonormal iris normalization, solvePnP head pose, tracking confidence | none | DONE |
| M2 | ML & Gaze Estimation / Calibration | Multi-point calibration, outlier rejection, RidgeCV/Huber models, LOPO CV & holdout validation, persistence | M1 contracts | DONE |
| M3 | Temporal Filtering & Real-Time Performance | Velocity-gated One-Euro & Kalman filters, threaded camera capture, low-latency pipeline | M1, M2 | DONE |
| M4 | UX, Visualization, CLI & Debug HUD | Screen canvas, animated targets, gaze heat trail, camera debug HUD, telemetry card, CLI/hotkeys | M1, M2, M3 | DONE |
| M_Final | Final Milestone & Verification | 100% E2E test pass (Tiers 1-5), Tier 5 adversarial hardening, Forensic Audit, documentation | M1, M2, M3, M4, E2E | DONE |

## Interface Contracts

### Module: `src/types.py`
Shared dataclasses across all modules:
- `NormalizedPoint`: `(x: float, y: float, z: float = 0.0)`
- `EyeData`: `(inner: Tuple[int, int], outer: Tuple[int, int], top: Tuple[int, int], bottom: Tuple[int, int], iris_center: Tuple[int, int], norm_x: float, norm_y: float, ear: float, is_open: bool, iris_diameter_px: float, circularity: float)`
- `HeadPoseData`: `(pitch: float, yaw: float, roll: float, rvec: np.ndarray, tvec: np.ndarray, feature_vector: np.ndarray)`
- `GazeFeatures`: `(left_eye: EyeData, right_eye: EyeData, avg_norm_x: float, avg_norm_y: float, head_pose: HeadPoseData, confidence: float, is_valid: bool, feature_vector: np.ndarray)`
- `GazePrediction`: `(screen_x: float, screen_y: float, norm_x: float, norm_y: float, confidence: float, is_valid: bool)`

### Module: `src/cv/` ↔ `src/calibration/` / `src/models/`
- `EyeExtractor.extract(landmarks: List[NormalizedPoint], img_w: int, img_h: int) -> GazeFeatures`
- `HeadPoseEstimator.estimate(landmarks: List[NormalizedPoint], img_w: int, img_h: int) -> HeadPoseData`
- `GazeFeatures.vector_8d`: `[norm_x_L, norm_y_L, norm_x_R, norm_y_R, pitch/45, yaw/45, roll/45, tz/1000]`.
- `GazeFeatures.vector_10d`: `[norm_x_L, norm_y_L, norm_x_R, norm_y_R, avg_norm_x, avg_norm_y, pitch/45, yaw/45, roll/45, tz/1000]`.

### Module: `src/models/` ↔ `src/filters/` ↔ `src/ui/`
- `BaseGazeRegressor.train(X: np.ndarray, y: np.ndarray, point_ids: Optional[np.ndarray] = None) -> Dict[str, float]`
- `BaseGazeRegressor.predict(feature_vector: np.ndarray) -> Optional[Tuple[float, float]]`
- `BaseGazeRegressor.save_profile(filepath: str) -> None`
- `BaseGazeRegressor.load_profile(filepath: str) -> bool`
- `OneEuroFilter2D.filter(point: Tuple[float, float], timestamp: Optional[float] = None) -> Tuple[float, float]`

## Code Layout
```
gaze-tracker/
├── main.py                     # Main application entry point and orchestration
├── README.md                   # Complete architectural, mathematical & usage guide
├── models/
│   ├── face_landmarker.task    # MediaPipe FaceLandmarker model bundle
│   └── calibration_model.pkl   # Serialized default calibration profile
├── src/
│   ├── __init__.py
│   ├── config.py               # GazeConfig dataclass and hyperparameter settings
│   ├── types.py                # Core data models and typed contracts
│   ├── pipeline.py             # Pipeline orchestrator coupling CV, ML, filtering
│   ├── camera_stream.py        # Threaded async video capture with device fallback
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── face_detector.py    # MediaPipe FaceLandmarker wrapper
│   │   ├── eye_extractor.py    # Orthonormal scale/roll invariant iris normalization
│   │   ├── head_pose.py        # solvePnP with corrected 3D model & FOV matrix
│   │   └── quality_tracker.py  # EAR, iris circularity, lighting, confidence scoring
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrator.py       # Multi-point calibration manager with saccade trimming
│   │   └── targets.py          # Target pattern generator (9, 13, 16 points)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── regressor.py        # RidgeCV, Huber, SVR gaze regressors
│   │   └── serializer.py       # Model profile serialization & validation
│   ├── filters/
│   │   ├── __init__.py
│   │   ├── one_euro.py         # Velocity-gated One-Euro adaptive filter
│   │   └── kalman.py           # 2D Constant Velocity Kalman Filter
│   └── ui/
│       ├── __init__.py
│       ├── canvas.py           # Screen Gaze Canvas & animated target visualizer
│       ├── hud.py              # Camera Debug HUD overlay & telemetry card
│       └── app.py              # Application controller & keyboard event loop
└── tests/
    ├── __init__.py
    ├── conftest.py             # Synthetic landmarks, frames, and calibration fixtures
    ├── test_tier1_units.py     # Tier 1: Unit & Component Integrity Tests
    ├── test_tier2_invariance.py# Tier 2: Rotation & Scale Invariance Tests
    ├── test_tier3_calibration.py# Tier 3: Calibration, LOPO CV & Accuracy Tests
    ├── test_tier4_performance.py# Tier 4: Latency, FPS & Stress Robustness Tests
    ├── test_m1_cv.py           # Milestone 1 CV Tests
    ├── test_m2_calibration_models.py # Milestone 2 ML Tests
    ├── test_m3_filters_pipeline.py # Milestone 3 Filters & Pipeline Tests
    ├── test_m4_ui_hud.py       # Milestone 4 UI & HUD Tests
    ├── test_challenger_m1.py   # Challenger M1 Empirical Stress Tests
    ├── test_challenger_m2.py   # Challenger M2 Empirical Stress Tests
    ├── test_challenger_m2_adversarial.py # Challenger M2 Latency & Serialization Tests
    ├── test_challenger_m3.py   # Challenger M3 Filter & Latency Stress Tests
    ├── test_challenger_m4.py   # Challenger M4 UI Resolution Stress Tests
    └── test_tier5_adversarial_hardening.py # Tier 5 White-Box Adversarial Coverage Hardening
```
