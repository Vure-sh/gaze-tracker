# Original User Request

## 2026-08-30T15:31:31Z

Audit, refactor, and substantially improve the existing real-time webcam eye and gaze tracking system at `/home/vure/gaze-tracker` into a production-grade, highly accurate, low-latency, and robust computer vision product.

Working directory: /home/vure/gaze-tracker
Integrity mode: development

You are operating as a coordinated senior engineering team of specialists:
- Computer Vision Engineer (Landmark quality, normalized iris features, head pose compensation, tracking stability)
- ML / Gaze Estimation Engineer (Gaze mapping, calibration methodology, regression models, prediction accuracy)
- Performance Engineer (FPS, latency, memory, efficient frame processing)
- UX Engineer (Calibration experience, visual cues, gaze cursor, debug HUD)
- QA / Reliability Engineer (Automated test suite, edge cases, lighting/blinks, failure recovery)

## Requirements

### R1. Technical Audit & Baseline Profiling
Conduct a thorough audit of the existing codebase (`main.py`, `src/` modules, models, and dependencies). Measure baseline FPS, latency, landmark extraction stability, calibration fitting quality, and identify critical edge-case failure modes without blindly rewriting working components.

### R2. Computer Vision & Robust Feature Engineering
Enhance facial and iris feature extraction:
- Implement dual-eye scale- and roll-invariant iris normalization.
- Refine 3D head-pose compensation (`solvePnP` Euler angles and translation vectors) to decouple head motion from gaze estimation.
- Implement gaze confidence scoring and tracking quality metrics (e.g. iris visibility, landmark jitter detection, lighting/contrast adequacy, and blink handling).

### R3. Calibration Methodology & Gaze Regression
Improve gaze-to-screen mapping and calibration accuracy:
- Enhance multi-point calibration with adaptive saccade latency trimming and robust outlier filtering.
- Implement and benchmark lightweight regression models (e.g. Regularized Polynomial Ridge, SVR, Huber regression) with cross-validation or point-holdout verification.
- Add a post-calibration validation mode to calculate and display live screen Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Ensure calibration profiles are seamlessly persistent and reloadable.

### R4. Temporal Filtering & Real-Time Performance
Optimize the real-time runtime pipeline:
- Implement adaptive temporal filtering (e.g., tuned One-Euro filter and velocity-aware smoothing) for steady fixations without introducing lag during rapid saccades.
- Profile and optimize OpenCV frame acquisition, MediaPipe processing, and matrix calculations to maintain high, stable framerates (30–60+ FPS) with low CPU usage.
- Ensure graceful recovery during temporary face loss, blinking, or rapid lighting changes.

### R5. UX, Visualization & Debugging Tools
Refine the user experience and visualization:
- Screen Gaze Canvas: Clean, high-visibility gaze cursor, trailing gaze heat, and intuitive calibration visual cues (pulsing animated targets with progress feedback).
- Camera Debug HUD: Live facial mesh contour overlays, iris bounding circles, 3D head pose orientation axes, and real-time telemetry (FPS, EAR, head angles, gaze coordinates, tracking confidence).
- Clear keyboard controls and intuitive CLI options.

### R6. Automated Testing & Verification Suite
Provide an automated test suite verifying:
- Pipeline component initialization and data integrity.
- Feature extraction invariance under rotation and scaling.
- Regression training, prediction, serialization, and deserialization.
- Temporal filter convergence and step-response.
- Handling of invalid/corrupted frame inputs without crashes.

## Acceptance Criteria

### Verification & Accuracy
- [ ] Automated test suite (`pytest` or test runner) executes and passes 100% of unit and integration tests.
- [ ] Calibration achieves low prediction error (target MAE < 35px on standard 1080p displays) under normal head poses.
- [ ] Gaze cursor demonstrates steady fixation (no micro-jitter) while remaining responsive during rapid saccades.
- [ ] Head pose compensation prevents gaze drift during moderate yaw/pitch rotations (±15°).

### Robustness & Performance
- [ ] Pipeline runs continuously in real-time (>= 30 FPS) with low frame latency (< 35ms processing time).
- [ ] Gracefully handles face occlusion, closed eyes/blinking, and off-screen lookaways without raising uncaught exceptions or drifting indefinitely.
- [ ] Calibration models serialize and deserialize without loss of accuracy or configuration mismatch.

### Documentation & Deliverables
- [ ] Updated `README.md` containing architectural diagram, installation steps, mathematical explanations, benchmark results, and usage guide.
- [ ] Codebase is modular, clean, PEP 8 compliant, and fully documented with type annotations.
