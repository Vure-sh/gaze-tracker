# BRIEFING — 2026-08-30T18:36:00Z

## Mission
Analyze computer vision and core pipeline (MediaPipe Face Mesh/Iris, 3D head pose via solvePnP, eye region extraction, iris normalization, EAR/blink/quality metrics, coordinate spaces, and failure modes under head roll/distance/extremes) to inform production refactoring.

## 🔒 My Identity
- Archetype: explorer
- Roles: CV & Core Pipeline Explorer
- Working directory: /home/vure/gaze-tracker/.agents/explorer_cv_1
- Original parent: babbda7c-3a89-4322-9d89-2c0ab8c31a32
- Milestone: exploration

## 🔒 Key Constraints
- Read-only investigation — do NOT implement changes in source code
- Produce self-contained handoff report adhering to the 5-component structure
- Detail exact file paths, line numbers, mathematical formulations, failure modes, and refactoring interfaces

## Current Parent
- Conversation ID: babbda7c-3a89-4322-9d89-2c0ab8c31a32
- Updated: 2026-08-30T18:36:00Z

## Investigation State
- **Explored paths**:
  - `src/config.py`, `src/face_mesh_detector.py`, `src/eye_extractor.py`, `src/head_pose.py`, `src/filters.py`, `src/gaze_regressor.py`, `src/calibrator.py`, `src/visualizer.py`, `main.py`, `models/`
- **Key findings**:
  1. *Horizontal Gaze Cancellation*: Opposite canthal vectors between left and right eyes causes `avg_norm_x` to remain 0.5 under horizontal eye movements (`Left delta: -0.25, Right delta: +0.25`).
  2. *solvePnP Pitch Gimbal Lock / Discontinuity*: Inverted $Y$ axis in `MODEL_POINTS` puts resting pitch at $-174.8^\circ$, right on the $\pm \pi$ branch cut, causing discontinuous $\pm 180^\circ$ jumps.
  3. *Vertical Normalization Coupling*: `norm_y` divided by moving eyelid height ($p_{\text{top}} - p_{\text{bottom}}$) causes noise amplification during squints/smiles and roll cross-axis leakage.
  4. *Head Pose Coupling in Regression*: 14D concatenated vector overfits static head calibration; polynomial cross-terms blow up under runtime head rotation.
  5. *MediaPipe Detector*: Currently in `IMAGE` mode (lacking temporal tracking & smoothing) and discarding blendshapes + 4x4 transform matrices.
- **Unexplored areas**: None within the CV pipeline scope.

## Key Decisions Made
- Formulated orthonormal dual-eye coordinate frame $(\vec{u}, \vec{u}_\perp)$ normalized by inter-canthal width and 5-point iris radius for true scale, roll, and blink invariance.
- Redefined standard 3D anthropometric face model points for `solvePnP` with camera-aligned frame (pitch/yaw/roll centered at $0^\circ$).
- Formulated decoupled geometric gaze model $\vec{G} = f_{\text{eye}}(\vec{x}_{\text{eye}}) + W_{\text{head}} \vec{x}_{\text{head}}$.
- Defined comprehensive modular CV architecture: `src/cv/landmark_detector.py`, `src/cv/eye_extractor.py`, `src/cv/head_pose.py`, `src/cv/quality_tracker.py`.

## Artifact Index
- `/home/vure/gaze-tracker/.agents/explorer_cv_1/progress.md` — Liveness & task progress
- `/home/vure/gaze-tracker/.agents/explorer_cv_1/handoff.md` — Final 5-component synthesis report
