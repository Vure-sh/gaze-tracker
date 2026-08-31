## 2026-08-30T15:37:40Z
You are the Computer Vision Specialist Worker for Milestone 1 (CV & Robust Feature Engineering) for the gaze-tracker project at `/home/vure/gaze-tracker`.
Your working directory is `/home/vure/gaze-tracker/.agents/worker_m1_1`.

Read the original request at `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md`.
Read the project architecture at `/home/vure/gaze-tracker/PROJECT.md`.
Read the CV exploration findings at `/home/vure/gaze-tracker/.agents/explorer_cv_1/handoff.md`.
Read the ML exploration findings at `/home/vure/gaze-tracker/.agents/explorer_ml_1/handoff.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your task for Milestone 1:
1. Create `src/types.py` defining core dataclasses: `NormalizedPoint`, `EyeData`, `HeadPoseData`, `GazeFeatures`, `GazePrediction`, `TrackingQuality`.
2. Refactor/Enhance `src/config.py` with typed configurations, FOV settings, model paths, and landmark indices.
3. Implement `src/cv/face_detector.py` (and maintain `src/face_mesh_detector.py` compatibility wrapper) with MediaPipe FaceLandmarker, supporting both static image and video tracking modes, blendshapes extraction, and 4x4 matrix extraction.
4. Implement `src/cv/eye_extractor.py` (and maintain `src/eye_extractor.py` compatibility wrapper) with:
   - Orthonormal scale- and roll-invariant iris normalization that fixes the horizontal vector cancellation bug (ensuring left and right canthal vectors consistently map horizontal gaze to +X).
   - 6-point EAR computation with adaptive baseline thresholding for reliable blink detection.
   - 5-point iris geometry with circularity metric and metric iris diameter/depth calculation.
5. Implement `src/cv/head_pose.py` (and maintain `src/head_pose.py` compatibility wrapper) with:
   - Corrected 3D anthropometric model points (`MODEL_POINTS_CORRECTED`) aligned with OpenCV camera coordinates (neutral face yields Pitch ≈ 0.0°, Yaw ≈ 0.0°, Roll ≈ 0.0° without branch-cut discontinuity).
   - FOV-based camera intrinsic matrix.
   - Decoupled head pose rotation & translation feature vector.
6. Implement `src/cv/quality_tracker.py` computing composite tracking confidence score based on EAR, iris circularity, lighting contrast, and landmark stability.
7. Ensure clean 8D/10D normalized feature vector generation in `GazeFeatures.feature_vector`.

You have exclusive write ownership of `src/types.py`, `src/config.py`, `src/cv/*`, `src/face_mesh_detector.py`, `src/eye_extractor.py`, `src/head_pose.py`.
Run tests and verification commands to verify correctness.
Write your completion handoff report to `/home/vure/gaze-tracker/.agents/worker_m1_1/handoff.md` and send a message when done.
