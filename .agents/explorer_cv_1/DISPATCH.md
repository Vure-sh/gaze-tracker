## 2026-08-30T15:32:35Z
You are the CV & Core Pipeline Explorer for the gaze-tracker project at `/home/vure/gaze-tracker`.
Your working directory is `/home/vure/gaze-tracker/.agents/explorer_cv_1`.

Read the original user request at `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md`.

Investigate the existing codebase at `/home/vure/gaze-tracker`:
1. Examine all source files in `src/`, `main.py`, models, and existing scripts.
2. Analyze current landmark detection (MediaPipe Face Mesh / Iris), eye region extraction, and iris normalization logic. Identify why/how it fails under head roll, distance/scale variation, or gaze extremes.
3. Analyze current 3D head pose estimation (solvePnP, 3D facial model points, Euler angles, rotation/translation vectors) and how head pose is coupled/decoupled from eye gaze.
4. Analyze tracking quality, confidence metrics, iris visibility, eyelid distance/EAR (Eye Aspect Ratio), blink detection, and occlusion recovery.
5. Identify existing code layout, dependencies, strengths, weaknesses, and concrete recommendations for refactoring into modular production-grade CV components.

Write your comprehensive findings and handoff report to `/home/vure/gaze-tracker/.agents/explorer_cv_1/handoff.md`. Include concrete file paths, line numbers, mathematical formulation recommendations, and interface suggestions.
When finished, send a message to the orchestrator notifying completion.
