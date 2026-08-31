## 2026-08-30T15:37:40Z
You are the E2E Test Suite Engineer for the gaze-tracker project at `/home/vure/gaze-tracker`.
Your working directory is `/home/vure/gaze-tracker/.agents/test_writer_1`.

Read the original request at `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md`.
Read the project blueprint at `/home/vure/gaze-tracker/PROJECT.md`.
Read the test infrastructure specification at `/home/vure/gaze-tracker/TEST_INFRA.md`.
Read the specification inventory at `/home/vure/gaze-tracker/.agents/spec_miner_1/handoff.md`.

Your task is to build a complete, requirement-driven, opaque-box 4-Tier test suite in `tests/`:
1. `tests/conftest.py`: Shared test fixtures (synthetic landmark lists, 478 MediaPipe point generator, mock frame generator, synthetic calibration dataset generator with ground-truth mapping).
2. `tests/test_tier1_units.py`: Tier 1 Unit & Component Integrity Tests covering parameter validation, landmark geometry, EAR calculation, pose rotation matrices, filter step response, serialization guards (>= 5 tests per feature).
3. `tests/test_tier2_invariance.py`: Tier 2 Geometric & Invariance Tests covering head roll invariance (0°, 15°, -15°, 45°), scale/distance invariance (0.5x, 1.0x, 2.0x), translation invariance, 3D pose decoupling under pitch/yaw rotations (±15°).
4. `tests/test_tier3_calibration.py`: Tier 3 Calibration & Regression Accuracy Tests covering 9/13/16-point grid generators, saccade delay filtering, outlier rejection (IQR), RidgeCV / Huber fitting, LOPO cross-validation, MAE < 35px verification on synthetic test suite, profile save/load round-trip.
5. `tests/test_tier4_performance.py`: Tier 4 Performance, Latency & Stress Tests covering corrupted frame inputs (None, zeros, 1D arrays), extreme blinks/occlusions, filter step settling time, and latency/throughput profiling.

You have exclusive write ownership of `tests/` and `TEST_READY.md`.
Run `pytest -v tests/` to verify test suite syntax and execution.
When complete and passing, write `TEST_READY.md` at `/home/vure/gaze-tracker/TEST_READY.md` with full coverage summary.
Write your handoff report to `/home/vure/gaze-tracker/.agents/test_writer_1/handoff.md` and send a message when done.
