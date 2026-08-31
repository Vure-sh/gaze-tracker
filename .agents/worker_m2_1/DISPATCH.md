## 2026-08-30T18:56:46Z
You are the ML & Calibration Specialist Worker for Milestone 2 (ML & Gaze Estimation / Calibration) for the gaze-tracker project at `/home/vure/gaze-tracker`.
Your working directory is `/home/vure/gaze-tracker/.agents/worker_m2_1`.

Read the original user request at `/home/vure/gaze-tracker/.agents/ORIGINAL_REQUEST.md`.
Read the project architecture at `/home/vure/gaze-tracker/PROJECT.md`.
Read the ML exploration findings at `/home/vure/gaze-tracker/.agents/explorer_ml_1/handoff.md`.
Read the CV exploration findings at `/home/vure/gaze-tracker/.agents/explorer_cv_1/handoff.md`.
Read the M1 Worker handoff at `/home/vure/gaze-tracker/.agents/worker_m1_1/handoff.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your task for Milestone 2:
1. Implement `src/calibration/targets.py` & `src/calibration/calibrator.py` (and maintain `src/calibrator.py` compatibility wrapper):
   - Multi-point calibration sequences (9, 13, 16 points) with Boustrophedon / serpentine ordering to reduce saccadic eye fatigue.
   - Wall-clock dwell timing: 350ms saccade delay (`saccade_delay_seconds`), collection window, and minimum valid frame threshold (`min_valid_samples = 15`).
   - Statistical outlier rejection: normalize features, calculate Euclidean distance to median vector, apply $1.5 \times \text{IQR}$ cutoff with zero-variance safety guards.
   - Dedicated post-calibration interactive holdout validation mode (4-point/5-point holdout) calculating live screen pixel MAE, RMSE, and visual angle error in degrees ($\theta < 1.0^\circ$).
2. Implement `src/models/regressor.py` (and maintain `src/gaze_regressor.py` compatibility wrapper):
   - `BaseGazeRegressor` abstract base class defining `train()`, `predict()`, `save_profile()`, `load_profile()`.
   - `PolynomialRidgeRegressor`: Degree-2 polynomial expansion with `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])`, training on normalized target screen coordinates $[0, 1]^2$ with automatic display resolution scaling and output clamping.
   - `RobustHuberRegressor`: Degree-2 polynomial expansion with `HuberRegressor` for outlier-robust loss.
   - `SVRGazeRegressor`: SVR alternative backend.
   - Leave-One-Point-Out (LOPO) Group Cross-Validation calculating spatial generalization MAE and RMSE during training.
3. Implement `src/models/serializer.py`:
   - Schema-versioned (`schema_version: "2.0"`) model profile serialization (JSON metadata + numpy arrays or safe pickle) preserving hyperparameters, polynomial degree, resolution, feature dimensions, and validation metrics.
   - Profile verification and backward compatibility loader.

You have exclusive write ownership of `src/calibration/*`, `src/calibrator.py`, `src/models/*`, `src/gaze_regressor.py`.
Run tests via `pytest` and verify all Tier 1–4 tests and calibration accuracy requirements (MAE < 35px, RMSE < 50px).
Write your completion handoff report to `/home/vure/gaze-tracker/.agents/worker_m2_1/handoff.md` and send a message when done.
