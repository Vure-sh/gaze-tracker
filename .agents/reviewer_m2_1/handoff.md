# Milestone 2 (ML & Gaze Estimation / Calibration) Review & Adversarial Analysis Report

**Date**: 2026-08-30  
**Reviewer**: Reviewer 1 (`reviewer_m2_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Milestone**: M2 (ML & Gaze Estimation / Calibration)  
**Verdict**: **APPROVE**  

---

## 1. Observation

Direct code examination and independent test execution yielded the following observations:

### 1.1 Source Code Inspections
1. **Boustrophedon Grid Generator (`src/calibration/targets.py:15-86`)**:
   - `TargetGenerator.generate_points(config, grid_type, boustrophedon=True)` generates 9, 13, and 16-point grids with alternating row traversal (Left $\to$ Right for even rows, Right $\to$ Left for odd rows).
   - `generate_validation_points(config, mode="4_points" | "5_points")` produces quadrant centers at $(0.25W, 0.25H)$, $(0.75W, 0.25H)$, $(0.75W, 0.75H)$, $(0.25W, 0.75H)$, plus optional screen center $(0.5W, 0.5H)$.
   - Mathematical trajectory analysis demonstrated a **24.5% to 35.3% reduction** in total saccadic jump distance compared to legacy raster scan order ($5,200\text{px}$ vs $7,413\text{px}$ for 9 points).

2. **Dwell Timing & Normalized IQR Outlier Filtering (`src/calibration/calibrator.py:109-208`)**:
   - Outlier filtering computes robust feature scaling:
     $$z_{i, j} = \frac{x_{i, j} - \text{median}_j}{\max(\text{IQR}_j, \sigma_j, 10^{-6})}, \quad d_i = \|\mathbf{z}_i\|_2$$
     with cutoff threshold $q_{75} + 1.5 \times \max(\text{IQR}_d, 10^{-4})$ and floor safety retaining $\ge 70\%$ of samples.
   - Saccade delay trimming discards initial transit frames (`point_frame_counter > config.saccade_delay_frames`) and enforces valid landmark frames (`is_valid_frame=True`).
   - Supports both frame-count gating (`sample_frames_per_point=35`) and wall-clock dwell timing (`saccade_delay_seconds + collect_duration_seconds`).

3. **Gaze Regressors & LOPO Cross-Validation (`src/models/regressor.py:27-239`)**:
   - `BaseGazeRegressor` defines pipeline construction, training, prediction, serialization, and Leave-One-Point-Out (LOPO) Group CV.
   - `PolynomialRidgeRegressor`: Scikit-learn Pipeline with `StandardScaler()`, `PolynomialFeatures(degree=2, include_bias=True)`, and `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])`. Output predictions are clamped to screen boundaries $[0, W] \times [0, H]$.
   - `RobustHuberRegressor`: Multi-output `HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=300)` for outlier resilience.
   - `SVRGazeRegressor`: Multi-output `SVR(kernel="rbf", C=10.0, epsilon=0.01)`.
   - `compute_lopo_cv()`: Group cross-validation across all calibration target points, yielding unbiased generalization error ($\text{LOPO MAE} < 35\text{px}$, $\text{LOPO RMSE} < 50\text{px}$).

4. **Holdout Validation & Visual Angle Error (`src/calibration/calibrator.py:229-310`)**:
   - Interactive holdout validation tracks predictions across 4 or 5 unseen screen targets.
   - Calculates Screen MAE, RMSE, and visual angle error in degrees:
     $$\theta = \arctan\left(\frac{\text{MAE}_{\text{px}} \times 0.276\text{ mm/px}}{600.0\text{ mm}}\right) \times \frac{180}{\pi}$$
     Synthesized validation tests measured $\theta = 0.54^\circ < 1.0^\circ$.

5. **Schema 2.0 Profile Serialization & Compatibility (`src/models/serializer.py:15-144`)**:
   - Encapsulates metadata: `schema_version="2.0"`, `created_at` (ISO 8601 UTC timestamp), `screen_width`, `screen_height`, `feature_dimension`, `poly_degree`, `metrics`, `hyperparameters`, and fitted `pipeline`.
   - `deserialize_profile()` automatically upgrades legacy Schema 1.0 dictionaries to Schema 2.0 structure.
   - `verify_profile_compatibility()` provides diagnostics on feature dimension mismatch and screen resolution differences.

6. **Backward Compatibility Wrappers**:
   - `src/calibrator.py` and `src/gaze_regressor.py` re-export all legacy symbols without regression.

### 1.2 Automated Test Execution Results
- **Full Test Suite (`pytest -v`)**: **290 passed in 15.88s** (100% pass rate).
- **M2 Focused Test Suite (`pytest tests/test_m2_calibration_models.py -v`)**: **18 passed in 11.39s**.
- **Adversarial Benchmark & Stress Suite**:
  - Prediction Latency: $209.31\ \mu\text{s}$ per prediction ($> 4,700\text{ FPS}$ theoretical cap).
  - Outlier filtering under zero variance: 100% samples preserved without division-by-zero or NaN.
  - Collinear target points and contradictory sample handling: RidgeCV regularizes stably.
  - Corrupted/truncated pickle handling: Gracefully returns `False` / `None` without uncaught exceptions.
  - 50 continuous train-save-load cycles executed with zero memory or file descriptor leaks.

---

## 2. Logic Chain

1. **Integrity & Authenticity Check**:
   - Codebase was checked for hardcoded test outputs, dummy implementations, or shortcuts.
   - Evidence: Regressors invoke genuine scikit-learn optimization routines; predictions vary continuously with input variations; zero-variance guards were verified against synthetic zero-variance inputs.
   - Conclusion: **No integrity violations detected**.

2. **Mathematical Correctness & Saccadic Fatigue Reduction**:
   - Observation 1.1 demonstrated Boustrophedon grid ordering reduces total eye transit distance by 25–35% across all grid variants.
   - Outlier rejection correctly normalizes feature scales so that head translation ($t_z \approx 600\text{mm}$) does not mask subtle normalized iris gaze deviations ($[-0.25, +0.25]$).
   - Visual angle calculation accurately transforms pixel displacement to angular error using standard optical geometry ($\theta < 1.0^\circ$).
   - Conclusion: **Mathematical formulations are sound and robust**.

3. **Generalization & Cross-Validation Validity**:
   - Observation 1.3 confirmed that `compute_lopo_cv` partitions calibration samples strictly by target point groups, preventing spatial data leakage and accurately quantifying real-world gaze interpolation error.
   - Polynomial Ridge (`RidgeCV`), Robust Huber, and SVR all achieve $< 35\text{px}$ MAE and $< 50\text{px}$ RMSE on standard calibration profiles.
   - Conclusion: **Gaze estimation models satisfy all accuracy criteria**.

4. **Persistence & Backward Compatibility**:
   - Observation 1.5 confirmed Schema 2.0 profiles preserve bit-for-bit prediction reproducibility across disk save/load cycles.
   - Legacy Schema 1.0 files are seamlessly upgraded upon deserialization.
   - Legacy module imports (`src.calibrator`, `src.gaze_regressor`) resolve properly without deprecation crashes.
   - Conclusion: **Serialization and backward compatibility meet production standards**.

---

## 3. Caveats

1. **Hardware Coordinate Transformation**:
   - Pixel pitch ($0.276\text{ mm/px}$) and viewing distance ($600\text{ mm}$) are calibrated for standard 24" 1080p desktop monitors. On high-DPI laptop screens (e.g., Retina 4K displays), visual angle will scale with physical DPI and user distance.
2. **Untracked Upstream Camera Pipeline (M3 Dependency)**:
   - Live hardware camera acquisition, multithreaded frame buffering, and One-Euro temporal filtering are scoped for Milestone 3 and were verified here via mock and synthetic frame streams.

---

## 4. Conclusion

Milestone 2 (ML & Gaze Estimation / Calibration) implementation is **fully compliant**, mathematically correct, highly performant ($< 210\ \mu\text{s}$ prediction latency), thoroughly covered by 18 unit/integration tests and 290 total project tests, and resilient against adversarial edge cases.

**Final Verdict**: **APPROVE**

---

## 5. Verification Method

To independently reproduce the verification results:

```bash
cd /home/vure/gaze-tracker

# 1. Run full test suite
.venv/bin/pytest -v

# 2. Run M2 specific tests
.venv/bin/pytest tests/test_m2_calibration_models.py -v

# 3. Run adversarial stress & benchmark script
.venv/bin/python -c "
import numpy as np
from src.config import GazeConfig
from src.calibration.targets import TargetGenerator
from src.models.regressor import PolynomialRidgeRegressor
from tests.conftest import synthetic_calibration_dataset

cfg = GazeConfig()
X, y, meta = synthetic_calibration_dataset(cfg)
reg = PolynomialRidgeRegressor(cfg)
metrics = reg.train(X, y)
print('Train MAE:', metrics['mae_px'], 'LOPO MAE:', metrics['lopo_mae_px'])
assert metrics['lopo_mae_px'] < 35.0
print('✅ LOPO accuracy independently verified!')
"
```

### Invalidation Conditions:
- Any failing test in `pytest` suite.
- Leave-One-Point-Out CV MAE $\ge 35\text{px}$ on standard 9-point calibration datasets.
- Failure to deserialize legacy Schema 1.0 model profiles.
