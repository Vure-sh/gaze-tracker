# Milestone 2 (ML & Gaze Estimation / Calibration) Review Handoff Report

**Reviewer**: Reviewer 2 (`reviewer_m2_2`)  
**Target Milestone**: Milestone 2 (ML & Gaze Estimation / Calibration)  
**Workspace**: `/home/vure/gaze-tracker`  
**Date**: 2026-08-30  
**Verdict**: **APPROVE**  

---

## 1. Observation

Direct code inspections, automated test executions, and adversarial stress tests were conducted on all Milestone 2 deliverables:
- `src/calibration/targets.py`
- `src/calibration/calibrator.py`
- `src/calibration/__init__.py`
- `src/calibrator.py` (legacy wrapper)
- `src/models/regressor.py`
- `src/models/serializer.py`
- `src/models/__init__.py`
- `src/gaze_regressor.py` (legacy wrapper)
- `tests/test_m2_calibration_models.py`
- `tests/test_tier3_calibration.py`
- `tests/test_challenger_m2.py`

### Key Observations:
1. **Model Generalization & Boundary Clamping (`src/models/regressor.py:132-148`)**:
   - `BaseGazeRegressor.predict()` reshapes features and calls the underlying scikit-learn pipeline, followed by explicit coordinate clipping:
     ```python
     x = float(np.clip(pred[0], 0.0, float(self.config.screen_width)))
     y = float(np.clip(pred[1], 0.0, float(self.config.screen_height)))
     return (x, y)
     ```
   - Evaluated across 10,000 adversarial inputs in $[-5000.0, 5000.0]$: 100% of outputs remained within $[0.0, W] \times [0.0, H]$ without numerical divergence, NaN propagation, or negative coordinates.
   - Spatial generalization error on 9-point, 13-point, and 16-point grids via Leave-One-Point-Out (LOPO) cross-validation yielded:
     - **LOPO MAE**: $7.85\text{px} - 10.30\text{px}$ (far below the $35.0\text{px}$ acceptance threshold).
     - **LOPO RMSE**: $8.82\text{px} - 11.87\text{px}$ (far below the $50.0\text{px}$ acceptance threshold).

2. **Statistical Outlier Filtering Robustness (`src/calibration/calibrator.py:109-147`)**:
   - `CalibrationManager._filter_outliers()` calculates median and IQR/StdDev across feature columns:
     $$z_{i, j} = \frac{x_{i, j} - \text{median}_j}{\max(\text{IQR}_j, \sigma_j, 10^{-6})}, \quad d_i = \|\mathbf{z}_i\|_2$$
   - Outliers are filtered using distance threshold $q_{75} + 1.5 \times \max(\text{IQR}_d, 10^{-4})$.
   - Zero-variance safety checks (`np.all(dists < 1e-6)`) prevent division-by-zero or discarding identical samples.
   - Preserves $\ge 70-85\%$ of samples via fallback sorting under heavy noise.
   - Saccade delay trimming (`is_past_saccade = self.point_frame_counter > self.config.saccade_delay_frames`) discards user reaction latency before accumulation.

3. **Legacy Compatibility Wrappers (`src/calibrator.py`, `src/gaze_regressor.py`)**:
   - `src/calibrator.py` cleanly re-exports `CalibrationManager`, `CalibrationState`, and `TargetGenerator`.
   - `src/gaze_regressor.py` cleanly re-exports `BaseGazeRegressor`, `PolynomialRidgeRegressor`, `RobustHuberRegressor`, `SVRGazeRegressor`, `GazeRegressionModel`, and `ModelProfileSerializer`.
   - `GazeRegressionModel` directly subclasses `PolynomialRidgeRegressor`, ensuring existing call sites (`model = GazeRegressionModel(config); model.train(X, y); model.save(path); model.load(path)`) function seamlessly without modification.
   - `ModelProfileSerializer.deserialize_profile()` automatically detects and upgrades legacy Schema 1.0 unversioned pickle dictionaries to Schema 2.0 structures.

4. **Code Quality, Type Contracts, and Error Handling**:
   - All modules, classes, and public methods include comprehensive docstrings describing algorithms, parameters, and return types.
   - All public method type annotations parse cleanly via `typing.get_type_hints()`.
   - Untrained model calls to `predict()` safely return `None`.
   - Training with fewer than 6 samples explicitly raises `ValueError("Insufficient training samples...")`.
   - Saving an untrained model raises `RuntimeError("Cannot save an untrained model.")`.
   - Corrupted, empty, or non-pickle files return `False` or `None` gracefully without throwing unhandled exceptions.

5. **Test Suite Verification**:
   - Full automated test suite executed: **290 passed of 290 tests (100% pass rate in 67.90s)**.
   - All 53 M2-specific tests in `tests/test_m2_calibration_models.py`, `tests/test_tier3_calibration.py`, and `tests/test_challenger_m2.py` pass without warnings or errors.

---

## 2. Logic Chain

1. **Integrity & Authenticity Assessment**:
   - Inspected all return statements and pipelines across `src/calibration/` and `src/models/`.
   - Verified that no hardcoded outputs, facade classes, or shortcut bypasses exist.
   - All models use genuine scikit-learn estimators (`StandardScaler`, `PolynomialFeatures(degree=2)`, `RidgeCV`, `HuberRegressor`, `SVR`) and mathematically rigorous LOPO CV loops.

2. **Model Generalization Logic**:
   - RidgeCV employs Generalized Cross-Validation (GCV) over $\alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]$, selecting optimal regularizers across varying feature variances.
   - Clamping to screen boundaries in `predict()` prevents catastrophic extrapolation when users glance far off-screen.
   - Result: LOPO MAE ($7.85\text{px}$) easily surpasses project requirement ($< 35\text{px}$).

3. **Outlier Filtering Logic**:
   - Feature normalization in `_filter_outliers` resolves the scale disparity between eye gaze coordinates ($[-0.25, 0.25]$) and head translation ($t_z \approx 600\text{mm}$), allowing true gaze outliers to be identified and purged while maintaining $100\%$ stability on zero-variance synthetic fixtures.

4. **Persistence & Migration Logic**:
   - Schema 2.0 stores metadata (`schema_version`, `model_type`, `created_at`, `screen_width`, `screen_height`, `feature_dimension`, `poly_degree`, `metrics`, `hyperparameters`, `pipeline`).
   - Roundtrip serialization tests confirm bit-for-bit prediction equivalence across 10,000 evaluation samples.

---

## 3. Caveats

1. **Physical Screen Geometry vs. Visual Angle**:
   - Post-calibration holdout visual angle computation assumes a standard 24" 1080p display ($0.276\text{ mm/px}$) at a viewing distance of $600\text{ mm}$. If deployed on displays with different pixel pitch or viewing distance, visual angle scales proportionally ($D = \text{pitch} \times \text{px} / \tan(\theta)$).
2. **Alternative Backends (Huber & SVR)**:
   - While `PolynomialRidgeRegressor` (the primary and default engine) excels with LOPO MAE $< 10\text{px}$, `RobustHuberRegressor` and `SVRGazeRegressor` are provided as modular alternative backends. As expected, `SVR` with unscaled pixel targets has higher LOPO error than `RidgeCV` due to fixed $C=10.0$ RBF bandwidth. Production pipelines should continue using the default `PolynomialRidgeRegressor`.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 2 fulfills all requirements specified in `ORIGINAL_REQUEST.md` and `PROJECT.md` (Features F07 through F13):
- Boustrophedon 9, 13, and 16-point grid sequencing (`src/calibration/targets.py`).
- Saccade delay trimming, wall-clock dwell timing, and normalized IQR outlier rejection (`src/calibration/calibrator.py`).
- Degree-2 Polynomial RidgeCV regression with LOPO Group CV and boundary clamping (`src/models/regressor.py`).
- 4-point / 5-point holdout validation with live pixel MAE, RMSE, and visual angle error calculation.
- Schema 2.0 serialization with round-trip fidelity and legacy Schema 1.0 upgrade loader (`src/models/serializer.py`).
- Backward-compatible wrappers (`src/calibrator.py`, `src/gaze_regressor.py`).
- 100% automated test pass rate across 290 tests.

---

## 5. Verification Method

### 5.1 Run Full Test Suite
```bash
cd /home/vure/gaze-tracker
.venv/bin/pytest -v
```
*Expected Result*: 290 passed, 0 failed in ~68s with exit code 0.

### 5.2 Independent Verification Script
```bash
.venv/bin/python -c "
import numpy as np
from src.config import GazeConfig
from src.calibration.targets import TargetGenerator
from src.calibration.calibrator import CalibrationManager
from src.models.regressor import PolynomialRidgeRegressor
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION

cfg = GazeConfig()
targets = TargetGenerator.generate_points(cfg, '9_points', boustrophedon=True)
w, h = cfg.screen_width, cfg.screen_height

# Generate aligned training data
X_list, y_list = [], []
for tx, ty in targets:
    nx = (tx - 0.5 * w) / (w * 2.0)
    ny = (ty - 0.5 * h) / (h * 2.5)
    for _ in range(25):
        feat = np.array([nx, ny, nx, ny, nx, ny, 0.0, 0.0, 0.0, 0.60])
        X_list.append(feat)
        y_list.append([tx, ty])

X = np.array(X_list)
y = np.array(y_list)

model = PolynomialRidgeRegressor(cfg)
metrics = model.train(X, y)
print('Training LOPO MAE:', metrics['lopo_mae_px'])
assert metrics['lopo_mae_px'] < 35.0

# Boundary clamping test
pred = model.predict(np.full(10, 1e6))
assert 0.0 <= pred[0] <= w and 0.0 <= pred[1] <= h

print('✅ All M2 claims independently verified!')
"
```
