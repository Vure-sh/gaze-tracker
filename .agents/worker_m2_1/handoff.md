# Milestone 2 (ML & Gaze Estimation / Calibration) Hard Handoff Report

**Date**: 2026-08-30  
**Author**: ML & Calibration Specialist (`worker_m2_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Milestone**: M2 (ML & Gaze Estimation / Calibration)  
**Status**: COMPLETE — Hard Handoff  

---

## 1. Observation

### 1.1 Baseline Audit & Identified Architectural Needs
1. **Legacy Grid Sequencing (`src/calibrator.py:43-59`)**:
   - Grid points were generated in static raster scan order ($0 \to 1 \to 2$, then jumping across the screen to start row 2).
   - This caused large diagonal saccadic transitions, eye fatigue, and prolonged fixation settling latency.
2. **Outlier Filtering Scale Disparity (`src/calibrator.py:88-105`)**:
   - Outlier Euclidean distance was computed on raw unnormalized vectors where translation ($t_z \approx 600\text{mm}$) dominated normalized iris gaze features ($[-0.25, +0.25]$), rendering IQR outlier filtering ineffective against true iris gaze noise.
3. **Model Architecture & Validation Metrics (`src/gaze_regressor.py:21-55`)**:
   - Legacy code used a single monolithic `Ridge(alpha=1.0)` model without cross-validated alpha selection (`RidgeCV`), robust loss options (`HuberRegressor`), or alternative estimator backends (`SVR`).
   - Training error was computed purely on resubstitution samples $(X, y)$, lacking spatial generalization metrics (Leave-One-Point-Out Group CV) and live visual angle error validation ($\theta < 1.0^\circ$).
4. **Model Serialization & Schema Versioning (`src/gaze_regressor.py:75-111`)**:
   - Raw dictionary dumps lacked schema version tags, timestamps, display resolution portability checks, and feature dimension validation guards.

---

## 2. Logic Chain

1. **Boustrophedon (Serpentine) Sequence Generator (`src/calibration/targets.py`)**:
   - *Observation 1.1*: Raster scanning caused long cross-screen saccades.
   - *Logic Step*: Implemented `TargetGenerator` with serpentine row ordering (Row 0: Left $\to$ Right, Row 1: Right $\to$ Left, Row 2: Left $\to$ Right) for 9, 13, and 16-point grids.
   - *Result*: Saccade trajectory distance between consecutive targets is minimized, reducing eye fatigue and improving fixation data stability.

2. **Normalized IQR Outlier Rejection (`src/calibration/calibrator.py`)**:
   - *Observation 1.2*: Disparate feature scales prevented effective iris gaze outlier detection.
   - *Logic Step*: Normalized each feature dimension using robust median and IQR/StdDev scaling before calculating Euclidean distances:
     $$z_{i, j} = \frac{x_{i, j} - \text{median}_j}{\max(\text{IQR}_j, \sigma_j, 10^{-6})}, \quad d_i = \|\mathbf{z}_i\|_2$$
     Filtered samples beyond $q_{75} + 1.5 \times \text{IQR}_d$ with zero-variance safety guards.
   - *Result*: Accurately discards gaze look-away outliers while preserving 100% of clean, tight fixation samples.

3. **Modular Regressors & LOPO Cross-Validation (`src/models/regressor.py`)**:
   - *Observation 1.3*: Resubstitution error underestimates spatial interpolation error.
   - *Logic Step*: Created `BaseGazeRegressor` abstract base class defining `train()`, `predict()`, `save_profile()`, `load_profile()`, and `compute_lopo_cv()`.
     - `PolynomialRidgeRegressor`: Degree-2 polynomial with `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])` and output coordinate clamping to $[0, W] \times [0, H]$.
     - `RobustHuberRegressor`: Degree-2 polynomial with `MultiOutputRegressor(HuberRegressor)` for outlier-robust fitting.
     - `SVRGazeRegressor`: Multi-output Support Vector Regression backend.
     - `GazeRegressionModel`: Backward-compatible subclass mapping to `PolynomialRidgeRegressor`.
   - *Result*: Training reports unbiased generalization LOPO MAE $< 35\text{px}$ and LOPO RMSE $< 50\text{px}$.

4. **Dedicated Interactive Holdout Validation Mode (`src/calibration/calibrator.py`)**:
   - *Logic Step*: Added `start_validation(mode="4_points" | "5_points")` and `_process_validation_frame()`.
   - *Metrics*: Calculates live Screen MAE (px), RMSE (px), and visual angle error in degrees:
     $$\theta = \arctan\left(\frac{\text{MAE}_{\text{px}} \times \text{pitch}_{\text{mm}}}{D_{\text{mm}}}\right) \times \frac{180}{\pi} < 1.0^\circ$$

5. **Schema 2.0 Profile Serialization (`src/models/serializer.py`)**:
   - *Observation 1.4*: Unversioned serialization prevented validation and compatibility checks.
   - *Logic Step*: Implemented `ModelProfileSerializer` with schema version `"2.0"`, timestamps, display resolution metadata, feature dimension verification, bit-for-bit prediction reproducibility, and automatic upgrade loader for legacy Schema 1.0 pickle files.

6. **Backward Compatibility Wrappers**:
   - `src/calibrator.py` and `src/gaze_regressor.py` maintain full API compatibility for legacy imports.

---

## 3. Caveats

1. **Visual Angle Assumptions**:
   - Visual angle calculation assumes standard 24" 1080p monitor pixel pitch ($0.276\text{ mm/px}$) and user eye-to-screen distance $D = 600\text{ mm}$. If physical screen dimensions or distance differ significantly, visual angle scales proportionally with distance.
2. **Head Movement During Holdout Validation**:
   - Holdout validation assumes user maintains natural head pose within $\pm 15^\circ$ during the verification phase.

---

## 4. Conclusion

Milestone 2 is complete and thoroughly validated:
- `src/calibration/targets.py`: Implemented 9, 13, 16-point grids with Boustrophedon sequencing and 4/5-point validation holdout patterns.
- `src/calibration/calibrator.py` & `src/calibrator.py`: Implemented wall-clock dwell timing, saccade delay trimming, normalized IQR outlier rejection, and holdout validation.
- `src/models/regressor.py` & `src/gaze_regressor.py`: Implemented `BaseGazeRegressor`, `PolynomialRidgeRegressor` (RidgeCV), `RobustHuberRegressor`, `SVRGazeRegressor`, and LOPO Group CV.
- `src/models/serializer.py`: Implemented Schema 2.0 serialization, compatibility validation, and legacy upgrade loader.
- `tests/test_m2_calibration_models.py`: 18 comprehensive tests.
- Full test suite: **290 tests passing (100% pass rate in 13.33s)**.

---

## 5. Verification Method

### 5.1 Run Full Test Suite
```bash
cd /home/vure/gaze-tracker
.venv/bin/pytest -v
```
*Expected Output*: `290 passed in ~13s` with exit code 0.

### 5.2 Specific M2 Verification Commands

1. **Verify Boustrophedon Grid Ordering & Validation Targets**:
```bash
.venv/bin/python -c "
from src.config import GazeConfig
from src.calibration.targets import TargetGenerator

cfg = GazeConfig()
pts9 = TargetGenerator.generate_points(cfg, '9_points', boustrophedon=True)
val4 = TargetGenerator.generate_validation_points(cfg, '4_points')

print('9-Point Boustrophedon Sequence:', pts9)
print('4-Point Holdout Targets:', val4)
assert len(pts9) == 9 and len(val4) == 4
# Row 0 increases in X, Row 1 decreases in X
assert pts9[0][0] < pts9[2][0] and pts9[3][0] > pts9[5][0]
print('✅ Boustrophedon grid sequencing verified!')
"
```

2. **Verify Regressor LOPO CV Accuracy & Clamping**:
```bash
.venv/bin/python -c "
import numpy as np
from src.config import GazeConfig
from src.models.regressor import PolynomialRidgeRegressor, RobustHuberRegressor
from tests.conftest import create_synthetic_landmarks

cfg = GazeConfig()
np.random.seed(42)

# Fit on synthetic dataset
from tests.conftest import synthetic_calibration_dataset
X, y, meta = synthetic_calibration_dataset(cfg)

reg = PolynomialRidgeRegressor(cfg)
metrics = reg.train(X, y)
print('Polynomial Ridge Metrics:', metrics)
assert metrics['mae_px'] < 35.0
assert metrics['lopo_mae_px'] < 35.0
assert metrics['lopo_rmse_px'] < 50.0

pred = reg.predict(X[0])
print('Prediction for sample 0:', pred)
assert 0.0 <= pred[0] <= cfg.screen_width and 0.0 <= pred[1] <= cfg.screen_height
print('✅ Regressor LOPO CV accuracy verified!')
"
```

3. **Verify Schema 2.0 Serialization & Round-Trip Fidelity**:
```bash
.venv/bin/python -c "
import tempfile, os
from src.config import GazeConfig
from src.models.regressor import PolynomialRidgeRegressor
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION
from tests.conftest import synthetic_calibration_dataset

cfg = GazeConfig()
X, y, meta = synthetic_calibration_dataset(cfg)
reg = PolynomialRidgeRegressor(cfg)
reg.train(X, y)

with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
    path = tmp.name

try:
    reg.save_profile(path)
    profile = ModelProfileSerializer.deserialize_profile(path)
    assert profile['schema_version'] == CURRENT_SCHEMA_VERSION
    
    loaded = PolynomialRidgeRegressor(cfg)
    assert loaded.load_profile(path) is True
    
    # Exact prediction match
    assert reg.predict(X[0]) == loaded.predict(X[0])
    print('✅ Schema 2.0 serialization roundtrip verified!')
finally:
    if os.path.exists(path): os.remove(path)
"
```

### 5.3 Invalidation Conditions
- If `pytest` has any failing tests (exit code != 0), this milestone is invalidated.
- If Leave-One-Point-Out cross-validation MAE exceeds 35px on standard calibration datasets, this milestone is invalidated.
- If Schema 2.0 serialization loses prediction precision upon deserialization, this milestone is invalidated.
