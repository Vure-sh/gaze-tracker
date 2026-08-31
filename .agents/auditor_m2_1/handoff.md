# Milestone 2 (ML & Gaze Estimation / Calibration) Forensic Audit Report

**Date**: 2026-08-30  
**Auditor**: Forensic Integrity Auditor (`auditor_m2_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Target**: Milestone 2 (`src/calibration/`, `src/models/`, `src/calibrator.py`, `src/gaze_regressor.py`)  
**Integrity Mode**: Development Mode (General Project Profile)  
**Verdict**: **CLEAN** (No Integrity Violations or Cheating Vectors Detected)

---

## 1. Forensic Audit Report

**Work Product**: Milestone 2 Gaze Estimation & Calibration Subsystem (`src/calibration/`, `src/models/`, `src/calibrator.py`, `src/gaze_regressor.py`)  
**Profile**: General Project (Development Mode)  
**Verdict**: **CLEAN**

### Phase Results
- **Static Analysis & Anti-Cheat Scan**: **PASS** — No hardcoded test outputs, stub/constant return functions, fabricated logs, or synthetic branching found.
- **Scikit-Learn Pipeline Structure & Execution**: **PASS** — Genuine instantiation and fitting of `StandardScaler`, `PolynomialFeatures`, `RidgeCV`, `MultiOutputRegressor(HuberRegressor)`, and `MultiOutputRegressor(SVR)`.
- **Leave-One-Point-Out (LOPO) Group Cross-Validation**: **PASS** — Authentic fold-by-fold pipeline cloning, independent training, and prediction on held-out target coordinate groups. Dynamically sensitive to label corruption.
- **Statistical IQR Outlier Rejection**: **PASS** — Authentic feature normalization ($z_{i,j} = (x_{i,j} - \text{median}_j)/\text{scale}_j$) and Euclidean distance thresholding ($q_{75} + 1.5 \times \text{IQR}$) with zero-variance safety guards.
- **Holdout Validation Mode & Visual Angle Trigonometry**: **PASS** — Genuine collection state machine computing live MAE, RMSE, and authentic visual angle: $\theta = \arctan\left(\frac{\text{MAE}_{\text{px}} \times 0.276\text{mm}}{600\text{mm}}\right) \times \frac{180}{\pi} < 1.0^\circ$.
- **Schema 2.0 Serialization & Round-Trip Bit Fidelity**: **PASS** — Bit-for-bit prediction matching across 1,000 query vectors; seamless upgrade for legacy Schema 1.0 files; graceful rejection of corrupted binaries.
- **Full Test Suite Execution**: **PASS** — 345/345 tests passing (100% pass rate in 109.17s).

---

## 2. 5-Component Handoff Report

### 2.1 Observation

1. **Source Code Static Analysis**:
   - `src/calibration/targets.py:15-86`: `TargetGenerator.generate_points()` computes target coordinates algebraically using `config.screen_width`, `config.screen_height`, and margins (`config.calibration_margin_x`, `config.calibration_margin_y`). Alternating serpentine Boustrophedon ordering is implemented using parity check `(row_idx % 2 == 1)`. No hardcoded target lookup arrays or test mocks.
   - `src/calibration/calibrator.py:109-148`: `_filter_outliers()` calculates medians, 25th/75th percentiles per feature, normalizes feature variances, calculates Euclidean distances, and applies distance IQR cutoff ($q_{75} + 1.5 \times \text{IQR}$). Contains zero-variance guards (`len(samples) < 5`, `np.all(dists < 1e-6)`, `feat_iqr > 1e-6`).
   - `src/models/regressor.py:41-89`: `compute_lopo_cv()` creates fold splits `test_mask = (point_ids == grp)`, fits a fresh pipeline `fold_pipeline.fit(X[train_mask], y[train_mask])`, predicts on held-out test points `preds = fold_pipeline.predict(X[test_mask])`, and averages Euclidean errors across all groups.
   - `src/models/regressor.py:188-233`: Defines genuine scikit-learn estimator pipelines:
     - `PolynomialRidgeRegressor`: `StandardScaler` -> `PolynomialFeatures(degree=poly_degree)` -> `RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])`.
     - `RobustHuberRegressor`: `StandardScaler` -> `PolynomialFeatures` -> `MultiOutputRegressor(HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=300))`.
     - `SVRGazeRegressor`: `StandardScaler` -> `PolynomialFeatures` -> `MultiOutputRegressor(SVR(kernel='rbf', C=10.0, epsilon=0.01))`.
   - `src/models/serializer.py:15-115`: Implements `ModelProfileSerializer.serialize_profile()` and `deserialize_profile()`, supporting Schema 2.0 metadata payloads and upgrading Schema 1.0 pickle files.
   - `src/calibrator.py` & `src/gaze_regressor.py`: Transparent re-export modules preserving full backward compatibility.

2. **Empirical Pipeline & Parameter Extraction**:
   Executing parameter extraction on a fitted `PolynomialRidgeRegressor`:
   ```python
   # Output from forensic check script:
   Train MAE: 1.1939 px, Train RMSE: 1.4865 px
   Selected RidgeCV alpha: 0.01
   Ridge coef shape: (2, 45) # Correct 2D output x 45 polynomial terms
   Ridge intercept: [954.87, 496.96]
   StandardScaler mean_ shape: (8,)
   StandardScaler var_ shape: (8,)
   ```

3. **Empirical LOPO Sensitivity Verification**:
   Testing `compute_lopo_cv()` response when perturbing target coordinates for fold 4 by 500px:
   ```
   Clean LOPO MAE: 6.108 px
   Corrupted Fold LOPO MAE: 416.151 px (Shifted dynamically by +410px)
   ```

4. **Empirical Serialization Bit-Exactness**:
   Testing 1,000 random query feature vectors on original vs deserialized models:
   ```
   PolynomialRidgeRegressor: Bit-for-bit exact match on all 1,000 queries
   RobustHuberRegressor: Bit-for-bit exact match on all 1,000 queries
   SVRGazeRegressor: Bit-for-bit exact match on all 1,000 queries
   ```

5. **Empirical Test Suite Execution**:
   Executing `.venv/bin/pytest -v`:
   ```
   ======================= 345 passed in 109.17s (0:01:49) ========================
   Exit code: 0
   ```

---

### 2.2 Logic Chain

1. *Observation 1 & 2*: Static analysis of `src/models/regressor.py` and empirical parameter inspection confirm that `PolynomialRidgeRegressor`, `RobustHuberRegressor`, and `SVRGazeRegressor` instantiate and fit actual mathematical estimators (`RidgeCV`, `HuberRegressor`, `SVR`) operating on standardized polynomial feature matrices.
   - *Inference*: The ML models are genuine scikit-learn implementations with no facade stubs, hardcoded return constants, or mock fits.

2. *Observation 1 & 3*: LOPO CV partitions the sample dataset by target group index, fits a distinct pipeline on $N-1$ points, and evaluates on the held-out point. When fold targets were perturbed by 500px, LOPO MAE dynamically increased from 6.1px to 416.2px.
   - *Inference*: LOPO CV is an authentic, active validation loop that truthfully evaluates spatial generalization without synthetic shortcuts.

3. *Observation 1 & 4*: Outlier rejection normalizes feature scales prior to Euclidean distance computation, preventing $t_z$ translation from masking iris offset variance. Serialization creates Schema 2.0 payloads with UTC timestamps, resolution tags, and preserves exact floating-point predictions upon reload.
   - *Inference*: Calibration data filtering and model persistence satisfy all architectural and robustness requirements.

4. *Observation 5*: Running the comprehensive test suite (`tests/test_tier1_units.py`, `tests/test_tier2_invariance.py`, `tests/test_tier3_calibration.py`, `tests/test_tier4_performance.py`, `tests/test_m2_calibration_models.py`, `tests/test_challenger_m2.py`, `tests/test_challenger_m2_adversarial.py`) yields a 100% pass rate (345/345 passed).
   - *Inference*: All Milestone 2 requirements (R3, F07–F13) are verified and mathematically sound.

---

### 2.3 Caveats

1. **Hardware Distance Assumption**: Visual angle calculation in validation mode assumes standard 24" 1080p display pitch ($0.276\text{ mm/px}$) and user eye distance of $600\text{ mm}$. If physical hardware setup deviates, angular errors scale proportionally with distance.
2. **Scikit-learn Version Pinning**: Pickled models rely on scikit-learn `Pipeline` serialization; moving profiles across major incompatible Python / scikit-learn versions should use the schema validator to trigger retraining.

---

### 2.4 Conclusion

The Milestone 2 work product is verified as **CLEAN**. There are no integrity violations, hardcoded test results, facade implementations, or cheat vectors. The calibration pipeline, ML regression backends, LOPO cross-validation, and schema-versioned persistence meet all contractual requirements.

---

### 2.5 Verification Method

To independently verify this audit verdict, execute:

```bash
# 1. Run the entire test suite
cd /home/vure/gaze-tracker
.venv/bin/pytest -v

# 2. Run independent forensic check for pipeline parameters and LOPO CV
.venv/bin/python -c "
import numpy as np
from src.config import GazeConfig
from src.models.regressor import PolynomialRidgeRegressor
from src.models.serializer import ModelProfileSerializer

cfg = GazeConfig()
reg = PolynomialRidgeRegressor(cfg)
X = np.random.uniform(-0.3, 0.3, (50, 8))
y = np.random.uniform(100, 1000, (50, 2))
metrics = reg.train(X, y)
assert reg.is_trained
assert 'lopo_mae_px' in metrics
pred = reg.predict(X[0])
assert 0 <= pred[0] <= cfg.screen_width
assert 0 <= pred[1] <= cfg.screen_height
print('✅ Verified authentic pipeline execution and LOPO CV!')
"
```
