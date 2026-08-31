# Milestone 1 (CV & Robust Feature Engineering) Review & Adversarial Challenge Report

**Date**: 2026-08-30  
**Reviewer**: Quality Reviewer & Adversarial Critic (`reviewer_m1_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Milestone**: M1 (CV & Robust Feature Engineering)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Direct Source Code Observations
1. **Orthonormal Dual-Eye Normalization (`src/cv/eye_extractor.py:103-164`)**:
   - Left eye inner canthus (133) and outer canthus (33) direction vector:
     $$\vec{w}_L = \vec{p}_{133} - \vec{p}_{33}$$
   - Right eye inner canthus (362) and outer canthus (263) direction vector:
     $$\vec{w}_R = \vec{p}_{263} - \vec{p}_{362}$$
   - Both canthal vectors point strictly left-to-right ($+X$ in camera coordinates), completely resolving the legacy symmetric sign cancellation bug.
   - Orthonormal basis constructed via:
     $$\vec{u} = \frac{\vec{w}}{\|\vec{w}\|}, \quad \vec{u}_{\perp} = \begin{bmatrix} -u_y \\ u_x \end{bmatrix}$$
   - Normalized coordinates are zero-centered at fissure midpoint $\vec{p}_{\text{mid}} = (\vec{p}_{\text{inner}} + \vec{p}_{\text{outer}}) / 2$:
     $$norm\_x = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}}{\|\vec{w}\|}, \quad norm\_y = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}_{\perp}}{\|\vec{w}\|}$$
   - Zero-division guard: `if eye_width < 1e-6: eye_width = 1e-6`.

2. **Corrected Anthropometric 3D Head Pose Model (`src/cv/head_pose.py:18-25, 96-107`)**:
   - Corrected 3D anthropometric face model:
     ```python
     MODEL_POINTS_CORRECTED = np.array([
         (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
         (0.0, 100.0, -20.0),      # Chin (Landmark 152) -> +Y is DOWN
         (-65.0, -50.0, -40.0),    # Left eye outer corner (Landmark 33)
         (65.0, -50.0, -40.0),     # Right eye outer corner (Landmark 263)
         (-40.0, 50.0, -30.0),     # Left mouth corner (Landmark 61)
         (40.0, 50.0, -30.0)       # Right mouth corner (Landmark 291)
     ], dtype=np.float64)
     ```
   - Aligned directly with OpenCV camera optical coordinates ($+X$ right, $+Y$ down, $+Z$ forward).
   - Upright resting face yields Euler angles $(\text{Pitch}, \text{Yaw}, \text{Roll}) = (0.00^\circ, 0.00^\circ, 0.00^\circ)$.
   - Eliminates the legacy $-180^\circ$ pitch inversion and branch-cut discontinuities.

3. **6-Point EAR & Dynamic Blink Detection (`src/cv/eye_extractor.py:22-41, 70-90`)**:
   - 6-point formulation using landmarks $[33, 133, 160, 144, 158, 153]$ (left) and $[362, 263, 385, 380, 387, 373]$ (right):
     $$\text{EAR} = \frac{\|p_{\text{top1}} - p_{\text{bottom1}}\| + \|p_{\text{top2}} - p_{\text{bottom2}}\|}{2 \cdot \|p_{\text{outer}} - p_{\text{inner}}\|}$$
   - Dynamic threshold adaptation uses running 90th percentile open baseline with minimum/maximum boundary clamping $[0.12, 0.28]$.

4. **Iris Circularity & Metric Depth (`src/cv/eye_extractor.py:42-69`)**:
   - Evaluates radial symmetry across 4 perimeter iris points relative to center point:
     $$\text{circularity} = \exp\left(-\frac{\text{Var}(r)}{\sigma^2}\right) \in [0.0, 1.0]$$
   - Pinhole camera depth calculated via human corneal baseline $D_{\text{metric}} = 11.7\text{ mm}$:
     $$Z_{\text{metric}} = \frac{f \cdot D_{\text{metric}}}{D_{\text{px}}}$$

5. **Tracking Quality Evaluation (`src/cv/quality_tracker.py:45-130`)**:
   - Evaluates composite confidence combining EAR aperture ($35\%$), iris circularity ($25\%$), periocular grayscale contrast ($20\%$), and landmark displacement jitter ($20\%$).
   - Returns structured `TrackingQuality` dataclass with gating on eye closure and extreme head rotations ($>45^\circ$).

6. **Core Types & Feature Vectors (`src/types.py:80-130`)**:
   - Provides clean dataclasses (`NormalizedPoint`, `EyeData`, `HeadPoseData`, `TrackingQuality`, `GazeFeatures`, `GazePrediction`, `FaceDetectionResult`).
   - `GazeFeatures.vector_8d`: 8D normalized vector `[norm_x_L, norm_y_L, norm_x_R, norm_y_R, pitch/45, yaw/45, roll/45, tz/1000]`.
   - `GazeFeatures.vector_10d` and `vector_14d` for backward compatibility.
   - Robust fallback values when `head_pose` is `None`.

### 1.2 Test Execution Results
- Full automated test suite: `uv run pytest -v`
  - **146 passed in 12.17s** (100% pass rate).
  - Tiers 1–4 unit, transformation invariance, calibration, and stress tests all pass.

---

## 2. Logic Chain & Stress-Testing

1. **Horizontal Gaze Decoupling & Conjugate Symmetry**:
   - *Observation*: Aligned canthal vectors point left-to-right on both eyes.
   - *Adversarial Challenge*: Tested horizontal iris shift ($\pm 6\text{px}$).
   - *Result*: Looking right produced $\text{norm\_x}_L = +0.1875, \text{norm\_x}_R = +0.1875, \text{avg\_norm\_x} = +0.1875$. Looking left produced negative values. No cancellation occurs.
   - *Invariance*: Tested across in-plane roll ($0^\circ$ to $360^\circ$) and scaling ($0.05\times$ to $10.0\times$). Relative deviation remained $< 10^{-3}$.

2. **SolvePnP 3D Model Continuity & Decoupling**:
   - *Observation*: Corrected 3D face model points $+Y$ downwards in camera space.
   - *Adversarial Challenge*: Swept pitch angle continuously from $-40^\circ$ to $+40^\circ$ in $2^\circ$ increments through synthetic camera projection.
   - *Result*: Estimated pitch matched ground truth with error $< 0.2^\circ$, and max step delta was $< 3.0^\circ$. No branch-cut jump or singularity observed.
   - *Yaw & Roll Sweeps*: Both sweeps from $-40^\circ$ to $+40^\circ$ demonstrated identical continuous linear tracking without axis cross-talk.

3. **Blink & Aperture Recovery**:
   - *Observation*: 6-point EAR with dynamic running history.
   - *Adversarial Challenge*: Injected synthetic blink drop ($\text{EAR} = 0.05$) followed by immediate reopening.
   - *Result*: Gaze feature `is_valid` dropped to `False` with confidence $\le 0.20$ during blink, and recovered instantly to `True` on the following open frame.

4. **Integrity & Code Inspection**:
   - Conducted regex and semantic scans across `src/` for hardcoded test fixtures, dummy facade implementations, or bypassed computations.
   - Verified that all computer vision and mathematical transformations perform genuine floating-point vector calculations.
   - No integrity violations or cheating mechanisms detected.

---

## 3. Caveats

1. **Pinhole Camera Optical Baseline**:
   - The default camera matrix assumes a standard webcam horizontal field of view ($65.0^\circ$). For cameras with wide-angle or fisheye lenses, `GazeConfig.camera_fov_h_deg` should be specified in configuration or CLI parameters.
2. **Extreme Single-Eye Occlusion**:
   - For head yaw angles exceeding $45^\circ$, one eye may experience nasal bridge occlusion. The pipeline marks tracking as invalid for multi-point gaze when both eyes are required, while individual `EyeData` instances remain accessible for single-eye fallback modes in M2.

---

## 4. Conclusion

**Verdict: APPROVE**

The implementation of Milestone 1 (CV & Robust Feature Engineering) fulfills all requirements of R1, R2, and the architectural contracts in `PROJECT.md`:
- Mathematical formulation of orthonormal iris normalization is robust, roll-invariant, scale-invariant, and free of horizontal cancellation.
- 3D anthropometric face model and `solvePnP` pose estimation eliminate branch-cut discontinuities around $(0, 0, 0)$.
- 6-point EAR and dynamic adaptive blink thresholding reliably detect eye closure and rapid recovery.
- Iris circularity and periocular lighting evaluation accurately measure signal quality.
- 100% of unit and integration tests (146/146) pass without failures.

Milestone 1 is approved for integration and downstream Milestone 2 (ML & Calibration) development.

---

## 5. Verification Method

### 5.1 Automated Test Suite
Run the full test suite verifying all 4 tiers and M1 modules:
```bash
uv run pytest -v
```
*Expected Result*: `146 passed` with exit code 0.

### 5.2 Python Invariance & Continuity Validation
Run the following verification script to independently validate mathematical properties:
```bash
uv run python -c "
import numpy as np, cv2, math
from src.config import GazeConfig
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from tests.test_m1_cv import build_synthetic_landmarks

config = GazeConfig()
ee = EyeExtractor(config)
hpe = HeadPoseEstimator(config)

# 1. Verify Neutral Head Pose
cam_mat = config.get_camera_matrix(640, 480)
proj_pts, _ = cv2.projectPoints(hpe.MODEL_POINTS_CORRECTED, np.zeros((3, 1)), np.array([[0.0],[0.0],[600.0]]), cam_mat, np.zeros((4, 1)))
lms = [config.left_eye_outer for _ in range(478)]
lms = [type('LM', (), {'x': 0.5, 'y': 0.5, 'z': 0.0})() for _ in range(478)]
for i, idx in enumerate(config.head_pose_mesh_indices):
    lms[idx] = type('LM', (), {'x': proj_pts[i, 0, 0]/640.0, 'y': proj_pts[i, 0, 1]/480.0, 'z': 0.0})()
hp = hpe.estimate(lms, 640, 480)
assert hp is not None and abs(hp.pitch) < 0.1 and abs(hp.yaw) < 0.1 and abs(hp.roll) < 0.1

# 2. Verify Directional Sensitivity & Roll Invariance
f_neutral = ee.extract(build_synthetic_landmarks(shift_x=0.0), 640, 480)
f_right = ee.extract(build_synthetic_landmarks(shift_x=5.0), 640, 480)
f_roll = ee.extract(build_synthetic_landmarks(shift_x=5.0, angle_deg=30.0), 640, 480)
assert f_right.avg_norm_x > f_neutral.avg_norm_x
assert math.isclose(f_roll.avg_norm_x, f_right.avg_norm_x, abs_tol=1e-3)
print('✅ Mathematical verification confirmed successfully!')
"
```

### 5.3 Invalidation Conditions
- If resting upright head pose yields pitch near $-180^\circ$ rather than $0.0^\circ$.
- If looking right yields opposite directional signs on left vs. right eye normalized iris coordinates.
- If any test in `pytest` fails.
