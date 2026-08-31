# Challenger 1 Report: Milestone 1 (CV & Robust Feature Engineering)

**Date**: 2026-08-30  
**Challenger**: Empirical Challenger 1 (`challenger_m1_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Target Milestone**: M1 (CV & Robust Feature Engineering)  
**Verdict**: **APPROVE**  

---

## 1. Observation

### 1.1 Direct Codebase & Interface Audit
1. **Orthonormal Dual-Eye Normalization (`src/cv/eye_extractor.py:107-164`)**:
   - Direction vectors correctly point left-to-right across the ocular fissure for both eyes:
     - Left eye: `canthal_vec = p_inner - p_outer` ($p_{133} - p_{33}$, $+X$ in observer frame)
     - Right eye: `canthal_vec = p_outer - p_inner` ($p_{263} - p_{362}$, $+X$ in observer frame)
   - Orthonormal basis: $\vec{u} = \frac{\vec{w}}{\|\vec{w}\|}$, $\vec{u}_{\perp} = [-u_y, u_x]^T$, $\vec{p}_{\text{mid}} = \frac{\vec{p}_{\text{inner}} + \vec{p}_{\text{outer}}}{2}$.
   - Normalized coordinates: $norm\_x = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}}{\|\vec{w}\|}$, $norm\_y = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}_{\perp}}{\|\vec{w}\|}$.
   - Guards against zero width: `if eye_width < 1e-6: eye_width = 1e-6`.

2. **Corrected Anthropometric 3D Model & Euler Angle Decomposition (`src/cv/head_pose.py:18-107`)**:
   - Aligned 3D model frame with OpenCV camera coordinate system ($+X$ right, $+Y$ down, $+Z$ forward):
     ```python
     MODEL_POINTS_CORRECTED = np.array([
         (0.0, 0.0, 0.0),          # Nose tip (1)
         (0.0, 100.0, -20.0),      # Chin (152)
         (-65.0, -50.0, -40.0),    # Left eye outer (33)
         (65.0, -50.0, -40.0),     # Right eye outer (263)
         (-40.0, 50.0, -30.0),     # Left mouth corner (61)
         (40.0, 50.0, -30.0)       # Right mouth corner (291)
     ], dtype=np.float64)
     ```
   - Euler angle decomposition via `cv2.solvePnP` + `cv2.Rodrigues` computes:
     $\text{pitch} = \text{arctan2}(R_{21}, R_{22})$, $\text{yaw} = \text{arctan2}(-R_{20}, \sqrt{R_{00}^2 + R_{10}^2})$, $\text{roll} = \text{arctan2}(R_{10}, R_{00})$, yielding $(0.0^\circ, 0.0^\circ, 0.0^\circ)$ on neutral upright head pose.

3. **Adaptive 6-Point EAR & Dynamic Blink Thresholding (`src/cv/eye_extractor.py:22-40, 70-90`)**:
   - 6-point formulation: $\text{EAR} = \frac{\|p_{top1} - p_{bottom1}\| + \|p_{top2} - p_{bottom2}\|}{2 \|p_{outer} - p_{inner}\|}$.
   - Running percentile baseline $\text{EAR}_{\text{open}} = \text{percentile}(\text{history}_{150}, 90\%)$ with dynamic threshold $\text{EAR}_{\text{threshold}} = \text{clip}(0.60 \times \text{EAR}_{\text{open}}, 0.12, 0.28)$.

---

### 1.2 Empirical Stress Test Execution Results
An adversarial test harness containing 113 stress cases (`tests/test_challenger_m1.py`) was implemented and executed against the implementation modules.

#### Stress Dimension 1: Head Roll Rotations [-90°, +90°] in 5° Steps (37 Discrete Angles)
- Target nominal gaze offset: $(norm\_x = 0.15625, norm\_y = -0.09375)$
- **Empirical Measurements**:
  - Maximum $norm\_x$ error across all 37 angles: **$8.05 \times 10^{-16}$** (Mean: $2.25 \times 10^{-16}$)
  - Maximum $norm\_y$ error across all 37 angles: **$1.30 \times 10^{-15}$** (Mean: $3.03 \times 10^{-16}$)
  - Maximum EAR error across all 37 angles: **$7.22 \times 10^{-16}$** (Mean: $1.92 \times 10^{-16}$)
  - Both `left_eye.is_open` and `right_eye.is_open` remained `True` across all 37 roll angles.

#### Stress Dimension 2: Head Scale Variations [0.2x to 5.0x] (10 Scale Factors)
- Tested scale factors: $[0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]$
- **Empirical Measurements**:
  - Maximum $norm\_x$ error across all scales: **$5.55 \times 10^{-16}$** (Mean: $9.16 \times 10^{-17}$)
  - Maximum $norm\_y$ error across all scales: **$5.55 \times 10^{-16}$** (Mean: $9.30 \times 10^{-17}$)
  - Metric depth inverse scaling consistency: Standard deviation of $(Z \times S) = \mathbf{1.76 \times 10^{-12}\text{ mm}}$.

#### Stress Dimension 3: Head Pose Continuous Sweeps [-50°, +50°] (101 Steps per Axis)
- Evaluated pure and compound rotations crossing $\pm 45^\circ$:
- **Empirical Measurements**:
  - **Pitch Sweep ($-50^\circ \to +50^\circ$)**: $\text{MAE} = \mathbf{0.0000^\circ}$, $\text{Max Error} = \mathbf{0.0000^\circ}$, $\text{Max Step Jump} = \mathbf{1.0000^\circ}$ (smooth, zero jump).
  - **Yaw Sweep ($-50^\circ \to +50^\circ$)**: $\text{MAE} = \mathbf{0.0000^\circ}$, $\text{Max Error} = \mathbf{0.0000^\circ}$, $\text{Max Step Jump} = \mathbf{1.0000^\circ}$ (smooth, zero jump).
  - **Roll Sweep ($-50^\circ \to +50^\circ$)**: $\text{MAE} = \mathbf{0.0000^\circ}$, $\text{Max Error} = \mathbf{0.0000^\circ}$, $\text{Max Step Jump} = \mathbf{1.0000^\circ}$ (smooth, zero jump).
  - Compound extreme rotations $(\pm 45^\circ, \pm 45^\circ, 0^\circ)$ and $(\pm 35^\circ, \pm 35^\circ, \pm 35^\circ)$: Euler angles recovered within $< 0.5^\circ$.
  - **Branch-Cut / Singularity Check**: Zero discontinuity near $\pm 45^\circ$, no gimbal lock divergence.

#### Stress Dimension 4: Blink Transitions & Adaptive Baseline
- Evaluated realistic 60-frame time series (open $\to$ closing $\to$ closed $\to$ opening $\to$ open):
  - Closed eye frames correctly flagged: `left_eye.is_open=False`, `right_eye.is_open=False`, `is_valid=False`, `confidence=0.20`.
  - Re-opened eye frames correctly recovered: `is_open=True`, `is_valid=True`, `confidence >= 0.70`.
  - Narrow-eyed subjects ($\text{EAR}_{\text{open}} \approx 0.20$): Adaptive threshold adapted down to $0.1688$, detecting full closure without false positives during open gaze.
  - Wide-eyed subjects ($\text{EAR}_{\text{open}} \approx 0.40$): Adaptive threshold adapted up to $0.2400$, detecting partial closure ($\text{EAR} = 0.14$) cleanly.

#### Stress Dimension 5: Degenerate & Adversarial Inputs
- **Collinear Landmarks ($y = x$)**: Handled gracefully without crash; `EyeExtractor` produced valid zero-centered features; `QualityTracker` returned valid `TrackingQuality`.
- **Zero Coordinates $(0, 0, 0)$**: Handled gracefully; `EyeExtractor` zero-width guard engaged safely (`norm_x = 0.0`); `QualityTracker` completed without exception.
- **Zero-Width Eye Bounding Box ($p_{\text{inner}} == p_{\text{outer}}$)**: Eye fissure width clamped to $10^{-6}$ epsilon; produced non-NaN output.
- **Out-of-Bounds Coordinates ($x=-9999, y=9999$)**: Executed without memory or arithmetic faults.
- **Truncated Landmark Lists ($N < 478$) / None Inputs**: All extractors returned `None` without uncaught exceptions.
- **Corrupted / Empty Video Frames (0x0, None, 1-channel, 4-channel)**: `FaceDetector.detect` returned `None` safely.

---

## 2. Logic Chain

1. *From Observation 1.1.1 & 1.2 (Stress Dimension 1)*: Aligned canthal vectors ($\vec{w}_L = \vec{p}_{133} - \vec{p}_{33}$, $\vec{w}_R = \vec{p}_{263} - \vec{p}_{362}$) and orthonormal projection $(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u} / \|\vec{w}\|$ eliminate horizontal gaze cancellation and guarantee exact mathematical invariance under roll rotations. Tested across 37 angles from $-90^\circ$ to $+90^\circ$, max observed error was $< 1.31 \times 10^{-15}$ (machine epsilon precision).
2. *From Observation 1.1.1 & 1.2 (Stress Dimension 2)*: Normalizing both iris vector and vertical basis by eye fissure width $\|\vec{w}\|$ guarantees strict scale invariance across distances. Tested from 0.2x to 5.0x scale, maximum error was $< 5.56 \times 10^{-16}$, and metric iris depth $Z = (f \cdot D_{\text{metric}}) / D_{\text{px}}$ scaled inversely with standard deviation $1.76 \times 10^{-12}\text{ mm}$.
3. *From Observation 1.1.2 & 1.2 (Stress Dimension 3)*: Corrected 3D face model aligned with camera $+Y$-down frame eliminates the legacy $180^\circ$ pitch inversion. Neutral head pose decomposes to $(0.0^\circ, 0.0^\circ, 0.0^\circ)$, moving the $[-\pi, \pi]$ branch cut to $\pm 180^\circ$, far outside the normal head rotation range ($\pm 45^\circ$). Continuous angle sweeps demonstrated $0.0000^\circ$ MAE and smooth $1.0000^\circ$ step changes with zero singularities.
4. *From Observation 1.1.3 & 1.2 (Stress Dimension 4)*: 6-point EAR coupled with a running 90th-percentile baseline adapts cleanly to narrow and wide eye geometries and responds dynamically to blink transitions without signal latching.
5. *From Observation 1.1.1 & 1.2 (Stress Dimension 5)*: Zero-division guards, dimension checks, and boundary handling prevent uncaught exceptions across degenerate, collinear, zero, and malformed inputs.
6. *Overall Test Suite*: The entire test suite (Tiers 1–4 plus the newly added Milestone 1 adversarial stress suite) passes 100% (**272 of 272 tests passing**).

---

## 3. Caveats

1. **Synthetic vs Optical Distortion**: Empirical invariance tests verified mathematical invariance under affine/projective transformations. Real-world fisheye distortion on extreme wide-angle webcams ($>120^\circ$ FOV) may introduce non-linear radial bending unless rectified via OpenCV distortion coefficients.
2. **Extreme Nasal Occlusion**: At extreme yaw angles exceeding $\pm 45^\circ$, the contralateral eye may become anatomically occluded by the nasal bridge; downstream gaze regressors (M2) should utilize individual eye coordinates when one eye is degraded.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 1 (CV & Robust Feature Engineering) passes all empirical stress challenges with zero critical bugs, zero mathematical singularities, machine-precision scale/roll invariance, and comprehensive input degeneracy protection. The module contracts in `src/types.py`, `src/cv/`, and `src/config.py` are robust and ready for Milestone 2 (Calibration & ML Gaze Regression).

---

## 5. Verification Method

### 5.1 Run Full Test Suite (272 Tests)
```bash
uv run pytest -v
```
*Expected Result*: `272 passed` with exit code 0.

### 5.2 Run Adversarial Challenger Stress Suite Directly (113 Tests)
```bash
uv run pytest -v tests/test_challenger_m1.py
```
*Expected Result*: `113 passed` in under 2 seconds.

### 5.3 Quantitative Metric Extraction Command
```bash
uv run python -c "
import numpy as np
from src.config import GazeConfig
from src.cv.eye_extractor import EyeExtractor
from tests.test_challenger_m1 import build_challenger_landmarks

config = GazeConfig()
ee = EyeExtractor(config)
roll_errs = [abs(ee.extract(build_challenger_landmarks(shift_x=10, angle_deg=r), 640, 480).avg_norm_x - (10/64)) for r in range(-90, 95, 5)]
print(f'Max Roll Error across [-90°, +90°]: {max(roll_errs):.4e}')
assert max(roll_errs) < 1e-12
print('✅ Verification succeeded!')
"
```

### 5.4 Invalidation Conditions
- If maximum roll invariance error exceeds $10^{-3}$ across $\pm 90^\circ$, this approval is invalidated.
- If maximum scale invariance error exceeds $10^{-3}$ across $0.2\text{x}-5.0\text{x}$, this approval is invalidated.
- If Euler angles experience a jump $> 5.0^\circ$ during a $1.0^\circ$ head pose step in $[-45^\circ, +45^\circ]$, this approval is invalidated.
- If any test in `uv run pytest -v` fails, this approval is invalidated.
