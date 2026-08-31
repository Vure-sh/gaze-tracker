# Milestone 1 (CV & Robust Feature Engineering) Handoff Report

**Date**: 2026-08-30  
**Author**: Computer Vision Specialist (`worker_m1_1`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Milestone**: M1 (CV & Robust Feature Engineering)  
**Status**: COMPLETE — Hard Handoff  

---

## 1. Observation

### 1.1 Direct Baseline Observations & Code Analysis
1. **Critical Horizontal Gaze Cancellation Bug in Legacy Code**:
   - In legacy `src/eye_extractor.py:64-82, 116-117`:
     ```python
     # Left eye: outer=33 (left in mirrored frame), inner=133 (right in mirrored frame)
     width_vec = p_outer - p_inner  # points left
     # Right eye: inner=362 (left in mirrored frame), outer=263 (right in mirrored frame)
     width_vec = p_outer - p_inner  # points right
     avg_norm_x = (left.norm_x + right.norm_x) / 2.0
     ```
   - When the subject looked right, `left.norm_x` decreased while `right.norm_x` increased symmetrically. Averaging them yielded `avg_norm_x == 0.500` constantly, wiping out horizontal gaze motion.
2. **3D Head Pose Branch-Cut Discontinuity**:
   - In legacy `src/head_pose.py:24-31`:
     ```python
     MODEL_POINTS = np.array([
         (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
         (0.0, -330.0, -65.0),     # Chin (Landmark 152) -> +Y was defined as UP
         ...
     ])
     ```
   - In OpenCV camera coordinates, $+Y$ is DOWN. Because the legacy 3D model was inverted $180^\circ$ relative to camera coordinates, `solvePnP` on an upright resting face yielded Pitch $\approx -174.8^\circ$ ($\approx -\pi$). Minor upward head tilt crossed the $[-\pi, +\pi]$ branch-cut, causing normalized features (`pitch / 45.0`) to jump discontinuously between $-3.9$ and $+3.9$.
3. **Camera Intrinsic Matrix & FOV**:
   - Legacy `src/head_pose.py:36-43` assumed focal length $f_x = f_y = W$ (e.g. 640px). For standard $65^\circ$ FOV webcams, the actual focal length is $f = \frac{W/2}{\tan(32.5^\circ)} \approx 502\text{px}$. Overestimating $f$ distorted 3D translation vectors ($t_z$).
4. **EAR Blink Detection**:
   - Legacy code used a single vertical pair ($p_{159} - p_{145}$) with a fixed hardcoded threshold of 0.18, susceptible to false blink triggers during partial squinting.
5. **MediaPipe Task Execution**:
   - Legacy `src/face_mesh_detector.py` discarded `result.face_blendshapes` (52 channels) and `result.facial_transformation_matrixes` (4x4 matrix), and ran exclusively in `RunningMode.IMAGE`.

---

## 2. Logic Chain

1. **Orthonormal Normalization Formulation (`src/cv/eye_extractor.py`)**:
   - *Observation 1.1.1*: Opposite directional vectors caused horizontal cancellation.
   - *Logic Step*: Aligned canthal vectors for both eyes to point consistently in the $+X$ direction (image left-to-right):
     - Left Eye: $\vec{w}_L = \vec{p}_{133} - \vec{p}_{33}$ (inner minus outer)
     - Right Eye: $\vec{w}_R = \vec{p}_{263} - \vec{p}_{362}$ (outer minus inner)
   - *Orthonormal Frame*:
     $$\vec{u} = \frac{\vec{w}}{\|\vec{w}\|}, \quad \vec{u}_{\perp} = \begin{bmatrix} -u_y \\ u_x \end{bmatrix}, \quad \vec{p}_{\text{mid}} = \frac{\vec{p}_{\text{inner}} + \vec{p}_{\text{outer}}}{2}$$
   - *Result*: Zero-centered coordinates $norm\_x = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}}{\|\vec{w}\|}$ and $norm\_y = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}_{\perp}}{\|\vec{w}\|}$. Looking right produces $+norm\_x$ on both eyes and $+avg\_norm\_x$, with mathematical invariance under head roll and camera distance scaling.

2. **Corrected Anthropometric 3D Model (`src/cv/head_pose.py`)**:
   - *Observation 1.1.2*: Inverted $+Y$ model caused resting Pitch $\approx -175^\circ$ on the branch cut.
   - *Logic Step*: Aligned 3D model frame directly with OpenCV camera coordinates ($+X$ right, $+Y$ down, $+Z$ forward):
     ```python
     MODEL_POINTS_CORRECTED = np.array([
         (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
         (0.0, 100.0, -20.0),      # Chin (Landmark 152)
         (-65.0, -50.0, -40.0),    # Left eye outer (Landmark 33)
         (65.0, -50.0, -40.0),     # Right eye outer (Landmark 263)
         (-40.0, 50.0, -30.0),     # Left mouth corner (Landmark 61)
         (40.0, 50.0, -30.0)       # Right mouth corner (Landmark 291)
     ], dtype=np.float64)
     ```
   - *Result*: Neutral upright face yields Pitch $\approx 0.0^\circ$, Yaw $\approx 0.0^\circ$, Roll $\approx 0.0^\circ$, completely eliminating branch-cut discontinuities within normal head movement ($\pm 45^\circ$).

3. **6-Point EAR with Dynamic Adaptive Baseline (`src/cv/eye_extractor.py`)**:
   - *Observation 1.1.4*: Fixed 0.18 threshold failed on small/narrow eyes.
   - *Logic Step*: Implemented Soukupova & Cech (2016) 6-point formulation using landmarks $[33, 133, 160, 144, 158, 153]$ (left) and $[362, 263, 385, 380, 387, 373]$ (right), combined with a running 90th percentile baseline $\text{EAR}_{\text{open}}$ and dynamic threshold $\text{EAR}_{\text{threshold}} = \text{clip}(0.60 \times \text{EAR}_{\text{open}}, 0.12, 0.28)$.

4. **5-Point Iris Circularity Geometry & Metric Depth (`src/cv/eye_extractor.py`)**:
   - *Logic Step*: Evaluated radial distances of 4 perimeter iris points $[469\dots472, 474\dots477]$ relative to iris centers $[468, 473]$. Computed circularity score $\exp(-\text{Var}(r)/\sigma^2) \in [0, 1]$ and metric camera depth $Z = \frac{f \cdot D_{\text{iris, metric}}}{D_{\text{px}}}$ where $D_{\text{iris, metric}} = 11.7\text{ mm}$.

5. **Multi-Dimensional Quality & Confidence Tracker (`src/cv/quality_tracker.py`)**:
   - *Logic Step*: Weighted combination of EAR aperture, iris circularity, periocular grayscale standard deviation contrast, and landmark temporal displacement stability, outputting structured `TrackingQuality` dataclass with diagnostic `failure_reasons`.

6. **Unified Typed Contracts & Clean Feature Vectors (`src/types.py`)**:
   - *Logic Step*: Defined `NormalizedPoint`, `EyeData`, `HeadPoseData`, `GazeFeatures`, `GazePrediction`, `TrackingQuality`, `FaceDetectionResult`. Exposes `GazeFeatures.vector_8d`, `vector_10d`, and `vector_14d`.

---

## 3. Caveats

1. **Physical Webcam Intrinsic Calibration**: Camera matrix uses horizontal FOV estimation ($65^\circ$). If user hardware differs significantly (e.g. $120^\circ$ ultra-wide lens), `CameraConfig.fov_h_deg` should be adjusted via CLI or config.
2. **Extreme Nasal Bridge Occlusion**: For extreme head yaw exceeding $45^\circ$, one eye is geometrically occluded; downstream regression handles single-eye fallback using individual eye norm coordinates.

---

## 4. Conclusion

Milestone 1 is complete. All core computer vision and feature engineering components have been implemented, tested, and validated:
- `src/types.py`: Fully typed core data models and clean feature vectors.
- `src/config.py`: Modular typed configuration and FOV camera intrinsic computation.
- `src/cv/face_detector.py` & `src/face_mesh_detector.py`: MediaPipe FaceLandmarker with IMAGE and VIDEO modes, blendshapes, and 4x4 matrix extraction.
- `src/cv/eye_extractor.py` & `src/eye_extractor.py`: Orthonormal scale/roll invariant normalization, 6-point adaptive EAR, 5-point iris geometry.
- `src/cv/head_pose.py` & `src/head_pose.py`: Corrected 3D anthropometric face model, solvePnP with continuous Euler angles around $(0, 0, 0)$.
- `src/cv/quality_tracker.py`: Composite tracking confidence tracking EAR, circularity, contrast, stability.
- `tests/test_m1_cv.py`: 18 comprehensive tests.
- Full test suite: **128 tests passing (100% pass rate)**.

---

## 5. Verification Method

### 5.1 Automated Test Suite Execution
Run the full test suite verifying Tiers 1–3 and Milestone 1 unit/invariance tests:
```bash
uv run pytest -v
```
*Expected Output*: `128 passed in ~8s` with exit code 0.

### 5.2 Specific CV Verification Commands
1. **Verify Head Pose Neutral Angle & Branch-Cut Fix**:
```bash
uv run python -c "
import numpy as np, cv2
from src.config import GazeConfig
from src.cv.head_pose import HeadPoseEstimator
from src.types import NormalizedPoint

config = GazeConfig()
hpe = HeadPoseEstimator(config)
cam_mat = config.get_camera_matrix(640, 480)
proj_pts, _ = cv2.projectPoints(hpe.MODEL_POINTS_CORRECTED, np.zeros((3, 1)), np.array([[0.0],[0.0],[600.0]]), cam_mat, np.zeros((4, 1)))

lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]
for i, idx in enumerate(config.head_pose_mesh_indices):
    lms[idx] = NormalizedPoint(x=proj_pts[i, 0, 0] / 640.0, y=proj_pts[i, 0, 1] / 480.0, z=0.0)

data = hpe.estimate(lms, 640, 480)
print('Neutral Head Pose Euler Angles:', f'Pitch={data.pitch:.2f}°, Yaw={data.yaw:.2f}°, Roll={data.roll:.2f}°')
assert abs(data.pitch) < 0.1 and abs(data.yaw) < 0.1 and abs(data.roll) < 0.1
print('✅ Head Pose neutral angle verified!')
"
```

2. **Verify Directional Sensitivity & Invariance**:
```bash
uv run python -c "
from src.config import GazeConfig
from src.cv.eye_extractor import EyeExtractor
from tests.test_m1_cv import build_synthetic_landmarks

config = GazeConfig()
ee = EyeExtractor(config)

f_neutral = ee.extract(build_synthetic_landmarks(shift_x=0.0), 640, 480)
f_right = ee.extract(build_synthetic_landmarks(shift_x=6.0), 640, 480)
f_rot = ee.extract(build_synthetic_landmarks(shift_x=6.0, angle_deg=30.0), 640, 480)
f_scaled = ee.extract(build_synthetic_landmarks(shift_x=6.0, scale=1.5), 640, 480)

print(f'Neutral avg_norm_x: {f_neutral.avg_norm_x:.4f}')
print(f'Looking Right avg_norm_x: {f_right.avg_norm_x:.4f} (L: {f_right.left_eye.norm_x:.4f}, R: {f_right.right_eye.norm_x:.4f})')
print(f'30° Roll avg_norm_x: {f_rot.avg_norm_x:.4f}')
print(f'1.5x Scale avg_norm_x: {f_scaled.avg_norm_x:.4f}')
assert f_right.avg_norm_x > f_neutral.avg_norm_x
assert abs(f_rot.avg_norm_x - f_right.avg_norm_x) < 1e-3
assert abs(f_scaled.avg_norm_x - f_right.avg_norm_x) < 1e-3
print('✅ Directional sensitivity, roll invariance, and scale invariance verified!')
"
```

### 5.3 Invalidation Conditions
- If neutral face `solvePnP` yields resting pitch near $-175^\circ$ rather than $0.0^\circ$, this milestone is invalidated.
- If looking right yields opposite directional signs for left vs right eye normalized iris positions, this milestone is invalidated.
- If any test in `pytest` fails (exit code != 0), this milestone is invalidated.
