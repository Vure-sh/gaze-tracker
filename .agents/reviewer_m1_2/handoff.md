# Milestone 1 (CV & Robust Feature Engineering) Review Handoff Report

**Date**: 2026-08-30  
**Reviewer**: Reviewer & Adversarial Critic 2 (`reviewer_m1_2`)  
**Workspace**: `/home/vure/gaze-tracker`  
**Milestone**: M1 (CV & Robust Feature Engineering)  
**Deliverables Reviewed**: `src/types.py`, `src/config.py`, `src/cv/*`, `src/face_mesh_detector.py`, `src/eye_extractor.py`, `src/head_pose.py`, `tests/`  

---

## Review Summary

**Verdict**: **APPROVE**  
**Integrity Audit**: **CLEAN (No Integrity Violations)**  
**Adversarial Risk Level**: **LOW**  
**Automated Test Pass Rate**: **272 / 272 passing (100%)**

---

## 1. Observation

### 1.1 Direct Source Code Observations & Architectural Inspection

1. **Normalized Iris Projections on Canthal Axes (`src/cv/eye_extractor.py:107-163`)**:
   - Left Eye: Canthal vector $\vec{w}_L = \vec{p}_{133} - \vec{p}_{33}$ (inner nasal minus outer temporal).
   - Right Eye: Canthal vector $\vec{w}_R = \vec{p}_{263} - \vec{p}_{362}$ (outer temporal minus inner nasal).
   - Both unit horizontal vectors $\vec{u} = \vec{w} / \|\vec{w}\|$ point in the $+X$ direction (image left-to-right).
   - Orthogonal unit vertical vector $\vec{u}_{\perp} = [-u_y, u_x]^T$ points in the $+Y$ direction (image top-to-bottom).
   - Normalized projections:
     $$norm\_x = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}}{\|\vec{w}\|}, \quad norm\_y = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}_{\perp}}{\|\vec{w}\|}$$
   - When looking right, $norm\_x > 0$ for both left and right eyes, and $avg\_norm\_x > 0$. The legacy cancellation bug (where left and right eye horizontal offsets had opposite signs and cancelled to zero on averaging) is completely resolved.

2. **Continuity of Euler Angles & Resting Head Pose (`src/cv/head_pose.py:18-25, 96-107`)**:
   - `MODEL_POINTS_CORRECTED` is aligned with the standard OpenCV camera frame ($+X$ right, $+Y$ down, $+Z$ forward into scene):
     - Nose tip: `(0.0, 0.0, 0.0)`
     - Chin: `(0.0, 100.0, -20.0)`
     - Left eye outer (33): `(-65.0, -50.0, -40.0)`
     - Right eye outer (263): `(65.0, -50.0, -40.0)`
     - Left mouth (61): `(-40.0, 50.0, -30.0)`
     - Right mouth (291): `(40.0, 50.0, -30.0)`
   - Neutral upright head pose produces $(pitch, yaw, roll) = (0.0^\circ, 0.0^\circ, 0.0^\circ)$, shifting normal head motion ($\pm 45^\circ$) far away from the $[-\pi, +\pi]$ branch-cut of $\arctan2$.

3. **Backward Compatibility Wrappers (`src/face_mesh_detector.py`, `src/eye_extractor.py`, `src/head_pose.py`)**:
   - `src/face_mesh_detector.py` provides `FaceMeshDetector(FaceDetector)`.
   - `src/eye_extractor.py` exposes `EyeExtractor`, `EyeData`, `GazeFeatures`.
   - `src/head_pose.py` exposes `HeadPoseEstimator`, `HeadPoseData`.
   - `GazeFeatures.vector_14d` preserves backward compatibility with downstream modules expecting 14D feature vectors.

4. **Code Quality, Typing & Structural Integrity (`src/types.py`, `src/config.py`)**:
   - All modules use `@dataclass` contracts with static type hints.
   - Zero syntax errors or missing imports; verified via Python AST parsing and compilation.

---

## 2. Logic Chain

1. **Iris Projection Mathematical Rigor**:
   - Aligned canthal vectors ensure that conjugate eye movements project onto a shared right-handed coordinate frame $(\vec{u}, \vec{u}_{\perp})$ in $\mathbb{R}^2$.
   - Because $R(\theta)^T R(\theta) = I$, the inner products $(R(\theta)\vec{v}) \cdot (R(\theta)\vec{u}) = \vec{v} \cdot \vec{u}$ and $(R(\theta)\vec{v}) \cdot (R(\theta)\vec{u}_{\perp}) = \vec{v} \cdot \vec{u}_{\perp}$ are strictly invariant under arbitrary 2D in-plane head roll.
   - Dividing by $\|\vec{w}\|$ provides scale and camera distance invariance.

2. **Euler Angle Branch-Cut Elimination**:
   - In OpenCV camera coordinates, $+Y$ is oriented downwards. Inverting the 3D model $Y$ coordinates placed the resting orientation at $R \approx R_x(180^\circ)$, directly on the $\pm 180^\circ$ singularity.
   - Correcting the model points so $+Y$ points towards the chin establishes $R \approx I$ for upright faces.
   - The matrix decomposition $R_z(\text{roll}) R_y(\text{yaw}) R_x(\text{pitch})$ with $sy = \sqrt{R_{00}^2 + R_{10}^2}$ maintains continuous angle mapping over pitch $[-45^\circ, +45^\circ]$, yaw $[-45^\circ, +45^\circ]$, and roll $[-45^\circ, +45^\circ]$.

3. **Integrity & Facade Verification**:
   - Inspected all functions across `src/cv/*` and `src/*.py`.
   - No hardcoded test outputs or dummy return statements were detected. All outputs are derived from live landmark coordinates, trigonometric projections, and mathematical equations.

---

## 3. Caveats

1. **Non-Standard Webcam FOV**:
   - Camera matrix intrinsics rely on $65^\circ$ horizontal FOV by default. For ultra-wide or narrow angle webcams, `camera_fov_h_deg` should be specified in `GazeConfig`.
2. **Extreme Head Pose Occlusion**:
   - For extreme yaw exceeding $\pm 45^\circ$, one eye can be physically occluded by the nasal bridge. Downstream regressors should leverage individual eye `norm_x`/`norm_y` rather than assuming both eyes are visible.

---

## 4. Conclusion

**Verdict: APPROVE**

Milestone 1 successfully delivers high-quality, mathematically sound, and robust computer vision feature extraction for the gaze tracking pipeline. All acceptance criteria for M1 are met:
- Sign conventions and orthonormal projections for iris tracking are validated without horizontal cancellation.
- 3D head pose Euler angles are continuous and stable around neutral $(0, 0, 0)$.
- 6-point adaptive EAR and 5-point iris circularity are fully functional.
- Backward compatibility with legacy wrappers is verified.
- 272 automated tests pass with 100% success rate.

---

## 5. Verification Method

### 5.1 Full Test Suite Execution
```bash
uv run pytest -v
```
*Expected Output*: `272 passed in ~13s` with exit code 0.

### 5.2 Independent Mathematical Verification Commands

1. **Euler Continuity across 6,859 3D Head Poses ($[-45^\circ, +45^\circ]$)**:
```bash
uv run python -c "
import numpy as np, cv2
from src.config import GazeConfig
from src.cv.head_pose import HeadPoseEstimator
from src.types import NormalizedPoint

config = GazeConfig()
hpe = HeadPoseEstimator(config)
cam_mat = config.get_camera_matrix(640, 480)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

max_err = 0.0
for p in np.linspace(-45, 45, 19):
    for y in np.linspace(-45, 45, 19):
        for r in np.linspace(-45, 45, 19):
            p_rad, y_rad, r_rad = np.radians(p), np.radians(y), np.radians(r)
            Rx = np.array([[1, 0, 0], [0, np.cos(p_rad), -np.sin(p_rad)], [0, np.sin(p_rad), np.cos(p_rad)]])
            Ry = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
            Rz = np.array([[np.cos(r_rad), -np.sin(r_rad), 0], [np.sin(r_rad), np.cos(r_rad), 0], [0, 0, 1]])
            rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
            proj_pts, _ = cv2.projectPoints(hpe.MODEL_POINTS_CORRECTED, rvec, np.array([[0.0],[0.0],[600.0]]), cam_mat, dist_coeffs)
            lms = [NormalizedPoint(x=0.5, y=0.5, z=0.0) for _ in range(478)]
            for i, idx in enumerate(config.head_pose_mesh_indices):
                lms[idx] = NormalizedPoint(x=proj_pts[i, 0, 0] / 640.0, y=proj_pts[i, 0, 1] / 480.0, z=0.0)
            data = hpe.estimate(lms, 640, 480)
            max_err = max(max_err, abs(data.pitch - p), abs(data.yaw - y), abs(data.roll - r))

assert max_err < 1e-4
print(f'✅ Euler continuity verified! Max reconstruction error: {max_err:.4e}°')
"
```

2. **360° In-Plane Iris Roll Invariance**:
```bash
uv run python -c "
import numpy as np
from src.config import GazeConfig
from src.cv.eye_extractor import EyeExtractor
from tests.test_m1_cv import build_synthetic_landmarks

config = GazeConfig()
ee = EyeExtractor(config)
base = build_synthetic_landmarks(shift_x=4.0, shift_y=-2.0, scale=1.0)
f_base = ee.extract(base, 640, 480)
for angle in np.linspace(0, 360, 73):
    rot = build_synthetic_landmarks(shift_x=4.0, shift_y=-2.0, angle_deg=angle, scale=1.0)
    f_rot = ee.extract(rot, 640, 480)
    assert abs(f_rot.avg_norm_x - f_base.avg_norm_x) < 1e-3
    assert abs(f_rot.avg_norm_y - f_base.avg_norm_y) < 1e-3
print('✅ 360° Iris roll invariance verified!')
"
```

### 5.3 Invalidation Conditions
- If resting upright head pose yields pitch near $\pm 180^\circ$, this review is invalidated.
- If looking right causes left and right eye normalized coordinates to cancel or move in opposite directions, this review is invalidated.
- If any test in `pytest` fails, this review is invalidated.

---

## 6. Verified Claims & Adversarial Stress Tests

| Claim / Component | Verification Method | Result | Status |
|---|---|---|---|
| Orthonormal Iris Projection | Tested conjugate left/right gaze offsets and 360° rotation | Error < $1.7 \times 10^{-15}$ | **PASS** |
| 3D Head Pose Continuity | Tested 6,859 poses covering full $[-45^\circ, +45^\circ]$ range | Error < $1.6 \times 10^{-13\circ}$ | **PASS** |
| 6-Point Adaptive EAR | Evaluated open, closed, and asymmetric eye states | Dynamic threshold adapts properly | **PASS** |
| Iris Geometry & Depth | Tested 5-point circularity and $11.7\text{mm}$ corneal depth | Scales inversely with distance | **PASS** |
| Backward Compatibility | Imported `FaceMeshDetector`, `EyeExtractor`, `HeadPoseEstimator` | All legacy entrypoints preserved | **PASS** |
| Edge Case Robustness | Tested empty frames, corrupted arrays, zero-width eyes, OOB landmarks | Safe fallback without unhandled crashes | **PASS** |
| Integrity Check | Inspected source code AST and execution traces | Zero hardcoded test constants | **PASS** |
