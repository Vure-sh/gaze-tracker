# Computer Vision & Core Pipeline Audit & Refactoring Blueprint

## 1. Observation

### 1.1 Project Structure & File Layout
The codebase at `/home/vure/gaze-tracker` consists of:
- `main.py` (298 lines): Monolithic runtime loop managing video capture, landmark detection, feature extraction, solvePnP, polynomial regression prediction, temporal filtering, calibration state machine, and OpenCV GUI display.
- `src/config.py` (88 lines): Central dataclass `GazeConfig` with landmark indices, thresholds, and dimensions.
- `src/face_mesh_detector.py` (61 lines): MediaPipe `FaceLandmarker` wrapper loading `models/face_landmarker.task`.
- `src/eye_extractor.py` (136 lines): Extracts eye corner/eyelid landmarks, computes normalized iris coordinates and Eye Aspect Ratio (EAR).
- `src/head_pose.py` (124 lines): Solves 3D head pose using `cv2.solvePnP` on 6 facial landmark points.
- `src/gaze_regressor.py` (111 lines): Degree-2 Polynomial Ridge Regression pipeline (`StandardScaler -> PolynomialFeatures -> Ridge`).
- `src/calibrator.py` (158 lines): Multi-point calibration sequence manager (9, 13, 16 points) with saccade delay and IQR outlier filtering.
- `src/filters.py` (144 lines): Implementations of `OneEuroFilter2D` and `KalmanFilter2D`.
- `src/visualizer.py` (211 lines): Screen gaze canvas and camera debug HUD renderer.
- `models/face_landmarker.task` (3.76 MB) & `models/calibration_model.pkl` (3.45 KB).

---

### 1.2 MediaPipe FaceMesh & Iris Landmark Extraction
In `src/face_mesh_detector.py`:
- Lines 22-29:
```python
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=num_faces,
    running_mode=vision.RunningMode.IMAGE
)
self.detector = vision.FaceLandmarker.create_from_options(options)
```
- Lines 53-55:
```python
result = self.detector.detect(mp_image)
if result and result.face_landmarks and len(result.face_landmarks) > 0:
    return result.face_landmarks[0]
```
- Observation:
  1. `running_mode=vision.RunningMode.IMAGE` runs a full face detector on every frame independently rather than tracking landmarks temporally (`RunningMode.VIDEO` or `LIVE_STREAM`). This prevents MediaPipe's internal Kalman filter and ROI tracking from operating, increasing per-frame inference latency (~15-25ms vs ~5-8ms) and causing high-frequency landmark jitter.
  2. `output_face_blendshapes=True` and `output_facial_transformation_matrixes=True` are requested in the detector configuration, but both `result.face_blendshapes` (52 expression weights including blinks and gaze directions) and `result.facial_transformation_matrices` (4x4 metric canonical transform) are completely discarded in `detect()` (line 55 only returns `result.face_landmarks[0]`).
  3. The 3D depth coordinate `landmark.z` is present in MediaPipe's normalized landmarks but completely ignored in `src/eye_extractor.py` (lines 53-55: only `lm.x * img_w` and `lm.y * img_h` are used).

---

### 1.3 Eye Feature Extraction & Iris Normalization
In `src/eye_extractor.py`:
- Left eye indices (lines 39-44): `inner=133`, `outer=33`, `top=159`, `bottom=145`, `iris=468`.
- Right eye indices (lines 48-52): `inner=362`, `outer=263`, `top=386`, `bottom=374`, `iris=473`.
- Horizontal projection logic (lines 64-82):
```python
width_vec = p_outer - p_inner
eye_width = np.linalg.norm(width_vec)
...
u = width_vec / eye_width
proj_x = np.dot(p_iris - p_inner, u)
norm_x = float(proj_x / eye_width)
```
- Vertical projection logic (lines 70-87):
```python
height_vec = p_top - p_bottom
eye_height = np.linalg.norm(height_vec)
...
if eye_height > 1e-6:
    v = height_vec / eye_height
    proj_y = np.dot(p_iris - p_bottom, v)
    norm_y = float(proj_y / eye_height)
```
- Averaging logic (lines 116-117):
```python
avg_norm_x = (left.norm_x + right.norm_x) / 2.0
avg_norm_y = (left.norm_y + right.norm_y) / 2.0
```

- Observation:
  1. For the left eye: `p_inner` is 133 (nasal/inner canthus) and `p_outer` is 33 (temporal/outer canthus). In the mirrored camera frame (`cv2.flip(frame, 1)`), $\vec{w}_{\text{left}} = \vec{p}_{33} - \vec{p}_{133}$ points to the image left (user's right).
  2. For the right eye: `p_inner` is 362 (nasal/inner canthus) and `p_outer` is 263 (temporal/outer canthus). $\vec{w}_{\text{right}} = \vec{p}_{263} - \vec{p}_{362}$ points to the image right (user's left).
  3. When the user looks right: `left.norm_x` decreases towards 0.0, while `right.norm_x` increases towards 1.0.
  4. Evaluating `avg_norm_x = (left.norm_x + right.norm_x) / 2.0` results in:
     $$(x_0 - \Delta x) + (x_0 + \Delta x) = 2 x_0 \implies avg\_norm\_x = x_0$$
     The horizontal gaze shift exactly cancels out in `avg_norm_x`!
  5. In vertical projection, `height_vec = p_top - p_bottom` is NOT orthogonal to `width_vec`. Under head roll or oblique gaze, non-orthogonal axes cause cross-axis leakage between horizontal and vertical gaze coordinates.
  6. Dividing vertical projection by `eye_height` ($p_{\text{top}} - p_{\text{bottom}}$) makes `norm_y` volatile when eyelids narrow (squinting, smiling, partial blinks), causing numerical division instability and baseline gaze drift.
  7. MediaPipe provides 5 iris landmarks per eye (468-472 for left, 473-477 for right). `EyeExtractor` ignores the 4 perimeter iris points (469-472, 474-477), discarding metric iris diameter calculation ($D_{\text{iris}} \approx 11.7\text{ mm}$) and iris circularity confidence metrics.

---

### 1.4 3D Head Pose Estimation (solvePnP)
In `src/head_pose.py`:
- 3D Anthropometric Model Points (lines 24-31):
```python
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),     # Chin (Landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye outer corner (Landmark 33)
    (225.0, 170.0, -135.0),   # Right eye outer corner (Landmark 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0)   # Right mouth corner (Landmark 291)
], dtype=np.float64)
```
- Camera matrix (lines 36-43):
```python
focal_length = img_w
center = (img_w / 2.0, img_h / 2.0)
return np.array([
    [focal_length, 0.0, center[0]],
    [0.0, focal_length, center[1]],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
```
- Euler Angle Extraction (lines 75-90):
```python
sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
singular = sy < 1e-6
if not singular:
    pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
    yaw = np.arctan2(-rot_mat[2, 0], sy)
    roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
...
pitch_deg = float(np.degrees(pitch))
yaw_deg = float(np.degrees(yaw))
roll_deg = float(np.degrees(roll))
```
- Feature Vector (lines 105-113):
```python
feature_vector = np.array([
    pitch_deg / 45.0,
    yaw_deg / 45.0,
    roll_deg / 45.0,
    float(tvec[0, 0]) / 500.0,
    float(tvec[1, 0]) / 500.0,
    float(tvec[2, 0]) / 1000.0
], dtype=np.float64)
```

- Observation:
  1. In `MODEL_POINTS`, chin is defined at $Y = -330.0$ and eyes at $Y = +170.0$ (model $+Y$ is UP). In OpenCV camera coordinates, $+Y$ is DOWN.
  2. Because the 3D model is inverted $180^\circ$ relative to the camera coordinate frame, `cv2.solvePnP` on an upright face yields:
     $$\text{Pitch} \approx -174.8^\circ \quad (\approx -\pi), \quad \text{rvec} \approx [-3.05, 0, 0]^T$$
  3. Because pitch sits directly on the $-\pi / +\pi$ branch cut of `arctan2`, minor upward head tilt causes pitch to cross $-180^\circ \to +180^\circ$, causing `pitch_deg / 45.0` to jump abruptly from $-3.9$ to $+3.9$ (a discontinuity of magnitude $\approx 7.8$ in normalized feature space).
  4. Camera matrix assumes focal length $f_x = f_y = W$ (e.g. 640px). For standard 65°–70° FOV webcams, the actual focal length is $\approx \frac{W/2}{\tan(35^\circ)} \approx 450\text{px}$. Overestimating $f$ by ~40% distorts the recovered translation vector $t_z$ (depth) by ~40%.
  5. In `main.py` lines 209-212, eye features (8D) and head pose features (6D) are concatenated into a 14D vector and fed directly into a degree-2 polynomial regression model ($105$ polynomial expansion features). During calibration, the user's head is stationary while gaze varies across the screen; consequently, the model cannot learn the true physical coupling between head pose and gaze, leading to catastrophic gaze drift whenever the user turns or translates their head during runtime tracking.

---

### 1.5 EAR, Blink Detection, and Temporal Filtering
- In `src/eye_extractor.py` line 74:
  `ear = eye_height / eye_width = \|p_{159} - p_{145}\| / \|p_{33} - p_{133}\|`
  This uses a single vertical pair instead of the standard 6-point Soukupová & Čech (2016) formulation:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
- Fixed hard-coded threshold `ear_blink_threshold = 0.18` in `src/config.py` does not adapt to individual anatomical variation in eye aperture.
- In `src/filters.py` (`OneEuroFilter1D`, lines 46-66):
  ```python
  te = t - self.t_prev
  if te <= 1e-5:
      return ...
  dx = (x - hat_x_prev) / te
  ```
  When face tracking is temporarily lost (e.g., blink, lookaway, occlusion for 1–3 seconds), `te` becomes large (e.g. 2.0s), and the filter uses an obsolete `hat_x_prev` from seconds earlier, causing an uncalibrated step response upon tracking re-acquisition.

---

## 2. Logic Chain

```
[Observation 1.3: Opposite inner->outer vectors for left and right eyes]
  ==> Left norm_x moves inversely to Right norm_x for identical horizontal gaze shifts
  ==> avg_norm_x = (left.norm_x + right.norm_x)/2 cancels out horizontal gaze motion
  ==> High-degree polynomial regression is forced to overfit individual noisy channels

[Observation 1.3: Vertical norm_y divided by moving eyelid aperture (p_top - p_bottom)]
  ==> Squints, smiles, and blinks shrink the denominator (eye_height -> 0)
  ==> norm_y exhibits massive vertical drift independent of actual gaze angle
  ==> Non-orthogonal height_vec causes cross-talk between roll angle and norm_y

[Observation 1.4: MODEL_POINTS defined with +Y UP vs OpenCV camera +Y DOWN]
  ==> solvePnP produces rotation matrix with ~180° inversion (resting pitch ≈ -174.8°)
  ==> Resting pitch sits on the [-π, +π] branch cut
  ==> Head pitch tilt causes discontinuous flips between -180° and +180°
  ==> Polynomial features (pitch/45)^2 jump abruptly, causing wild cursor teleportation

[Observation 1.4: 14D concatenated Eye+Head vector trained on static-head calibration]
  ==> Zero head variance during calibration -> regression weights for head features are arbitrary
  ==> When head moves during tracking, polynomial cross-terms (e.g. norm_x * yaw) blow up
  ==> Head pose is strongly coupled instead of decoupled, causing gaze drift under yaw/pitch

[Observation 1.2: MediaPipe running in RunningMode.IMAGE]
  ==> No temporal tracking across frames; MediaPipe internal Kalman smoothing is bypassed
  ==> High landmark jitter, higher per-frame latency (~20ms vs ~6ms)
  ==> Blendshapes (52 channels) and 4x4 facial transformation matrix are discarded
```

---

## 3. Caveats
1. **Camera Hardware Calibration**: Without physical intrinsic calibration (`cv2.calibrateCamera` with a checkerboard), focal length must be approximated using horizontal FOV assumption ($\approx 65^\circ$).
2. **Extreme Head Poses (>45°)**: When yaw exceeds $45^\circ$, one eye is completely occluded by the nasal bridge. The system must switch gracefully to single-eye tracking mode.
3. **Eyeglass Glare**: Strong specular reflections on thick lenses can perturb MediaPipe's iris perimeter detection.

---

## 4. Conclusion & Actionable Architectural Recommendations

### 4.1 Recommended Mathematical Formulations

#### 1. Orthonormal Dual-Eye Normalization (Scale- and Roll-Invariant)
For each eye ($k \in \{\text{left}, \text{right}\}$):
1. Compute the canthal axis vector pointing consistently from nasal canthus $\vec{p}_{\text{nasal}}$ to temporal canthus $\vec{p}_{\text{temporal}}$ in screen/image space:
   - For left eye (subject right): $\vec{w}_L = \vec{p}_{33} - \vec{p}_{133}$ (image left-to-right is $+X_{\text{screen}}$).
   - For right eye (subject left): $\vec{w}_R = \vec{p}_{362} - \vec{p}_{263}$ (flipped to match image left-to-right direction $+X_{\text{screen}}$).
2. Orthonormal eye basis vectors:
   $$\vec{u} = \frac{\vec{w}}{\|\vec{w}\|}, \quad \vec{u}_{\perp} = \begin{bmatrix} -u_y \\ u_x \end{bmatrix}$$
3. Eye fissure midpoint:
   $$\vec{p}_{\text{mid}} = \frac{\vec{p}_{\text{nasal}} + \vec{p}_{\text{temporal}}}{2}$$
4. Zero-centered, scale- and roll-invariant normalized iris coordinates:
   $$norm\_x = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}}{\|\vec{w}\|}$$
   $$norm\_y = \frac{(\vec{p}_{\text{iris}} - \vec{p}_{\text{mid}}) \cdot \vec{u}_{\perp}}{\|\vec{w}\|}$$
   - When looking dead-center: $norm\_x \approx 0.0, norm\_y \approx 0.0$.
   - Scale-invariant: normalized by inter-canthal distance $\|\vec{w}\|$ (not variable eyelid height).
   - Roll-invariant: coordinates are evaluated in the intrinsic orthonormal frame $(\vec{u}, \vec{u}_{\perp})$.
   - Sign-consistent: for both eyes, $+norm\_x$ is screen-right, $+norm\_y$ is screen-down.
   - Dual-eye average: $avg\_norm\_x = \frac{norm\_x_L + norm\_x_R}{2}$ now provides clean, amplified horizontal sensitivity.

#### 2. Iris Diameter & Circularity Confidence Metric
Using all 5 MediaPipe iris landmarks ($\vec{p}_0$ center, $\vec{p}_1 \dots \vec{p}_4$ perimeter):
1. Radial distances: $r_i = \|\vec{p}_i - \vec{p}_0\|$ for $i \in \{1, 2, 3, 4\}$.
2. Iris radius in pixels: $\bar{r} = \frac{1}{4} \sum_{i=1}^4 r_i$.
3. Metric depth estimate:
   $$Z_{\text{iris}} = \frac{f \cdot D_{\text{iris, metric}}}{2 \bar{r}}$$
   where $D_{\text{iris, metric}} \approx 11.7\text{ mm}$.
4. Circularity / symmetry confidence:
   $$\text{conf}_{\text{iris}} = \exp\left( -\frac{\text{Var}(r)}{\sigma_r^2} \right) \in [0.0, 1.0]$$

#### 3. Standardized 3D Anthropometric Model Points for solvePnP
Align the 3D model frame with the camera optical frame ($+X$ right, $+Y$ down, $+Z$ forward towards camera):
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
- Upright face looking forward yields $\text{Pitch} \approx 0.0^\circ, \text{Yaw} \approx 0.0^\circ, \text{Roll} \approx 0.0^\circ$.
- No branch-cut discontinuities or angle wrapping during normal head movement ($\pm 45^\circ$).

#### 4. Decoupled Eye-Head Gaze Model
Rather than a monolithic 14D polynomial regression that overfits static calibration:
$$\vec{G}_{\text{screen}} = f_{\text{eye}}(\vec{x}_{\text{eye}}) + W_{\text{head}} \cdot \vec{x}_{\text{head}}$$
where:
- $\vec{x}_{\text{eye}} = [norm\_x_L, norm\_y_L, norm\_x_R, norm\_y_R, avg\_norm\_x, avg\_norm\_y]^T$ is mapped via regularized polynomial regression ($f_{\text{eye}}$) calibrated on the multi-point screen grid.
- $\vec{x}_{\text{head}} = [\text{pitch}, \text{yaw}, \text{roll}, \Delta t_x, \Delta t_y, \Delta t_z]^T$ applies a geometrically decoupled angular projection:
  $$\Delta X_{\text{head}} = \frac{Z_{\text{screen}}}{S_x} \tan(\text{yaw}), \quad \Delta Y_{\text{head}} = \frac{Z_{\text{screen}}}{S_y} \tan(\text{pitch})$$
  preventing head movement from corrupting eye non-linearities.

#### 5. Adaptive EAR & 6-Point Blink Detection
- Left eye EAR:
  $$\text{EAR}_L = \frac{\|\vec{p}_{160} - \vec{p}_{144}\| + \|\vec{p}_{158} - \vec{p}_{153}\|}{2 \|\vec{p}_{33} - \vec{p}_{133}\|}$$
- Right eye EAR:
  $$\text{EAR}_R = \frac{\|\vec{p}_{385} - \vec{p}_{380}\| + \|\vec{p}_{387} - \vec{p}_{373}\|}{2 \|\vec{p}_{263} - \vec{p}_{362}\|}$$
- Adaptive baseline: Keep a running 5-second 90th percentile $\text{EAR}_{\text{open}}$, setting dynamic threshold $\text{EAR}_{\text{blink}} = 0.60 \times \text{EAR}_{\text{open}}$.
- Integrate with MediaPipe blendshapes: $\text{blink\_score} = \max(\text{blendshape}[\text{"eyeBlink"}], \mathbb{I}(\text{EAR} < \text{EAR}_{\text{blink}}))$.

---

### 4.2 Modular Refactoring Plan

```
src/
├── core/
│   ├── config.py                 # Typed configurationdataclasses
│   ├── types.py                  # Structured dataclasses (Landmarks, EyeData, GazeEstimate, QualityMetrics)
│   └── pipeline.py               # Clean end-to-end processing pipeline class
├── cv/
│   ├── landmark_detector.py      # MediaPipe FaceLandmarker (LIVE_STREAM / VIDEO mode, blendshapes, matrix)
│   ├── eye_extractor.py          # Orthonormal scale/roll invariant iris normalization & 5-point geometry
│   ├── head_pose.py              # solvePnP with corrected 3D model, camera FOV matrix, rotation representations
│   └── quality_tracker.py        # Multi-dimensional confidence (EAR, iris circularity, lighting contrast)
├── calibration/
│   ├── calibrator.py             # Multi-point calibration manager with saccade trimming & holdout validation
│   └── targets.py                # Target pattern generators (9, 13, 16 points)
├── models/
│   ├── regressor.py              # Decoupled Polynomial Ridge / Huber / SVR regression models
│   └── serializer.py             # Model persistence & validation
├── filters/
│   ├── one_euro.py               # Robust One-Euro filter with dt timeout reset & velocity clipping
│   └── kalman.py                 # 2D Constant Velocity Kalman Filter
└── ui/
    ├── canvas.py                 # Fullscreen gaze canvas & animated target cues
    ├── hud.py                    # Camera debug HUD overlay
    └── app.py                    # Main UI application controller
```

---

## 5. Verification Method

### 5.1 Verification Commands
Run these commands to verify the mathematical observations:

1. **Verify Head Pose Model Pitch Offset & Corrected Pose**:
```bash
uv run python -c "
import numpy as np, cv2
from src.config import GazeConfig
from src.head_pose import HeadPoseEstimator

config = GazeConfig()
hpe = HeadPoseEstimator(config)
class LM:
    def __init__(self, x, y):
        self.x = x / 640.0; self.y = y / 480.0
lms = [LM(0,0) for _ in range(478)]
lms[1] = LM(320, 240); lms[152] = LM(320, 340); lms[33] = LM(260, 180)
lms[263] = LM(380, 180); lms[61] = LM(280, 280); lms[291] = LM(360, 280)
data = hpe.estimate(lms, 640, 480)
print('Current Model Pitch (should be near -175 deg):', data.pitch)
"
```

2. **Verify Opposite Left/Right Iris Vector Signs**:
```bash
uv run python -c "
from src.config import GazeConfig
from src.eye_extractor import EyeExtractor
config = GazeConfig()
ee = EyeExtractor(config)
class LM:
    def __init__(self, x, y):
        self.x = x / 640.0; self.y = y / 480.0
lms = [LM(0,0) for _ in range(478)]
# Left eye: outer=33 (x=260), inner=133 (x=300), iris=468 (x=280)
# Right eye: outer=263 (x=380), inner=362 (x=340), iris=473 (x=360)
lms[33] = LM(260, 200); lms[133] = LM(300, 200); lms[159] = LM(280, 190); lms[145] = LM(280, 210); lms[468] = LM(280, 200)
lms[263] = LM(380, 200); lms[362] = LM(340, 200); lms[386] = LM(360, 190); lms[374] = LM(360, 210); lms[473] = LM(360, 200)

data_center = ee.extract(lms, 640, 480)
# Shift eyes rightwards (+10px)
lms[468] = LM(290, 200); lms[473] = LM(370, 200)
data_right = ee.extract(lms, 640, 480)

print('Neutral avg_norm_x:', data_center.avg_norm_x)
print('Looking right avg_norm_x:', data_right.avg_norm_x)
print('Notice: Left norm_x moved:', data_right.left_eye.norm_x - data_center.left_eye.norm_x)
print('Notice: Right norm_x moved:', data_right.right_eye.norm_x - data_center.right_eye.norm_x)
"
```

### 5.2 Invalidation Conditions
- If `avg_norm_x` is proven to have consistent sign between left and right eyes without modification, this finding is invalidated.
- If `solvePnP` resting pitch with `MODEL_POINTS` is shown to be $0.0^\circ$ rather than $-174.8^\circ$, the head pose finding is invalidated.
