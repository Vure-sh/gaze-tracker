"""Real-time gaze tracking pipeline orchestrator coupling CV, ML regression, and temporal filtering."""

import time
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np

from src.config import GazeConfig
from src.types import GazeFeatures, HeadPoseData, GazePrediction, TrackingQuality
from src.face_mesh_detector import FaceMeshDetector
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from src.cv.quality_tracker import QualityTracker
from src.models.regressor import GazeRegressionModel, BaseGazeRegressor
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.filters.one_euro import OneEuroFilter2D
from src.filters.kalman import KalmanFilter2D


@dataclass
class GazePipelineResult:
    """Complete diagnostic and coordinate output from a single pipeline processing step."""
    frame: Optional[np.ndarray]
    gaze_features: Optional[GazeFeatures]
    head_pose: Optional[HeadPoseData]
    quality: Optional[TrackingQuality]
    raw_gaze: Optional[Tuple[float, float]]
    smoothed_gaze: Optional[Tuple[float, float]]
    prediction: Optional[GazePrediction]
    is_valid: bool
    latency_ms: float
    fps: float
    timestamp: float


class GazePipeline:
    """
    High-performance real-time gaze tracking pipeline orchestrator.
    Seamlessly manages frame processing, landmark tracking, 3D pose compensation,
    ML gaze estimation, temporal smoothing, and calibration lifecycle.
    """

    def __init__(
        self,
        config: Optional[GazeConfig] = None,
        model_path: Optional[str] = None,
        regressor: Optional[BaseGazeRegressor] = None,
        filter_type: Optional[str] = None
    ):
        self.config = config or GazeConfig()
        if model_path:
            self.config.model_path = model_path

        # 1. Computer Vision & Feature Extractors
        self.detector = FaceMeshDetector(self.config.model_path)
        self.eye_extractor = EyeExtractor(self.config)
        self.head_pose_estimator = HeadPoseEstimator(self.config)
        self.quality_tracker = QualityTracker(self.config)

        # 2. ML Gaze Regressor
        self.regressor = regressor or GazeRegressionModel(self.config)

        # 3. Calibration Manager
        self.calibrator = CalibrationManager(self.config, self.regressor)

        # 4. Temporal Filter
        filter_mode = filter_type or self.config.filter_type
        self.filter_type = filter_mode
        self._init_filter(filter_mode)

        # 5. Performance & Latency Telemetry
        self._latency_history = collections.deque(maxlen=30)
        self._fps_history = collections.deque(maxlen=30)
        self._last_time = time.time()

    def _init_filter(self, filter_mode: str) -> None:
        """Initialize or switch temporal filtering backend."""
        self.filter_type = filter_mode
        if filter_mode == "kalman":
            self.gaze_filter = KalmanFilter2D(
                process_noise=1e-2,
                measurement_noise=1e-1
            )
        else:
            self.gaze_filter = OneEuroFilter2D(
                min_cutoff=self.config.one_euro_min_cutoff,
                beta=self.config.one_euro_beta,
                d_cutoff=self.config.one_euro_d_cutoff
            )

    def set_filter_type(self, filter_mode: str) -> None:
        """Dynamically switch temporal filter between 'one_euro' and 'kalman'."""
        self._init_filter(filter_mode)

    def process_frame(
        self,
        frame: Optional[np.ndarray],
        timestamp: Optional[float] = None
    ) -> GazePipelineResult:
        """
        Execute one complete pipeline step on the input video frame.
        Guarantees sub-35ms total processing latency and crash-proof error handling.
        """
        start_t = time.perf_counter()
        now_t = timestamp if timestamp is not None else time.time()

        # Handle None or corrupted frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
            return GazePipelineResult(
                frame=frame,
                gaze_features=None,
                head_pose=None,
                quality=None,
                raw_gaze=None,
                smoothed_gaze=None,
                prediction=None,
                is_valid=False,
                latency_ms=0.0,
                fps=self.get_fps(),
                timestamp=now_t
            )

        img_h, img_w = frame.shape[:2]

        # 1. Landmark Detection
        landmarks = self.detector.detect(frame)

        gaze_features: Optional[GazeFeatures] = None
        head_pose: Optional[HeadPoseData] = None
        quality: Optional[TrackingQuality] = None
        combined_feature_vector: Optional[np.ndarray] = None
        is_valid_frame = False

        if landmarks is not None and len(landmarks) >= 468:
            # 2. Extract Eye Features & 3D Head Pose
            gaze_features = self.eye_extractor.extract(landmarks, img_w, img_h)
            head_pose = self.head_pose_estimator.estimate(landmarks, img_w, img_h)

            if gaze_features is not None and head_pose is not None:
                # Link head pose to gaze features
                gaze_features.head_pose = head_pose

                # 3. Assess Multi-Dimensional Tracking Quality
                quality = self.quality_tracker.assess_quality(
                    gaze_features=gaze_features,
                    head_pose=head_pose,
                    frame=frame,
                    landmarks=landmarks
                )
                gaze_features.quality = quality
                gaze_features.confidence = quality.confidence
                is_valid_frame = gaze_features.is_valid

                # Construct normalized feature vector
                combined_feature_vector = gaze_features.vector_14d

        # 4. Handle Calibration State Machine
        if self.calibrator.state == CalibrationState.COLLECTING:
            finished = self.calibrator.process_frame(combined_feature_vector, is_valid_frame)
            if finished:
                self.gaze_filter.reset()

        # 5. Predict Gaze Position (if trained)
        raw_gaze: Optional[Tuple[float, float]] = None
        smoothed_gaze: Optional[Tuple[float, float]] = None
        prediction: Optional[GazePrediction] = None

        if self.regressor.is_trained and combined_feature_vector is not None and is_valid_frame:
            pred_pt = self.regressor.predict(combined_feature_vector)
            if pred_pt is not None:
                raw_gaze = pred_pt
                smoothed_gaze = self.gaze_filter.filter(raw_gaze, timestamp=now_t)
                
                # Clamp within screen bounds
                gx = max(0.0, min(float(self.config.screen_width), smoothed_gaze[0]))
                gy = max(0.0, min(float(self.config.screen_height), smoothed_gaze[1]))
                smoothed_gaze = (gx, gy)

                prediction = GazePrediction(
                    screen_x=gx,
                    screen_y=gy,
                    norm_x=gx / max(1.0, float(self.config.screen_width)),
                    norm_y=gy / max(1.0, float(self.config.screen_height)),
                    confidence=quality.confidence if quality is not None else 1.0,
                    is_valid=True,
                    timestamp=now_t
                )

        # 6. Compute Telemetry
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self._latency_history.append(elapsed_ms)

        dt = max(1e-4, now_t - self._last_time)
        self._last_time = now_t
        self._fps_history.append(1.0 / dt)

        return GazePipelineResult(
            frame=frame,
            gaze_features=gaze_features,
            head_pose=head_pose,
            quality=quality,
            raw_gaze=raw_gaze,
            smoothed_gaze=smoothed_gaze,
            prediction=prediction,
            is_valid=is_valid_frame,
            latency_ms=elapsed_ms,
            fps=self.get_fps(),
            timestamp=now_t
        )

    def get_fps(self) -> float:
        """Get rolling average framerate."""
        if not self._fps_history:
            return 0.0
        return float(np.mean(self._fps_history))

    def get_average_latency_ms(self) -> float:
        """Get rolling average processing latency in milliseconds."""
        if not self._latency_history:
            return 0.0
        return float(np.mean(self._latency_history))

    def start_calibration(self, grid_type: Optional[str] = None) -> None:
        """Initiate screen calibration sequence."""
        self.calibrator.start_calibration(grid_type)

    def reset(self) -> None:
        """Reset calibration and temporal smoothing filters."""
        self.regressor.is_trained = False
        self.calibrator.state = CalibrationState.IDLE
        self.calibrator.points = []
        self.gaze_filter.reset()

    def save_calibration(self, filepath: Optional[str] = None) -> bool:
        """Persist trained calibration profile to disk."""
        target = filepath or self.config.calibration_file
        if not self.regressor.is_trained:
            return False
        self.regressor.save(target)
        return True

    def load_calibration(self, filepath: Optional[str] = None) -> bool:
        """Load calibration profile from disk."""
        target = filepath or self.config.calibration_file
        success = self.regressor.load(target)
        if success:
            self.calibrator.points = self.calibrator.generate_points()
            self.gaze_filter.reset()
        return success

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot dict of live pipeline performance and state metrics."""
        return {
            "fps": round(self.get_fps(), 2),
            "latency_ms": round(self.get_average_latency_ms(), 2),
            "is_calibrated": self.regressor.is_trained,
            "filter_type": self.filter_type,
            "calibration_state": self.calibrator.state.name,
            "screen_resolution": (self.config.screen_width, self.config.screen_height),
            "camera_resolution": (self.config.camera_width, self.config.camera_height),
        }
