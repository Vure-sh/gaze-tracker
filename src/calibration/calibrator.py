"""Multi-point calibration manager with wall-clock dwell timing, outlier rejection, and holdout validation."""

from __future__ import annotations
import time
import math
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.config import GazeConfig
from src.models.regressor import BaseGazeRegressor, GazeRegressionModel
from src.calibration.targets import TargetGenerator


class CalibrationState(Enum):
    """Enumeration of calibration and validation state machine modes."""
    IDLE = auto()
    COUNTDOWN = auto()
    COLLECTING = auto()
    POINT_COMPLETE = auto()
    FINISHED = auto()
    VALIDATING = auto()
    VALIDATION_COMPLETE = auto()


class CalibrationManager:
    """
    Manages multi-point calibration sequences, wall-clock dwell timing,
    saccade trimming, normalized feature outlier filtering, and holdout validation.
    """

    def __init__(self, config: GazeConfig, regressor: Optional[BaseGazeRegressor] = None):
        self.config = config
        self.regressor = regressor if regressor is not None else GazeRegressionModel(config)
        self.state = CalibrationState.IDLE

        self.points: List[Tuple[int, int]] = []
        self.current_point_idx: int = 0
        self.current_point_samples: List[np.ndarray] = []

        self.all_features: List[np.ndarray] = []
        self.all_targets: List[Tuple[int, int]] = []
        self.all_point_ids: List[int] = []

        self.point_frame_counter: int = 0
        self.point_start_time: float = 0.0

        # Post-calibration validation tracking
        self.validation_points: List[Tuple[int, int]] = []
        self.validation_point_idx: int = 0
        self.validation_errors: List[float] = []
        self.validation_metrics: Dict[str, float] = {}

    def generate_points(
        self,
        grid_type: Optional[str] = None,
        boustrophedon: bool = True
    ) -> List[Tuple[int, int]]:
        """Generates target screen coordinates in Boustrophedon serpentine order."""
        self.points = TargetGenerator.generate_points(
            config=self.config,
            grid_type=grid_type,
            boustrophedon=boustrophedon
        )
        return self.points

    def start_calibration(
        self,
        grid_type: Optional[str] = None,
        boustrophedon: bool = True
    ):
        """Initializes and starts a fresh multi-point calibration session."""
        self.generate_points(grid_type, boustrophedon=boustrophedon)
        self.current_point_idx = 0
        self.current_point_samples = []
        self.all_features = []
        self.all_targets = []
        self.all_point_ids = []
        self.point_frame_counter = 0
        self.point_start_time = time.time()
        self.state = CalibrationState.COLLECTING

    def get_current_target(self) -> Optional[Tuple[int, int]]:
        """Returns the active (X, Y) pixel target on screen."""
        if self.state in (CalibrationState.COLLECTING, CalibrationState.COUNTDOWN, CalibrationState.POINT_COMPLETE):
            if 0 <= self.current_point_idx < len(self.points):
                return self.points[self.current_point_idx]
        elif self.state == CalibrationState.VALIDATING:
            if 0 <= self.validation_point_idx < len(self.validation_points):
                return self.validation_points[self.validation_point_idx]
        return None

    def get_progress(self) -> Tuple[int, int, float]:
        """
        Returns:
            (current_point_index_1_based, total_points, point_progress_0_to_1)
        """
        if self.state == CalibrationState.VALIDATING:
            total = len(self.validation_points)
            idx = min(self.validation_point_idx + 1, total)
            progress = min(1.0, self.point_frame_counter / max(1, self.config.sample_frames_per_point))
            return (idx, total, progress)

        total = len(self.points)
        idx = min(self.current_point_idx + 1, total)
        target_samples = max(15, self.config.sample_frames_per_point - self.config.saccade_delay_frames)
        sample_prog = len(self.current_point_samples) / max(1, target_samples)
        frame_prog = self.point_frame_counter / max(1, self.config.sample_frames_per_point)
        progress = min(1.0, max(sample_prog, frame_prog))
        return (idx, total, progress)

    def _filter_outliers(self, samples: List[np.ndarray]) -> List[np.ndarray]:
        """
        Rejects statistical outliers using normalized feature Euclidean distance to median with 1.5*IQR threshold.
        Includes zero-variance safety guards to prevent division by zero or discarding identical samples.
        """
        if len(samples) < 5:
            return list(samples)

        arr = np.array(samples, dtype=np.float64)

        # Feature normalization to equalize variance across disparate scales
        medians = np.median(arr, axis=0)
        q75_feat, q25_feat = np.percentile(arr, [75, 25], axis=0)
        feat_iqr = q75_feat - q25_feat

        # Guard against zero-variance features
        std_feat = np.std(arr, axis=0)
        scale = np.where(feat_iqr > 1e-6, feat_iqr, np.where(std_feat > 1e-6, std_feat, 1.0))

        norm_diffs = (arr - medians) / scale
        dists = np.linalg.norm(norm_diffs, axis=1)

        # Check if all distances are zero (identical samples)
        if np.all(dists < 1e-6):
            return list(samples)

        q75, q25 = np.percentile(dists, [75, 25])
        iqr = q75 - q25
        cutoff = q75 + 1.5 * max(iqr, 1e-4)

        clean_samples = [samples[i] for i in range(len(samples)) if dists[i] <= cutoff]

        # Safety fallback: preserve at least 70% of samples if aggressive filtering occurs
        if len(clean_samples) < max(3, int(0.7 * len(samples))):
            sorted_indices = np.argsort(dists)
            keep_count = max(3, int(0.85 * len(samples)))
            clean_samples = [samples[i] for i in sorted_indices[:keep_count]]

        return clean_samples

    def process_frame(
        self,
        feature_vector: Optional[np.ndarray],
        is_valid_frame: bool,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Processes a video frame for calibration or validation.

        Args:
            feature_vector: Input eye & head pose feature vector.
            is_valid_frame: True if frame passed landmark confidence and blink checks.
            timestamp: Optional wall-clock timestamp (seconds).

        Returns:
            True if the calibration sequence or validation finished on this frame.
        """
        if self.state == CalibrationState.VALIDATING:
            return self._process_validation_frame(feature_vector, is_valid_frame, timestamp=timestamp)

        if self.state != CalibrationState.COLLECTING:
            return False

        now = timestamp if timestamp is not None else time.time()
        elapsed = now - self.point_start_time

        target = self.points[self.current_point_idx]
        # Progress tracking based on valid samples collected
        target_samples = max(15, self.config.sample_frames_per_point - self.config.saccade_delay_frames)

        # Saccade delay trimming: ignore initial reaction latency
        self.point_frame_counter += 1
        is_past_saccade = (self.point_frame_counter > self.config.saccade_delay_frames)
        
        if is_past_saccade:
            if is_valid_frame and feature_vector is not None:
                self.current_point_samples.append(feature_vector.copy())

        # Complete point when target samples gathered, OR on frame timeout if samples available
        sample_complete = len(self.current_point_samples) >= target_samples
        frame_timeout = (self.point_frame_counter >= self.config.sample_frames_per_point) and len(self.current_point_samples) >= 5
        hard_timeout = (self.point_frame_counter >= 60)  # Max 2 seconds per point

        point_complete = sample_complete or frame_timeout or hard_timeout

        if point_complete:
            clean_samples = self._filter_outliers(self.current_point_samples)

            for s in clean_samples:
                self.all_features.append(s)
                self.all_targets.append(target)
                self.all_point_ids.append(self.current_point_idx)

            # Advance to next target point
            self.current_point_idx += 1
            self.current_point_samples = []
            self.point_frame_counter = 0
            self.point_start_time = now

            # Check if sequence is complete
            if self.current_point_idx >= len(self.points):
                self._finish_and_train()
                return True

        return False

    def _finish_and_train(self):
        """Fits regressor on collected samples and serializes profile."""
        min_required = 6
        if len(self.all_features) >= min_required:
            X = np.array(self.all_features, dtype=np.float64)
            y = np.array(self.all_targets, dtype=np.float64)
            pt_ids = np.array(self.all_point_ids, dtype=np.int32)

            metrics = self.regressor.train(X, y, point_ids=pt_ids)
            lopo_str = f", LOPO MAE: {metrics['lopo_mae_px']:.1f}px" if "lopo_mae_px" in metrics else ""
            print(f"Calibration Complete! Model trained on {len(X)} samples. Train MAE: {metrics['mae_px']:.1f}px{lopo_str}")

            self.regressor.save()
            self.state = CalibrationState.FINISHED
        else:
            print(f"⚠️ Calibration failed: Insufficient clean samples ({len(self.all_features)} < {min_required}).")
            self.state = CalibrationState.IDLE

    # =========================================================================
    # Holdout Validation Mode
    # =========================================================================

    def start_validation(self, mode: str = "4_points"):
        """Starts post-calibration holdout validation mode."""
        self.validation_points = TargetGenerator.generate_validation_points(self.config, mode)
        self.validation_point_idx = 0
        self.validation_errors = []
        self.current_point_samples = []
        self.point_frame_counter = 0
        self.point_start_time = time.time()
        self.state = CalibrationState.VALIDATING

    def _process_validation_frame(
        self,
        feature_vector: Optional[np.ndarray],
        is_valid_frame: bool,
        timestamp: Optional[float] = None
    ) -> bool:
        """Collects validation predictions on holdout targets."""
        if self.state != CalibrationState.VALIDATING:
            return False

        now = timestamp if timestamp is not None else time.time()
        elapsed = now - self.point_start_time

        target = self.validation_points[self.validation_point_idx]
        self.point_frame_counter += 1

        is_past_saccade = (self.point_frame_counter > self.config.saccade_delay_frames)
        if is_past_saccade:
            if is_valid_frame and feature_vector is not None and self.regressor.is_trained:
                pred = self.regressor.predict(feature_vector)
                if pred is not None:
                    err_px = float(math.hypot(pred[0] - target[0], pred[1] - target[1]))
                    self.validation_errors.append(err_px)

        total_dwell_s = self.config.saccade_delay_seconds + self.config.collect_duration_seconds
        time_finished = (timestamp is not None) and (elapsed >= total_dwell_s)
        frame_finished = (self.point_frame_counter >= self.config.sample_frames_per_point)

        if frame_finished or time_finished:
            self.validation_point_idx += 1
            self.point_frame_counter = 0
            self.point_start_time = now

            if self.validation_point_idx >= len(self.validation_points):
                self._finish_validation()
                return True

        return False

    def _finish_validation(self):
        """Computes live validation MAE, RMSE, and visual angle error in degrees."""
        if not self.validation_errors:
            self.validation_metrics = {"val_mae_px": 0.0, "val_rmse_px": 0.0, "visual_angle_deg": 0.0}
            self.state = CalibrationState.VALIDATION_COMPLETE
            return

        errs = np.array(self.validation_errors, dtype=np.float64)
        mae_px = float(np.mean(errs))
        rmse_px = float(np.sqrt(np.mean(errs ** 2)))

        # Approximate pixel pitch for 24" 1080p display (~0.276 mm/px) and 600mm user distance
        pixel_pitch_mm = 0.276
        user_distance_mm = 600.0
        error_mm = mae_px * pixel_pitch_mm
        visual_angle_deg = float(math.degrees(math.atan2(error_mm, user_distance_mm)))

        self.validation_metrics = {
            "val_mae_px": mae_px,
            "val_rmse_px": rmse_px,
            "visual_angle_deg": visual_angle_deg,
            "samples_count": float(len(errs))
        }

        print(
            f"🎯 Validation Complete: MAE = {mae_px:.1f}px, RMSE = {rmse_px:.1f}px, "
            f"Visual Angle = {visual_angle_deg:.2f}°"
        )
        self.state = CalibrationState.VALIDATION_COMPLETE
