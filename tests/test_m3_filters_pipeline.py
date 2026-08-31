"""Milestone 3 Verification: Temporal Filtering, Threaded Camera Stream & Real-Time Performance Pipeline."""

import time
import numpy as np
import pytest

from src.config import GazeConfig
from src.filters.one_euro import LowPassFilter, OneEuroFilter1D, OneEuroFilter2D
from src.filters.kalman import KalmanFilter2D
from src.camera_stream import ThreadedCameraStream, open_camera_device, ensure_tablet_stream
from src.pipeline import GazePipeline, GazePipelineResult
from src.calibration.calibrator import CalibrationState
from tests.conftest import create_synthetic_landmarks


# ============================================================================
# 1. One-Euro Adaptive Filter & Low-Pass Filter
# ============================================================================

class TestOneEuroTemporalFiltering:
    """Tests for One-Euro velocity adaptation, deadband, timestamping, and jitter reduction."""

    def test_low_pass_filter_alpha_variation(self):
        """Verify LowPassFilter behaves properly with dynamic alpha adjustments."""
        lpf = LowPassFilter(alpha=0.2)
        assert lpf.filter(100.0) == 100.0
        # Step with alpha 0.5
        v2 = lpf.filter(200.0, alpha=0.5)
        assert v2 == 150.0  # 0.5*200 + 0.5*100
        lpf.reset()
        assert lpf.hat_x_prev is None

    def test_one_euro_1d_deadband(self):
        """Verify deadband absorbs micro-fluctuations below threshold."""
        f = OneEuroFilter1D(min_cutoff=0.1, beta=0.1, deadband=2.0)
        v1 = f.filter(100.0, timestamp=0.0)
        assert v1 == 100.0

        # Sub-deadband change (100.0 -> 101.5, diff = 1.5 < 2.0)
        v2 = f.filter(101.5, timestamp=0.033)
        assert v2 == 100.0

        # Change exceeding deadband (100.0 -> 105.0, diff = 5.0 >= 2.0)
        v3 = f.filter(105.0, timestamp=0.066)
        assert v3 > 100.0

    def test_one_euro_2d_fixation_jitter_below_threshold(self):
        """Verify OneEuroFilter2D keeps steady fixation jitter below 1.1px variance."""
        f2d = OneEuroFilter2D(min_cutoff=0.02, beta=0.01, d_cutoff=1.0)
        np.random.seed(42)

        target_center = (960.0, 540.0)
        dt = 0.033  # ~30 FPS
        filtered_points = []

        t = 0.0
        for _ in range(120):
            t += dt
            # Gaussian noise with sigma = 3px
            noisy_x = target_center[0] + np.random.normal(0, 3.0)
            noisy_y = target_center[1] + np.random.normal(0, 3.0)
            filtered = f2d.filter((noisy_x, noisy_y), timestamp=t)
            filtered_points.append(filtered)

        # Discard warmup
        stable_pts = np.array(filtered_points[20:])
        var_x = float(np.var(stable_pts[:, 0]))
        var_y = float(np.var(stable_pts[:, 1]))

        print(f"\nFixation Filter Variance: X={var_x:.4f}px², Y={var_y:.4f}px²")
        assert var_x < 1.1, f"Fixation X variance {var_x:.3f} exceeded 1.1px"
        assert var_y < 1.1, f"Fixation Y variance {var_y:.3f} exceeded 1.1px"

    def test_one_euro_2d_saccade_fast_settling(self):
        """Verify fast settling time during 1000px saccade jump (<= 3 frames)."""
        f2d = OneEuroFilter2D(min_cutoff=0.1, beta=0.05, d_cutoff=1.0)
        dt = 0.033

        # Warmup at (200, 200)
        t = 0.0
        for _ in range(15):
            t += dt
            f2d.filter((200.0, 200.0), timestamp=t)

        # Saccade jump to (1200, 800)
        saccade_target = (1200.0, 800.0)
        settled_frame = None

        for frame_idx in range(1, 10):
            t += dt
            pt = f2d.filter(saccade_target, timestamp=t)
            dist = np.hypot(pt[0] - saccade_target[0], pt[1] - saccade_target[1])
            if dist < 20.0:
                settled_frame = frame_idx
                break

        assert settled_frame is not None
        assert settled_frame <= 3, f"Saccade settling took {settled_frame} frames (> 3)"

    def test_one_euro_timestamp_monotonicity_guards(self):
        """Verify graceful handling when timestamps jump backward or have zero delta."""
        f = OneEuroFilter2D()
        p1 = f.filter((100.0, 100.0), timestamp=1.0)
        # Duplicate timestamp
        p2 = f.filter((200.0, 200.0), timestamp=1.0)
        assert p2 == p1
        # Backward timestamp
        p3 = f.filter((300.0, 300.0), timestamp=0.5)
        assert isinstance(p3, tuple)


# ============================================================================
# 2. Kalman Filter 2D
# ============================================================================

class TestKalmanFilter2D:
    """Tests for 2D constant-velocity Kalman filter dynamics and state updates."""

    def test_kalman_initialization_and_first_measurement(self):
        """Verify Kalman filter matches initial measurement exactly on step 1."""
        kf = KalmanFilter2D(process_noise=1e-3, measurement_noise=1e-1)
        assert kf.is_initialized is False
        pt = kf.filter((640.0, 480.0), timestamp=0.0)
        assert pt == (640.0, 480.0)
        assert kf.is_initialized is True
        assert kf.state[0, 0] == 640.0
        assert kf.state[1, 0] == 480.0

    def test_kalman_velocity_tracking_linear_motion(self):
        """Verify Kalman filter accurately tracks constant velocity linear motion."""
        kf = KalmanFilter2D(process_noise=1e-2, measurement_noise=1e-1)
        dt = 0.033
        velocity = (300.0, 150.0)  # px/s

        t = 0.0
        pos = np.array([100.0, 100.0])
        filtered_pt = (100.0, 100.0)

        for _ in range(50):
            t += dt
            pos = pos + np.array(velocity) * dt
            # Add small measurement noise
            noisy_pos = pos + np.random.normal(0, 0.5, 2)
            filtered_pt = kf.filter(tuple(noisy_pos), timestamp=t)

        # After 50 steps, estimated state velocity should closely match true velocity
        est_vx = kf.state[2, 0]
        est_vy = kf.state[3, 0]
        assert abs(est_vx - velocity[0]) < 25.0
        assert abs(est_vy - velocity[1]) < 25.0
        assert abs(filtered_pt[0] - pos[0]) < 10.0
        assert abs(filtered_pt[1] - pos[1]) < 10.0

    def test_kalman_reset(self):
        """Verify reset clears state vector and initialization flag."""
        kf = KalmanFilter2D()
        kf.filter((500.0, 500.0), timestamp=1.0)
        kf.filter((510.0, 510.0), timestamp=1.033)
        assert kf.is_initialized is True

        kf.reset()
        assert kf.is_initialized is False
        assert kf.last_time is None


# ============================================================================
# 3. Threaded Camera Stream & Device Discovery
# ============================================================================

class TestCameraStream:
    """Tests for asynchronous threaded capture and device fallback handling."""

    def test_open_camera_device_fallback_safe(self):
        """Verify open_camera_device handles non-existent device without throwing uncaught exceptions."""
        cap, dev = open_camera_device("non_existent_video_device_999", fallback_devices=[99, 98])
        assert cap is None or hasattr(cap, "isOpened")

    def test_threaded_camera_stream_init_properties(self):
        """Verify ThreadedCameraStream instance configuration."""
        stream = ThreadedCameraStream(camera_src=9999, width=1280, height=720, fps=30)
        assert stream.width == 1280
        assert stream.height == 720
        assert stream.target_fps == 30
        assert stream.is_opened is False

    def test_threaded_camera_stream_stop_idempotent(self):
        """Verify calling stop() multiple times is safe and idempotent."""
        stream = ThreadedCameraStream(camera_src=9999)
        stream.stop()
        stream.stop()
        stream.release()
        assert stream._running is False


# ============================================================================
# 4. End-to-End Real-Time Pipeline
# ============================================================================

class TestGazePipeline:
    """Tests for GazePipeline orchestration, calibration lifecycle, filter switching, and latency."""

    def test_pipeline_initialization(self, gaze_config: GazeConfig):
        """Verify GazePipeline initializes all sub-components with default config."""
        pipeline = GazePipeline(config=gaze_config)
        assert pipeline.config == gaze_config
        assert pipeline.detector is not None
        assert pipeline.eye_extractor is not None
        assert pipeline.head_pose_estimator is not None
        assert pipeline.quality_tracker is not None
        assert pipeline.regressor is not None
        assert pipeline.calibrator is not None
        assert pipeline.gaze_filter is not None
        assert pipeline.filter_type in ("one_euro", "kalman")

    def test_pipeline_process_none_and_empty_frame(self, gaze_config: GazeConfig):
        """Verify pipeline returns valid GazePipelineResult on None or empty frame."""
        pipeline = GazePipeline(config=gaze_config)
        
        # Test None
        res_none = pipeline.process_frame(None)
        assert isinstance(res_none, GazePipelineResult)
        assert res_none.is_valid is False
        assert res_none.raw_gaze is None
        assert res_none.smoothed_gaze is None

        # Test Empty frame
        res_empty = pipeline.process_frame(np.zeros((0, 0), dtype=np.uint8))
        assert isinstance(res_empty, GazePipelineResult)
        assert res_empty.is_valid is False

    def test_pipeline_process_synthetic_frame_uncalibrated(
        self, gaze_config: GazeConfig, mock_bgr_frame
    ):
        """Verify pipeline execution on uncalibrated state processes features without crash."""
        pipeline = GazePipeline(config=gaze_config)
        result = pipeline.process_frame(mock_bgr_frame, timestamp=1.0)

        assert isinstance(result, GazePipelineResult)
        assert result.latency_ms >= 0.0
        assert pipeline.get_average_latency_ms() >= 0.0

    def test_pipeline_filter_switching(self, gaze_config: GazeConfig):
        """Verify dynamically switching between One-Euro and Kalman filters."""
        pipeline = GazePipeline(config=gaze_config, filter_type="one_euro")
        assert isinstance(pipeline.gaze_filter, OneEuroFilter2D)

        pipeline.set_filter_type("kalman")
        assert isinstance(pipeline.gaze_filter, KalmanFilter2D)
        assert pipeline.filter_type == "kalman"

        pipeline.set_filter_type("one_euro")
        assert isinstance(pipeline.gaze_filter, OneEuroFilter2D)
        assert pipeline.filter_type == "one_euro"

    def test_pipeline_calibration_and_prediction_cycle(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, tmp_path
    ):
        """Verify pipeline trains on calibration data, saves, loads, and predicts coordinates."""
        pipeline = GazePipeline(config=gaze_config)
        X, y, meta = synthetic_calibration_dataset

        # Train regressor
        metrics = pipeline.regressor.train(X, y)
        assert metrics["mae_px"] < 35.0
        assert pipeline.regressor.is_trained is True

        # Save calibration
        save_file = str(tmp_path / "test_pipeline_calib.pkl")
        assert pipeline.save_calibration(save_file) is True

        # Fresh pipeline instance
        pipeline2 = GazePipeline(config=gaze_config)
        assert pipeline2.regressor.is_trained is False
        assert pipeline2.load_calibration(save_file) is True
        assert pipeline2.regressor.is_trained is True

    def test_pipeline_latency_under_35ms_benchmark(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, mock_bgr_frame
    ):
        """Benchmark: Verify GazePipeline process_frame cycle executes well below 35ms SLA."""
        pipeline = GazePipeline(config=gaze_config)
        X, y, meta = synthetic_calibration_dataset
        pipeline.regressor.train(X, y)

        n_frames = 50
        start = time.perf_counter()
        for i in range(n_frames):
            _ = pipeline.process_frame(mock_bgr_frame, timestamp=i * 0.033)
        total_time = time.perf_counter() - start

        avg_latency_ms = (total_time / n_frames) * 1000.0
        fps = n_frames / total_time

        print(f"\n⚡ GazePipeline Benchmark: {avg_latency_ms:.3f}ms per frame ({fps:.1f} FPS)")
        assert avg_latency_ms < 35.0, f"Latency {avg_latency_ms:.2f}ms exceeded 35ms SLA"
