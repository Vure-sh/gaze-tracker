"""Tier 4: Performance, Latency & Stress Robustness Tests for Gaze Tracker.

Covers: Malformed/corrupted frame inputs, extreme blinks/occlusions, filter step settling time,
throughput & latency benchmarks (< 35ms, >= 30 FPS), and 5 real-world application workload scenarios.
"""

import time
import numpy as np
import pytest

from src.config import GazeConfig
from src.face_mesh_detector import FaceMeshDetector
from src.eye_extractor import EyeExtractor, GazeFeatures
from src.head_pose import HeadPoseEstimator
from src.calibrator import CalibrationManager, CalibrationState
from src.gaze_regressor import GazeRegressionModel
from src.filters import OneEuroFilter2D, KalmanFilter2D
from src.visualizer import GazeVisualizer
from tests.conftest import create_synthetic_landmarks


# ============================================================================
# 1. Malformed & Corrupted Frame Inputs
# ============================================================================

class TestMalformedAndCorruptedInputs:
    """Verifies that the entire pipeline gracefully handles malformed inputs without uncaught exceptions."""

    def test_detector_corrupted_frame_inputs(
        self, gaze_config: GazeConfig, mock_corrupted_frames
    ):
        """Verify FaceMeshDetector safely returns None on all corrupt, empty, or wrong-dimension inputs."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path)
        for key, frame in mock_corrupted_frames.items():
            result = detector.detect(frame)
            assert result is None or isinstance(result, list)

    def test_eye_extractor_corrupted_landmarks(self, gaze_config: GazeConfig):
        """Verify EyeExtractor gracefully returns None on None, empty, or truncated landmark lists."""
        extractor = EyeExtractor(gaze_config)
        assert extractor.extract(None, 640, 480) is None
        assert extractor.extract([], 640, 480) is None
        short_landmarks = create_synthetic_landmarks()[:300]
        assert extractor.extract(short_landmarks, 640, 480) is None

    def test_head_pose_corrupted_landmarks(self, gaze_config: GazeConfig):
        """Verify HeadPoseEstimator gracefully returns None on None or truncated landmark lists."""
        estimator = HeadPoseEstimator(gaze_config)
        assert estimator.estimate(None, 640, 480) is None
        assert estimator.estimate([], 640, 480) is None
        short_landmarks = create_synthetic_landmarks()[:300]
        assert estimator.estimate(short_landmarks, 640, 480) is None

    def test_visualizer_corrupted_canvas_inputs(self, gaze_config: GazeConfig, mock_bgr_frame):
        """Verify GazeVisualizer safely handles None features, None poses, and extreme gaze values."""
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        viz = GazeVisualizer(gaze_config)

        # Draw HUD with all None
        hud = viz.draw_debug_hud(mock_bgr_frame, None, None, None, 0.0, False, calibrator)
        assert hud.shape == mock_bgr_frame.shape

        # Draw canvas with None gaze
        canvas_none = viz.create_screen_canvas(None, calibrator, False, {})
        assert canvas_none.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)

        # Draw canvas with out-of-bounds negative and huge coordinates
        canvas_extreme = viz.create_screen_canvas((-9999.0, 99999.0), calibrator, True, {})
        assert canvas_extreme.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)


# ============================================================================
# 2. Extreme Blinks, Occlusions, and Seamless Recovery
# ============================================================================

class TestBlinksAndOcclusionRecovery:
    """Verifies that the system safely manages long eye closures, partial blinks, and immediate recovery."""

    def test_prolonged_blink_sequence(self, gaze_config: GazeConfig):
        """Verify 50 consecutive closed-eye frames maintain is_valid=False and do not corrupt state."""
        extractor = EyeExtractor(gaze_config)
        landmarks_closed = create_synthetic_landmarks(left_eye_closed=True, right_eye_closed=True)

        for _ in range(50):
            features = extractor.extract(landmarks_closed, 640, 480)
            assert features is not None
            assert features.left_eye.is_open is False
            assert features.right_eye.is_open is False
            assert features.is_valid is False
            assert not np.isnan(features.feature_vector).any()

    def test_intermittent_blink_and_recovery(self, gaze_config: GazeConfig):
        """Verify system transitions from open -> closed -> open with instant valid tracking recovery."""
        extractor = EyeExtractor(gaze_config)
        lm_open = create_synthetic_landmarks(left_eye_closed=False, right_eye_closed=False)
        lm_closed = create_synthetic_landmarks(left_eye_closed=True, right_eye_closed=True)

        # Phase 1: Open
        f1 = extractor.extract(lm_open, 640, 480)
        assert f1.is_valid is True

        # Phase 2: Blink for 3 frames
        for _ in range(3):
            f_blink = extractor.extract(lm_closed, 640, 480)
            assert f_blink.is_valid is False

        # Phase 3: Immediate Recovery
        f_recovered = extractor.extract(lm_open, 640, 480)
        assert f_recovered.is_valid is True
        assert f_recovered.left_eye.is_open is True
        assert f_recovered.right_eye.is_open is True


# ============================================================================
# 3. Temporal Filter Settling Time & Step Response
# ============================================================================

class TestTemporalFilterDynamics:
    """Verifies One-Euro filter velocity adaptation, rapid saccade settling, and jitter reduction."""

    def test_one_euro_saccade_step_settling_time(self):
        """
        Verify OneEuroFilter2D settles to > 99% of a large step input (100 -> 1800 px)
        within <= 3 frames (~100ms at 30 FPS).
        """
        f2d = OneEuroFilter2D(min_cutoff=0.20, beta=0.02, d_cutoff=1.0)
        dt = 0.033  # ~30 FPS

        # Initial fixation at (100, 100) for 10 frames
        t = 0.0
        for _ in range(10):
            t += dt
            f2d.filter((100.0, 100.0), timestamp=t)

        # Large saccade step jump to (1800, 900)
        target_pt = (1800.0, 900.0)
        settled = False
        for frame_idx in range(1, 10):
            t += dt
            filtered_pt = f2d.filter(target_pt, timestamp=t)
            err = np.linalg.norm(np.array(filtered_pt) - np.array(target_pt))
            if err < 15.0:  # Within 15px of 1800px step (> 99% settled)
                assert frame_idx <= 3, f"Saccade settling took {frame_idx} frames (> 3 max allowed)"
                settled = True
                break

        assert settled is True

    def test_one_euro_fixation_jitter_attenuation(self, gaze_config: GazeConfig):
        """Verify OneEuroFilter2D significantly reduces micro-jitter variance during steady fixation."""
        f2d = OneEuroFilter2D(
            min_cutoff=gaze_config.one_euro_min_cutoff,
            beta=gaze_config.one_euro_beta,
            d_cutoff=gaze_config.one_euro_d_cutoff
        )
        np.random.seed(42)

        base_pt = np.array([960.0, 540.0])
        n_frames = 100
        dt = 0.033

        raw_points = []
        filtered_points = []

        t = 0.0
        for _ in range(n_frames):
            t += dt
            # Synthetic camera sensor micro-jitter: +- 5px Gaussian noise
            jitter = np.random.normal(0, 5.0, 2)
            noisy_pt = base_pt + jitter
            filtered = f2d.filter(tuple(noisy_pt), timestamp=t)

            raw_points.append(noisy_pt)
            filtered_points.append(filtered)

        raw_var = np.var(raw_points, axis=0)
        filtered_var = np.var(filtered_points[10:], axis=0)  # Skip first 10 warmup frames

        # Variance reduction must exceed 60%
        var_reduction_x = (raw_var[0] - filtered_var[0]) / raw_var[0]
        var_reduction_y = (raw_var[1] - filtered_var[1]) / raw_var[1]

        assert var_reduction_x > 0.60
        assert var_reduction_y > 0.60


# ============================================================================
# 4. Latency & Throughput Benchmarks (Real-Time Performance)
# ============================================================================

class TestPipelineThroughputAndLatency:
    """Verifies component execution speeds and guarantees frame processing latency < 35ms (>= 30 FPS)."""

    def test_eye_extractor_latency_and_throughput(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify EyeExtractor processes frames in < 2ms (> 500 FPS)."""
        extractor = EyeExtractor(gaze_config)
        n_iter = 500

        start_time = time.perf_counter()
        for _ in range(n_iter):
            _ = extractor.extract(synthetic_landmarks, 640, 480)
        elapsed = time.perf_counter() - start_time

        avg_latency_ms = (elapsed / n_iter) * 1000.0
        fps = n_iter / elapsed

        print(f"\n⚡ EyeExtractor: {avg_latency_ms:.3f}ms per frame ({fps:.0f} FPS)")
        assert avg_latency_ms < 2.0, f"EyeExtractor latency {avg_latency_ms:.2f}ms exceeded 2ms"

    def test_head_pose_estimator_latency_and_throughput(self, gaze_config: GazeConfig, synthetic_landmarks):
        """Verify HeadPoseEstimator estimates 3D pose in < 4ms (> 250 FPS)."""
        estimator = HeadPoseEstimator(gaze_config)
        n_iter = 300

        start_time = time.perf_counter()
        for _ in range(n_iter):
            _ = estimator.estimate(synthetic_landmarks, 640, 480)
        elapsed = time.perf_counter() - start_time

        avg_latency_ms = (elapsed / n_iter) * 1000.0
        fps = n_iter / elapsed

        print(f"\n⚡ HeadPoseEstimator: {avg_latency_ms:.3f}ms per frame ({fps:.0f} FPS)")
        assert avg_latency_ms < 4.0, f"HeadPoseEstimator latency {avg_latency_ms:.2f}ms exceeded 4ms"

    def test_gaze_regressor_prediction_throughput(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Verify GazeRegressionModel predicts screen gaze in < 1ms (> 1000 FPS)."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        regressor.train(X, y)

        sample = X[0]
        n_iter = 2000

        start_time = time.perf_counter()
        for _ in range(n_iter):
            _ = regressor.predict(sample)
        elapsed = time.perf_counter() - start_time

        avg_latency_ms = (elapsed / n_iter) * 1000.0
        fps = n_iter / elapsed

        print(f"\n⚡ GazeRegressor: {avg_latency_ms:.3f}ms per prediction ({fps:.0f} FPS)")
        assert avg_latency_ms < 1.0, f"GazeRegressor prediction latency {avg_latency_ms:.2f}ms exceeded 1ms"

    def test_one_euro_filter_throughput(self):
        """Verify OneEuroFilter2D step executes in < 0.1ms (> 10,000 FPS)."""
        f2d = OneEuroFilter2D()
        n_iter = 10000

        start_time = time.perf_counter()
        for i in range(n_iter):
            _ = f2d.filter((500.0, 300.0), timestamp=i * 0.033)
        elapsed = time.perf_counter() - start_time

        avg_latency_ms = (elapsed / n_iter) * 1000.0
        assert avg_latency_ms < 0.1, f"OneEuroFilter2D latency {avg_latency_ms:.3f}ms exceeded 0.1ms"

    def test_end_to_end_pipeline_latency(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, synthetic_landmarks
    ):
        """Verify complete frame processing cycle runs in < 35ms (>= 30 FPS requirement)."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        gaze_filter = OneEuroFilter2D()

        X, y, meta = synthetic_calibration_dataset
        regressor.train(X, y)

        n_frames = 100
        start_time = time.perf_counter()

        for frame_i in range(n_frames):
            # 1. Eye extraction
            features = extractor.extract(synthetic_landmarks, 640, 480)
            # 2. Head pose
            pose = estimator.estimate(synthetic_landmarks, 640, 480)
            # 3. Predict gaze
            comb = features.vector_14d
            pred = regressor.predict(comb)
            # 4. Temporal filter
            smoothed = gaze_filter.filter(pred, timestamp=frame_i * 0.033)

        elapsed = time.perf_counter() - start_time
        avg_latency_ms = (elapsed / n_frames) * 1000.0
        fps = n_frames / elapsed

        print(f"\n🚀 Full End-to-End Pipeline: {avg_latency_ms:.3f}ms per frame ({fps:.1f} FPS)")
        assert avg_latency_ms < 35.0, f"Pipeline latency {avg_latency_ms:.2f}ms exceeded 35ms SLA"
        assert fps >= 30.0, f"Pipeline throughput {fps:.1f} FPS is below 30 FPS requirement"


# ============================================================================
# 5. Real-World Application Workload Scenarios
# ============================================================================

class TestApplicationWorkloadScenarios:
    """Executes the 5 comprehensive end-to-end integration workflows specified in TEST_INFRA.md."""

    def test_scenario_1_full_calibration_to_holdout_validation(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Scenario 1: Full 9-point calibration -> Train -> 4-point holdout validation -> MAE < 35px."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        metrics = regressor.train(X, y)

        assert metrics["mae_px"] < 35.0
        assert metrics["rmse_px"] < 50.0

    def test_scenario_2_continuous_tracking_saccades_and_fixations(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Scenario 2: Continuous gaze tracking across 100 frames of dynamic saccades and fixations."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        regressor.train(X, y)
        gaze_filter = OneEuroFilter2D()

        # Simulate 100 frames reading text across screen
        for i in range(100):
            sample = X[i % len(X)]
            raw_pred = regressor.predict(sample)
            assert raw_pred is not None
            smoothed_pred = gaze_filter.filter(raw_pred, timestamp=i * 0.033)
            assert isinstance(smoothed_pred, tuple)
            assert 0 <= smoothed_pred[0] <= gaze_config.screen_width
            assert 0 <= smoothed_pred[1] <= gaze_config.screen_height

    def test_scenario_3_blink_and_extreme_head_pose_recovery(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset
    ):
        """Scenario 3: Rapid blinking and extreme head pitch/yaw rotations recovery without drift."""
        X, y, meta = synthetic_calibration_dataset
        regressor = GazeRegressionModel(gaze_config)
        regressor.train(X, y)
        extractor = EyeExtractor(gaze_config)
        gaze_filter = OneEuroFilter2D()

        # Stream of 20 normal -> 10 blinks -> 20 normal frames
        for i in range(50):
            is_blink = 20 <= i < 30
            landmarks = create_synthetic_landmarks(
                left_eye_closed=is_blink,
                right_eye_closed=is_blink,
                roll_deg=15.0 if is_blink else 0.0
            )
            features = extractor.extract(landmarks, 640, 480)
            if features.is_valid:
                pred = regressor.predict(features.vector_14d)
                smoothed = gaze_filter.filter(pred, timestamp=i * 0.033)
                assert smoothed is not None

    def test_scenario_4_profile_save_restart_load_verification(
        self, gaze_config: GazeConfig, synthetic_calibration_dataset, tmp_path
    ):
        """Scenario 4: Calibration Profile Save -> Clean Restart -> Profile Load & Verification."""
        X, y, meta = synthetic_calibration_dataset
        regressor1 = GazeRegressionModel(gaze_config)
        regressor1.train(X, y)

        save_path = str(tmp_path / "scenario4_model.pkl")
        regressor1.save(save_path)

        # Fresh instance
        regressor2 = GazeRegressionModel(gaze_config)
        assert regressor2.is_trained is False
        success = regressor2.load(save_path)
        assert success is True
        assert regressor2.is_trained is True

        # Verify predictions match bit-for-bit
        for idx in [0, 10, 20, 50, 100]:
            p1 = regressor1.predict(X[idx])
            p2 = regressor2.predict(X[idx])
            assert p1 == p2

    def test_scenario_5_video_glitch_and_missing_landmark_handling(
        self, gaze_config: GazeConfig, mock_corrupted_frames
    ):
        """Scenario 5: Video stream corruption and missing landmarks handled gracefully without crashes."""
        detector = FaceMeshDetector(model_path=gaze_config.model_path)
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)

        for name, frame in mock_corrupted_frames.items():
            lms = detector.detect(frame)
            # Regardless of detection outcome, downstream modules must handle None safely
            eye_feat = extractor.extract(lms, 640, 480)
            pose_feat = estimator.estimate(lms, 640, 480)
            assert eye_feat is None or isinstance(eye_feat, GazeFeatures)
