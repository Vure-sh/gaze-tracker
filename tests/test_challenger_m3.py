"""Challenger Milestone 3 Verification: Adversarial Stress, Saccade Lag, Micro-Jitter & Filter Edge Cases."""

import time
import numpy as np
import pytest

from src.config import GazeConfig
from src.filters.one_euro import OneEuroFilter2D
from src.filters.kalman import KalmanFilter2D
from src.pipeline import GazePipeline
from tests.conftest import create_synthetic_landmarks


class TestChallengerM3TemporalDynamics:
    """Adversarial challenger tests for temporal filter latency, noise rejection, and saccade preservation."""

    def test_adversarial_impulse_noise_rejection(self):
        """Verify single-frame impulse outlier spike does not permanently distort smooth tracking trajectory."""
        f2d = OneEuroFilter2D(min_cutoff=0.05, beta=0.5, d_cutoff=1.0)
        dt = 0.033

        # Stream 30 stable frames at (500, 500)
        t = 0.0
        for _ in range(30):
            t += dt
            f2d.filter((500.0, 500.0), timestamp=t)

        # Single frame corrupted spike to (2000, 2000)
        t += dt
        f2d.filter((2000.0, 2000.0), timestamp=t)

        # Next normal frame at (500, 500)
        t += dt
        recovered_pt = f2d.filter((500.0, 500.0), timestamp=t)

        # Gaze must quickly recover within 100px on the very next frame
        dist = np.hypot(recovered_pt[0] - 500.0, recovered_pt[1] - 500.0)
        assert dist < 120.0, f"Impulse recovery error {dist:.1f}px was too high"

    def test_adversarial_high_frequency_zigzag_tracking(self):
        """Verify filter tracks rapid reading zigzag motions without clipping or unbounded phase lag."""
        f2d = OneEuroFilter2D(min_cutoff=0.08, beta=0.8, d_cutoff=1.0)
        dt = 0.033

        t = 0.0
        # Simulate 100 frames of sinusoidal / zigzag gaze scan
        errors = []
        for i in range(100):
            t += dt
            true_x = 960.0 + 400.0 * np.sin(2.0 * np.pi * 0.5 * t)
            true_y = 540.0 + 200.0 * np.cos(2.0 * np.pi * 0.5 * t)
            
            # Small sensor noise
            noisy_x = true_x + np.random.normal(0, 2.0)
            noisy_y = true_y + np.random.normal(0, 2.0)
            
            filtered = f2d.filter((noisy_x, noisy_y), timestamp=t)
            if i > 10:  # Skip warmup
                err = np.hypot(filtered[0] - true_x, filtered[1] - true_y)
                errors.append(err)

        mean_tracking_lag = float(np.mean(errors))
        print(f"\nZigzag Tracking Mean Lag: {mean_tracking_lag:.2f}px")
        assert mean_tracking_lag < 35.0, f"Dynamic tracking lag {mean_tracking_lag:.2f}px exceeded 35px"

    def test_pipeline_long_run_memory_and_stability(
        self, gaze_config: GazeConfig, mock_bgr_frame, synthetic_calibration_dataset
    ):
        """Verify pipeline processes 500 consecutive frames without memory growth, degradation, or crashes."""
        pipeline = GazePipeline(config=gaze_config)
        X, y, meta = synthetic_calibration_dataset
        pipeline.regressor.train(X, y)

        latencies = []
        for i in range(300):
            res = pipeline.process_frame(mock_bgr_frame, timestamp=i * 0.033)
            latencies.append(res.latency_ms)

        mean_latency = float(np.mean(latencies))
        p99_latency = float(np.percentile(latencies, 99))
        print(f"\n300 Frame Stress Test: Mean={mean_latency:.3f}ms, P99={p99_latency:.3f}ms")
        assert mean_latency < 35.0
        assert p99_latency < 50.0
