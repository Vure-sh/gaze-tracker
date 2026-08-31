"""Adversarial stress-test harness for Milestone 1 (CV & Robust Feature Engineering).

Evaluates:
1. Periocular lighting contrast variations (low contrast, saturated glare, uniform gray, underexposure).
2. Iris circularity metric under perturbed/deformed iris landmarks.
3. Temporal landmark jitter stability metrics under simulated Gaussian noise.
4. High-throughput execution latency and profiling across 1,000 synthetic frames.
"""

from __future__ import annotations
import math
import time
import numpy as np
import pytest
import cv2

from src.config import GazeConfig, QualityConfig
from src.types import NormalizedPoint, EyeData, HeadPoseData, GazeFeatures, TrackingQuality
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from src.cv.quality_tracker import QualityTracker
from tests.test_m1_cv import build_synthetic_landmarks


# ============================================================================
# 1. Periocular Lighting Contrast Variations
# ============================================================================

class TestLightingContrastAdversarial:
    """Stress tests QualityTracker photometric evaluation across adverse lighting."""

    @pytest.fixture
    def setup_pipeline(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        tracker = QualityTracker(config)
        landmarks = build_synthetic_landmarks()
        features = extractor.extract(landmarks, 640, 480)
        return config, extractor, tracker, landmarks, features

    def test_uniform_gray_zero_contrast(self, setup_pipeline):
        config, _, tracker, landmarks, features = setup_pipeline
        # All pixels = 128 (stddev = 0.0)
        gray_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        quality = tracker.evaluate(features, landmarks, frame=gray_frame, img_w=640, img_h=480)

        assert quality.contrast_score == 0.0
        assert "Low periocular lighting contrast" in quality.failure_reasons
        # Composite score must be penalized by contrast_weight * 0
        expected_penalty = config.quality.contrast_weight * 1.0
        assert quality.confidence <= (1.0 - expected_penalty + 1e-4)

    def test_saturated_overexposure_glare(self, setup_pipeline):
        config, _, tracker, landmarks, features = setup_pipeline
        # Overexposed frame: saturated white (255) with minimal noise
        glare_frame = np.clip(np.random.normal(254, 0.5, (480, 640, 3)), 0, 255).astype(np.uint8)

        quality = tracker.evaluate(features, landmarks, frame=glare_frame, img_w=640, img_h=480)

        assert quality.contrast_score < 0.10
        assert "Low periocular lighting contrast" in quality.failure_reasons

    def test_saturated_underexposure_darkness(self, setup_pipeline):
        config, _, tracker, landmarks, features = setup_pipeline
        # Completely dark frame: 0-1 pixel values
        dark_frame = np.clip(np.random.normal(1, 0.5, (480, 640, 3)), 0, 255).astype(np.uint8)

        quality = tracker.evaluate(features, landmarks, frame=dark_frame, img_w=640, img_h=480)

        assert quality.contrast_score < 0.10
        assert "Low periocular lighting contrast" in quality.failure_reasons

    def test_contrast_sweep_monotonicity(self, setup_pipeline):
        config, _, tracker, landmarks, features = setup_pipeline
        std_targets = [0.0, 5.0, 10.0, 15.0, 25.0, 35.0, 50.0]
        scores = []

        for std in std_targets:
            # Create synthetic eye region texture with specific standard deviation
            frame = np.clip(np.random.normal(128, std, (480, 640, 3)), 0, 255).astype(np.uint8)
            tracker.reset()
            q = tracker.evaluate(features, landmarks, frame=frame, img_w=640, img_h=480)
            scores.append(q.contrast_score)

        # Ensure monotonically non-decreasing up to saturation at 1.0
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1] + 0.05

    def test_corrupted_frame_dimensions_resilience(self, setup_pipeline):
        _, _, tracker, landmarks, features = setup_pipeline
        # Single-channel grayscale, empty frame, None frame, wrong dimensions
        frames = [
            np.full((480, 640), 128, dtype=np.uint8),
            np.zeros((0, 0, 3), dtype=np.uint8),
            None,
            np.zeros((10, 10, 3), dtype=np.uint8)  # Tiny ROI
        ]
        for f in frames:
            tracker.reset()
            q = tracker.evaluate(features, landmarks, frame=f, img_w=640, img_h=480)
            assert isinstance(q, TrackingQuality)
            assert 0.0 <= q.confidence <= 1.0


# ============================================================================
# 2. Iris Circularity Metric under Perturbed / Deformed Iris Landmarks
# ============================================================================

class TestIrisCircularityAdversarial:
    """Stress tests iris circularity under severe landmark deformations and occlusions."""

    @pytest.fixture
    def setup_extractor(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        tracker = QualityTracker(config)
        return config, extractor, tracker

    def _build_lms_with_custom_iris(
        self,
        left_iris_pts: list[tuple[float, float]],
        right_iris_pts: list[tuple[float, float]],
        img_w: int = 640,
        img_h: int = 480
    ):
        lms = build_synthetic_landmarks(img_w=img_w, img_h=img_h)
        # Left iris center (468) and perimeter (469, 470, 471, 472)
        lms[468] = NormalizedPoint(x=left_iris_pts[0][0] / img_w, y=left_iris_pts[0][1] / img_h, z=0.0)
        for i, idx in enumerate([469, 470, 471, 472]):
            lms[idx] = NormalizedPoint(x=left_iris_pts[i + 1][0] / img_w, y=left_iris_pts[i + 1][1] / img_h, z=0.0)

        # Right iris center (473) and perimeter (474, 475, 476, 477)
        lms[473] = NormalizedPoint(x=right_iris_pts[0][0] / img_w, y=right_iris_pts[0][1] / img_h, z=0.0)
        for i, idx in enumerate([474, 475, 476, 477]):
            lms[idx] = NormalizedPoint(x=right_iris_pts[i + 1][0] / img_w, y=right_iris_pts[i + 1][1] / img_h, z=0.0)

        return lms

    def test_perfect_circular_iris(self, setup_extractor):
        config, extractor, tracker = setup_extractor
        # Radius = 6.0 in all 4 cardinal directions
        c_L = (260.0, 200.0)
        iris_L = [c_L, (c_L[0] + 6.0, c_L[1]), (c_L[0] - 6.0, c_L[1]), (c_L[0], c_L[1] - 6.0), (c_L[0], c_L[1] + 6.0)]
        c_R = (380.0, 200.0)
        iris_R = [c_R, (c_R[0] + 6.0, c_R[1]), (c_R[0] - 6.0, c_R[1]), (c_R[0], c_R[1] - 6.0), (c_R[0], c_R[1] + 6.0)]

        lms = self._build_lms_with_custom_iris(iris_L, iris_R)
        feat = extractor.extract(lms, 640, 480)

        assert math.isclose(feat.left_eye.circularity, 1.0, abs_tol=1e-4)
        assert math.isclose(feat.right_eye.circularity, 1.0, abs_tol=1e-4)
        assert feat.left_eye.iris_diameter_px == 12.0

        q = tracker.evaluate(feat, lms, img_w=640, img_h=480)
        assert q.circularity_score > 0.95
        assert "Iris contour deformation or partial occlusion" not in q.failure_reasons

    def test_elliptical_iris_squish(self, setup_extractor):
        config, extractor, tracker = setup_extractor
        # Major axis = 9.0, Minor axis = 3.0 (aspect ratio 3:1)
        c_L = (260.0, 200.0)
        iris_L = [c_L, (c_L[0] + 9.0, c_L[1]), (c_L[0] - 9.0, c_L[1]), (c_L[0], c_L[1] - 3.0), (c_L[0], c_L[1] + 3.0)]
        c_R = (380.0, 200.0)
        iris_R = [c_R, (c_R[0] + 9.0, c_R[1]), (c_R[0] - 9.0, c_R[1]), (c_R[0], c_R[1] - 3.0), (c_R[0], c_R[1] + 3.0)]

        lms = self._build_lms_with_custom_iris(iris_L, iris_R)
        feat = extractor.extract(lms, 640, 480)

        # Radii are [9, 9, 3, 3] -> mean=6.0, var=9.0
        # circularity = exp(-9.0 / 4.0) = exp(-2.25) ~= 0.1054
        assert feat.left_eye.circularity < 0.20

        q = tracker.evaluate(feat, lms, img_w=640, img_h=480)
        assert q.circularity_score < 0.20
        assert "Iris contour deformation or partial occlusion" in q.failure_reasons

    def test_partial_eyelid_occlusion_asymmetry(self, setup_extractor):
        config, extractor, tracker = setup_extractor
        # Top landmark occluded / pulled inward (r_top = 2.0 instead of 6.0)
        c_L = (260.0, 200.0)
        iris_L = [c_L, (c_L[0] + 6.0, c_L[1]), (c_L[0] - 6.0, c_L[1]), (c_L[0], c_L[1] - 2.0), (c_L[0], c_L[1] + 6.0)]
        c_R = (380.0, 200.0)
        iris_R = [c_R, (c_R[0] + 6.0, c_R[1]), (c_R[0] - 6.0, c_R[1]), (c_R[0], c_R[1] - 2.0), (c_R[0], c_R[1] + 6.0)]

        lms = self._build_lms_with_custom_iris(iris_L, iris_R)
        feat = extractor.extract(lms, 640, 480)

        # Radii [6, 6, 2, 6] -> mean = 5.0, var = 3.0
        # circularity = exp(-3.0 / 4.0) = exp(-0.75) ~= 0.472
        assert 0.40 < feat.left_eye.circularity < 0.50
        q = tracker.evaluate(feat, lms, img_w=640, img_h=480)
        assert "Iris contour deformation or partial occlusion" in q.failure_reasons

    def test_severe_random_perturbation(self, setup_extractor):
        config, extractor, tracker = setup_extractor
        np.random.seed(123)
        c_L = (260.0, 200.0)
        # Random chaotic offsets
        iris_L = [c_L] + [(c_L[0] + np.random.uniform(-15, 15), c_L[1] + np.random.uniform(-15, 15)) for _ in range(4)]
        c_R = (380.0, 200.0)
        iris_R = [c_R] + [(c_R[0] + np.random.uniform(-15, 15), c_R[1] + np.random.uniform(-15, 15)) for _ in range(4)]

        lms = self._build_lms_with_custom_iris(iris_L, iris_R)
        feat = extractor.extract(lms, 640, 480)

        assert feat.left_eye.circularity < 0.10
        q = tracker.evaluate(feat, lms, img_w=640, img_h=480)
        assert q.circularity_score < 0.10


# ============================================================================
# 3. Temporal Landmark Jitter Stability Metrics under Simulated Noise
# ============================================================================

class TestTemporalJitterStabilityAdversarial:
    """Stress tests QualityTracker temporal jitter detection and recovery under Gaussian noise."""

    def test_zero_noise_stability(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        tracker = QualityTracker(config)
        lms = build_synthetic_landmarks()

        for _ in range(15):
            feat = extractor.extract(lms, 640, 480)
            q = tracker.evaluate(feat, lms, img_w=640, img_h=480)

        assert math.isclose(q.stability_score, 1.0, abs_tol=1e-3)
        assert "Landmark high-frequency tracking jitter" not in q.failure_reasons

    def test_gaussian_noise_jitter_scaling(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        np.random.seed(42)

        noise_sigmas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        avg_stabilities = []

        for sigma in noise_sigmas:
            tracker = QualityTracker(config)
            scores = []
            for _ in range(25):
                # Add Gaussian pixel noise to synthetic landmarks
                base_lms = build_synthetic_landmarks()
                noisy_lms = []
                for pt in base_lms:
                    noisy_lms.append(NormalizedPoint(
                        x=pt.x + np.random.normal(0, sigma / 640.0),
                        y=pt.y + np.random.normal(0, sigma / 480.0),
                        z=pt.z
                    ))
                feat = extractor.extract(noisy_lms, 640, 480)
                q = tracker.evaluate(feat, noisy_lms, img_w=640, img_h=480)
                scores.append(q.stability_score)

            mean_stab = float(np.mean(scores[10:]))  # After warmup
            avg_stabilities.append(mean_stab)

        # Monotonic drop in stability as noise increases
        for i in range(len(avg_stabilities) - 1):
            assert avg_stabilities[i] >= avg_stabilities[i + 1] - 0.05

        # At sigma >= 8px, stability must drop below threshold
        assert avg_stabilities[-1] < 0.20

    def test_jitter_burst_and_recovery(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        tracker = QualityTracker(config)
        np.random.seed(99)

        # Phase 1: 15 clean frames (Stability ~ 1.0)
        base_lms = build_synthetic_landmarks()
        for _ in range(15):
            feat = extractor.extract(base_lms, 640, 480)
            q_clean = tracker.evaluate(feat, base_lms, img_w=640, img_h=480)
        assert q_clean.stability_score > 0.95

        # Phase 2: 10 frames of severe jitter (sigma = 12px)
        for _ in range(10):
            noisy_lms = [
                NormalizedPoint(
                    x=pt.x + np.random.normal(0, 12.0 / 640.0),
                    y=pt.y + np.random.normal(0, 12.0 / 480.0),
                    z=pt.z
                ) for pt in base_lms
            ]
            feat = extractor.extract(noisy_lms, 640, 480)
            q_jitter = tracker.evaluate(feat, noisy_lms, img_w=640, img_h=480)
        assert q_jitter.stability_score < 0.30
        assert "Landmark high-frequency tracking jitter" in q_jitter.failure_reasons

        # Phase 3: 15 clean frames recovery
        for _ in range(15):
            feat = extractor.extract(base_lms, 640, 480)
            q_rec = tracker.evaluate(feat, base_lms, img_w=640, img_h=480)
        assert q_rec.stability_score > 0.95
        assert "Landmark high-frequency tracking jitter" not in q_rec.failure_reasons


# ============================================================================
# 4. High-Throughput Execution Benchmark (1,000 Synthetic Frames)
# ============================================================================

class TestHighThroughputBenchmark:
    """Rigorous 1,000-frame throughput and latency benchmark across CV feature extraction."""

    def test_1000_frames_throughput_and_latency_distribution(self):
        config = GazeConfig()
        extractor = EyeExtractor(config)
        estimator = HeadPoseEstimator(config)
        tracker = QualityTracker(config)

        landmarks = build_synthetic_landmarks()
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)

        extract_latencies = []
        estimate_latencies = []
        evaluate_latencies = []
        total_latencies = []

        n_frames = 1000

        # Warm up JIT / caches
        for _ in range(50):
            hp = estimator.estimate(landmarks, 640, 480)
            feat = extractor.extract(landmarks, 640, 480, head_pose=hp)
            _ = tracker.evaluate(feat, landmarks, frame=frame, img_w=640, img_h=480)

        # 1,000 benchmark iterations
        for _ in range(n_frames):
            t0 = time.perf_counter()
            hp = estimator.estimate(landmarks, 640, 480)
            t1 = time.perf_counter()
            feat = extractor.extract(landmarks, 640, 480, head_pose=hp)
            t2 = time.perf_counter()
            quality = tracker.evaluate(feat, landmarks, frame=frame, img_w=640, img_h=480)
            t3 = time.perf_counter()

            estimate_latencies.append((t1 - t0) * 1000.0)
            extract_latencies.append((t2 - t1) * 1000.0)
            evaluate_latencies.append((t3 - t2) * 1000.0)
            total_latencies.append((t3 - t0) * 1000.0)

        mean_total = float(np.mean(total_latencies))
        p50_total = float(np.percentile(total_latencies, 50))
        p95_total = float(np.percentile(total_latencies, 95))
        p99_total = float(np.percentile(total_latencies, 99))
        max_total = float(np.max(total_latencies))
        fps = 1000.0 / mean_total

        print(f"\n📊 1,000 Frame CV Feature Pipeline Latency Metrics:")
        print(f"  - HeadPose.estimate(): mean = {np.mean(estimate_latencies):.4f}ms, P95 = {np.percentile(estimate_latencies, 95):.4f}ms")
        print(f"  - EyeExtractor.extract(): mean = {np.mean(extract_latencies):.4f}ms, P95 = {np.percentile(extract_latencies, 95):.4f}ms")
        print(f"  - QualityTracker.evaluate(): mean = {np.mean(evaluate_latencies):.4f}ms, P95 = {np.percentile(evaluate_latencies, 95):.4f}ms")
        print(f"  - Combined Pipeline: mean = {mean_total:.4f}ms, P50 = {p50_total:.4f}ms, P95 = {p95_total:.4f}ms, P99 = {p99_total:.4f}ms, max = {max_total:.4f}ms")
        print(f"  - Throughput: {fps:.1f} FPS (Budget requirement: >= 30 FPS, Latency < 35ms)")

        assert mean_total < 5.0, f"Mean latency {mean_total:.3f}ms exceeded 5.0ms"
        assert p99_total < 15.0, f"P99 latency {p99_total:.3f}ms exceeded 15.0ms"
        assert fps > 200.0, f"Throughput {fps:.1f} FPS is below 200 FPS"
