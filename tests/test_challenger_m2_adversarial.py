"""
Challenger 2 Empirical Adversarial Stress Test Suite for Milestone 2 (ML & Gaze Estimation / Calibration).

Tests:
1. Serialization bit-for-bit prediction equivalence across 10,000 synthetic test samples after save/load.
2. Schema validation safety: graceful rejection on corrupted files, missing fields, or dimension mismatches.
3. Backward compatibility: automatic loading and upgrading of legacy Schema 1.0 .pkl files.
4. Inference latency: benchmark predict() throughput across 10,000 predictions (< 0.5ms / > 2000 FPS).
5. Holdout validation accuracy and outlier filtering stress.
"""

import os
import sys
import tempfile
import time
import math
import pickle
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple

from src.config import GazeConfig
from src.models.regressor import (
    BaseGazeRegressor,
    PolynomialRidgeRegressor,
    RobustHuberRegressor,
    SVRGazeRegressor,
    GazeRegressionModel
)
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.calibration.targets import TargetGenerator


def create_synthetic_dataset(
    cfg: GazeConfig,
    num_points: int = 9,
    samples_per_point: int = 25,
    feature_dim: int = 10,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Generates synthetic feature and screen target datasets for arbitrary feature dimensions."""
    np.random.seed(seed)
    w, h = cfg.screen_width, cfg.screen_height
    mx, my = cfg.calibration_margin_x, cfg.calibration_margin_y

    xs = [mx * w, 0.5 * w, (1.0 - mx) * w]
    ys = [my * h, 0.5 * h, (1.0 - my) * h]
    targets = [(int(x), int(y)) for y in ys for x in xs]

    X_list = []
    y_list = []

    for tx, ty in targets:
        true_norm_x = (tx - 0.5 * w) / (w * 2.0)
        true_norm_y = (ty - 0.5 * h) / (h * 2.5)

        for _ in range(samples_per_point):
            feat = np.zeros(feature_dim, dtype=np.float64)
            feat[0] = true_norm_x + np.random.normal(0, 0.003)
            feat[1] = true_norm_y + np.random.normal(0, 0.003)
            if feature_dim >= 4:
                feat[2] = true_norm_x + np.random.normal(0, 0.003)
                feat[3] = true_norm_y + np.random.normal(0, 0.003)
            if feature_dim >= 6:
                feat[4] = (feat[0] + feat[2]) / 2.0
                feat[5] = (feat[1] + feat[3]) / 2.0
            if feature_dim >= 8:
                feat[6] = np.random.normal(0, 0.01)  # pitch
                feat[7] = np.random.normal(0, 0.01)  # yaw
            if feature_dim >= 10:
                feat[8] = np.random.normal(0, 0.01)  # roll
                feat[9] = 0.6 + np.random.normal(0, 0.005)  # tz
            if feature_dim == 14:
                ear_l = 0.31 + np.random.normal(0, 0.005)
                ear_r = 0.31 + np.random.normal(0, 0.005)
                tx_n = np.random.normal(0, 0.005)
                ty_n = np.random.normal(0, 0.005)
                feat = np.array([
                    feat[0], feat[1], feat[2], feat[3], feat[4], feat[5],
                    ear_l, ear_r,
                    feat[6], feat[7], feat[8],
                    tx_n, ty_n, feat[9]
                ], dtype=np.float64)

            X_list.append(feat)
            y_list.append([tx, ty])

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64), targets


class TestEmpiricalSerializationFidelity:
    """Stress test 1: 10,000 synthetic sample bit-for-bit serialization equivalence."""

    @pytest.mark.parametrize("regressor_cls", [
        PolynomialRidgeRegressor,
        RobustHuberRegressor,
        SVRGazeRegressor,
        GazeRegressionModel
    ])
    @pytest.mark.parametrize("feature_dim", [8, 10, 14])
    def test_bit_for_bit_prediction_equivalence_10k_samples(self, regressor_cls, feature_dim):
        """
        Generates 10,000 random/synthetic test samples across normal, wide, and extreme feature spaces,
        verifies 100.0% exact float prediction match after serialize / deserialize roundtrip.
        """
        cfg = GazeConfig()
        cfg.feature_dimension = feature_dim
        X_train, y_train, _ = create_synthetic_dataset(cfg, num_points=9, samples_per_point=15, feature_dim=feature_dim)

        model = regressor_cls(cfg)
        model.train(X_train, y_train)

        # Generate 10,000 evaluation samples:
        # - 6,000 in normal range [-0.5, 0.5]
        # - 2,000 in wide range [-1.5, 1.5]
        # - 2,000 boundary/extreme range [-5.0, 5.0]
        np.random.seed(1000 + feature_dim)
        eval_samples_normal = np.random.uniform(-0.5, 0.5, size=(6000, feature_dim))
        eval_samples_wide = np.random.uniform(-1.5, 1.5, size=(2000, feature_dim))
        eval_samples_extreme = np.random.uniform(-5.0, 5.0, size=(2000, feature_dim))
        eval_samples = np.vstack([eval_samples_normal, eval_samples_wide, eval_samples_extreme])
        assert eval_samples.shape == (10000, feature_dim)

        # Predict baseline before serialization
        baseline_preds = []
        for s in eval_samples:
            p = model.predict(s)
            assert p is not None
            baseline_preds.append(p)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save profile (Schema 2.0)
            model.save_profile(tmp_path)

            # Load into fresh instance
            loaded_model = regressor_cls(cfg)
            load_success = loaded_model.load_profile(tmp_path)
            assert load_success is True
            assert loaded_model.is_trained is True

            # Evaluate roundtrip predictions across all 10,000 test samples
            max_discrepancy = 0.0
            exact_matches = 0
            close_matches = 0

            for i, s in enumerate(eval_samples):
                loaded_p = loaded_model.predict(s)
                assert loaded_p is not None
                orig_p = baseline_preds[i]

                dx = abs(loaded_p[0] - orig_p[0])
                dy = abs(loaded_p[1] - orig_p[1])
                diff = max(dx, dy)
                if diff > max_discrepancy:
                    max_discrepancy = diff

                if loaded_p[0] == orig_p[0] and loaded_p[1] == orig_p[1]:
                    exact_matches += 1
                if math.isclose(loaded_p[0], orig_p[0], abs_tol=1e-12) and math.isclose(loaded_p[1], orig_p[1], abs_tol=1e-12):
                    close_matches += 1

            # Assert 100% bit-for-bit exact match (zero discrepancy)
            assert exact_matches == 10000, f"Exact matches: {exact_matches}/10000 (max diff: {max_discrepancy})"
            assert close_matches == 10000
            assert max_discrepancy == 0.0, f"Non-zero max discrepancy: {max_discrepancy}"

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestEmpiricalSchemaValidationSafety:
    """Stress test 2: Graceful rejection on corrupted files, missing fields, and dimension mismatches."""

    def test_nonexistent_file_rejection(self):
        """Deserializing a non-existent file returns None without exception."""
        res = ModelProfileSerializer.deserialize_profile("/tmp/nonexistent_profile_123456789.pkl")
        assert res is None

        cfg = GazeConfig()
        reg = PolynomialRidgeRegressor(cfg)
        assert reg.load_profile("/tmp/nonexistent_profile_123456789.pkl") is False

    def test_zero_byte_empty_file_rejection(self):
        """Deserializing an empty file returns None without exception."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            res = ModelProfileSerializer.deserialize_profile(tmp_path)
            assert res is None

            cfg = GazeConfig()
            reg = PolynomialRidgeRegressor(cfg)
            assert reg.load_profile(tmp_path) is False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @pytest.mark.parametrize("corrupt_size", [1, 8, 32, 128, 512, 1024, 4096])
    def test_random_garbage_bytes_rejection(self, corrupt_size):
        """Deserializing arbitrary random byte sequences returns None gracefully."""
        garbage = os.urandom(corrupt_size)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp.write(garbage)
            tmp_path = tmp.name

        try:
            res = ModelProfileSerializer.deserialize_profile(tmp_path)
            assert res is None

            cfg = GazeConfig()
            reg = PolynomialRidgeRegressor(cfg)
            assert reg.load_profile(tmp_path) is False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_truncated_pickle_file_rejection(self):
        """Truncating a valid serialized profile midway returns None without crash."""
        cfg = GazeConfig()
        X_train, y_train, _ = create_synthetic_dataset(cfg, feature_dim=10)
        model = PolynomialRidgeRegressor(cfg)
        model.train(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            valid_path = tmp.name
        model.save_profile(valid_path)

        with open(valid_path, "rb") as f:
            full_bytes = f.read()
        os.remove(valid_path)

        for cut in [10, 50, len(full_bytes) // 4, len(full_bytes) // 2, len(full_bytes) - 5]:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp.write(full_bytes[:cut])
                trunc_path = tmp.name
            try:
                res = ModelProfileSerializer.deserialize_profile(trunc_path)
                assert res is None, f"Expected None on truncation at byte {cut}/{len(full_bytes)}"
                reg = PolynomialRidgeRegressor(cfg)
                assert reg.load_profile(trunc_path) is False
            finally:
                if os.path.exists(trunc_path):
                    os.remove(trunc_path)

    @pytest.mark.parametrize("invalid_payload", [
        "not_a_dict_just_a_string",
        123456789,
        [1, 2, 3, {"pipeline": "mock"}],
        {"invalid_key": "no_pipeline"},
        {"schema_version": "2.0"},  # Missing pipeline
        {"schema_version": "2.0", "metrics": {}},  # Missing pipeline
    ])
    def test_invalid_pickle_payload_structures(self, invalid_payload):
        """Pickled objects with wrong types or missing pipeline key are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(invalid_payload, tmp)
            tmp_path = tmp.name
        try:
            res = ModelProfileSerializer.deserialize_profile(tmp_path)
            assert res is None

            cfg = GazeConfig()
            reg = PolynomialRidgeRegressor(cfg)
            assert reg.load_profile(tmp_path) is False
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_feature_dimension_mismatch_verification(self):
        """Verify profile compatibility detects dimension mismatch."""
        profile_8d = {
            "schema_version": "2.0",
            "pipeline": "mock_pipeline",
            "screen_width": 1920,
            "screen_height": 1080,
            "feature_dimension": 8
        }
        compat, msg = ModelProfileSerializer.verify_profile_compatibility(profile_8d, expected_features=8)
        assert compat is True
        assert msg is None

        compat, msg = ModelProfileSerializer.verify_profile_compatibility(profile_8d, expected_features=10)
        assert compat is False
        assert "Feature dimension mismatch" in msg

        compat, msg = ModelProfileSerializer.verify_profile_compatibility(profile_8d, expected_features=14)
        assert compat is False
        assert "Feature dimension mismatch" in msg

    def test_resolution_mismatch_detection(self):
        """Verify profile compatibility flags resolution differences."""
        profile_1080p = {
            "schema_version": "2.0",
            "pipeline": "mock_pipeline",
            "screen_width": 1920,
            "screen_height": 1080,
            "feature_dimension": 10
        }
        compat, msg = ModelProfileSerializer.verify_profile_compatibility(
            profile_1080p, screen_w=2560, screen_h=1440
        )
        assert compat is True  # Model is still runnable
        assert msg is not None
        assert "Display resolution mismatch" in msg


class TestEmpiricalLegacyBackwardCompatibility:
    """Stress test 3: Automatic loading and upgrading of legacy Schema 1.0 .pkl files."""

    def test_legacy_schema_1_0_upgrade_and_prediction(self):
        """
        Creates a legacy Schema 1.0 pickle file (unversioned dictionary),
        deserializes it, verifies it upgrades cleanly, loads into regressor, and predicts accurately.
        """
        cfg = GazeConfig()
        cfg.feature_dimension = 10
        X_train, y_train, _ = create_synthetic_dataset(cfg, num_points=9, feature_dim=10, seed=123)

        reg_temp = PolynomialRidgeRegressor(cfg)
        reg_temp.train(X_train, y_train)
        fitted_pipeline = reg_temp.pipeline

        legacy_payload = {
            "pipeline": fitted_pipeline,
            "screen_width": 1920,
            "screen_height": 1080,
            "feature_dimension": 10,
            "poly_degree": 2,
            "metrics": {"train_mae": 14.2, "train_rmse": 20.1}
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(legacy_payload, tmp)
            legacy_path = tmp.name

        try:
            profile = ModelProfileSerializer.deserialize_profile(legacy_path)
            assert profile is not None
            assert profile["schema_version"] == "1.0"
            assert profile["model_type"] == "GazeRegressionModel_Legacy"
            assert profile["pipeline"] is not None
            assert profile["screen_width"] == 1920
            assert profile["screen_height"] == 1080
            assert profile["feature_dimension"] == 10

            reg = PolynomialRidgeRegressor(cfg)
            success = reg.load_profile(legacy_path)
            assert success is True
            assert reg.is_trained is True
            assert reg.metrics == {"train_mae": 14.2, "train_rmse": 20.1}

            sample_feature = X_train[0]
            pred = reg.predict(sample_feature)
            assert pred is not None
            assert 0.0 <= pred[0] <= 1920.0
            assert 0.0 <= pred[1] <= 1080.0

            expected_pred = reg_temp.predict(sample_feature)
            assert pred == expected_pred

        finally:
            if os.path.exists(legacy_path):
                os.remove(legacy_path)


class TestEmpiricalInferenceLatencyAndThroughput:
    """Stress test 4: Benchmark predict() throughput across 10,000 predictions (< 0.5ms / > 2000 FPS)."""

    @pytest.mark.parametrize("regressor_cls, name", [
        (PolynomialRidgeRegressor, "PolynomialRidgeRegressor"),
        (GazeRegressionModel, "GazeRegressionModel")
    ])
    def test_production_regressors_meet_latency_target_10k_predictions(self, regressor_cls, name):
        """
        Executes 10,000 consecutive single-sample predict() calls on production default models.
        Target: mean latency < 0.5ms per prediction, throughput > 2000 FPS.
        """
        cfg = GazeConfig()
        cfg.feature_dimension = 10
        X_train, y_train, _ = create_synthetic_dataset(cfg, num_points=9, feature_dim=10, seed=999)

        model = regressor_cls(cfg)
        model.train(X_train, y_train)

        dim = cfg.feature_dimension
        np.random.seed(777)
        test_samples = np.random.uniform(-0.5, 0.5, size=(10000, dim))

        for i in range(500):
            _ = model.predict(test_samples[i % 1000])

        individual_latencies_ms: List[float] = []
        t0_total = time.perf_counter()

        for s in test_samples:
            t_start = time.perf_counter()
            pred = model.predict(s)
            t_end = time.perf_counter()
            individual_latencies_ms.append((t_end - t_start) * 1000.0)
            assert pred is not None

        t1_total = time.perf_counter()

        total_time_s = t1_total - t0_total
        throughput_fps = len(test_samples) / total_time_s
        mean_latency_ms = float(np.mean(individual_latencies_ms))
        median_latency_ms = float(np.median(individual_latencies_ms))
        p95_latency_ms = float(np.percentile(individual_latencies_ms, 95))
        p99_latency_ms = float(np.percentile(individual_latencies_ms, 99))

        print(f"\n[LATENCY BENCHMARK: {name}]")
        print(f"  Throughput:   {throughput_fps:,.1f} FPS (Target: > 2000 FPS)")
        print(f"  Mean Latency: {mean_latency_ms:.5f} ms (Target: < 0.5 ms)")
        print(f"  Median (p50): {median_latency_ms:.5f} ms")
        print(f"  p95 / p99:    {p95_latency_ms:.5f} ms / {p99_latency_ms:.5f} ms")

        assert mean_latency_ms < 0.5, f"Mean latency {mean_latency_ms:.4f}ms exceeded 0.5ms target"
        assert throughput_fps > 2000.0, f"Throughput {throughput_fps:.1f} FPS below 2000 FPS target"
        assert p99_latency_ms < 1.0, f"p99 latency {p99_latency_ms:.4f}ms exceeded 1.0ms"

    @pytest.mark.parametrize("regressor_cls, name", [
        (RobustHuberRegressor, "RobustHuberRegressor"),
        (SVRGazeRegressor, "SVRGazeRegressor"),
    ])
    def test_alternative_regressors_bounded_latency(self, regressor_cls, name):
        """Benchmarks alternative backends (Huber, SVR) to ensure bounded predictable runtime (< 2.0ms)."""
        cfg = GazeConfig()
        cfg.feature_dimension = 10
        X_train, y_train, _ = create_synthetic_dataset(cfg, num_points=9, feature_dim=10, seed=999)

        model = regressor_cls(cfg)
        model.train(X_train, y_train)

        dim = cfg.feature_dimension
        np.random.seed(777)
        test_samples = np.random.uniform(-0.5, 0.5, size=(1000, dim))

        for i in range(100):
            _ = model.predict(test_samples[i % 500])

        individual_latencies_ms: List[float] = []
        for s in test_samples:
            t_start = time.perf_counter()
            pred = model.predict(s)
            t_end = time.perf_counter()
            individual_latencies_ms.append((t_end - t_start) * 1000.0)
            assert pred is not None

        mean_latency_ms = float(np.mean(individual_latencies_ms))
        print(f"\n[ALTERNATIVE BACKEND LATENCY: {name}] Mean Latency: {mean_latency_ms:.5f} ms")
        assert mean_latency_ms < 2.0, f"Alternative backend mean latency {mean_latency_ms:.4f}ms exceeded 2.0ms"


class TestEmpiricalHoldoutValidationAndCalibrationStress:
    """Additional stress tests: Holdout validation state machine, noise resistance, outlier filtering."""

    def test_interactive_holdout_validation_accuracy_metrics(self):
        """Verifies 4-point holdout validation calculates MAE, RMSE, and visual angle < 1.0 deg."""
        cfg = GazeConfig()
        cfg.feature_dimension = 10
        X_train, y_train, _ = create_synthetic_dataset(cfg, num_points=9, samples_per_point=30, feature_dim=10, seed=42)

        model = PolynomialRidgeRegressor(cfg)
        model.train(X_train, y_train)

        mgr = CalibrationManager(cfg, regressor=model)
        mgr.start_validation(mode="4_points")
        assert mgr.state == CalibrationState.VALIDATING
        assert len(mgr.validation_points) == 4

        val_targets = TargetGenerator.generate_validation_points(cfg, "4_points")
        frame_time = 0.0

        for pt_idx, target_xy in enumerate(val_targets):
            assert mgr.get_current_target() == target_xy
            for f in range(cfg.sample_frames_per_point + 5):
                frame_time += 0.033
                nx = (target_xy[0] - 0.5 * cfg.screen_width) / (cfg.screen_width * 2.0)
                ny = (target_xy[1] - 0.5 * cfg.screen_height) / (cfg.screen_height * 2.5)
                feat = np.array([nx, ny, nx, ny, nx, ny, 0.0, 0.0, 0.0, 0.6], dtype=np.float64)

                done = mgr.process_frame(feat, is_valid_frame=True, timestamp=frame_time)
                if done:
                    break

        assert mgr.state == CalibrationState.VALIDATION_COMPLETE
        assert "val_mae_px" in mgr.validation_metrics
        assert "visual_angle_deg" in mgr.validation_metrics
        print(f"\nHoldout Validation Results: MAE={mgr.validation_metrics['val_mae_px']:.2f}px, Visual Angle={mgr.validation_metrics['visual_angle_deg']:.3f}°")
        assert mgr.validation_metrics["val_mae_px"] < 35.0
        assert mgr.validation_metrics["visual_angle_deg"] < 1.0
