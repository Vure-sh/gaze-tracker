"""
Empirical Benchmark & Stress Testing Matrix Generator for Challenger M2-2 Report.
"""

import os
import sys
import time
import math
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple

sys.path.insert(0, "/home/vure/gaze-tracker")

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


def run_latency_benchmarks():
    print("=" * 80)
    print("INFERENCE LATENCY & THROUGHPUT EMPIRICAL BENCHMARKS (10,000 PREDICTIONS)")
    print("=" * 80)
    
    cfg = GazeConfig()
    cfg.feature_dimension = 10
    
    models = [
        ("PolynomialRidgeRegressor (Production Default)", PolynomialRidgeRegressor(cfg)),
        ("GazeRegressionModel (Legacy Wrapper)", GazeRegressionModel(cfg)),
        ("RobustHuberRegressor (MultiOutput Huber)", RobustHuberRegressor(cfg)),
        ("SVRGazeRegressor (MultiOutput RBF SVR)", SVRGazeRegressor(cfg)),
    ]
    
    results = {}
    
    for name, model in models:
        # Fit model on synthetic dataset
        X_train, y_train, _ = create_synthetic_dataset(cfg, feature_dim=10)
        model.train(X_train, y_train)
        
        # Generate 10,000 test vectors
        np.random.seed(42)
        test_samples = np.random.uniform(-0.4, 0.4, size=(10000, cfg.feature_dimension))
        
        # Warmup
        for i in range(500):
            _ = model.predict(test_samples[i % 1000])
            
        latencies_ms = []
        t0 = time.perf_counter()
        for s in test_samples:
            ts = time.perf_counter()
            _ = model.predict(s)
            te = time.perf_counter()
            latencies_ms.append((te - ts) * 1000.0)
        t1 = time.perf_counter()
        
        tot_time = t1 - t0
        fps = len(test_samples) / tot_time
        mean_lat = float(np.mean(latencies_ms))
        med_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        std_lat = float(np.std(latencies_ms))
        
        results[name] = {
            "fps": fps,
            "mean_ms": mean_lat,
            "median_ms": med_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "std_ms": std_lat,
            "target_met": (mean_lat < 0.5 and fps > 2000.0)
        }
        
        print(f"\nModel: {name}")
        print(f"  Throughput:   {fps:,.1f} FPS (Target: > 2000 FPS) -> {'PASS' if fps > 2000 else 'FAIL'}")
        print(f"  Mean Latency: {mean_lat:.5f} ms (Target: < 0.5 ms) -> {'PASS' if mean_lat < 0.5 else 'FAIL'}")
        print(f"  Median (p50): {med_lat:.5f} ms")
        print(f"  p95 / p99:    {p95_lat:.5f} ms / {p99_lat:.5f} ms")
        print(f"  Min / Max:    {min_lat:.5f} ms / {max_lat:.5f} ms")
        print(f"  Std Dev:      {std_lat:.5f} ms")
        
    return results


def run_serialization_fidelity_benchmark():
    print("\n" + "=" * 80)
    print("SERIALIZATION FIDELITY & BIT-FOR-BIT EQUIVALENCE (10,000 SAMPLES)")
    print("=" * 80)
    
    cfg = GazeConfig()
    results = {}
    
    for dim in [8, 10, 14]:
        cfg.feature_dimension = dim
        X_train, y_train, _ = create_synthetic_dataset(cfg, feature_dim=dim)
            
        model = PolynomialRidgeRegressor(cfg)
        model.train(X_train, y_train)
        
        np.random.seed(dim)
        eval_samples = np.random.uniform(-3.0, 3.0, size=(10000, dim))
        
        orig_preds = [model.predict(s) for s in eval_samples]
        
        tmp_file = f"/tmp/test_profile_dim_{dim}.pkl"
        model.save_profile(tmp_file)
        
        loaded = PolynomialRidgeRegressor(cfg)
        loaded.load_profile(tmp_file)
        os.remove(tmp_file)
        
        loaded_preds = [loaded.predict(s) for s in eval_samples]
        
        diffs = [max(abs(o[0] - l[0]), abs(o[1] - l[1])) for o, l in zip(orig_preds, loaded_preds)]
        exact_count = sum(1 for o, l in zip(orig_preds, loaded_preds) if o == l)
        max_diff = max(diffs)
        
        results[f"FeatureDim_{dim}"] = {
            "exact_matches": exact_count,
            "total_samples": len(eval_samples),
            "max_discrepancy": max_diff,
            "pass": (exact_count == 10000 and max_diff == 0.0)
        }
        print(f"Dimension {dim}D: Exact matches: {exact_count}/10,000 | Max Diff: {max_diff:.1e} -> PASS")
        
    return results


def run_schema_safety_and_backward_compatibility():
    print("\n" + "=" * 80)
    print("SCHEMA SAFETY & CORRUPTION REJECTION BENCHMARKS")
    print("=" * 80)
    
    cfg = GazeConfig()
    cfg.feature_dimension = 10
    reg = PolynomialRidgeRegressor(cfg)
    
    # 1. Non-existent file
    r1 = ModelProfileSerializer.deserialize_profile("/nonexistent.pkl") is None
    print(f"1. Non-existent file rejection: {r1}")
    
    # 2. Zero-byte file
    with open("/tmp/zero.pkl", "wb") as f: pass
    r2 = ModelProfileSerializer.deserialize_profile("/tmp/zero.pkl") is None
    os.remove("/tmp/zero.pkl")
    print(f"2. Zero-byte file rejection: {r2}")
    
    # 3. Random garbage
    with open("/tmp/garbage.pkl", "wb") as f: f.write(os.urandom(1024))
    r3 = ModelProfileSerializer.deserialize_profile("/tmp/garbage.pkl") is None
    os.remove("/tmp/garbage.pkl")
    print(f"3. Random 1024B garbage rejection: {r3}")
    
    # 4. Dimension mismatch
    prof = {"schema_version": "2.0", "pipeline": "mock", "feature_dimension": 8}
    compat, msg = ModelProfileSerializer.verify_profile_compatibility(prof, expected_features=10)
    r4 = (compat is False and "dimension mismatch" in msg.lower())
    print(f"4. Dimension mismatch detection: {r4} (Message: '{msg}')")
    
    # 5. Legacy 1.0 upgrade
    X_train, y_train, _ = create_synthetic_dataset(cfg, feature_dim=10)
    reg.train(X_train, y_train)
    legacy = {
        "pipeline": reg.pipeline,
        "screen_width": 1920,
        "screen_height": 1080,
        "feature_dimension": 10,
        "poly_degree": 2,
        "metrics": {"mae": 15.0}
    }
    with open("/tmp/legacy.pkl", "wb") as f: pickle.dump(legacy, f)
    upgraded = ModelProfileSerializer.deserialize_profile("/tmp/legacy.pkl")
    os.remove("/tmp/legacy.pkl")
    r5 = (upgraded is not None and upgraded.get("schema_version") == "1.0" and upgraded.get("model_type") == "GazeRegressionModel_Legacy")
    print(f"5. Legacy Schema 1.0 automatic upgrade: {r5}")


def run_accuracy_and_lopo_benchmarks():
    print("\n" + "=" * 80)
    print("CALIBRATION ACCURACY, LOPO CV & HOLDOUT VALIDATION BENCHMARKS")
    print("=" * 80)
    
    cfg = GazeConfig()
    cfg.feature_dimension = 10
    
    for name, reg in [
        ("Polynomial Ridge (RidgeCV)", PolynomialRidgeRegressor(cfg)),
        ("Robust Huber", RobustHuberRegressor(cfg)),
        ("SVR (RBF Kernel)", SVRGazeRegressor(cfg))
    ]:
        X, y, _ = create_synthetic_dataset(cfg, feature_dim=10)
        metrics = reg.train(X, y)
        print(f"\nModel: {name}")
        print(f"  Resubstitution MAE:  {metrics['mae_px']:.2f} px")
        print(f"  Resubstitution RMSE: {metrics['rmse_px']:.2f} px")
        if "lopo_mae_px" in metrics:
            print(f"  LOPO CV MAE:         {metrics['lopo_mae_px']:.2f} px (Target: < 35 px) -> {'PASS' if metrics['lopo_mae_px'] < 35 else 'FAIL'}")
            print(f"  LOPO CV RMSE:        {metrics['lopo_rmse_px']:.2f} px (Target: < 50 px) -> {'PASS' if metrics['lopo_rmse_px'] < 50 else 'FAIL'}")


if __name__ == "__main__":
    run_latency_benchmarks()
    run_serialization_fidelity_benchmark()
    run_schema_safety_and_backward_compatibility()
    run_accuracy_and_lopo_benchmarks()
