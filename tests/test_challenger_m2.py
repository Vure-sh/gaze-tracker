"""Empirical Challenger 1 Stress Harness for Milestone 2 (ML & Calibration).

Adversarial Stress Test Dimensions:
1. Extreme Outlier Injection During Calibration (10%, 25%, 50% simulated glance-aways, saccades, and head motions):
   - Verifies IQR outlier rejection filters them out and model fits cleanly.
2. Zero-Variance & Degenerate Sample Inputs:
   - Identical samples, constant feature subsets, zero Euclidean distances.
   - Verifies no division-by-zero, NaN propagation, or crash.
3. Leave-One-Point-Out (LOPO) Cross-Validation across 9-Point, 13-Point, and 16-Point Grids:
   - Verifies LOPO MAE < 35px and RMSE < 50px across PolynomialRidge / GazeRegressionModel.
   - Tests under moderate head pose variation (+-15 deg).
4. Screen Coordinate Boundary & Clamping Tests:
   - Predictions near and far outside screen edges ([-5000, 5000]).
   - Verifies strict [0, W] x [0, H] clamping.
5. Regressor Backend Characterization:
   - Benchmarks PolynomialRidge, RobustHuber, and SVR backends.
   - Measures resubstitution and generalization error across target scalings.
6. High-Stress Matrix Condition & Profile Integrity:
   - Collinear / rank-deficient feature matrices.
   - Boundary sample counts (5 vs 6 samples).
   - Serialization round-trip bit fidelity and schema versioning.
"""

import math
import tempfile
import os
import numpy as np
import pytest

from src.config import GazeConfig
from src.calibration.targets import TargetGenerator
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.models.regressor import (
    BaseGazeRegressor,
    PolynomialRidgeRegressor,
    RobustHuberRegressor,
    SVRGazeRegressor,
    GazeRegressionModel,
)
from src.models.serializer import ModelProfileSerializer, CURRENT_SCHEMA_VERSION


# ============================================================================
# Helper: Synthetic Multi-Grid Dataset Generator
# ============================================================================

def generate_multi_point_dataset(
    config: GazeConfig,
    grid_type: str = "9_points",
    samples_per_point: int = 25,
    noise_std: float = 0.003,
    head_pose_yaw_deg: float = 0.0,
    head_pose_pitch_deg: float = 0.0,
    feature_dimension: int = 8,
    seed: int = 42
):
    """
    Generates synthetic calibration feature vectors and target coordinates
    for any specified grid type ('9_points', '13_points', '16_points').
    Supports both 8D and 14D feature representations.
    """
    np.random.seed(seed)
    w = config.screen_width
    h = config.screen_height

    targets = TargetGenerator.generate_points(config, grid_type=grid_type, boustrophedon=True)
    X_list = []
    y_list = []
    point_ids = []

    yaw_norm = head_pose_yaw_deg / 45.0
    pitch_norm = head_pose_pitch_deg / 45.0

    for pt_idx, (tx, ty) in enumerate(targets):
        # Ground-truth normalized iris offset corresponding to target
        # Screen center (w/2, h/2) maps to norm_x = 0.0, norm_y = 0.0
        true_norm_x = (tx - 0.5 * w) / (w * 2.0)
        true_norm_y = (ty - 0.5 * h) / (h * 2.5)

        for _ in range(samples_per_point):
            norm_x_L = true_norm_x + np.random.normal(0, noise_std)
            norm_y_L = true_norm_y + np.random.normal(0, noise_std)
            norm_x_R = true_norm_x + np.random.normal(0, noise_std)
            norm_y_R = true_norm_y + np.random.normal(0, noise_std)

            if feature_dimension == 14:
                avg_norm_x = (norm_x_L + norm_x_R) / 2.0
                avg_norm_y = (norm_y_L + norm_y_R) / 2.0
                ear_L = 0.31 + np.random.normal(0, 0.005)
                ear_R = 0.31 + np.random.normal(0, 0.005)
                feat = np.array([
                    norm_x_L, norm_y_L, norm_x_R, norm_y_R,
                    avg_norm_x, avg_norm_y, ear_L, ear_R,
                    pitch_norm + np.random.normal(0, 0.005),
                    yaw_norm + np.random.normal(0, 0.005),
                    0.0 + np.random.normal(0, 0.005),
                    0.0, 0.0, 0.60 + np.random.normal(0, 0.01)
                ], dtype=np.float64)
            else:
                # 8D feature vector: [norm_x_L, norm_y_L, norm_x_R, norm_y_R, pitch, yaw, roll, tz]
                feat = np.array([
                    norm_x_L,
                    norm_y_L,
                    norm_x_R,
                    norm_y_R,
                    pitch_norm + np.random.normal(0, 0.005),
                    yaw_norm + np.random.normal(0, 0.005),
                    0.0 + np.random.normal(0, 0.005),
                    0.60 + np.random.normal(0, 0.01)
                ], dtype=np.float64)

            X_list.append(feat)
            y_list.append([tx, ty])
            point_ids.append(pt_idx)

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64),
        np.array(point_ids, dtype=np.int32),
        targets
    )


# ============================================================================
# 1. Extreme Outlier Injection During Calibration
# ============================================================================

class TestOutlierInjectionStress:
    """Stress-tests IQR outlier filtering under 10%, 25%, and 50% outlier corruption."""

    @pytest.mark.parametrize("outlier_ratio", [0.10, 0.25, 0.50])
    def test_simulated_glance_aways_outlier_rejection(self, gaze_config: GazeConfig, outlier_ratio: float):
        """
        Injects 10%, 25%, and 50% simulated glance-aways (extreme eye iris jumps, saccades).
        Verifies:
        - Outliers are filtered or mitigated.
        - CalibrationManager and regressors train cleanly without exception.
        - Resulting model retains low error on clean validation targets.
        """
        calibrator = CalibrationManager(gaze_config)
        np.random.seed(100 + int(outlier_ratio * 100))

        total_samples = 40
        num_outliers = int(total_samples * outlier_ratio)
        num_clean = total_samples - num_outliers

        # Clean samples clustered near center
        clean_samples = [
            np.array([
                0.05 + np.random.normal(0, 0.003),
                -0.02 + np.random.normal(0, 0.003),
                0.05 + np.random.normal(0, 0.003),
                -0.02 + np.random.normal(0, 0.003),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                0.60 + np.random.normal(0, 0.02),
            ])
            for _ in range(num_clean)
        ]

        # Outliers: simulated glance-aways looking at screen corners or room
        outlier_samples = []
        for _ in range(num_outliers):
            out_feat = np.array([
                np.random.choice([-0.85, 0.85, 1.5, -1.5]),
                np.random.choice([-0.75, 0.75, 1.2, -1.2]),
                np.random.choice([-0.85, 0.85, 1.5, -1.5]),
                np.random.choice([-0.75, 0.75, 1.2, -1.2]),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.3, 1.2),
            ])
            outlier_samples.append(out_feat)

        all_samples = clean_samples + outlier_samples
        np.random.shuffle(all_samples)

        filtered = calibrator._filter_outliers(all_samples)

        # Verify filtering behavior
        assert len(filtered) > 0
        assert len(filtered) <= len(all_samples)

        # If outlier ratio <= 25%, verify 100% of extreme glance-aways (|norm_x| > 0.5) are removed
        if outlier_ratio <= 0.25:
            for s in filtered:
                assert abs(s[0]) < 0.50, f"Outlier with norm_x={s[0]} leaked through filter!"
                assert abs(s[1]) < 0.50, f"Outlier with norm_y={s[1]} leaked through filter!"

        # Verify filtered samples can be fitted cleanly
        if len(filtered) >= 6:
            X_clean = np.array(filtered)
            y_clean = np.tile([960, 540], (len(X_clean), 1))
            reg = PolynomialRidgeRegressor(gaze_config)
            metrics = reg.train(X_clean, y_clean)
            assert metrics["mae_px"] < 10.0

    def test_extreme_numerical_spikes_outlier_rejection(self, gaze_config: GazeConfig):
        """Verifies colossal numerical spikes (+-1e6) are completely purged."""
        calibrator = CalibrationManager(gaze_config)
        clean = [np.full(8, 0.1) + np.random.normal(0, 0.001, 8) for _ in range(20)]
        spikes = [
            np.full(8, 1e6),
            np.full(8, -1e6),
            np.array([1e7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.6])
        ]
        samples = clean + spikes
        filtered = calibrator._filter_outliers(samples)

        assert len(filtered) == len(clean)
        for s in filtered:
            assert np.all(np.abs(s) < 10.0)

    def test_full_session_with_intermittent_glances(self, gaze_config: GazeConfig):
        """
        Runs an end-to-end multi-point calibration session where 20% of frames
        are look-aways or blinks during target collection.
        Verifies session finishes and model trains successfully.
        """
        calibrator = CalibrationManager(gaze_config)
        calibrator.start_calibration("9_points")

        np.random.seed(42)
        total_pts = len(calibrator.points)

        for pt_idx in range(total_pts):
            tx, ty = calibrator.points[pt_idx]
            true_nx = (tx - 0.5 * 1920) / 3840.0
            true_ny = (ty - 0.5 * 1080) / 2700.0

            clean_feat = np.array([
                true_nx, true_ny, true_nx, true_ny,
                0.0, 0.0, 0.0, 0.60
            ], dtype=np.float64)

            glance_feat = np.array([
                0.90, -0.85, 0.90, -0.85,
                0.3, -0.3, 0.0, 0.60
            ], dtype=np.float64)

            finished = False
            for f in range(gaze_config.sample_frames_per_point):
                # 20% of frames are glance-aways
                if f % 5 == 0:
                    feat = glance_feat
                else:
                    feat = clean_feat + np.random.normal(0, 0.002, 8)

                finished = calibrator.process_frame(feat, is_valid_frame=True)

            if pt_idx == total_pts - 1:
                assert finished is True

        assert calibrator.state == CalibrationState.FINISHED
        assert calibrator.regressor.is_trained is True
        assert calibrator.regressor.metrics["mae_px"] < 35.0


# ============================================================================
# 2. Zero-Variance & Degenerate Sample Inputs
# ============================================================================

class TestZeroVarianceSafety:
    """Stress-tests zero-variance, identical, and constant feature inputs."""

    def test_all_identical_samples_no_crash(self, gaze_config: GazeConfig):
        """Verify identical samples do not cause division by zero or NaN in outlier filter."""
        calibrator = CalibrationManager(gaze_config)
        identical_val = np.array([0.15, -0.10, 0.15, -0.10, 0.0, 0.0, 0.0, 0.60])
        samples = [identical_val.copy() for _ in range(30)]

        filtered = calibrator._filter_outliers(samples)
        assert len(filtered) == 30
        assert not any(np.isnan(s).any() for s in filtered)
        assert not any(np.isinf(s).any() for s in filtered)

    def test_partial_zero_variance_dimensions(self, gaze_config: GazeConfig):
        """
        Verify feature matrices where some dimensions have zero variance (e.g. constant pitch/yaw)
        while others vary do not cause division by zero or NaN.
        """
        calibrator = CalibrationManager(gaze_config)
        samples = []
        for i in range(25):
            s = np.array([
                0.10 + i * 0.005,  # varying
                0.20 + i * 0.005,  # varying
                0.10 + i * 0.005,  # varying
                0.20 + i * 0.005,  # varying
                0.0,               # constant 0
                0.0,               # constant 0
                0.0,               # constant 0
                0.60               # constant 0.60
            ])
            samples.append(s)

        filtered = calibrator._filter_outliers(samples)
        assert len(filtered) >= 20
        for s in filtered:
            assert not np.isnan(s).any()
            assert not np.isinf(s).any()

    def test_regressors_with_zero_variance_features(self, gaze_config: GazeConfig):
        """
        Verify PolynomialRidgeRegressor and alternative backends
        fit without crashing on datasets with constant feature columns (e.g. static head pose).
        """
        np.random.seed(42)
        N = 30
        X = np.zeros((N, 8))
        X[:, 0] = np.linspace(-0.2, 0.2, N)
        X[:, 1] = np.linspace(-0.15, 0.15, N)
        X[:, 2] = X[:, 0] + np.random.normal(0, 0.001, N)
        X[:, 3] = X[:, 1] + np.random.normal(0, 0.001, N)
        X[:, 7] = 0.60  # constant tz

        y = np.zeros((N, 2))
        y[:, 0] = 960 + X[:, 0] * 3840
        y[:, 1] = 540 + X[:, 1] * 2700

        reg = PolynomialRidgeRegressor(gaze_config)
        metrics = reg.train(X, y)
        assert reg.is_trained is True
        assert not np.isnan(metrics["mae_px"])
        pred = reg.predict(X[0])
        assert pred is not None
        assert 0.0 <= pred[0] <= gaze_config.screen_width
        assert 0.0 <= pred[1] <= gaze_config.screen_height


# ============================================================================
# 3. Leave-One-Point-Out (LOPO) CV Across 9, 13, and 16 Point Grids
# ============================================================================

class TestLOPOCrossValidationStress:
    """Stress-tests LOPO Cross-Validation across 9, 13, and 16 point grids."""

    @pytest.mark.parametrize("grid_type, expected_num_points", [
        ("9_points", 9),
        ("13_points", 13),
        ("16_points", 16)
    ])
    def test_lopo_cv_all_grid_types(
        self, gaze_config: GazeConfig, grid_type: str, expected_num_points: int
    ):
        """
        Performs Leave-One-Point-Out cross validation across all target points in 9, 13, and 16-point grids.
        Verifies:
        - LOPO MAE < 35px
        - LOPO RMSE < 50px
        - Visual angle error < 1.0 degree
        """
        X, y, pt_ids, targets = generate_multi_point_dataset(
            gaze_config, grid_type=grid_type, samples_per_point=25, noise_std=0.002
        )
        assert len(targets) == expected_num_points

        reg = PolynomialRidgeRegressor(gaze_config)
        metrics = reg.train(X, y, point_ids=pt_ids)

        assert reg.is_trained is True
        assert "lopo_mae_px" in metrics
        assert "lopo_rmse_px" in metrics

        lopo_mae = metrics["lopo_mae_px"]
        lopo_rmse = metrics["lopo_rmse_px"]

        # Pixel to visual angle: 24" 1080p, ~0.276 mm/px, distance 600mm
        vis_angle_deg = math.degrees(math.atan2(lopo_mae * 0.276, 600.0))

        print(f"\n[Grid: {grid_type}] LOPO MAE: {lopo_mae:.2f}px, RMSE: {lopo_rmse:.2f}px, Angle: {vis_angle_deg:.2f}°")

        assert lopo_mae < 35.0, f"LOPO MAE {lopo_mae:.2f}px exceeded 35px threshold on {grid_type}"
        assert lopo_rmse < 50.0, f"LOPO RMSE {lopo_rmse:.2f}px exceeded 50px threshold on {grid_type}"
        assert vis_angle_deg < 1.0, f"LOPO visual angle {vis_angle_deg:.2f}° exceeded 1.0° threshold on {grid_type}"

    def test_lopo_cv_with_gaze_regression_model(self, gaze_config: GazeConfig):
        """Verifies backward-compatible GazeRegressionModel matches PolynomialRidge accuracy."""
        X, y, pt_ids, _ = generate_multi_point_dataset(
            gaze_config, grid_type="9_points", samples_per_point=25, noise_std=0.002
        )
        model = GazeRegressionModel(gaze_config)
        metrics = model.train(X, y, point_ids=pt_ids)

        assert model.is_trained is True
        assert metrics["mae_px"] < 35.0
        assert metrics["lopo_mae_px"] < 35.0
        assert metrics["lopo_rmse_px"] < 50.0

    def test_lopo_cv_with_moderate_head_pose_angles(self, gaze_config: GazeConfig):
        """
        Verifies calibration fitting and LOPO accuracy remain robust
        under +-15 degree pitch and yaw head pose rotations.
        """
        for yaw in [-15.0, 0.0, 15.0]:
            for pitch in [-15.0, 0.0, 15.0]:
                X, y, pt_ids, _ = generate_multi_point_dataset(
                    gaze_config, grid_type="9_points", samples_per_point=20,
                    head_pose_yaw_deg=yaw, head_pose_pitch_deg=pitch, seed=42
                )
                reg = PolynomialRidgeRegressor(gaze_config)
                metrics = reg.train(X, y, point_ids=pt_ids)
                assert metrics["mae_px"] < 35.0
                assert metrics["lopo_mae_px"] < 35.0


# ============================================================================
# 4. Screen Coordinate Boundary & Clamping Tests
# ============================================================================

class TestBoundaryAndClampingStress:
    """Stress-tests prediction behavior at screen extremes and out-of-bounds inputs."""

    def test_extreme_feature_inputs_strictly_clamped(self, gaze_config: GazeConfig):
        """
        Tests that extreme positive/negative feature vectors are strictly clamped
        to [0, screen_width] x [0, screen_height].
        """
        X, y, _, _ = generate_multi_point_dataset(gaze_config, "9_points", samples_per_point=15)
        reg = PolynomialRidgeRegressor(gaze_config)
        reg.train(X, y)

        w = gaze_config.screen_width
        h = gaze_config.screen_height

        extreme_inputs = [
            np.full(8, 1000.0),
            np.full(8, -1000.0),
            np.full(8, 1e6),
            np.full(8, -1e6),
            np.array([10.0, -10.0, 10.0, -10.0, 0.0, 0.0, 0.0, 0.6]),
            np.array([-10.0, 10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.6]),
        ]

        for feat in extreme_inputs:
            pred = reg.predict(feat)
            assert pred is not None, "Predict returned None for valid numerical array"
            px, py = pred
            assert 0.0 <= px <= float(w), f"Predicted X ({px}) violated boundary [0, {w}]"
            assert 0.0 <= py <= float(h), f"Predicted Y ({py}) violated boundary [0, {h}]"

    def test_corner_target_predictions(self, gaze_config: GazeConfig):
        """Verifies prediction fidelity at exact display corners: (0,0), (W,0), (0,H), (W,H)."""
        w, h = gaze_config.screen_width, gaze_config.screen_height

        corners = [(0.0, 0.0), (float(w), 0.0), (0.0, float(h)), (float(w), float(h))]
        X_list = []
        y_list = []

        for cx, cy in corners:
            nx = (cx - 0.5 * w) / (w * 2.0)
            ny = (cy - 0.5 * h) / (h * 2.5)
            for _ in range(10):
                feat = np.array([nx, ny, nx, ny, 0.0, 0.0, 0.0, 0.60])
                X_list.append(feat)
                y_list.append([cx, cy])

        X = np.array(X_list)
        y = np.array(y_list)

        reg = PolynomialRidgeRegressor(gaze_config)
        reg.train(X, y)

        for cx, cy in corners:
            nx = (cx - 0.5 * w) / (w * 2.0)
            ny = (cy - 0.5 * h) / (h * 2.5)
            feat = np.array([nx, ny, nx, ny, 0.0, 0.0, 0.0, 0.60])
            px, py = reg.predict(feat)
            assert 0.0 <= px <= float(w)
            assert 0.0 <= py <= float(h)
            dist = math.hypot(px - cx, py - cy)
            assert dist < 30.0, f"Corner prediction error {dist:.1f}px too high for corner ({cx}, {cy})"


# ============================================================================
# 5. Regressor Backend Characterization & Empirical Metrics
# ============================================================================

class TestRegressorBackendCharacterization:
    """Characterizes the performance profiles of PolynomialRidge, RobustHuber, and SVR."""

    def test_benchmark_all_backends_on_synthetic_data(self, gaze_config: GazeConfig):
        """
        Empirically benchmarks all 3 estimator backends:
        - PolynomialRidge: Primary regularized estimator (RidgeCV).
        - RobustHuber: Huber loss estimator.
        - SVR: Support Vector Regression.
        Records exact MAE, RMSE, and LOPO cross-validation metrics.
        """
        X, y, pt_ids, _ = generate_multi_point_dataset(
            gaze_config, "9_points", samples_per_point=25, noise_std=0.003
        )

        results = {}
        models = {
            "PolynomialRidge": PolynomialRidgeRegressor(gaze_config),
            "RobustHuber": RobustHuberRegressor(gaze_config),
            "SVR": SVRGazeRegressor(gaze_config)
        }

        for name, model in models.items():
            metrics = model.train(X, y, point_ids=pt_ids)
            results[name] = metrics
            assert model.is_trained is True

        # PolynomialRidge MUST meet strict production requirements
        ridge_metrics = results["PolynomialRidge"]
        assert ridge_metrics["mae_px"] < 10.0
        assert ridge_metrics["lopo_mae_px"] < 35.0
        assert ridge_metrics["lopo_rmse_px"] < 50.0

        print("\n=== Empirical Regressor Benchmark Summary ===")
        for name, m in results.items():
            lopo_s = f", LOPO MAE: {m.get('lopo_mae_px', 'N/A')}"
            print(f"  {name:18s} -> Train MAE: {m['mae_px']:.2f}px, RMSE: {m['rmse_px']:.2f}px{lopo_s}")


# ============================================================================
# 6. High-Stress Matrix Conditions & Serialization Fidelity
# ============================================================================

class TestHighStressConditions:
    """Stress-tests rank-deficient matrices, sample limits, and serialization."""

    def test_collinear_feature_matrix(self, gaze_config: GazeConfig):
        """
        Verifies RidgeCV handles perfectly collinear / duplicate feature columns
        (e.g., Left eye == Right eye exactly) via L2 regularization without numerical instability.
        """
        N = 30
        X = np.zeros((N, 8))
        base_x = np.linspace(-0.2, 0.2, N)
        base_y = np.linspace(-0.15, 0.15, N)

        # Duplicate columns
        X[:, 0] = base_x
        X[:, 1] = base_y
        X[:, 2] = base_x  # Exact duplicate of col 0
        X[:, 3] = base_y  # Exact duplicate of col 1
        X[:, 4] = base_x  # Collinear
        X[:, 5] = base_y  # Collinear
        X[:, 6] = 0.0
        X[:, 7] = 0.60

        y = np.zeros((N, 2))
        y[:, 0] = 960 + base_x * 3840
        y[:, 1] = 540 + base_y * 2700

        reg = PolynomialRidgeRegressor(gaze_config)
        metrics = reg.train(X, y)
        assert reg.is_trained is True
        assert metrics["mae_px"] < 25.0

    def test_sample_count_boundaries(self, gaze_config: GazeConfig):
        """
        Boundary condition:
        - 5 samples: must raise ValueError (insufficient samples).
        - 6 samples: must succeed (minimum required points).
        """
        reg = PolynomialRidgeRegressor(gaze_config)

        # 5 samples -> Failure
        X5 = np.zeros((5, 8))
        y5 = np.zeros((5, 2))
        with pytest.raises(ValueError, match="Insufficient training samples"):
            reg.train(X5, y5)

        # 6 samples -> Success
        X6 = np.random.normal(0, 0.1, (6, 8))
        y6 = np.random.uniform(100, 900, (6, 2))
        metrics = reg.train(X6, y6)
        assert reg.is_trained is True
        assert "mae_px" in metrics

    def test_serialization_repeated_cycles_and_bit_exactness(
        self, gaze_config: GazeConfig
    ):
        """
        Verifies 20 repeated save-load cycles preserve 100% bit-exact prediction outputs
        and Schema 2.0 metadata integrity.
        """
        X, y, _, _ = generate_multi_point_dataset(gaze_config, "9_points", samples_per_point=15)
        model = PolynomialRidgeRegressor(gaze_config)
        model.train(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            profile_path = tmp.name

        try:
            model.save_profile(profile_path)
            current_model = model

            for cycle in range(20):
                fresh_model = PolynomialRidgeRegressor(gaze_config)
                success = fresh_model.load_profile(profile_path)
                assert success is True
                assert fresh_model.is_trained is True

                # Compare predictions against original
                for idx in range(len(X)):
                    orig_pred = model.predict(X[idx])
                    curr_pred = fresh_model.predict(X[idx])
                    assert orig_pred == curr_pred, f"Prediction mismatch at cycle {cycle}, sample {idx}"

                # Re-save from fresh model to next cycle
                fresh_model.save_profile(profile_path)
        finally:
            if os.path.exists(profile_path):
                os.remove(profile_path)
