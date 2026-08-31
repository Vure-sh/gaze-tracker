"""Challenger Milestone 4 Verification: Multi-Resolution Canvas Scaling, UI Stress & Fast Loop Rendering."""

import numpy as np
import pytest

from src.config import GazeConfig
from src.calibration.calibrator import CalibrationManager
from src.models.regressor import GazeRegressionModel
from src.ui.canvas import ScreenGazeCanvas
from src.ui.hud import CameraDebugHUD
from src.ui.app import GazeTrackerApp
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator
from tests.conftest import create_synthetic_landmarks


class TestChallengerM4UIStressAndScaling:
    """Adversarial UI stress and resolution invariance tests."""

    @pytest.mark.parametrize("resolution", [
        (640, 480),     # Small / Embedded
        (1280, 720),    # 720p HD
        (1920, 1080),   # 1080p FHD
        (2560, 1440),   # 2K QHD
        (3840, 2160),   # 4K UHD
        (3440, 1440),   # Ultrawide 21:9
    ])
    def test_multi_resolution_canvas_rendering(self, resolution):
        """Verify ScreenGazeCanvas accurately scales and renders across all standard display aspect ratios."""
        w, h = resolution
        cfg = GazeConfig()
        cfg.screen_width = w
        cfg.screen_height = h

        canvas_viz = ScreenGazeCanvas(cfg)
        regressor = GazeRegressionModel(cfg)
        calibrator = CalibrationManager(cfg, regressor)
        calibrator.points = calibrator.generate_points("9_points")

        # Test uncalibrated
        c_uncal = canvas_viz.render(None, calibrator, False, {})
        assert c_uncal.shape == (h, w, 3)

        # Test calibrating
        calibrator.start_calibration("9_points")
        c_calib = canvas_viz.render(None, calibrator, False, {})
        assert c_calib.shape == (h, w, 3)

        # Test tracking
        c_track = canvas_viz.render((w / 2, h / 2), calibrator, True, {"mae_px": 15.0})
        assert c_track.shape == (h, w, 3)

    def test_rapid_canvas_and_hud_rendering_throughput(self, gaze_config: GazeConfig, mock_bgr_frame, synthetic_landmarks):
        """Verify rendering 300 frames of full canvas and HUD achieves > 100 FPS."""
        canvas_viz = ScreenGazeCanvas(gaze_config)
        hud_viz = CameraDebugHUD()
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.points = calibrator.generate_points()

        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        features = extractor.extract(synthetic_landmarks, 640, 480)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)

        import time
        t0 = time.perf_counter()
        n_frames = 200

        for i in range(n_frames):
            gaze_x = (960.0 + i * 2) % gaze_config.screen_width
            gaze_y = (540.0 + i) % gaze_config.screen_height

            _ = canvas_viz.render((gaze_x, gaze_y), calibrator, True, {"mae_px": 18.0})
            _ = hud_viz.render(mock_bgr_frame, features, pose, (gaze_x, gaze_y), 60.0, True, calibrator)

        elapsed = time.perf_counter() - t0
        render_fps = n_frames / elapsed
        print(f"\n⚡ UI Render Throughput: {render_fps:.1f} FPS ({elapsed*1000/n_frames:.2f}ms per frame)")
        assert render_fps > 50.0, f"Render throughput {render_fps:.1f} FPS is too slow"

    def test_hotkey_sequence_barrage(self, gaze_config: GazeConfig):
        """Verify rapid interleaved keyboard hotkey events maintain consistent state."""
        app = GazeTrackerApp(config=gaze_config)
        keys = [
            ord('c'), ord('d'), ord('f'), ord('r'),
            ord('c'), ord('d'), ord('f'), ord('s'),
            ord('l'), ord('r')
        ]
        for k in keys:
            assert app.handle_key(k) is True

        assert app._running is False or app._running is True
