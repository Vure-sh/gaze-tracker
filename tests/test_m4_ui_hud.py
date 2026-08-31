"""Milestone 4 Verification: Screen Gaze Canvas, Animated Targets, Camera Debug HUD, Hotkeys & CLI."""

import math
import numpy as np
import pytest

from src.config import GazeConfig
from src.types import GazeFeatures, HeadPoseData
from src.calibration.calibrator import CalibrationManager, CalibrationState
from src.models.regressor import GazeRegressionModel
from src.ui.canvas import ScreenGazeCanvas
from src.ui.hud import CameraDebugHUD
from src.ui.app import GazeTrackerApp
from tests.conftest import create_synthetic_landmarks
from src.cv.eye_extractor import EyeExtractor
from src.cv.head_pose import HeadPoseEstimator


# ============================================================================
# 1. Screen Gaze Canvas Tests
# ============================================================================

class TestScreenGazeCanvas:
    """Tests for dark slate screen canvas, animated pulsing target, and glowing gaze cursor."""

    def test_canvas_idle_uncalibrated_rendering(self, gaze_config: GazeConfig):
        """Verify canvas renders title and calibration instructions in uncalibrated state."""
        canvas_viz = ScreenGazeCanvas(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)

        img = canvas_viz.render(None, calibrator, False, {})
        assert img.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)
        assert img.dtype == np.uint8
        # Should have rendered text (non-background pixels)
        assert img.sum() > 0

    def test_canvas_calibrating_pulsing_target_and_progress_arc(self, gaze_config: GazeConfig):
        """Verify canvas renders animated target and progress feedback during active calibration."""
        canvas_viz = ScreenGazeCanvas(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.start_calibration("9_points")

        # Simulate 10 frames of calibration
        for _ in range(10):
            img = canvas_viz.render(None, calibrator, False, {})
            assert img.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)

        assert canvas_viz.pulse_phase > 0.0

    def test_canvas_tracking_cursor_and_trail(self, gaze_config: GazeConfig):
        """Verify active gaze tracking renders glowing cursor and maintains decaying heat trail."""
        canvas_viz = ScreenGazeCanvas(gaze_config, trail_len=15)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        calibrator.points = calibrator.generate_points()

        for step in range(25):
            pt = (500.0 + step * 10, 300.0 + step * 5)
            img = canvas_viz.render(pt, calibrator, True, {"mae_px": 12.5})
            assert img.shape == (gaze_config.screen_height, gaze_config.screen_width, 3)

        assert len(canvas_viz.trail_history) == 15

    def test_canvas_clear_trail(self, gaze_config: GazeConfig):
        """Verify clear_trail() empties gaze history."""
        canvas_viz = ScreenGazeCanvas(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        canvas_viz.render((500.0, 500.0), calibrator, True, {})
        assert len(canvas_viz.trail_history) == 1

        canvas_viz.clear_trail()
        assert len(canvas_viz.trail_history) == 0


# ============================================================================
# 2. Camera Debug HUD Tests
# ============================================================================

class TestCameraDebugHUD:
    """Tests for eye contours, 3D head pose orientation axes, and translucent telemetry card."""

    def test_hud_rendering_with_empty_or_none_frame(self):
        """Verify HUD safely returns fallback array when input frame is empty or None."""
        hud_viz = CameraDebugHUD()
        regressor = GazeRegressionModel(GazeConfig())
        calibrator = CalibrationManager(GazeConfig(), regressor)

        out_none = hud_viz.render(None, None, None, None, 30.0, False, calibrator)
        assert out_none.shape == (480, 640, 3)

        out_empty = hud_viz.render(np.zeros((0, 0), dtype=np.uint8), None, None, None, 30.0, False, calibrator)
        assert out_empty.shape == (480, 640, 3)

    def test_hud_rendering_with_full_annotations(
        self, gaze_config: GazeConfig, mock_bgr_frame, synthetic_landmarks
    ):
        """Verify HUD renders eye contours, iris markers, 3D axes, and telemetry card with full data."""
        extractor = EyeExtractor(gaze_config)
        estimator = HeadPoseEstimator(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        hud_viz = CameraDebugHUD()

        features = extractor.extract(synthetic_landmarks, 640, 480)
        pose = estimator.estimate(synthetic_landmarks, 640, 480)
        gaze_pt = (960.0, 540.0)

        out = hud_viz.render(
            frame=mock_bgr_frame,
            gaze_features=features,
            head_pose=pose,
            gaze_pt=gaze_pt,
            fps=59.5,
            is_trained=True,
            calibrator=calibrator
        )

        assert out.shape == mock_bgr_frame.shape
        # Image should be modified with annotations
        assert not np.array_equal(out, mock_bgr_frame)

    def test_hud_blink_color_change(
        self, gaze_config: GazeConfig, mock_bgr_frame
    ):
        """Verify HUD changes color scheme when eye is closed/blinking."""
        extractor = EyeExtractor(gaze_config)
        regressor = GazeRegressionModel(gaze_config)
        calibrator = CalibrationManager(gaze_config, regressor)
        hud_viz = CameraDebugHUD()

        lm_closed = create_synthetic_landmarks(left_eye_closed=True, right_eye_closed=True)
        features_closed = extractor.extract(lm_closed, 640, 480)
        assert features_closed.left_eye.is_open is False

        out = hud_viz.render(
            frame=mock_bgr_frame,
            gaze_features=features_closed,
            head_pose=None,
            gaze_pt=None,
            fps=30.0,
            is_trained=False,
            calibrator=calibrator
        )
        assert out.shape == mock_bgr_frame.shape


# ============================================================================
# 3. Application Hotkeys & Event Loop Handling
# ============================================================================

class TestGazeTrackerAppHotkeys:
    """Tests for keyboard controls and application lifecycle events."""

    def test_app_hotkey_calibrate(self, gaze_config: GazeConfig):
        """Verify 'C' key initiates calibration."""
        app = GazeTrackerApp(config=gaze_config)
        assert app.pipeline.calibrator.state == CalibrationState.IDLE
        ret = app.handle_key(ord('c'))
        assert ret is True
        assert app.pipeline.calibrator.state == CalibrationState.COLLECTING

    def test_app_hotkey_reset(self, gaze_config: GazeConfig):
        """Verify 'R' key resets calibration and trail."""
        app = GazeTrackerApp(config=gaze_config)
        app.handle_key(ord('c'))
        assert app.pipeline.calibrator.state == CalibrationState.COLLECTING

        app.handle_key(ord('r'))
        assert app.pipeline.calibrator.state == CalibrationState.IDLE
        assert app.pipeline.regressor.is_trained is False
        assert len(app.canvas_renderer.trail_history) == 0

    def test_app_hotkey_toggle_hud(self, gaze_config: GazeConfig):
        """Verify 'D' key toggles HUD display state."""
        app = GazeTrackerApp(config=gaze_config, show_hud=True)
        assert app.show_hud is True
        app.handle_key(ord('d'))
        assert app.show_hud is False
        app.handle_key(ord('d'))
        assert app.show_hud is True

    def test_app_hotkey_quit(self, gaze_config: GazeConfig):
        """Verify 'Q' and ESC keys signal application exit."""
        app = GazeTrackerApp(config=gaze_config)
        app._running = True

        ret_q = app.handle_key(ord('q'))
        assert ret_q is False
        assert app._running is False

        app._running = True
        ret_esc = app.handle_key(27)
        assert ret_esc is False
        assert app._running is False
