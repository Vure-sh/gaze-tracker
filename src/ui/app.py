"""Main desktop application controller and event loop for the Real-Time Gaze Tracker."""

import time
import sys
from typing import Optional, Union, Any
import cv2
import numpy as np

from src.config import GazeConfig
from src.pipeline import GazePipeline, GazePipelineResult
from src.camera_stream import ThreadedCameraStream
from src.ui.canvas import ScreenGazeCanvas
from src.ui.hud import CameraDebugHUD


class GazeTrackerApp:
    """
    Complete application controller coupling camera capture, pipeline execution,
    real-time UI rendering, and interactive keyboard event dispatch.
    """

    def __init__(
        self,
        config: Optional[GazeConfig] = None,
        camera_src: Union[str, int] = "/dev/video9",
        fullscreen: bool = False,
        show_hud: bool = True,
        grid_type: str = "9_points",
        filter_type: str = "one_euro",
        load_profile: Optional[str] = None
    ):
        self.config = config or GazeConfig()
        self.config.calibration_grid_type = grid_type
        self.config.filter_type = filter_type

        self.camera_src = camera_src
        self.fullscreen = fullscreen
        self.show_hud = show_hud
        self.load_profile_path = load_profile

        # Components
        self.pipeline = GazePipeline(self.config, filter_type=filter_type)
        self.canvas_renderer = ScreenGazeCanvas(self.config)
        self.hud_renderer = CameraDebugHUD()
        self.camera_stream: Optional[ThreadedCameraStream] = None

        # Window identifiers
        self.canvas_window = "Screen Gaze Canvas"
        self.hud_window = "Camera Debug HUD"
        self._running = False

    def setup(self) -> None:
        """Initialize windows, load initial profiles, and configure displays."""
        # Try loading initial calibration profile if requested
        load_target = self.load_profile_path or self.config.calibration_file
        if self.pipeline.load_calibration(load_target):
            print(f"✅ Loaded calibration profile from {load_target}")

        # Setup Canvas Window
        cv2.namedWindow(self.canvas_window, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.canvas_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            w = min(1280, self.config.screen_width)
            h = min(720, self.config.screen_height)
            cv2.resizeWindow(self.canvas_window, w, h)

        # Setup HUD Window (if enabled)
        if self.show_hud:
            cv2.namedWindow(self.hud_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.hud_window, 640, 480)

    def run(self) -> None:
        """Execute the real-time application processing loop."""
        self.setup()
        self._running = True

        print("=" * 60)
        print("🎯 REAL-TIME WEBCAM EYE & GAZE TRACKER RUNNING")
        print("Keyboard Controls:")
        print("  [C] - Start Multi-Point Screen Calibration")
        print("  [R] - Reset Current Calibration")
        print("  [S] - Save Calibration Profile to File")
        print("  [L] - Load Calibration Profile from File")
        print("  [D] - Toggle Camera Debug HUD Window")
        print("  [F] - Toggle Fullscreen Canvas")
        print("  [Q] / [ESC] - Exit Application")
        print("=" * 60)

        # Start background threaded capture
        self.camera_stream = ThreadedCameraStream(self.camera_src)
        self.camera_stream.start()

        try:
            while self._running:
                has_frame, frame = self.camera_stream.read(wait_timeout=0.03)
                if not has_frame or frame is None:
                    time.sleep(0.005)
                    continue

                # 1. Execute full pipeline step
                result: GazePipelineResult = self.pipeline.process_frame(frame)

                # 2. Render Screen Canvas
                canvas = self.canvas_renderer.render(
                    gaze_pt=result.smoothed_gaze,
                    calibrator=self.pipeline.calibrator,
                    is_trained=self.pipeline.regressor.is_trained,
                    metrics=self.pipeline.regressor.metrics
                )
                cv2.imshow(self.canvas_window, canvas)

                # 3. Render Camera Debug HUD (if enabled)
                if self.show_hud:
                    hud = self.hud_renderer.render(
                        frame=frame,
                        gaze_features=result.gaze_features,
                        head_pose=result.head_pose,
                        gaze_pt=result.smoothed_gaze,
                        fps=result.fps,
                        is_trained=self.pipeline.regressor.is_trained,
                        calibrator=self.pipeline.calibrator
                    )
                    cv2.imshow(self.hud_window, hud)

                # 4. Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_key(key)

        except KeyboardInterrupt:
            print("\nShutting down gracefully on KeyboardInterrupt...")
        finally:
            self.teardown()

    def handle_key(self, key: int) -> bool:
        """
        Process keyboard hotkey events.
        Returns False if exit requested, True otherwise.
        """
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            self._running = False
            return False
        elif key in (ord('c'), ord('C')):
            from src.calibration.calibrator import CalibrationState
            if self.pipeline.calibrator.state != CalibrationState.COLLECTING:
                print("\n🎯 Starting screen calibration...")
                self.pipeline.start_calibration(self.config.calibration_grid_type)
            else:
                print("ℹ️ Calibration already in progress. Focus on the pulsing dots.")
        elif key in (ord('r'), ord('R')):
            print("🧹 Resetting calibration...")
            self.pipeline.reset()
            self.canvas_renderer.clear_trail()
        elif key in (ord('s'), ord('S')):
            if self.pipeline.save_calibration():
                print("💾 Saved calibration profile to disk.")
        elif key in (ord('l'), ord('L')):
            if self.pipeline.load_calibration():
                print("📂 Loaded calibration profile from disk.")
                self.canvas_renderer.clear_trail()
        elif key in (ord('d'), ord('D')):
            self.show_hud = not self.show_hud
            try:
                if not self.show_hud:
                    cv2.destroyWindow(self.hud_window)
                else:
                    cv2.namedWindow(self.hud_window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.hud_window, 640, 480)
            except Exception:
                pass
        elif key in (ord('f'), ord('F')):
            self.fullscreen = not self.fullscreen
            try:
                if self.fullscreen:
                    cv2.setWindowProperty(self.canvas_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.canvas_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            except Exception:
                pass

        return True

    def teardown(self) -> None:
        """Release all hardware devices and destroy UI windows."""
        self._running = False
        if self.camera_stream is not None:
            self.camera_stream.stop()
            self.camera_stream = None

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Gaze Tracker Application closed cleanly.")
