"""Screen Gaze Canvas visualizer with glowing cursor, heat trail, and animated pulsing targets."""

import time
import math
import collections
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from src.config import GazeConfig
from src.calibration.calibrator import CalibrationManager, CalibrationState


class ScreenGazeCanvas:
    """
    Renders high-visibility modern dark slate screen canvas.
    Features:
    - Animated pulsing concentric calibration targets with 360-degree progress arc
    - Multi-ring glowing gaze dot cursor
    - Alpha-faded decaying 20-frame gaze heat trail
    - Top instruction banners and bottom status telemetry
    - Memory pre-allocation for zero allocation overhead per frame
    """

    def __init__(self, config: GazeConfig, trail_len: int = 20):
        self.config = config
        self.trail_history: collections.deque = collections.deque(maxlen=trail_len)
        self.pulse_phase: float = 0.0
        self._canvas_buffer: Optional[np.ndarray] = None
        self._bg_color = (20, 22, 28)  # Modern dark slate

    def _get_canvas_buffer(self) -> np.ndarray:
        """Reuse or allocate canvas buffer matching configured screen resolution."""
        w, h = self.config.screen_width, self.config.screen_height
        if self._canvas_buffer is None or self._canvas_buffer.shape[:2] != (h, w):
            self._canvas_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        self._canvas_buffer[:] = self._bg_color
        return self._canvas_buffer

    def render(
        self,
        gaze_pt: Optional[Tuple[float, float]],
        calibrator: CalibrationManager,
        is_trained: bool,
        metrics: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Render screen canvas based on current tracker and calibration state.
        """
        w, h = self.config.screen_width, self.config.screen_height
        canvas = self._get_canvas_buffer()
        metrics = metrics or {}

        self.pulse_phase += 0.1

        # 1. State: Calibration in progress
        if calibrator.state == CalibrationState.COLLECTING:
            target = calibrator.get_current_target()
            pt_idx, total_pts, progress = calibrator.get_progress()

            if target is not None:
                tx, ty = target
                # Pulsing target rings
                pulse_r = int(24 + 6 * math.sin(self.pulse_phase))
                
                # Outer glow & inner core
                cv2.circle(canvas, (tx, ty), pulse_r + 16, (60, 120, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), pulse_r, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 6, (255, 255, 255), -1, cv2.LINE_AA)

                # 360° progress arc ring around target
                angle = int(360 * progress)
                cv2.ellipse(
                    canvas, (tx, ty), (pulse_r + 8, pulse_r + 8),
                    -90, 0, angle, (0, 255, 120), 4, cv2.LINE_AA
                )

            # Top instructions banner
            cv2.rectangle(canvas, (0, 0), (w, 80), (30, 34, 44), -1)
            msg = f"CALIBRATION IN PROGRESS: Point {pt_idx} of {total_pts} — Focus your gaze directly on the pulsing target"
            cv2.putText(canvas, msg, (40, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Progress bar
            bar_w = int((w - 80) * (pt_idx - 1 + progress) / max(1, total_pts))
            cv2.rectangle(canvas, (40, 68), (40 + bar_w, 74), (0, 220, 255), -1)

            return canvas

        # 2. State: Calibrated & Active Gaze Tracking
        if is_trained:
            # Draw subtle calibration anchor dots
            for pt in calibrator.points:
                cv2.circle(canvas, pt, 4, (45, 52, 65), -1, cv2.LINE_AA)

            if gaze_pt is not None:
                gx, gy = int(round(gaze_pt[0])), int(round(gaze_pt[1]))
                self.trail_history.append((gx, gy))

                # Draw decaying alpha-faded heat trail
                n_trail = len(self.trail_history)
                for i, (tx, ty) in enumerate(self.trail_history):
                    alpha = (i + 1) / max(1, n_trail)
                    radius = int(8 + alpha * 14)
                    color = (int(0 * alpha), int(160 * alpha), int(255 * alpha))
                    cv2.circle(canvas, (tx, ty), radius, color, 1, cv2.LINE_AA)

                # Draw multi-ring glowing gaze dot cursor
                cv2.circle(canvas, (gx, gy), 26, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (gx, gy), 14, (0, 255, 180), -1, cv2.LINE_AA)
                cv2.circle(canvas, (gx, gy), 4, (255, 255, 255), -1, cv2.LINE_AA)

                # Coordinate overlay tag
                coord_text = f"X: {gx}px  Y: {gy}px"
                cv2.putText(canvas, coord_text, (gx + 30, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

            # Bottom status telemetry banner
            cv2.rectangle(canvas, (0, h - 50), (w, h), (28, 32, 40), -1)
            mae = float(metrics.get("mae_px", 0.0))
            status_text = f"🟢 Tracking Active | Calibration MAE: {mae:.1f}px | [C] Calibrate  [R] Reset  [S] Save  [L] Load  [D] HUD  [F] Fullscreen  [Q] Quit"
            cv2.putText(canvas, status_text, (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)

        else:
            # 3. State: Uncalibrated / Idle
            center_x, center_y = w // 2, h // 2
            cv2.putText(canvas, "REAL-TIME EYE & GAZE TRACKER", (center_x - 300, center_y - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'C' to start multi-point screen calibration", (center_x - 270, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'L' to load existing calibration file", (center_x - 240, center_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 205), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'Q' to quit", (center_x - 90, center_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 140, 155), 1, cv2.LINE_AA)

        return canvas

    def clear_trail(self) -> None:
        """Clear gaze trail history."""
        self.trail_history.clear()
