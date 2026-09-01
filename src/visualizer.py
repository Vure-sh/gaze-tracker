"""Visualizer for rendering the Screen Gaze Canvas, Calibration Targets, and Debug HUD."""

import time
import math
import collections
from typing import Optional, Tuple, List, Any
import numpy as np
import cv2
from .config import GazeConfig
from .eye_extractor import GazeFeatures
from .head_pose import HeadPoseData
from .calibrator import CalibrationManager, CalibrationState

class GazeVisualizer:
    """Renders both the full-screen gaze canvas and the webcam debug HUD."""

    def __init__(self, config: GazeConfig):
        self.config = config
        self.trail_history = collections.deque(maxlen=20)
        self.fps_history = collections.deque(maxlen=30)
        self.last_frame_time = time.time()
        self.pulse_phase = 0.0

    def compute_fps(self) -> float:
        now = time.time()
        dt = max(1e-4, now - self.last_frame_time)
        self.last_frame_time = now
        self.fps_history.append(1.0 / dt)
        return float(np.mean(self.fps_history))

    def create_screen_canvas(
        self,
        gaze_pt: Optional[Tuple[float, float]],
        calibrator: CalibrationManager,
        is_trained: bool,
        metrics: dict
    ) -> np.ndarray:
        """
        Generates the fullscreen screen gaze canvas with calibration cues or real-time gaze cursor.
        """
        w, h = self.config.screen_width, self.config.screen_height
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 28)  # Modern dark slate background

        self.pulse_phase += 0.1

        # 1. State: Calibration in progress
        if calibrator.state == CalibrationState.COLLECTING:
            target = calibrator.get_current_target()
            pt_idx, total_pts, progress = calibrator.get_progress()

            if target:
                tx, ty = target
                # Pulsing target rings
                pulse_r = int(24 + 6 * math.sin(self.pulse_phase))
                
                # Outer glow
                cv2.circle(canvas, (tx, ty), pulse_r + 16, (60, 120, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), pulse_r, (0, 220, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 6, (255, 255, 255), -1, cv2.LINE_AA)

                # Progress arc ring around target
                angle = int(360 * progress)
                cv2.ellipse(canvas, (tx, ty), (pulse_r + 8, pulse_r + 8), -90, 0, angle, (0, 255, 120), 4, cv2.LINE_AA)

            # Top instructions banner
            cv2.rectangle(canvas, (0, 0), (w, 80), (30, 34, 44), -1)
            msg = f"CALIBRATION IN PROGRESS: Point {pt_idx} of {total_pts} - Focus your gaze directly on the pulsing dot"
            cv2.putText(canvas, msg, (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Progress bar
            bar_w = int((w - 80) * (pt_idx - 1 + progress) / total_pts)
            cv2.rectangle(canvas, (40, 70), (40 + bar_w, 76), (0, 220, 255), -1)

            return canvas

        # 2. State: Normal Gaze Tracking Mode
        if is_trained:
            # Draw subtle screen grid / calibration markers
            for pt in calibrator.points:
                cv2.circle(canvas, pt, 4, (45, 52, 65), -1, cv2.LINE_AA)

            if gaze_pt is not None:
                gx, gy = int(gaze_pt[0]), int(gaze_pt[1])
                self.trail_history.append((gx, gy))

                # Draw smooth fading gaze trail
                for i, (tx, ty) in enumerate(self.trail_history):
                    alpha = (i + 1) / len(self.trail_history)
                    radius = int(8 + alpha * 14)
                    color = (int(0 * alpha), int(160 * alpha), int(255 * alpha))
                    cv2.circle(canvas, (tx, ty), radius, color, 1, cv2.LINE_AA)

                # Draw main gaze dot cursor
                cv2.circle(canvas, (gx, gy), 26, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (gx, gy), 14, (0, 255, 180), -1, cv2.LINE_AA)
                cv2.circle(canvas, (gx, gy), 4, (255, 255, 255), -1, cv2.LINE_AA)

                # Target coordinates label
                coord_text = f"X: {gx}px  Y: {gy}px"
                cv2.putText(canvas, coord_text, (gx + 30, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

            # Bottom status banner
            cv2.rectangle(canvas, (0, h - 50), (w, h), (28, 32, 40), -1)
            mae = metrics.get('mae_px', 0)
            status_text = f"🟢 Tracking Active | Calibration MAE: {mae:.1f}px | [C] Re-Calibrate  [R] Reset  [S] Save  [L] Load  [Q] Quit"
            cv2.putText(canvas, status_text, (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

        else:
            # 3. State: Uncalibrated
            center_x, center_y = w // 2, h // 2
            cv2.putText(canvas, "REAL-TIME EYE & GAZE TRACKER", (center_x - 300, center_y - 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'C' to start 9-point screen calibration", (center_x - 260, center_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'L' to load existing calibration file", (center_x - 240, center_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 205), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press 'Q' to quit", (center_x - 90, center_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 140, 155), 1, cv2.LINE_AA)

        return canvas

    def draw_debug_hud(
        self,
        frame: np.ndarray,
        gaze_features: Optional[GazeFeatures],
        head_pose: Optional[HeadPoseData],
        gaze_pt: Optional[Tuple[float, float]],
        fps: float,
        is_trained: bool,
        calibrator: CalibrationManager
    ) -> np.ndarray:
        """Renders the webcam monitor feed with face/iris overlays, head pose axes, and HUD metrics."""
        hud = frame.copy()
        h, w = hud.shape[:2]

        if gaze_features is not None:
            # 1. Draw Eye Contours and Iris Centers
            for eye in (gaze_features.left_eye, gaze_features.right_eye):
                # Contour
                for i in range(len(eye.contour_px)):
                    p1 = eye.contour_px[i]
                    p2 = eye.contour_px[(i + 1) % len(eye.contour_px)]
                    cv2.line(hud, p1, p2, (0, 255, 120), 1, cv2.LINE_AA)

                # Key landmark points
                cv2.circle(hud, eye.inner_corner_px, 2, (0, 255, 255), -1)
                cv2.circle(hud, eye.outer_corner_px, 2, (0, 255, 255), -1)
                cv2.circle(hud, eye.top_eyelid_px, 2, (255, 120, 0), -1)
                cv2.circle(hud, eye.bottom_eyelid_px, 2, (255, 120, 0), -1)

                # Iris Center & Radius
                iris_color = (0, 200, 255) if eye.is_open else (0, 0, 255)
                cv2.circle(hud, eye.iris_center_px, 4, iris_color, -1, cv2.LINE_AA)
                cv2.circle(hud, eye.iris_center_px, 8, (255, 255, 255), 1, cv2.LINE_AA)

        if head_pose is not None:
            # 2. Draw 3D Head Pose Axes from Nose Tip
            nose = head_pose.nose_2d_px
            axes = head_pose.axes_2d_px
            if len(axes) == 3:
                cv2.line(hud, nose, axes[0], (0, 0, 255), 2, cv2.LINE_AA)  # X: Red
                cv2.line(hud, nose, axes[1], (0, 255, 0), 2, cv2.LINE_AA)  # Y: Green
                cv2.line(hud, nose, axes[2], (255, 0, 0), 2, cv2.LINE_AA)  # Z: Blue
                cv2.circle(hud, nose, 3, (255, 255, 255), -1)

        # 3. HUD Dashboard Overlay (Top-Left translucent card)
        card_w, card_h = 280, 175
        overlay = hud.copy()
        cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (15, 18, 25), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
        cv2.rectangle(hud, (10, 10), (10 + card_w, 10 + card_h), (60, 70, 85), 1)

        # Text Metrics
        y_offset = 30
        cv2.putText(hud, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
        
        if head_pose:
            y_offset += 20
            pose_str = f"Head: P:{head_pose.pitch:.1f}° Y:{head_pose.yaw:.1f}° R:{head_pose.roll:.1f}°"
            cv2.putText(hud, pose_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        if gaze_features:
            y_offset += 20
            ear_str = f"EAR: L:{gaze_features.left_eye.ear:.2f} R:{gaze_features.right_eye.ear:.2f}"
            blink_color = (0, 255, 120) if gaze_features.is_valid else (0, 0, 255)
            cv2.putText(hud, ear_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, blink_color, 1, cv2.LINE_AA)

            y_offset += 20
            norm_str = f"Iris Norm: ({gaze_features.avg_norm_x:.2f}, {gaze_features.avg_norm_y:.2f})"
            cv2.putText(hud, norm_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        y_offset += 22
        if calibrator.state == CalibrationState.COLLECTING:
            status_text = "Status: CALIBRATING..."
            status_color = (0, 220, 255)
        elif is_trained:
            status_text = "Status: TRACKING"
            status_color = (0, 255, 120)
        else:
            status_text = "Status: UNCALIBRATED"
            status_color = (0, 160, 255)
        cv2.putText(hud, status_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

        if gaze_pt and is_trained:
            y_offset += 20
            gaze_str = f"Screen: ({int(gaze_pt[0])}, {int(gaze_pt[1])})"
            cv2.putText(hud, gaze_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return hud

    def draw_confidence_bar(
        self,
        frame: np.ndarray,
        confidence: float,
        x: int = 10,
        y: int = 200,
        width: int = 120,
        height: int = 10,
    ) -> np.ndarray:
        """Draw a color-coded tracking confidence bar onto *frame* in-place.

        The bar transitions from red (low) through yellow to green (high) and
        is annotated with a percentage label.

        Args:
            frame: BGR image to draw on.
            confidence: Value in [0.0, 1.0].
            x, y: Top-left corner of the bar.
            width, height: Dimensions of the bar.

        Returns:
            The modified frame (same object, mutated in-place).
        """
        conf = max(0.0, min(1.0, float(confidence)))
        fill_w = int(width * conf)
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        # Color: red -> yellow -> green
        r = int(255 * (1.0 - conf))
        g = int(255 * conf)
        bar_color = (0, g, r)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (120, 120, 120), 1)
        # Label
        label = f"{int(conf * 100)}%"
        cv2.putText(frame, label, (x + width + 6, y + height - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        return frame
