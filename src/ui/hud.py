"""Camera Debug HUD overlay with eye contours, 3D pose axes, and alpha-blended telemetry dashboard."""

from typing import Optional, Tuple, List, Any
import numpy as np
import cv2

from src.types import GazeFeatures, HeadPoseData
from src.calibration.calibrator import CalibrationManager, CalibrationState


class CameraDebugHUD:
    """
    Renders diagnostic computer vision overlays on live webcam monitor feed:
    - 16-point eyelid contours, canthal axes, and 5-point iris metric circles
    - 3D solvePnP orthogonal head pose orientation vector axes (RGB from nose)
    - Translucent alpha-blended HUD telemetry dashboard card
    """

    def __init__(self, card_w: int = 280, card_h: int = 185):
        self.card_w = card_w
        self.card_h = card_h

    def render(
        self,
        frame: np.ndarray,
        gaze_features: Optional[GazeFeatures],
        head_pose: Optional[HeadPoseData],
        gaze_pt: Optional[Tuple[float, float]],
        fps: float,
        is_trained: bool,
        calibrator: CalibrationManager
    ) -> np.ndarray:
        """
        Draw all CV annotations and HUD dashboard on the input camera frame.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        hud = frame.copy()
        img_h, img_w = hud.shape[:2]

        # 1. Draw Eye Contours and Iris Centers
        if gaze_features is not None:
            for eye in (gaze_features.left_eye, gaze_features.right_eye):
                # 16-point eyelid contour polygon
                if eye.contour_px and len(eye.contour_px) > 1:
                    pts = np.array(eye.contour_px, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(hud, [pts], isClosed=True, color=(0, 255, 120), thickness=1, lineType=cv2.LINE_AA)

                # Key anatomical landmarks
                cv2.circle(hud, eye.inner_corner_px, 2, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(hud, eye.outer_corner_px, 2, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(hud, eye.top_eyelid_px, 2, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(hud, eye.bottom_eyelid_px, 2, (255, 120, 0), -1, cv2.LINE_AA)

                # Iris Center & Radius Indicator
                iris_color = (0, 220, 255) if eye.is_open else (0, 0, 255)
                cv2.circle(hud, eye.iris_center_px, 4, iris_color, -1, cv2.LINE_AA)
                radius = max(6, int(round(eye.iris_diameter_px / 2.0))) if eye.iris_diameter_px > 0 else 8
                cv2.circle(hud, eye.iris_center_px, radius, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw 3D Head Pose Axes from Nose Tip
        if head_pose is not None:
            nose = head_pose.nose_2d_px
            axes = head_pose.axes_2d_px
            if len(axes) >= 3:
                cv2.line(hud, nose, axes[0], (0, 0, 255), 2, cv2.LINE_AA)  # X-axis: Red
                cv2.line(hud, nose, axes[1], (0, 255, 0), 2, cv2.LINE_AA)  # Y-axis: Green
                cv2.line(hud, nose, axes[2], (255, 0, 0), 2, cv2.LINE_AA)  # Z-axis: Blue
                cv2.circle(hud, nose, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Translucent Telemetry Card Overlay
        card_w, card_h = self.card_w, self.card_h
        if img_w > card_w + 20 and img_h > card_h + 20:
            overlay = hud.copy()
            cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (15, 18, 25), -1)
            cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
            cv2.rectangle(hud, (10, 10), (10 + card_w, 10 + card_h), (60, 70, 85), 1)

            # Telemetry Metrics Text
            y_offset = 30
            cv2.putText(hud, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
            
            if head_pose:
                y_offset += 20
                pose_str = f"Head: P:{head_pose.pitch:.1f}° Y:{head_pose.yaw:.1f}° R:{head_pose.roll:.1f}°"
                cv2.putText(hud, pose_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

            if gaze_features:
                y_offset += 20
                ear_str = f"EAR: L:{gaze_features.left_eye.ear:.2f} R:{gaze_features.right_eye.ear:.2f}"
                blink_color = (0, 255, 120) if gaze_features.is_valid else (0, 0, 255)
                cv2.putText(hud, ear_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, blink_color, 1, cv2.LINE_AA)

                y_offset += 20
                norm_str = f"Iris Norm: ({gaze_features.avg_norm_x:.2f}, {gaze_features.avg_norm_y:.2f})"
                cv2.putText(hud, norm_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

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
                gaze_str = f"Screen: ({int(round(gaze_pt[0]))}, {int(round(gaze_pt[1]))})"
                cv2.putText(hud, gaze_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return hud
