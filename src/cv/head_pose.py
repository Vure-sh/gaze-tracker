"""3D Head pose estimation using Perspective-n-Point (solvePnP) with corrected anthropometric model."""

from __future__ import annotations
from typing import Optional, Tuple, List, Any
import numpy as np
import cv2

from src.config import GazeConfig
from src.types import HeadPoseData


class HeadPoseEstimator:
    """Estimates head yaw, pitch, and roll in real-time from 2D facial landmarks."""

    # Corrected 3D anthropometric face model aligned with OpenCV camera optical frame:
    # +X points Right, +Y points Down, +Z points Forward into camera
    # All dimensions in millimeters (based on standard human anthropometric averages)
    MODEL_POINTS_CORRECTED = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
        (0.0, 100.0, -20.0),      # Chin (Landmark 152)
        (-65.0, -50.0, -40.0),    # Left eye outer corner (Landmark 33)
        (65.0, -50.0, -40.0),     # Right eye outer corner (Landmark 263)
        (-40.0, 50.0, -30.0),     # Left mouth corner (Landmark 61)
        (40.0, 50.0, -30.0)       # Right mouth corner (Landmark 291)
    ], dtype=np.float64)

    # Legacy 3D model for backwards compatibility testing
    MODEL_POINTS_LEGACY = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (Landmark 1)
        (0.0, -330.0, -65.0),     # Chin (Landmark 152)
        (-225.0, 170.0, -135.0),  # Left eye outer corner (Landmark 33)
        (225.0, 170.0, -135.0),   # Right eye outer corner (Landmark 263)
        (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
        (150.0, -150.0, -125.0)   # Right mouth corner (Landmark 291)
    ], dtype=np.float64)

    # Default alias to corrected model
    MODEL_POINTS = MODEL_POINTS_CORRECTED

    def __init__(self, config: Optional[GazeConfig] = None):
        self.config = config or GazeConfig()
        if getattr(self.config, "head_pose_use_corrected_model", True):
            self.model_points = self.MODEL_POINTS_CORRECTED
        else:
            self.model_points = self.MODEL_POINTS_LEGACY

    def _get_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        return self.config.get_camera_matrix(img_w, img_h)

    def estimate(
        self,
        landmarks: List[Any],
        img_w: int,
        img_h: int
    ) -> Optional[HeadPoseData]:
        """
        Estimates 3D head pose from detected facial landmarks.

        Args:
            landmarks: List of at least 468 NormalizedPoint/Landmark objects.
            img_w: Frame width in pixels.
            img_h: Frame height in pixels.

        Returns:
            HeadPoseData instance or None if solvePnP fails.
        """
        if landmarks is None or len(landmarks) < 468:
            return None

        # Extract 2D image coordinates corresponding to the 3D model points
        image_points = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in self.config.head_pose_mesh_indices
        ], dtype=np.float64)

        camera_matrix = self._get_camera_matrix(img_w, img_h)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success or rvec is None or tvec is None:
                return None

            # Convert rotation vector to rotation matrix
            rot_mat, _ = cv2.Rodrigues(rvec)

            # Decompose rotation matrix into Euler angles (Pitch, Yaw, Roll)
            # Using standard intrinsic ZYX / extrinsic XYZ convention
            sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
                yaw = np.arctan2(-rot_mat[2, 0], sy)
                roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
                yaw = np.arctan2(-rot_mat[2, 0], sy)
                roll = 0.0

            pitch_deg = float(np.degrees(pitch))
            yaw_deg = float(np.degrees(yaw))
            roll_deg = float(np.degrees(roll))

            # Project 3D axis arrows from nose tip for visualization
            axis_length = 80.0
            axis_3d = np.array([
                (axis_length, 0.0, 0.0),   # X axis (Red) -> Right
                (0.0, axis_length, 0.0),   # Y axis (Green) -> Down
                (0.0, 0.0, -axis_length)   # Z axis (Blue) -> Forward out of face
            ], dtype=np.float64)

            proj_axes, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
            nose_pt = (int(round(image_points[0][0])), int(round(image_points[0][1])))
            axes_pts = [(int(round(p[0][0])), int(round(p[0][1]))) for p in proj_axes]

            # Normalized head pose feature vector (Euler angles + normalized translation)
            feature_vector = np.array([
                pitch_deg / 45.0,
                yaw_deg / 45.0,
                roll_deg / 45.0,
                float(tvec[0, 0]) / 500.0,
                float(tvec[1, 0]) / 500.0,
                float(tvec[2, 0]) / 1000.0
            ], dtype=np.float64)

            return HeadPoseData(
                pitch=pitch_deg,
                yaw=yaw_deg,
                roll=roll_deg,
                rvec=rvec,
                tvec=tvec,
                nose_2d_px=nose_pt,
                axes_2d_px=axes_pts,
                feature_vector=feature_vector,
                rot_mat=rot_mat
            )
        except Exception:
            return None
