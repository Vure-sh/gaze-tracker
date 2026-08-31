"""2D Constant-Velocity Kalman Filter for screen gaze coordinate smoothing."""

import time
from typing import Tuple, Optional
import numpy as np


class KalmanFilter2D:
    """
    2D Constant-Velocity Kalman Filter for Screen Gaze Coordinates.
    State vector: x = [pos_x, pos_y, vel_x, vel_y]^T
    Measurement: z = [pos_x, pos_y]^T
    """

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        
        # State: [x, y, vx, vy]^T
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0  # Initial state covariance
        self.Q = np.eye(4, dtype=np.float64) * self.process_noise  # Process noise covariance
        self.R = np.eye(2, dtype=np.float64) * self.measurement_noise  # Measurement noise covariance
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        self.last_time: Optional[float] = None
        self.is_initialized = False

    def filter(
        self,
        pt: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Predict and update the 2D position given noisy measurement pt=(x, y).
        """
        t = timestamp if timestamp is not None else time.time()

        if not self.is_initialized:
            self.state = np.array([[float(pt[0])], [float(pt[1])], [0.0], [0.0]], dtype=np.float64)
            self.P = np.eye(4, dtype=np.float64) * 100.0
            self.last_time = t
            self.is_initialized = True
            return (float(pt[0]), float(pt[1]))

        dt = max(1e-4, t - (self.last_time if self.last_time is not None else t))
        self.last_time = t

        # State transition matrix F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Discrete process noise matrix Q with continuous white noise acceleration model
        q_pos = (dt ** 3) / 3.0 * self.process_noise
        q_pos_vel = (dt ** 2) / 2.0 * self.process_noise
        q_vel = dt * self.process_noise
        Q = np.array([
            [q_pos,     0.0,       q_pos_vel, 0.0      ],
            [0.0,       q_pos,     0.0,       q_pos_vel],
            [q_pos_vel, 0.0,       q_vel,     0.0      ],
            [0.0,       q_pos_vel, 0.0,       q_vel    ]
        ], dtype=np.float64)

        # 1. Predict
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        # 2. Update (Measurement)
        z = np.array([[float(pt[0])], [float(pt[1])]], dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return (float(self.state[0, 0]), float(self.state[1, 0]))

    def reset(self) -> None:
        """Reset internal filter state and covariance."""
        self.is_initialized = False
        self.last_time = None
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
