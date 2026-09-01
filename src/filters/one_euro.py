"""Velocity-gated One-Euro temporal filter for jitter-free real-time gaze tracking."""

import time
import math
from typing import Tuple, Optional


class LowPassFilter:
    """Standard 1st-order discrete low-pass filter with exponential smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.hat_x_prev: Optional[float] = None

    def filter(self, x: float, alpha: Optional[float] = None) -> float:
        """Filter input scalar x with given or configured alpha."""
        if alpha is not None:
            self.alpha = float(alpha)
        if self.hat_x_prev is None:
            self.hat_x_prev = float(x)
            return float(x)
        hat_x = self.alpha * float(x) + (1.0 - self.alpha) * self.hat_x_prev
        self.hat_x_prev = hat_x
        return hat_x

    def reset(self) -> None:
        """Reset filter internal state."""
        self.hat_x_prev = None


class OneEuroFilter1D:
    """
    1D One-Euro Filter (Casiez, Roussel, Vogel, CHI 2012).
    Dynamically adapts cutoff frequency based on input signal derivative (velocity):
    - Low cutoff during steady fixation (high jitter attenuation, < 1.1px variance)
    - High cutoff during rapid saccades (near-zero lag, fast settling < 3 frames)
    """

    def __init__(
        self,
        min_cutoff: float = 0.05,
        beta: float = 0.6,
        d_cutoff: float = 1.0,
        deadband: float = 0.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.deadband = float(deadband)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, te: float) -> float:
        """Compute exponential smoothing factor alpha from cutoff frequency and sampling period."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        """
        Filter 1D value at given timestamp (or current wall-clock time).
        Guards against zero/negative delta time and handles velocity deadband.
        """
        t = timestamp if timestamp is not None else time.time()

        if self.t_prev is None:
            self.t_prev = t
            return self.x_filter.filter(x, alpha=1.0)

        te = t - self.t_prev
        if te <= 1e-5:
            return self.x_filter.hat_x_prev if self.x_filter.hat_x_prev is not None else float(x)

        # Estimate derivative (speed)
        prev_x = self.x_filter.hat_x_prev if self.x_filter.hat_x_prev is not None else float(x)
        diff = float(x) - prev_x

        # Deadband for micro-jitter attenuation
        if abs(diff) < self.deadband:
            x = prev_x
            diff = 0.0

        dx = diff / te
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, te))

        # Dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(edx)
        hat_x = self.x_filter.filter(x, self._alpha(cutoff, te))

        self.t_prev = t
        return hat_x

    def reset(self) -> None:
        """Reset filter history and timestamps."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.t_prev = None


class OneEuroFilter2D:
    """
    2D Velocity-Gated One-Euro Filter for 2D Screen Gaze Coordinates.
    Filters (X, Y) coordinates independently with shared or individual parameters.
    """

    def __init__(
        self,
        min_cutoff: float = 0.04,
        beta: float = 0.6,
        d_cutoff: float = 1.0,
        deadband: float = 0.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.deadband = deadband
        self.fx = OneEuroFilter1D(min_cutoff, beta, d_cutoff, deadband)
        self.fy = OneEuroFilter1D(min_cutoff, beta, d_cutoff, deadband)

    def filter(
        self,
        pt: Tuple[float, float],
        timestamp: Optional[float] = None
    ) -> Tuple[float, float]:
        """Filter a 2D coordinate tuple (x, y) with optional timestamp."""
        t = timestamp if timestamp is not None else time.time()
        rx = self.fx.filter(pt[0], t)
        ry = self.fy.filter(pt[1], t)
        return (float(rx), float(ry))

    def reset(self) -> None:
        """Reset both X and Y filter channels."""
        self.fx.reset()
        self.fy.reset()

    def update_params(
        self,
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: Optional[float] = None,
        deadband: Optional[float] = None,
    ) -> None:
        """Update filter hyper-parameters at runtime without resetting history.

        Useful for live tuning during interactive calibration sessions.
        Only provided (non-None) arguments are updated.

        Args:
            min_cutoff: Minimum cutoff frequency in Hz. Lower values smooth more
                during fixation but increase lag.
            beta: Velocity coupling coefficient. Higher values reduce saccade lag.
            d_cutoff: Derivative low-pass cutoff frequency in Hz.
            deadband: Absolute pixel deadband for micro-jitter suppression.
        """
        for attr, val in [("min_cutoff", min_cutoff), ("beta", beta),
                          ("d_cutoff", d_cutoff), ("deadband", deadband)]:
            if val is not None:
                setattr(self, attr, float(val))
                setattr(self.fx, attr, float(val))
                setattr(self.fy, attr, float(val))
