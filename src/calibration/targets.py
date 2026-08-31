"""Target generation for multi-point calibration grids and validation patterns.

Implements standard grids (9, 13, 16 points) with Boustrophedon (serpentine) ordering
to reduce saccadic eye fatigue during calibration, as well as holdout validation patterns.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

from src.config import GazeConfig


class TargetGenerator:
    """Generates screen calibration and validation target sequences."""

    @staticmethod
    def generate_points(
        config: GazeConfig,
        grid_type: Optional[str] = None,
        boustrophedon: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Generates a sequence of screen target coordinates (X, Y) in pixels.

        Args:
            config: GazeConfig containing screen dimensions and margins.
            grid_type: Grid specification ('9_points', '13_points', '16_points').
                       Defaults to config.calibration_grid_type. Falls back to '9_points'.
            boustrophedon: If True, uses serpentine row alternation (Left->Right, Right->Left)
                           to minimize saccade jump distance and fatigue.

        Returns:
            List of (x, y) integer pixel tuples on screen.
        """
        gtype = grid_type or config.calibration_grid_type
        w = config.screen_width
        h = config.screen_height
        mx = config.calibration_margin_x
        my = config.calibration_margin_y

        if gtype == "16_points":
            xs = np.linspace(mx * w, (1.0 - mx) * w, 4).tolist()
            ys = np.linspace(my * h, (1.0 - my) * h, 4).tolist()
            pts: List[Tuple[int, int]] = []
            for row_idx, y in enumerate(ys):
                row_xs = list(reversed(xs)) if (boustrophedon and row_idx % 2 == 1) else xs
                for x in row_xs:
                    pts.append((int(round(x)), int(round(y))))
            return pts

        elif gtype == "13_points":
            xs = [mx * w, 0.5 * w, (1.0 - mx) * w]
            ys = [my * h, 0.5 * h, (1.0 - my) * h]
            pts = []
            for row_idx, y in enumerate(ys):
                row_xs = list(reversed(xs)) if (boustrophedon and row_idx % 2 == 1) else xs
                for x in row_xs:
                    pts.append((int(round(x)), int(round(y))))

            # 4 inner quadrant points (serpentine order)
            inner_xs = [0.35 * w, 0.65 * w]
            inner_ys = [0.35 * h, 0.65 * h]
            if boustrophedon:
                pts.append((int(round(inner_xs[0])), int(round(inner_ys[0]))))
                pts.append((int(round(inner_xs[1])), int(round(inner_ys[0]))))
                pts.append((int(round(inner_xs[1])), int(round(inner_ys[1]))))
                pts.append((int(round(inner_xs[0])), int(round(inner_ys[1]))))
            else:
                for y in inner_ys:
                    for x in inner_xs:
                        pts.append((int(round(x)), int(round(y))))
            return pts

        else:
            # Default to 9 points (3x3 grid)
            xs = [mx * w, 0.5 * w, (1.0 - mx) * w]
            ys = [my * h, 0.5 * h, (1.0 - my) * h]
            pts = []
            for row_idx, y in enumerate(ys):
                row_xs = list(reversed(xs)) if (boustrophedon and row_idx % 2 == 1) else xs
                for x in row_xs:
                    pts.append((int(round(x)), int(round(y))))
            return pts

    @staticmethod
    def generate_validation_points(
        config: GazeConfig,
        mode: str = "4_points"
    ) -> List[Tuple[int, int]]:
        """
        Generates holdout targets for post-calibration accuracy validation.

        Args:
            config: GazeConfig with screen resolution.
            mode: '4_points' (quadrant centers) or '5_points' (quadrant centers + screen center).

        Returns:
            List of (x, y) target coordinates.
        """
        w = config.screen_width
        h = config.screen_height

        # Quadrant centers
        q_xs = [0.25 * w, 0.75 * w]
        q_ys = [0.25 * h, 0.75 * h]

        pts = [
            (int(round(q_xs[0])), int(round(q_ys[0]))),
            (int(round(q_xs[1])), int(round(q_ys[0]))),
            (int(round(q_xs[1])), int(round(q_ys[1]))),
            (int(round(q_xs[0])), int(round(q_ys[1]))),
        ]

        if mode == "5_points":
            pts.append((int(round(0.50 * w)), int(round(0.50 * h))))

        return pts

    @staticmethod
    def to_normalized_coordinates(
        points: List[Tuple[int, int]],
        screen_w: int,
        screen_h: int
    ) -> np.ndarray:
        """Converts pixel target points to normalized [0.0, 1.0]^2 array."""
        arr = np.array(points, dtype=np.float64)
        arr[:, 0] /= max(1.0, float(screen_w))
        arr[:, 1] /= max(1.0, float(screen_h))
        return arr
