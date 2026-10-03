"""Motion detection utilities for face tracking stability."""

import numpy as np
from collections import deque


class MotionDetector:
    """Detects excessive face motion that could compromise rPPG signal quality."""

    def __init__(
        self,
        motion_threshold: float = 15.0,
        history_size: int = 30,
    ):
        """Initialize motion detector.

        Args:
            motion_threshold: Pixel displacement threshold for motion flag.
            history_size: Number of frames to keep for motion analysis.
        """
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        self.positions: deque[tuple[float, float]] = deque(maxlen=history_size)
        self.displacements: deque[float] = deque(maxlen=history_size)

    def update(self, center_x: float, center_y: float) -> dict:
        """Update with new face center position.

        Args:
            center_x: Face bounding box center X coordinate.
            center_y: Face bounding box center Y coordinate.

        Returns:
            Dict with 'motion_detected' (bool), 'displacement' (float),
            'motion_score' (float 0-1).
        """
        displacement = 0.0
        if self.positions:
            prev_x, prev_y = self.positions[-1]
            displacement = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

        self.positions.append((center_x, center_y))
        self.displacements.append(displacement)

        motion_detected = displacement > self.motion_threshold
        motion_score = self._compute_motion_score()

        return {
            "motion_detected": motion_detected,
            "displacement": float(displacement),
            "motion_score": motion_score,
        }

    def _compute_motion_score(self) -> float:
        """Compute motion score (0-1, higher = less motion = better).

        Returns:
            Motion quality score.
        """
        if len(self.displacements) < 2:
            return 1.0

        avg_displacement = np.mean(list(self.displacements))
        # Normalize: 0 displacement → 1.0, threshold+ → 0.0
        score = max(0.0, 1.0 - avg_displacement / self.motion_threshold)
        return float(score)

    def get_position_variance(self) -> float:
        """Compute variance of face positions.

        Returns:
            Position variance (sum of X and Y variances).
        """
        if len(self.positions) < 2:
            return 0.0
        positions = np.array(list(self.positions))
        return float(np.var(positions[:, 0]) + np.var(positions[:, 1]))

    def reset(self):
        """Reset motion detector state."""
        self.positions.clear()
        self.displacements.clear()
