"""Smoothing utilities for HR estimation stabilization."""

import numpy as np
from collections import deque


class ExponentialMovingAverage:
    """Exponential moving average filter for HR smoothing."""

    def __init__(self, alpha: float = 0.3):
        """Initialize EMA filter.

        Args:
            alpha: Smoothing factor (0-1). Higher = more responsive, lower = smoother.
        """
        self.alpha = alpha
        self.value: float | None = None

    def update(self, new_value: float) -> float:
        """Update the EMA with a new value.

        Args:
            new_value: New measurement to incorporate.

        Returns:
            Smoothed value.
        """
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset the EMA state."""
        self.value = None


class HRSmoother:
    """Heart rate smoother with outlier rejection and EMA."""

    def __init__(
        self,
        max_hr_change: float = 15.0,
        ema_alpha: float = 0.3,
        history_size: int = 30,
    ):
        """Initialize HR smoother.

        Args:
            max_hr_change: Maximum allowed HR change between updates (BPM).
            ema_alpha: EMA smoothing factor.
            history_size: Number of recent HR values to keep.
        """
        self.max_hr_change = max_hr_change
        self.ema = ExponentialMovingAverage(alpha=ema_alpha)
        self.history: deque[float] = deque(maxlen=history_size)
        self.last_valid_hr: float | None = None

    def update(self, hr: float) -> float | None:
        """Update with a new HR estimate, applying outlier rejection and smoothing.

        Args:
            hr: Raw HR estimate in BPM.

        Returns:
            Smoothed HR value, or None if rejected as outlier.
        """
        # Basic physiological range check
        if hr < 50 or hr > 180:
            return self.last_valid_hr

        # Outlier rejection: reject if HR change too large
        if self.last_valid_hr is not None:
            if abs(hr - self.last_valid_hr) > self.max_hr_change:
                return self.last_valid_hr

        # Apply EMA smoothing
        smoothed = self.ema.update(hr)
        self.history.append(smoothed)
        self.last_valid_hr = smoothed
        return smoothed

    def get_stability(self) -> float:
        """Compute stability of recent HR estimates.

        Returns:
            Stability score (0-1). Higher = more stable.
        """
        if len(self.history) < 3:
            return 0.0
        std = np.std(list(self.history))
        # Normalize: std of 0 → stability 1.0, std of 20+ → stability ~0
        stability = max(0.0, 1.0 - std / 20.0)
        return float(stability)

    def get_average_hr(self) -> float | None:
        """Get average HR from history."""
        if not self.history:
            return None
        return float(np.mean(list(self.history)))

    def get_median_hr(self) -> float | None:
        """Get median HR from history."""
        if not self.history:
            return None
        return float(np.median(list(self.history)))

    def is_stable(self, threshold: float = 5.0) -> bool:
        """Check if recent HR readings are stable.

        Inspired by habom2310: only consider HR valid when the max deviation
        from the mean is within threshold bpm over recent history.

        Args:
            threshold: Maximum allowed deviation from mean (BPM).

        Returns:
            True if HR is stable (low variance).
        """
        if len(self.history) < 10:
            return False
        recent = list(self.history)[-min(20, len(self.history)):]
        arr = np.array(recent)
        return bool(max(arr - np.mean(arr)) < threshold)

    def get_stable_hr(self, threshold: float = 5.0) -> float | None:
        """Get HR only when readings are stable.

        Returns the mean of recent HR values if stable, None otherwise.
        This prevents displaying wildly fluctuating (unreliable) values.

        Args:
            threshold: Stability threshold in BPM.

        Returns:
            Stable HR value in BPM, or None if not yet stable.
        """
        if not self.is_stable(threshold):
            return None
        recent = list(self.history)[-min(20, len(self.history)):]
        return float(np.mean(recent))

    def reset(self):
        """Reset all state."""
        self.ema.reset()
        self.history.clear()
        self.last_valid_hr = None
