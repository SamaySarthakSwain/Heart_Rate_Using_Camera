"""Thread-safe signal buffer for temporal RGB storage."""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Sample:
    """A single RGB sample with timestamp."""
    timestamp: float
    r: float
    g: float
    b: float
    brightness: float = 0.0


class SignalBuffer:
    """Thread-safe circular buffer for RGB signal samples.

    Includes sudden-change rejection (to prevent motion artifact spikes)
    and timestamp interpolation (to fix spectral leakage from uneven frame rates).
    """

    def __init__(
        self,
        max_size: int = 300,
        min_size: int = 90,
        outlier_threshold: float = 10.0,
    ):
        """Initialize signal buffer.

        Args:
            max_size: Maximum number of samples (e.g., 10 sec × 30 fps = 300).
            min_size: Minimum samples needed before processing (e.g., 3 sec × 30 fps).
            outlier_threshold: Max allowed deviation from buffer mean for each
                channel. Values beyond this are clamped to the last valid sample.
        """
        self.max_size = max_size
        self.min_size = min_size
        self.outlier_threshold = outlier_threshold
        self._buffer: deque[Sample] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._last_valid_rgb: tuple[float, float, float] | None = None

    def add_sample(self, r: float, g: float, b: float, brightness: float = 0.0):
        """Add an RGB sample to the buffer with sudden-change rejection.

        If any channel deviates from the recent buffer mean by more than
        outlier_threshold, that channel's value is replaced with the last
        valid value. This prevents motion-caused spikes from poisoning FFT.

        Args:
            r: Red channel mean value.
            g: Green channel mean value.
            b: Blue channel mean value.
            brightness: Mean brightness value.
        """
        with self._lock:
            # Sudden-change rejection: compare against buffer mean
            if len(self._buffer) >= 10 and self._last_valid_rgb is not None:
                recent = list(self._buffer)[-min(30, len(self._buffer)):]
                mean_r = np.mean([s.r for s in recent])
                mean_g = np.mean([s.g for s in recent])
                mean_b = np.mean([s.b for s in recent])

                if abs(r - mean_r) > self.outlier_threshold:
                    r = self._last_valid_rgb[0]
                if abs(g - mean_g) > self.outlier_threshold:
                    g = self._last_valid_rgb[1]
                if abs(b - mean_b) > self.outlier_threshold:
                    b = self._last_valid_rgb[2]

            self._last_valid_rgb = (r, g, b)

            sample = Sample(
                timestamp=time.time(),
                r=r,
                g=g,
                b=b,
                brightness=brightness,
            )
            self._buffer.append(sample)

    def get_window(self) -> dict | None:
        """Get the current signal window as numpy arrays.

        Performs interpolation to evenly-spaced timestamps to fix spectral
        leakage caused by inconsistent webcam frame timing.

        Returns:
            Dict with 'R', 'G', 'B' arrays (interpolated), 'timestamps',
            'brightness', 'fps' (estimated), and 'n_samples'.
            Returns None if buffer doesn't have minimum samples.
        """
        with self._lock:
            if len(self._buffer) < self.min_size:
                return None

            samples = list(self._buffer)

        timestamps = np.array([s.timestamp for s in samples])
        r_vals = np.array([s.r for s in samples])
        g_vals = np.array([s.g for s in samples])
        b_vals = np.array([s.b for s in samples])
        brightness = np.array([s.brightness for s in samples])

        n = len(timestamps)

        # Estimate FPS from timestamps
        if n > 1:
            dt = np.diff(timestamps)
            fps = 1.0 / np.mean(dt) if np.mean(dt) > 0 else 30.0
        else:
            fps = 30.0

        # Interpolate to evenly-spaced timestamps to reduce spectral leakage
        if n > 2:
            even_times = np.linspace(timestamps[0], timestamps[-1], n)
            r_vals = np.interp(even_times, timestamps, r_vals)
            g_vals = np.interp(even_times, timestamps, g_vals)
            b_vals = np.interp(even_times, timestamps, b_vals)
            brightness = np.interp(even_times, timestamps, brightness)
            timestamps = even_times

        return {
            "R": r_vals,
            "G": g_vals,
            "B": b_vals,
            "timestamps": timestamps,
            "brightness": brightness,
            "fps": float(fps),
            "n_samples": n,
        }

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for processing."""
        with self._lock:
            return len(self._buffer) >= self.min_size

    def get_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)

    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
            self._last_valid_rgb = None
