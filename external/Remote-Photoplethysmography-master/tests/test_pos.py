"""Unit tests for POS rPPG algorithm."""

import numpy as np
import pytest
from backend.services.pos_rppg import pos_rppg, pos_rppg_windowed


def generate_synthetic_pulse(
    freq_hz: float = 1.2,
    fps: float = 30.0,
    duration_s: float = 10.0,
    amplitude: float = 0.005,
    noise_level: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic RGB signals with a simulated pulse."""
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    pulse = amplitude * np.sin(2 * np.pi * freq_hz * t)

    r = 150.0 + 0.3 * pulse * 150 + noise_level * np.random.randn(n)
    g = 120.0 + pulse * 120 + noise_level * np.random.randn(n)
    b = 100.0 + 0.2 * pulse * 100 + noise_level * np.random.randn(n)

    return r, g, b


class TestPOS:
    """Test suite for POS algorithm."""

    def test_output_shape(self):
        """Output should have same length as input."""
        r, g, b = generate_synthetic_pulse()
        signal = pos_rppg(r, g, b)
        assert len(signal) == len(r)

    def test_zero_mean(self):
        """Output should be approximately zero-mean."""
        r, g, b = generate_synthetic_pulse()
        signal = pos_rppg(r, g, b)
        assert abs(np.mean(signal)) < 1e-10

    def test_dominant_frequency(self):
        """Dominant frequency should be near the input pulse frequency."""
        freq_hz = 1.2  # 72 BPM
        fps = 30.0
        r, g, b = generate_synthetic_pulse(freq_hz=freq_hz, fps=fps, duration_s=10.0)
        signal = pos_rppg(r, g, b)

        from scipy.fft import rfft, rfftfreq
        n = len(signal)
        freqs = rfftfreq(n, d=1.0 / fps)
        power = np.abs(rfft(signal)) ** 2

        mask = (freqs >= 0.7) & (freqs <= 4.0)
        peak_idx = np.argmax(power[mask])
        peak_freq = freqs[mask][peak_idx]

        assert abs(peak_freq - freq_hz) < 0.2, f"Peak at {peak_freq} Hz, expected {freq_hz} Hz"

    def test_short_signal(self):
        """Should handle very short signals gracefully."""
        signal = pos_rppg(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert len(signal) == 2

    def test_constant_signal(self):
        """Should return zeros for constant input."""
        n = 100
        r = np.ones(n) * 150
        g = np.ones(n) * 120
        b = np.ones(n) * 100
        signal = pos_rppg(r, g, b)
        assert np.allclose(signal, 0, atol=1e-10)

    def test_windowed_shape(self):
        """Windowed POS should maintain output shape."""
        r, g, b = generate_synthetic_pulse(duration_s=10.0)
        signal = pos_rppg_windowed(r, g, b, window_size=45)
        assert len(signal) == len(r)
