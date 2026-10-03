"""Unit tests for HR estimator."""

import numpy as np
import pytest
from backend.services.hr_estimator import HREstimator


def make_sinusoid(freq_hz: float, fps: float = 30.0, duration_s: float = 10.0) -> np.ndarray:
    """Create a clean sinusoidal signal."""
    n = int(fps * duration_s)
    t = np.arange(n) / fps
    return np.sin(2 * np.pi * freq_hz * t)


class TestHREstimator:
    """Test suite for HR estimation."""

    def setup_method(self):
        self.estimator = HREstimator()

    def test_clean_sinusoid_72bpm(self):
        """Should estimate 72 BPM from a 1.2 Hz signal."""
        signal = make_sinusoid(1.2, fps=30.0, duration_s=10.0)
        result = self.estimator.estimate(signal, fps=30.0, apply_filter=False)

        assert result["valid"]
        assert abs(result["hr_bpm"] - 72.0) < 3.0, f"Got {result['hr_bpm']}, expected ~72"

    def test_clean_sinusoid_60bpm(self):
        """Should estimate 60 BPM from a 1.0 Hz signal."""
        signal = make_sinusoid(1.0, fps=30.0, duration_s=10.0)
        result = self.estimator.estimate(signal, fps=30.0, apply_filter=False)

        assert result["valid"]
        assert abs(result["hr_bpm"] - 60.0) < 3.0, f"Got {result['hr_bpm']}, expected ~60"

    def test_clean_sinusoid_90bpm(self):
        """Should estimate 90 BPM from a 1.5 Hz signal."""
        signal = make_sinusoid(1.5, fps=30.0, duration_s=10.0)
        result = self.estimator.estimate(signal, fps=30.0, apply_filter=False)

        assert result["valid"]
        assert abs(result["hr_bpm"] - 90.0) < 3.0, f"Got {result['hr_bpm']}, expected ~90"

    def test_short_signal_invalid(self):
        """Should return invalid for very short signals."""
        signal = np.sin(np.linspace(0, 1, 20))
        result = self.estimator.estimate(signal, fps=30.0)
        assert not result["valid"]

    def test_snr_positive_for_clean_signal(self):
        """Clean signal should have positive SNR."""
        signal = make_sinusoid(1.2, fps=30.0, duration_s=10.0)
        result = self.estimator.estimate(signal, fps=30.0, apply_filter=False)
        assert result["snr"] > 1.0

    def test_fusion(self):
        """Fused estimate should be valid when both inputs are valid."""
        chrom = make_sinusoid(1.2, fps=30.0, duration_s=10.0)
        pos = make_sinusoid(1.2, fps=30.0, duration_s=10.0) * 0.9

        result = self.estimator.estimate_fused(chrom, pos, fps=30.0)
        assert result["valid"]
        assert abs(result["hr_bpm"] - 72.0) < 5.0

    def test_fusion_one_invalid(self):
        """Fusion should use valid signal when one is invalid."""
        chrom = make_sinusoid(1.2, fps=30.0, duration_s=10.0)
        pos = np.random.randn(10)  # Too short → invalid

        result = self.estimator.estimate_fused(chrom, pos, fps=30.0)
        assert result["valid"]
