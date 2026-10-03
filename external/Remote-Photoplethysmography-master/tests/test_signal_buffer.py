"""Unit tests for signal buffer."""

import time
import pytest
from backend.services.signal_buffer import SignalBuffer


class TestSignalBuffer:
    """Test suite for SignalBuffer."""

    def test_add_and_size(self):
        """Buffer size should increase with samples."""
        buf = SignalBuffer(max_size=100, min_size=10)
        assert buf.get_size() == 0

        for i in range(20):
            buf.add_sample(100.0, 120.0, 80.0)

        assert buf.get_size() == 20

    def test_circular_overflow(self):
        """Buffer should not exceed max_size."""
        buf = SignalBuffer(max_size=10, min_size=3)

        for i in range(50):
            buf.add_sample(float(i), float(i), float(i))

        assert buf.get_size() == 10

    def test_is_ready(self):
        """is_ready should return True when min_size reached."""
        buf = SignalBuffer(max_size=100, min_size=5)
        assert not buf.is_ready()

        for i in range(5):
            buf.add_sample(100.0, 120.0, 80.0)

        assert buf.is_ready()

    def test_get_window_none_when_not_ready(self):
        """get_window should return None when insufficient samples."""
        buf = SignalBuffer(max_size=100, min_size=10)
        buf.add_sample(100.0, 120.0, 80.0)
        assert buf.get_window() is None

    def test_get_window_data(self):
        """get_window should return properly structured data."""
        buf = SignalBuffer(max_size=100, min_size=5)

        for i in range(10):
            buf.add_sample(100.0 + i, 120.0 + i, 80.0 + i, brightness=100.0)
            time.sleep(0.01)

        window = buf.get_window()
        assert window is not None
        assert len(window["R"]) == 10
        assert len(window["G"]) == 10
        assert len(window["B"]) == 10
        assert window["fps"] > 0
        assert window["n_samples"] == 10

    def test_clear(self):
        """Clear should empty the buffer."""
        buf = SignalBuffer(max_size=100, min_size=5)
        for i in range(10):
            buf.add_sample(100.0, 120.0, 80.0)

        buf.clear()
        assert buf.get_size() == 0
        assert not buf.is_ready()
