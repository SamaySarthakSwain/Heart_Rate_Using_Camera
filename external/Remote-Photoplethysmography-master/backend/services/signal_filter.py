"""Signal filtering for rPPG signals — detrending, bandpass, and smoothing."""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


# Physiological heart rate range
HR_LOW_HZ = 0.83   # 50 BPM
HR_HIGH_HZ = 3.0   # 180 BPM


def detrend_signal(signal: np.ndarray, order: int = 5) -> np.ndarray:
    """Remove slow trends from signal using polynomial detrending.

    Args:
        signal: 1D signal array.
        order: Polynomial order for detrending.

    Returns:
        Detrended signal.
    """
    n = len(signal)
    if n < order + 1:
        return signal - np.mean(signal)

    t = np.arange(n)
    coeffs = np.polyfit(t, signal, order)
    trend = np.polyval(coeffs, t)
    return signal - trend


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to isolate heart rate frequencies.

    Args:
        signal: 1D signal array.
        fps: Sampling rate in Hz.
        low_hz: Lower cutoff frequency (Hz).
        high_hz: Upper cutoff frequency (Hz).
        order: Filter order.

    Returns:
        Bandpass-filtered signal.
    """
    nyquist = fps / 2.0

    if nyquist <= low_hz:
        return signal - np.mean(signal)

    # Clamp high frequency to just below Nyquist
    high_hz = min(high_hz, nyquist * 0.95)

    if low_hz >= high_hz:
        return signal - np.mean(signal)

    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")

    # Use filtfilt for zero-phase filtering
    try:
        filtered = filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))
    except ValueError:
        return signal - np.mean(signal)

    return filtered


def process_rppg_signal(
    signal: np.ndarray,
    fps: float,
    detrend_order: int = 5,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
) -> np.ndarray:
    """Full signal processing pipeline: detrend → bandpass → window.

    Args:
        signal: Raw rPPG signal.
        fps: Sampling rate in Hz.
        detrend_order: Polynomial order for detrending.
        low_hz: Lower cutoff frequency.
        high_hz: Upper cutoff frequency.

    Returns:
        Processed signal ready for FFT analysis.
    """
    # Step 1: Remove polynomial trend
    detrended = detrend_signal(signal, order=detrend_order)

    # Step 2: Bandpass filter to HR range
    filtered = bandpass_filter(detrended, fps, low_hz, high_hz)

    return filtered
