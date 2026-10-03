"""FFT utility functions for rPPG signal analysis."""

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_psd(signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the power spectral density of a signal.

    Args:
        signal: 1D time-domain signal array.
        fps: Sampling rate in Hz.

    Returns:
        Tuple of (frequencies, power_spectrum) arrays.
    """
    n = len(signal)
    # Apply Hamming window to reduce spectral leakage
    windowed = signal * np.hamming(n)
    fft_vals = rfft(windowed)
    freqs = rfftfreq(n, d=1.0 / fps)
    power = np.abs(fft_vals) ** 2
    return freqs, power


def find_dominant_freq(
    freqs: np.ndarray,
    psd: np.ndarray,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
) -> tuple[float, float]:
    """Find the dominant frequency within a specified band.

    Args:
        freqs: Frequency array from compute_psd.
        psd: Power spectrum array from compute_psd.
        low_hz: Lower bound of frequency band (Hz).
        high_hz: Upper bound of frequency band (Hz).

    Returns:
        Tuple of (peak_frequency_hz, peak_power).
    """
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0, 0.0
    band_freqs = freqs[mask]
    band_power = psd[mask]
    
    max_idx = np.argmax(band_power)
    max_f = band_freqs[max_idx]
    max_p = band_power[max_idx]
    
    # Harmonic Rejection: If peak is > 90 BPM (1.5 Hz), check for fundamental at half freq
    if max_f > 1.5:
        fund_target = max_f / 2.0
        if fund_target >= low_hz:
            # Search in a +/- 0.15 Hz window around half the peak
            sub_mask = (band_freqs >= fund_target - 0.15) & (band_freqs <= fund_target + 0.15)
            if np.any(sub_mask):
                sub_freqs = band_freqs[sub_mask]
                sub_powers = band_power[sub_mask]
                sub_max_idx = np.argmax(sub_powers)
                sub_max_p = sub_powers[sub_max_idx]
                
                # If the sub-peak is at least 30% of the main peak's power, assume it's the fundamental
                if sub_max_p >= 0.3 * max_p:
                    return float(sub_freqs[sub_max_idx]), float(sub_max_p)
                    
    return float(max_f), float(max_p)


def compute_snr(
    freqs: np.ndarray,
    psd: np.ndarray,
    peak_freq: float,
    peak_bandwidth: float = 0.15,
    low_hz: float = 0.7,
    high_hz: float = 4.0,
) -> float:
    """Compute signal-to-noise ratio at the peak frequency.

    Args:
        freqs: Frequency array.
        psd: Power spectrum array.
        peak_freq: Dominant peak frequency (Hz).
        peak_bandwidth: Half-width of the signal band around peak (Hz).
        low_hz: Lower bound of physiological range.
        high_hz: Upper bound of physiological range.

    Returns:
        SNR value (linear scale). Returns 0 if no valid signal.
    """
    if peak_freq <= 0:
        return 0.0

    # Signal power: within peak_bandwidth of peak
    signal_mask = (freqs >= peak_freq - peak_bandwidth) & (
        freqs <= peak_freq + peak_bandwidth
    )
    # Noise power: rest of physiological band
    noise_mask = (freqs >= low_hz) & (freqs <= high_hz) & ~signal_mask

    signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 0.0
    noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1.0

    if noise_power <= 0:
        return 0.0

    return float(signal_power / noise_power)
