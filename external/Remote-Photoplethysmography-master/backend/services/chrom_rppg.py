"""CHROM rPPG algorithm - De Haan & Jeanne (2013).

Chrominance-based remote photoplethysmography signal extraction.
Robust to illumination variation and moderate motion.
"""

import numpy as np


def chrom_rppg(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extract rPPG signal using the CHROM algorithm.

    Args:
        r: Red channel time series (normalized or raw).
        g: Green channel time series.
        b: Blue channel time series.

    Returns:
        1D array containing the CHROM rPPG signal.
    """
    n = len(r)
    if n < 3:
        return np.zeros(n)

    # Step 1: Normalize by temporal mean
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    if r_mean == 0 or g_mean == 0 or b_mean == 0:
        return np.zeros(n)

    rn = r / r_mean
    gn = g / g_mean
    bn = b / b_mean

    # Step 2: Chrominance projection
    x_s = 3.0 * rn - 2.0 * gn
    y_s = 1.5 * rn + gn - 1.5 * bn

    # Step 3: Adaptive combination
    std_x = np.std(x_s)
    std_y = np.std(y_s)

    if std_y == 0:
        return x_s - np.mean(x_s)

    alpha = std_x / std_y
    signal = x_s - alpha * y_s

    # Zero-mean the output
    signal = signal - np.mean(signal)

    return signal


def chrom_rppg_windowed(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    window_size: int = 45,
) -> np.ndarray:
    """Apply CHROM with overlapping windows for better temporal adaptation.

    Uses overlapping windows to handle slow lighting changes more gracefully.

    Args:
        r: Red channel time series.
        g: Green channel time series.
        b: Blue channel time series.
        window_size: Size of each processing window (in samples).

    Returns:
        1D array containing the windowed CHROM rPPG signal.
    """
    n = len(r)
    if n < window_size:
        return chrom_rppg(r, g, b)

    signal = np.zeros(n)
    step = window_size // 2  # 50% overlap

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window_signal = chrom_rppg(r[start:end], g[start:end], b[start:end])

        # Apply Hanning window for smooth overlap-add
        hann = np.hanning(window_size)
        signal[start:end] += window_signal * hann

    # Normalize by overlap count
    overlap_count = np.zeros(n)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        overlap_count[start:end] += np.hanning(window_size)

    overlap_count[overlap_count == 0] = 1.0
    signal /= overlap_count

    return signal
