"""POS rPPG algorithm - Wang et al. (2017).

Plane-Orthogonal-to-Skin method for remote photoplethysmography.
Strong noise suppression, well-suited for real-time pipelines.
"""

import numpy as np


def pos_rppg(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extract rPPG signal using the POS algorithm.

    Args:
        r: Red channel time series.
        g: Green channel time series.
        b: Blue channel time series.

    Returns:
        1D array containing the POS rPPG signal.
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

    # Step 2: Skin-tone projection (plane orthogonal to skin)
    s1 = gn - bn
    s2 = gn + bn - 2.0 * rn

    # Step 3: Weighted combination
    std_s1 = np.std(s1)
    std_s2 = np.std(s2)

    if std_s2 == 0:
        return s1 - np.mean(s1)

    alpha = std_s1 / std_s2
    signal = s1 + alpha * s2

    # Zero-mean the output
    signal = signal - np.mean(signal)

    return signal


def pos_rppg_windowed(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    window_size: int = 45,
) -> np.ndarray:
    """Apply POS with overlapping windows for temporal adaptation.

    Args:
        r: Red channel time series.
        g: Green channel time series.
        b: Blue channel time series.
        window_size: Size of each processing window (in samples).

    Returns:
        1D array containing the windowed POS rPPG signal.
    """
    n = len(r)
    if n < window_size:
        return pos_rppg(r, g, b)

    signal = np.zeros(n)
    step = window_size // 2

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window_signal = pos_rppg(r[start:end], g[start:end], b[start:end])

        hann = np.hanning(window_size)
        signal[start:end] += window_signal * hann

    overlap_count = np.zeros(n)
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        overlap_count[start:end] += np.hanning(window_size)

    overlap_count[overlap_count == 0] = 1.0
    signal /= overlap_count

    return signal
