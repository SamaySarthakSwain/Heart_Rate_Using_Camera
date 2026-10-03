"""Heart rate estimation via FFT frequency analysis."""

import numpy as np
from backend.utils.fft_utils import compute_psd, find_dominant_freq, compute_snr
from backend.services.signal_filter import process_rppg_signal


class HREstimator:
    """Estimates heart rate from rPPG signals using FFT."""

    def __init__(
        self,
        low_hz: float = 0.83,
        high_hz: float = 3.0,
    ):
        """Initialize HR estimator.

        Args:
            low_hz: Minimum HR frequency (Hz). 0.83 Hz = 50 BPM.
            high_hz: Maximum HR frequency (Hz). 3.0 Hz = 180 BPM.
        """
        self.low_hz = low_hz
        self.high_hz = high_hz

    def estimate(
        self,
        signal: np.ndarray,
        fps: float,
        apply_filter: bool = True,
    ) -> dict:
        """Estimate heart rate from an rPPG signal.

        Args:
            signal: 1D rPPG signal array.
            fps: Sampling rate in Hz.
            apply_filter: Whether to apply bandpass filtering first.

        Returns:
            Dict with 'hr_bpm', 'peak_freq', 'snr', 'confidence', 'valid'.
        """
        if len(signal) < 30:
            return self._empty_result()

        # Filter the signal if requested
        if apply_filter:
            signal = process_rppg_signal(signal, fps, low_hz=self.low_hz, high_hz=self.high_hz)

        # Compute power spectral density
        freqs, psd = compute_psd(signal, fps)

        # Find dominant frequency in HR range
        peak_freq, peak_power = find_dominant_freq(freqs, psd, self.low_hz, self.high_hz)

        if peak_freq <= 0:
            return self._empty_result()

        # Compute SNR
        snr = compute_snr(freqs, psd, peak_freq, low_hz=self.low_hz, high_hz=self.high_hz)

        # Convert to BPM
        hr_bpm = peak_freq * 60.0

        # Basic confidence from SNR
        confidence = min(1.0, max(0.0, (snr - 1.0) / 4.0))

        return {
            "hr_bpm": float(hr_bpm),
            "peak_freq": float(peak_freq),
            "snr": float(snr),
            "confidence": float(confidence),
            "valid": True,
        }

    def estimate_fused(
        self,
        chrom_signal: np.ndarray,
        pos_signal: np.ndarray,
        fps: float,
    ) -> dict:
        """Estimate HR using fused CHROM + POS signals.

        Args:
            chrom_signal: CHROM algorithm output.
            pos_signal: POS algorithm output.
            fps: Sampling rate in Hz.

        Returns:
            Dict with fused HR estimate.
        """
        # Get individual estimates
        chrom_result = self.estimate(chrom_signal, fps)
        pos_result = self.estimate(pos_signal, fps)

        if not chrom_result["valid"] and not pos_result["valid"]:
            return self._empty_result()

        if not chrom_result["valid"]:
            return pos_result

        if not pos_result["valid"]:
            return chrom_result

        # SNR-weighted fusion
        total_snr = chrom_result["snr"] + pos_result["snr"]
        if total_snr <= 0:
            # Simple average fallback
            fused_hr = (chrom_result["hr_bpm"] + pos_result["hr_bpm"]) / 2.0
            fused_snr = max(chrom_result["snr"], pos_result["snr"])
        else:
            w_chrom = chrom_result["snr"] / total_snr
            w_pos = pos_result["snr"] / total_snr
            fused_hr = w_chrom * chrom_result["hr_bpm"] + w_pos * pos_result["hr_bpm"]
            fused_snr = max(chrom_result["snr"], pos_result["snr"])

        confidence = min(1.0, max(0.0, (fused_snr - 1.0) / 4.0))

        return {
            "hr_bpm": float(fused_hr),
            "peak_freq": float(fused_hr / 60.0),
            "snr": float(fused_snr),
            "confidence": float(confidence),
            "valid": True,
            "chrom_hr": float(chrom_result["hr_bpm"]),
            "pos_hr": float(pos_result["hr_bpm"]),
        }

    def _empty_result(self) -> dict:
        return {
            "hr_bpm": 0.0,
            "peak_freq": 0.0,
            "snr": 0.0,
            "confidence": 0.0,
            "valid": False,
        }
