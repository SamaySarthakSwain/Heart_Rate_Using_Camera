import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from collections import deque

class BandpassFilter:
    def __init__(self, lowcut=0.5, highcut=4.0, fps=30.0, order=3):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fps = fps
        self.order = order
        nyquist = self.fps / 2.0
        low = max(0.01, min(self.lowcut / nyquist, 0.99))
        high = max(low + 0.01, min(self.highcut / nyquist, 0.99))
        self.b, self.a = signal.butter(self.order, [low, high], btype="band")

    def filter(self, data):
        if len(data) < 3 * self.order + 1:
            return data
        try:
            return signal.filtfilt(self.b, self.a, data, padlen=3 * self.order)
        except ValueError:
            return data

def detrend_signal(signal_data, lambda_val=10):
    n = len(signal_data)
    if n < 3:
        return signal_data
    window = min(n // 2, int(n * 0.1)) or 1
    if window < 3:
        return signal_data - np.mean(signal_data)
    smoothed = np.convolve(signal_data, np.ones(window) / window, mode="same")
    return signal_data - smoothed

class GreenMethod:
    @staticmethod
    def extract_pulse(r, g, b):
        if len(g) < 3: return np.array([])
        g_mean = np.mean(g)
        if g_mean < 1e-8: return np.array([])
        pulse = g / g_mean - 1.0
        return pulse - np.mean(pulse)

class CHROMMethod:
    @staticmethod
    def extract_pulse(r, g, b):
        if len(r) < 3: return np.array([])
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        if r_mean < 1e-8 or g_mean < 1e-8 or b_mean < 1e-8: return np.array([])
        
        r_n = r / r_mean
        g_n = g / g_mean
        b_n = b / b_mean
        Xs = 3 * r_n - 2 * g_n
        Ys = 1.5 * r_n + g_n - 1.5 * b_n
        std_Ys = np.std(Ys)
        if std_Ys < 1e-8:
            return Xs - np.mean(Xs)
        alpha = np.std(Xs) / std_Ys
        pulse = Xs - alpha * Ys
        return pulse - np.mean(pulse)

class POSMethod:
    @staticmethod
    def extract_pulse(r, g, b):
        if len(r) < 3: return np.array([])
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
        if r_mean < 1e-8 or g_mean < 1e-8 or b_mean < 1e-8: return np.array([])
        
        r_n = r / r_mean
        g_n = g / g_mean
        b_n = b / b_mean
        Xs = g_n - b_n
        Ys = -2 * r_n + g_n + b_n
        std_Ys = np.std(Ys)
        if std_Ys < 1e-8:
            return Xs - np.mean(Xs)
        alpha = np.std(Xs) / std_Ys
        pulse = Xs + alpha * Ys
        return pulse - np.mean(pulse)

# Unified Interface implementation requested in Step 6 / 9

def extract_rppg(r_trace, g_trace, b_trace, method="chrom", fps=30.0, timestamps=None):
    """
    Extracts the rPPG signal using the specified method.
    Replaces extract_rppg(video, method) logic by decoupling ROI extraction.
    """
    methods = {
        "green": GreenMethod,
        "chrom": CHROMMethod,
        "pos": POSMethod
    }
    
    if method.lower() not in methods:
        raise ValueError(f"Method {method} not supported.")
        
    extractor = methods[method.lower()]
    raw_pulse = extractor.extract_pulse(np.array(r_trace), np.array(g_trace), np.array(b_trace))
    
    if len(raw_pulse) == 0:
        return np.array([])
        
    # Correct network jitter using Cubic Spline Interpolation
    if timestamps is not None and len(timestamps) == len(raw_pulse):
        t = np.array(timestamps)
        # Ensure timestamps are strictly increasing
        _, unique_indices = np.unique(t, return_index=True)
        t = t[np.sort(unique_indices)]
        raw_pulse = raw_pulse[np.sort(unique_indices)]
        
        if len(t) > 3:
            t = t - t[0]
            duration = t[-1]
            num_ideal_frames = int(duration * fps)
            if num_ideal_frames > 3:
                t_ideal = np.linspace(0, duration, num_ideal_frames)
                cs = CubicSpline(t, raw_pulse)
                raw_pulse = cs(t_ideal)
        
    # Standard preprocessing pipeline
    detrended = detrend_signal(raw_pulse)
    bp_filter = BandpassFilter(fps=fps)
    return bp_filter.filter(detrended)

def estimate_hr(signal_data, fps=30.0):
    """
    Estimates heart rate in BPM using FFT with Harmonic Rejection.
    """
    if len(signal_data) < 30 or np.isnan(signal_data).any() or np.isinf(signal_data).any():
        return 0.0
        
    n = len(signal_data)
    # Apply Hamming window to reduce spectral leakage
    windowed = signal_data * np.hamming(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    
    # 0.7 to 3.0 Hz (42 to 180 BPM)
    hr_range = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(hr_range):
        return 0.0
        
    hr_power = power[hr_range]
    hr_freqs = freqs[hr_range]
    
    if len(hr_power) == 0:
        return 0.0
        
    peak_idx = np.argmax(hr_power)
    peak_freq = hr_freqs[peak_idx]
    max_p = hr_power[peak_idx]
    
    if max_p < 1e-8:
        return 0.0
    
    # Harmonic Rejection: If peak is > 90 BPM (1.5 Hz), check for fundamental at half freq
    if peak_freq > 1.5:
        fund_target = peak_freq / 2.0
        if fund_target >= 0.7:
            # Search in a +/- 0.15 Hz window around half the peak
            sub_mask = (hr_freqs >= fund_target - 0.15) & (hr_freqs <= fund_target + 0.15)
            if np.any(sub_mask):
                sub_freqs = hr_freqs[sub_mask]
                sub_powers = hr_power[sub_mask]
                sub_max_idx = np.argmax(sub_powers)
                sub_max_p = sub_powers[sub_max_idx]
                
                # If the sub-peak is at least 15% of the main peak's power, assume it's the fundamental
                if sub_max_p >= 0.15 * max_p:
                    peak_freq = sub_freqs[sub_max_idx]
    
    return peak_freq * 60.0

def calculate_sqi(signal_data, fps=30.0):
    """
    Signal Quality Index based on SNR in frequency domain.
    Returns value between 0.0 and 1.0
    """
    if len(signal_data) < 30 or np.isnan(signal_data).any() or np.isinf(signal_data).any():
        return 0.0
        
    n = len(signal_data)
    windowed = signal_data * np.hamming(n)
    freqs = np.fft.rfftfreq(n, 1.0 / fps)
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    
    hr_range = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(hr_range):
        return 0.0
        
    hr_power = power[hr_range]
    hr_freqs = freqs[hr_range]
    
    if len(hr_power) == 0:
        return 0.0
        
    peak_idx = np.argmax(hr_power)
    peak_freq = hr_freqs[peak_idx]
    
    signal_mask = np.abs(hr_freqs - peak_freq) < 0.2
    signal_power = np.sum(hr_power[signal_mask])
    noise_power = np.sum(hr_power[~signal_mask]) + 1e-8
    
    if noise_power <= 0 or signal_power <= 0:
        return 0.0
        
    snr = 10 * np.log10(signal_power / noise_power)
    snr = max(0.0, min(snr, 30.0))
    
    confidence = 1 / (1 + np.exp(-(snr - 5) / 3))
    return float(confidence)

def evaluate(prediction, ground_truth):
    """
    Wraps the evaluation metrics from phase 8.
    """
    from our_project.evaluation.metrics import evaluate_all
    return evaluate_all(ground_truth, prediction)

if __name__ == "__main__":
    # Test unified interface
    t = np.arange(0, 10, 1/30.0) # 10 seconds at 30 fps
    hr_hz = 1.2 # 72 BPM
    pulse = 0.02 * np.sin(2 * np.pi * hr_hz * t)
    
    r = 150 + 0.5 * pulse + 0.01 * np.random.randn(len(t))
    g = 120 + pulse + 0.01 * np.random.randn(len(t))
    b = 100 + 0.3 * pulse + 0.01 * np.random.randn(len(t))
    
    methods = ["green", "chrom", "pos"]
    for m in methods:
        rppg_sig = extract_rppg(r, g, b, method=m)
        hr = estimate_hr(rppg_sig)
        sqi = calculate_sqi(rppg_sig)
        print(f"Method: {m.upper()} | HR: {hr:.1f} BPM | SQI: {sqi:.2f}")
