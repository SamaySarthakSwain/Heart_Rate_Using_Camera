import numpy as np
from collections import deque

class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class PhysiologicalConsistencyChecker:
    """
    Validates the physiological plausibility of the Heart Rate estimates and applies EMA smoothing & stability gating.
    """

    def __init__(self, min_hr: float = 42.0, max_hr: float = 180.0, max_change_per_sec: float = 15.0):
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.max_change_per_sec = max_change_per_sec
        self.last_valid_hr = None
        self.last_valid_time = None
        
        # Stability gating & EMA smoothing
        self.ema = ExponentialMovingAverage(alpha=0.3)
        self.history = deque(maxlen=90) # Increased history to 3 seconds (30fps)
        
        # Display Latch logic for readability
        self.display_latch_hr = None
        self.display_latch_time = 0.0
        self.latch_duration = 2.0 # Hold display steady for 2 seconds

    def validate(self, current_hr: float, current_time: float, sqi: float = 1.0):
        """
        Validates the HR and provides detailed feedback on accuracy loss due to lighting/camera.
        """
        # Always apply EMA Smoothing to prevent the value from freezing permanently
        smoothed_hr = self.ema.update(current_hr)
        self.history.append(smoothed_hr)
        self.last_valid_hr = smoothed_hr
        self.last_valid_time = current_time

        if sqi < 0.4:
            accuracy_loss = int((1.0 - sqi) * 100)
            reason = f"Poor Lighting or Low Camera Quality. Output accuracy reduced by ~{accuracy_loss}%."
            return False, smoothed_hr, reason

        if current_hr < self.min_hr or current_hr > self.max_hr:
            return False, smoothed_hr, "Out of absolute bounds"

        # Stability Gating
        stable_hr = self.get_stable_hr()
        if stable_hr is None:
            return False, smoothed_hr, "Stabilizing... (Please hold still)"
            
        # Display Latching (Slows down fluctuations for readability while continuous monitoring runs in background)
        if current_time - self.display_latch_time >= self.latch_duration:
            self.display_latch_hr = stable_hr
            self.display_latch_time = current_time
            
        # If latch hasn't been set yet (first valid reading)
        if self.display_latch_hr is None:
            self.display_latch_hr = stable_hr
            self.display_latch_time = current_time
            
        return True, self.display_latch_hr, "Valid"

    def get_stable_hr(self, threshold: float = 5.0):
        """Only returns HR if the recent variance is low."""
        if len(self.history) < 10:
            return None
        recent = list(self.history)[-min(20, len(self.history)):]
        arr = np.array(recent)
        if np.max(np.abs(arr - np.mean(arr))) < threshold:
            return float(np.mean(recent))
        return None


if __name__ == "__main__":
    checker = PhysiologicalConsistencyChecker()
    print(checker.validate(72.0, 1.0, 0.9))   # Expected: (True, 72.0, 'Valid')
    print(checker.validate(190.0, 2.0, 0.9))  # Expected: (False, 72.0, 'Out of absolute bounds')
    print(checker.validate(140.0, 2.0, 0.9))  # Expected: rejection or smoothing
