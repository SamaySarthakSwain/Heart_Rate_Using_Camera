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
        
        # Stability gating & EMA smoothing from Remote-Photoplethysmography-master
        self.ema = ExponentialMovingAverage(alpha=0.3)
        self.history = deque(maxlen=30)

    def validate(self, current_hr: float, current_time: float, sqi: float = 1.0):
        """
        Validates the HR based on absolute limits and temporal continuity.

        Returns:
            (is_valid, filtered_hr, reason)
        """
        if current_hr < self.min_hr or current_hr > self.max_hr:
            return False, self.last_valid_hr, "Out of absolute bounds"

        if sqi < 0.3:
            return False, self.last_valid_hr, "SQI too low"

        # Outlier rejection based on rate of change
        if self.last_valid_hr is not None and self.last_valid_time is not None:
            time_delta = current_time - self.last_valid_time
            if time_delta > 0:
                change = abs(current_hr - self.last_valid_hr)
                max_allowed_change = self.max_change_per_sec * time_delta

                if change > max_allowed_change:
                    if sqi > 0.8:
                        direction = np.sign(current_hr - self.last_valid_hr)
                        current_hr = self.last_valid_hr + direction * max_allowed_change
                    else:
                        return False, self.get_stable_hr(), "Physiologically implausible rate of change"

        # Apply EMA Smoothing
        smoothed_hr = self.ema.update(current_hr)
        self.history.append(smoothed_hr)
        
        self.last_valid_hr = smoothed_hr
        self.last_valid_time = current_time
        
        # Stability Gating
        stable_hr = self.get_stable_hr()
        if stable_hr is None:
            return False, smoothed_hr, "Stabilizing..."
            
        return True, stable_hr, "Valid"

    def get_stable_hr(self, threshold: float = 5.0):
        """Only returns HR if the recent variance is low."""
        if len(self.history) < 10:
            return None
        recent = list(self.history)[-min(20, len(self.history)):]
        arr = np.array(recent)
        if max(arr - np.mean(arr)) < threshold:
            return float(np.mean(recent))
        return None


if __name__ == "__main__":
    checker = PhysiologicalConsistencyChecker()
    print(checker.validate(72.0, 1.0, 0.9))   # Expected: (True, 72.0, 'Valid')
    print(checker.validate(190.0, 2.0, 0.9))  # Expected: (False, 72.0, 'Out of absolute bounds')
    print(checker.validate(140.0, 2.0, 0.9))  # Expected: rejection or smoothing
