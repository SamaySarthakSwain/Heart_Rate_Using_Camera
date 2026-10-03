import numpy as np


class PhysiologicalConsistencyChecker:
    """
    Validates the physiological plausibility of the Heart Rate estimates. (Phase 17)
    """

    def __init__(self, min_hr: float = 42.0, max_hr: float = 180.0, max_change_per_sec: float = 15.0):
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.max_change_per_sec = max_change_per_sec
        self.last_valid_hr: float | None = None
        self.last_valid_time: float | None = None

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

        if self.last_valid_hr is not None and self.last_valid_time is not None:
            time_delta = current_time - self.last_valid_time
            if time_delta > 0:
                change = abs(current_hr - self.last_valid_hr)
                max_allowed_change = self.max_change_per_sec * time_delta

                if change > max_allowed_change:
                    # Smoothing instead of strict rejection when signal quality is high
                    if sqi > 0.8:
                        direction = np.sign(current_hr - self.last_valid_hr)
                        smoothed_hr = self.last_valid_hr + direction * max_allowed_change
                        self.last_valid_hr = smoothed_hr
                        self.last_valid_time = current_time
                        return True, smoothed_hr, "Smoothed due to rapid change"
                    else:
                        return False, self.last_valid_hr, "Physiologically implausible rate of change"

        self.last_valid_hr = current_hr
        self.last_valid_time = current_time
        return True, current_hr, "Valid"


if __name__ == "__main__":
    checker = PhysiologicalConsistencyChecker()
    print(checker.validate(72.0, 1.0, 0.9))   # Expected: (True, 72.0, 'Valid')
    print(checker.validate(190.0, 2.0, 0.9))  # Expected: (False, 72.0, 'Out of absolute bounds')
    print(checker.validate(140.0, 2.0, 0.9))  # Expected: rejection or smoothing
