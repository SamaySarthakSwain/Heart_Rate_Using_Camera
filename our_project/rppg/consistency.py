class PhysiologicalConsistencyChecker:
    """
    Validates the physiological plausibility of the Heart Rate estimates. (Phase 17)
    """
    def __init__(self, min_hr=42.0, max_hr=180.0, max_change_per_sec=15.0):
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.max_change_per_sec = max_change_per_sec
        self.last_valid_hr = None
        self.last_valid_time = None
        
    def validate(self, current_hr, current_time, sqi=1.0):
        """
        Validates the HR based on absolute limits and temporal continuity.
        Returns: (is_valid, filtered_hr, reason)
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
                    # Smoothing instead of strict rejection if SQI is high
                    if sqi > 0.8:
                        smoothed_hr = self.last_valid_hr + np.sign(current_hr - self.last_valid_hr) * max_allowed_change
                        self.last_valid_hr = smoothed_hr
                        self.last_valid_time = current_time
                        return True, smoothed_hr, "Smoothed due to rapid change"
                    else:
                        return False, self.last_valid_hr, "Physiologically implausible rate of change"
                        
        self.last_valid_hr = current_hr
        self.last_valid_time = current_time
        return True, current_hr, "Valid"

if __name__ == "__main__":
    import numpy as np
    checker = PhysiologicalConsistencyChecker()
    print(checker.validate(72.0, 1.0, 0.9)) # Valid
    print(checker.validate(190.0, 2.0, 0.9)) # Out of bounds
    print(checker.validate(140.0, 2.0, 0.9)) # Implausible jump from 72 to 140 in 1 second
