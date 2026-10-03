import numpy as np

class IlluminationAnalyzer:
    """
    Analyzes ROI RGB traces for illumination quality and stability.
    """
    def __init__(self, target_brightness=128):
        self.target_brightness = target_brightness

    def evaluate_roi_illumination(self, rgb_mean):
        """
        Evaluates a single frame's RGB mean for a specific ROI.
        Returns a score from 0.0 (poor) to 1.0 (excellent).
        """
        r, g, b = rgb_mean
        
        # Calculate perceived brightness (luminance)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Penalize if too dark (<40) or too bright (>240)
        if luminance < 40 or luminance > 240:
            brightness_score = 0.1
        else:
            # Score peaks at target_brightness
            brightness_score = 1.0 - (abs(luminance - self.target_brightness) / 128.0)
            brightness_score = max(0.0, brightness_score)
            
        return float(brightness_score)

    def evaluate_temporal_illumination(self, luminance_history):
        """
        Evaluates lighting stability over time (variance).
        """
        if len(luminance_history) < 10:
            return 1.0
            
        variance = np.var(luminance_history)
        
        # High variance means flickering or changing light
        # Max acceptable variance is roughly 400 (std=20)
        stability_score = 1.0 - min(1.0, variance / 400.0)
        return float(stability_score)

if __name__ == "__main__":
    analyzer = IlluminationAnalyzer()
    print("Normal lighting score:", analyzer.evaluate_roi_illumination((120, 130, 110)))
    print("Dark lighting score:", analyzer.evaluate_roi_illumination((20, 20, 20)))
    print("Overexposed score:", analyzer.evaluate_roi_illumination((250, 250, 250)))
