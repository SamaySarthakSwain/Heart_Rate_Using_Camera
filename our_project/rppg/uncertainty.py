import numpy as np

class UncertaintyEstimator:
    """
    Evaluates the confidence/uncertainty of the final Heart Rate estimate. (Phase 29)
    """
    def __init__(self, variance_threshold=5.0):
        self.variance_threshold = variance_threshold

    def calculate_uncertainty(self, method_outputs, fused_sqi=None):
        """
        Estimates uncertainty based on the agreement between different algorithms.
        If CHROM, POS, and PhysNet all output ~72 BPM, uncertainty is low.
        If they output 60, 90, and 120, uncertainty is extremely high.
        
        Args:
            method_outputs: dict of method results e.g. {"chrom": {"hr": 72.1, "sqi": 0.8}, ...}
            fused_sqi: Optional overarching SQI
            
        Returns:
            uncertainty_score (0.0 to 1.0, where 1.0 is maximum uncertainty/failure)
            confidence_percentage (0 to 100)
        """
        hrs = []
        weights = []
        
        for m, data in method_outputs.items():
            # Only consider methods that had a somewhat valid SQI
            if data["sqi"] > 0.3:
                hrs.append(data["hr"])
                weights.append(data["sqi"])
                
        if len(hrs) < 2:
            # If only 1 or 0 methods survived the SQI check, uncertainty is extremely high
            if len(hrs) == 1 and weights[0] > 0.8:
                return 0.5, 50.0 # Moderate uncertainty
            return 1.0, 0.0 # Total failure
            
        # Calculate weighted standard deviation of the HR estimates
        mean_hr = np.average(hrs, weights=weights)
        variance = np.average((hrs - mean_hr)**2, weights=weights)
        std_dev = np.sqrt(variance)
        
        # Normalize uncertainty to [0, 1]
        # A standard deviation of > `variance_threshold` BPM indicates total disagreement
        uncertainty = min(1.0, std_dev / self.variance_threshold)
        
        # Factor in the actual SQI scores (if all methods agree but their SQI is garbage, we are still uncertain)
        avg_sqi = np.mean(weights)
        sqi_penalty = 1.0 - avg_sqi
        
        # Combine metric disagreement with raw signal quality
        final_uncertainty = 0.6 * uncertainty + 0.4 * sqi_penalty
        
        confidence_percent = (1.0 - final_uncertainty) * 100.0
        
        return float(final_uncertainty), float(confidence_percent)

if __name__ == "__main__":
    estimator = UncertaintyEstimator()
    
    # Case 1: High agreement, high SQI
    good_methods = {
        "chrom": {"hr": 72.1, "sqi": 0.9},
        "pos": {"hr": 71.8, "sqi": 0.85},
        "physnet": {"hr": 72.0, "sqi": 0.95}
    }
    unc_good, conf_good = estimator.calculate_uncertainty(good_methods)
    print(f"Good Signal -> Uncertainty: {unc_good:.2f}, Confidence: {conf_good:.1f}%")
    
    # Case 2: Complete disagreement, low SQI
    bad_methods = {
        "chrom": {"hr": 60.0, "sqi": 0.4},
        "pos": {"hr": 95.0, "sqi": 0.5},
        "physnet": {"hr": 120.0, "sqi": 0.6}
    }
    unc_bad, conf_bad = estimator.calculate_uncertainty(bad_methods)
    print(f"Bad Signal  -> Uncertainty: {unc_bad:.2f}, Confidence: {conf_bad:.1f}%")
