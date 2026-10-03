import numpy as np
from our_project.motion.analyzer import MotionAnalyzer
from our_project.illumination.analyzer import IlluminationAnalyzer

class AdaptiveROIFusion:
    """
    Dynamically fuses multiple ROIs based on instantaneous signal quality,
    motion penalty, and illumination quality. (Phase 13)
    """
    def __init__(self):
        self.motion_analyzer = MotionAnalyzer()
        self.illum_analyzer = IlluminationAnalyzer()
        
    def fuse_roi_signals(self, roi_rgb_means, current_landmarks):
        """
        Takes the RGB means from multiple ROIs (forehead, left_cheek, right_cheek),
        analyzes their quality, and produces a single optimized RGB mean.
        
        Args:
            roi_rgb_means: dict { 'forehead_refined': (r,g,b), 'left_cheek': ... }
            current_landmarks: MediaPipe landmarks for motion analysis
            
        Returns:
            best_rgb: (r, g, b) tuple of the dynamically fused signal
            weights: dict of the weights applied to each ROI
        """
        # 1. Global Motion Score (0 to 1, higher is worse)
        motion_score = self.motion_analyzer.calculate_motion_score(current_landmarks)
        self.last_motion_score = motion_score
        
        weights = {}
        total_weight = 0.0
        avg_illum = 0.0
        
        for region, rgb in roi_rgb_means.items():
            # 2. ROI-specific Illumination Score (0 to 1, higher is better)
            illum_score = self.illum_analyzer.evaluate_roi_illumination(rgb)
            avg_illum += illum_score
            
            # Base weight is heavily dependent on illumination
            base_weight = illum_score
            
            # 3. Dynamic adjustment based on region heuristics
            # Forehead is generally most stable and vascularized, unless covered by hair
            # Cheeks are prone to motion (talking, smiling)
            if region == 'forehead_refined':
                # Less penalized by global motion than cheeks
                weight = base_weight * (1.0 - (motion_score * 0.5))
            else:
                # Cheeks highly penalized by motion (talking artifact)
                # Doubled the penalty (2.0) to aggressively reject talking/smiling
                weight = base_weight * (1.0 - (motion_score * 2.0))
                
            # Ensure weight doesn't drop below a tiny epsilon to prevent div by zero
            weight = max(0.01, weight)
            
            weights[region] = weight
            total_weight += weight
            
        # Normalize weights
        for region in weights:
            weights[region] /= total_weight
            
        self.last_illum_score = avg_illum / max(1, len(roi_rgb_means))
            
        # Fuse RGB signals
        fused_r, fused_g, fused_b = 0.0, 0.0, 0.0
        
        for region, rgb in roi_rgb_means.items():
            w = weights[region]
            fused_r += rgb[0] * w
            fused_g += rgb[1] * w
            fused_b += rgb[2] * w
            
        return (fused_r, fused_g, fused_b), weights

if __name__ == "__main__":
    fusion = AdaptiveROIFusion()
    
    # Dummy data
    means = {
        'forehead_refined': (120, 130, 110), # Good
        'left_cheek': (20, 20, 20),          # Shadow
        'right_cheek': (120, 130, 110)       # Good
    }
    
    dummy_lms = [[0.5, 0.5, 0] for _ in range(478)]
    
    fused, weights = fusion.fuse_roi_signals(means, dummy_lms)
    print("Fused RGB:", fused)
    print("ROI Weights:", weights)
