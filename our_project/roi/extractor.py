import cv2
import numpy as np

class ROIExtractor:
    """
    Extracts specific Region of Interest (ROI) from facial landmarks.
    Supports Forehead, Left Cheek, Right Cheek extraction.
    """
    
    # MediaPipe FaceMesh indices for specific regions
    REGIONS = {
        'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        # Refining to just upper forehead for better rPPG signal
        'forehead_refined': [10, 338, 297, 332, 284, 251, 389, 356, 71, 68, 104, 69, 108],
        'left_cheek': [118, 119, 100, 126, 209, 49, 50, 205, 207, 214, 212, 216, 206],
        'right_cheek': [347, 348, 329, 355, 429, 279, 280, 425, 427, 434, 432, 436, 426]
    }

    def __init__(self, regions=["forehead_refined", "left_cheek", "right_cheek"]):
        self.regions_to_extract = regions

    def get_roi_masks(self, frame, landmarks):
        """
        Creates binary masks for each requested ROI.
        """
        h, w, _ = frame.shape
        masks = {}
        
        for region_name in self.regions_to_extract:
            indices = self.REGIONS.get(region_name, [])
            if not indices:
                continue
                
            pts = np.array([
                (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h))
                for idx in indices
            ], dtype=np.int32)
            
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 1)
            masks[region_name] = mask
            
        return masks

    def extract_rgb_means(self, frame, landmarks):
        """
        Extracts the spatial mean of R, G, B channels for each ROI.
        Returns a dictionary { region_name: (R, G, B) }
        """
        masks = self.get_roi_masks(frame, landmarks)
        rgb_means = {}
        
        for region, mask in masks.items():
            # BGR to RGB extraction
            b = np.sum(frame[:,:,0] * mask) / (np.sum(mask) + 1e-6)
            g = np.sum(frame[:,:,1] * mask) / (np.sum(mask) + 1e-6)
            r = np.sum(frame[:,:,2] * mask) / (np.sum(mask) + 1e-6)
            rgb_means[region] = (r, g, b)
            
        return rgb_means

if __name__ == "__main__":
    extractor = ROIExtractor()
    # Dummy test
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    dummy_lms = [(0.5, 0.5, 0.0) for _ in range(478)] # Center of screen
    means = extractor.extract_rgb_means(dummy_frame, dummy_lms)
    print("Dummy extracted means:", means)
