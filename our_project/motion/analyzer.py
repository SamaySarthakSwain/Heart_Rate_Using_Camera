import numpy as np

class MotionAnalyzer:
    """
    Analyzes facial landmarks to detect motion.
    Computes a motion score between 0.0 (no motion) and 1.0 (high motion).
    """
    def __init__(self, history_size=30):
        self.history_size = history_size
        self.landmark_history = []
        
    def calculate_motion_score(self, current_landmarks):
        """
        Takes current normalized landmarks and returns a motion score.
        """
        if not current_landmarks:
            return 1.0 # High motion/failure if no landmarks detected
            
        # We track a subset of stable landmarks (e.g., nose tip, corners of eyes)
        # For simplicity, we just use the first point (nose tip is usually idx 1)
        stable_points = np.array([current_landmarks[1], current_landmarks[33], current_landmarks[263]])
        
        if len(self.landmark_history) == 0:
            self.landmark_history.append(stable_points)
            return 0.0
            
        prev_points = self.landmark_history[-1]
        
        # Calculate Euclidean distance of these stable points between frames
        displacements = np.linalg.norm(stable_points - prev_points, axis=1)
        avg_displacement = np.mean(displacements)
        
        self.landmark_history.append(stable_points)
        if len(self.landmark_history) > self.history_size:
            self.landmark_history.pop(0)
            
        # Normalize displacement to a [0, 1] score
        # Assuming max reasonable inter-frame displacement is ~0.05 in normalized coords
        motion_score = min(1.0, avg_displacement / 0.05)
        
        return float(motion_score)

if __name__ == "__main__":
    analyzer = MotionAnalyzer()
    dummy_lms1 = [[0.5, 0.5, 0] for _ in range(478)]
    dummy_lms2 = [[0.51, 0.51, 0] for _ in range(478)]
    
    score1 = analyzer.calculate_motion_score(dummy_lms1)
    score2 = analyzer.calculate_motion_score(dummy_lms2)
    print(f"Motion score 1: {score1}")
    print(f"Motion score 2: {score2}")
