import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    """
    Robust Face and Landmark Detection using MediaPipe FaceMesh (478 landmarks).
    Provides stability and precise ROI bounding boxes even under motion.
    """
    def __init__(self, max_faces=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        try:
            import mediapipe as mp
            # Try legacy solutions first
            if hasattr(mp, 'solutions'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.use_legacy = True
            else:
                self.use_legacy = False
                print("Warning: MediaPipe solutions not found. Face detector will run in mock mode.")
        except ImportError:
            self.use_legacy = False
            print("Warning: MediaPipe not found. Face detector will run in mock mode.")
            
    def detect_landmarks(self, frame_bgr):
        """
        Detects facial landmarks.
        Returns the raw MediaPipe results and a list of normalized landmark coordinates.
        """
        if not self.use_legacy:
            # Mock mode
            return None, [[(0.5, 0.5, 0.0) for _ in range(478)]]
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                landmarks.append(points)
                
        return results, landmarks

if __name__ == "__main__":
    detector = FaceDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results, lms = detector.detect_landmarks(dummy_frame)
    print(f"Detected {len(lms)} faces in dummy frame.")
