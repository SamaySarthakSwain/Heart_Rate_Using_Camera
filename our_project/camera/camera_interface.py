import cv2
import time
import numpy as np

class CameraAbstraction:
    """
    Device-agnostic camera abstraction layer.
    Handles capability detection, frame acquisition, and timestamp normalization.
    """
    def __init__(self, source=0, target_fps=30.0, target_resolution=(640, 480)):
        self.source = source
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.cap = cv2.VideoCapture(self.source)
        
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps == 0 or np.isnan(self.actual_fps):
            self.actual_fps = 30.0  # Fallback
            
        # Detect capabilities
        self.capabilities = {
            "fps": self.actual_fps,
            "resolution": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "backend": self.cap.getBackendName(),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }
        
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = time.time()
        
    def read_frame(self):
        """
        Reads a frame and normalizes resolution and timestamp.
        Returns: (success, frame, timestamp, dropped_frame_flag)
        """
        if not self.cap.isOpened():
            return False, None, 0.0, False
            
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        # Detect dropped frames if running significantly slower than expected
        dropped_frame = time_diff > (self.frame_interval * 1.5)
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None, current_time, dropped_frame
            
        # Normalize resolution
        if (frame.shape[1], frame.shape[0]) != self.target_resolution:
            frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_AREA)
            
        self.last_frame_time = current_time
        return True, frame, current_time, dropped_frame
        
    def release(self):
        self.cap.release()

if __name__ == "__main__":
    cam = CameraAbstraction(source=0)
    print("Camera Capabilities detected:", cam.capabilities)
    ret, frame, ts, dropped = cam.read_frame()
    if ret:
        print(f"Successfully read frame of shape {frame.shape} at timestamp {ts}")
