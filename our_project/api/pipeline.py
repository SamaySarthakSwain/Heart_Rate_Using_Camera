import numpy as np
import cv2
import time
import os
from collections import deque

from our_project.camera.camera_interface import CameraAbstraction
from our_project.face.detector import FaceDetector
from our_project.roi.extractor import ROIExtractor
from our_project.rppg.adaptive_fusion import AdaptiveROIFusion
from our_project.rppg.baselines import extract_rppg, estimate_hr, calculate_sqi
from our_project.models.dl_wrapper import DLWrapper

class RealTimePipeline:
    """
    End-to-End Real-Time rPPG Pipeline orchestrator.
    Combines Camera -> Face Detection -> Multi-ROI Extraction -> 
    Adaptive Fusion -> rPPG Extraction (Classical or Deep Learning) -> HR Estimation
    """
    def __init__(self, method="chrom", buffer_size=300, min_frames=150, fps=30.0):
        self.method = method
        self.fps = fps
        # PhysNet specifically expects T=128 for its temporal convolutions
        self.buffer_size = 128 if method == "physnet" else buffer_size
        self.min_frames = 128 if method == "physnet" else min_frames
        
        # Initialize modules
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor()
        self.fusion = AdaptiveROIFusion()
        
        # Temporal buffers for the classical fused RGB signal
        self.r_buffer = deque(maxlen=self.buffer_size)
        self.g_buffer = deque(maxlen=self.buffer_size)
        self.b_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Temporal buffer for raw face frames (Deep Learning requirement)
        self.raw_face_buffer = deque(maxlen=self.buffer_size)
        
        # Initialize Deep Learning Wrapper if needed
        self.dl_wrapper = None
        self.dl_model = None
        if self.method == "physnet":
            repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'rPPG-Toolbox-main'))
            self.dl_wrapper = DLWrapper(repo_path)
            weights_path = os.path.join(repo_path, "final_model_release", "UBFC-rPPG_PhysNet_DiffNormalized.pth")
            try:
                print("Loading PhysNet model... (This may take a moment)")
                self.dl_model = self.dl_wrapper.load_physnet(weights_path)
                print("PhysNet loaded successfully.")
            except Exception as e:
                print(f"Failed to load PhysNet: {e}")
                self.method = "chrom" # Fallback if loading fails
        
    def _crop_face(self, frame_bgr, landmarks):
        """
        Calculates bounding box from landmarks and crops the face from the frame.
        """
        h, w, _ = frame_bgr.shape
        x_coords = [lm[0] * w for lm in landmarks]
        y_coords = [lm[1] * h for lm in landmarks]
        
        x_min, x_max = max(0, int(min(x_coords))), min(w, int(max(x_coords)))
        y_min, y_max = max(0, int(min(y_coords))), min(h, int(max(y_coords)))
        
        # Expand bounding box slightly (10%)
        w_pad = int((x_max - x_min) * 0.1)
        h_pad = int((y_max - y_min) * 0.1)
        
        x_min = max(0, x_min - w_pad)
        y_min = max(0, y_min - h_pad)
        x_max = min(w, x_max + w_pad)
        y_max = min(h, y_max + h_pad)
        
        face_crop = frame_bgr[y_min:y_max, x_min:x_max]
        return face_crop

    def process_frame(self, frame_bgr, timestamp):
        """
        Process a single frame through the entire pipeline.
        Returns the current HR estimation and SQI if buffer is ready.
        """
        # 1. Face & Landmark Detection
        results, landmarks_list = self.face_detector.detect_landmarks(frame_bgr)
        
        if not landmarks_list:
            return None, 0.0, "No human detected", 0.0, 0.0, 1.0
            
        landmarks = landmarks_list[0]
        
        # 2. Extract multi-ROI RGB means (Classical pathway)
        roi_means = self.roi_extractor.extract_rgb_means(frame_bgr, landmarks)
        
        # 3. Adaptive Fusion (Motion & Illumination)
        fused_rgb, weights = self.fusion.fuse_roi_signals(roi_means, landmarks)
        
        # 4. Buffer Updates
        self.r_buffer.append(fused_rgb[0])
        self.g_buffer.append(fused_rgb[1])
        self.b_buffer.append(fused_rgb[2])
        self.timestamps.append(timestamp)
        
        # For Deep Learning, we also need to buffer the raw face crop
        if self.method == "physnet":
            face_crop = self._crop_face(frame_bgr, landmarks)
            if face_crop.size > 0:
                self.raw_face_buffer.append(face_crop)
        
        # 5. HR Estimation (only if buffer is sufficiently full)
        if len(self.g_buffer) >= self.min_frames:
            
            if self.method in ["green", "chrom", "pos"]:
                # Classical rPPG
                rppg_wave = extract_rppg(self.r_buffer, self.g_buffer, self.b_buffer, 
                                         method=self.method, fps=self.fps, timestamps=list(self.timestamps))
                                         
                if len(rppg_wave) > 0:
                    hr = estimate_hr(rppg_wave, fps=self.fps)
                    sqi = calculate_sqi(rppg_wave, fps=self.fps)
                    
                    motion = getattr(self.fusion, 'last_motion_score', 0.0)
                    illum = getattr(self.fusion, 'last_illum_score', 1.0)
                    return hr, sqi, f"Success ({self.method.upper()})", float(rppg_wave[-1]), motion, illum
                    
            elif self.method == "physnet" and self.dl_model is not None:
                # Deep Learning Inference
                if len(self.raw_face_buffer) == self.buffer_size:
                    # frames is a list of cropped BGR numpy arrays
                    frames_list = list(self.raw_face_buffer)
                    # predict_physnet handles resizing to 128x128 and tensor conversion
                    rppg_wave = self.dl_wrapper.predict_physnet(self.dl_model, frames_list)
                    
                    if len(rppg_wave) > 0:
                        # Post-process the raw output from the neural net (detrend, bandpass, FFT)
                        # We can reuse the classical frequency estimation block for this!
                        # The output from PhysNet is already a temporal signal.
                        # We just need to bandpass filter it and find the peak frequency.
                        
                        # Use the same evaluation metrics
                        hr = estimate_hr(rppg_wave, fps=self.fps)
                        sqi = calculate_sqi(rppg_wave, fps=self.fps)
                        motion = getattr(self.fusion, 'last_motion_score', 0.0)
                        illum = getattr(self.fusion, 'last_illum_score', 1.0)
                        return hr, sqi, "Success (PhysNet)", float(rppg_wave[-1]), motion, illum
                
        return None, 0.0, "Buffering...", 0.0, 0.0, 1.0

if __name__ == "__main__":
    # Smoke test of the entire pipeline WITH PhysNet
    print("Initializing pipeline with PhysNet...")
    pipeline = RealTimePipeline(method="physnet", buffer_size=128)
    
    print("Testing End-to-End Pipeline with synthetic data...")
    for i in range(130):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_frame, (200, 150), (440, 330), (120, 130, 140), -1)
        
        # Simulate successful detection with face in center
        dummy_lms = [(0.5, 0.5, 0.0) for _ in range(478)] 
        
        hr, sqi, status, raw_wave = pipeline.process_frame(dummy_frame, time.time())
        if hr is not None:
            print(f"Frame {i} | HR: {hr:.1f} BPM | SQI: {sqi:.2f} | Status: {status}")
