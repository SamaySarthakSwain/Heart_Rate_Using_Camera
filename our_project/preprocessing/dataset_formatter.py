import os
import cv2
import numpy as np
import h5py

class DatasetFormatter:
    """
    Pre-processing pipeline to format raw dataset videos (UBFC, PURE, MMPD) 
    into standardized HDF5 formats compatible with our PyTorch DataLoaders. (Phase 20)
    """
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def format_video(self, video_path, output_h5_path, ground_truth_hr_array):
        """
        Reads a video, crops/resizes frames to target_size, normalizes, 
        and saves as an HDF5 dataset along with ground truth labels.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            
        cap.release()
        
        frames = np.array(frames) # Shape: (T, H, W, C)
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video.")
            
        if len(frames) != len(ground_truth_hr_array):
            print(f"Warning: Frame count ({len(frames)}) != GT length ({len(ground_truth_hr_array)}). Truncating.")
            min_len = min(len(frames), len(ground_truth_hr_array))
            frames = frames[:min_len]
            ground_truth_hr_array = ground_truth_hr_array[:min_len]

        # Save to HDF5
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('video', data=frames, compression="gzip")
            f.create_dataset('ground_truth', data=np.array(ground_truth_hr_array), compression="gzip")
            
        return output_h5_path

if __name__ == "__main__":
    print("DatasetFormatter initialized.")
    # Usage: formatter = DatasetFormatter()
    # formatter.format_video("raw_data/sub1/vid.avi", "processed/sub1.h5", [72.1, 72.1, ...])
