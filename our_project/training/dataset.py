import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import numpy as np

class RPPGDataset(Dataset):
    """
    PyTorch Dataset for loading standardized HDF5 rPPG data. (Phase 21)
    """
    def __init__(self, data_dir, clip_length=128):
        self.data_dir = data_dir
        self.clip_length = clip_length
        
        # Discover all h5 files
        self.h5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        
        # Create an index mapping for clips
        self.clips = []
        for file_path in self.h5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    num_frames = f['video'].shape[0]
                    # Create sliding window clips without overlap for training
                    for start_idx in range(0, num_frames - self.clip_length + 1, self.clip_length):
                        self.clips.append((file_path, start_idx))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        file_path, start_idx = self.clips[idx]
        
        with h5py.File(file_path, 'r') as f:
            # Shape: (T, H, W, C) -> PyTorch wants (C, T, H, W)
            video_clip = f['video'][start_idx:start_idx + self.clip_length]
            gt_clip = f['ground_truth'][start_idx:start_idx + self.clip_length]
            
        video_tensor = torch.from_numpy(video_clip).permute(3, 0, 1, 2).float()
        gt_tensor = torch.from_numpy(gt_clip).float()
        
        return video_tensor, gt_tensor

def get_dataloader(data_dir, batch_size=4, clip_length=128, shuffle=True):
    if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
        print(f"Warning: Data directory {data_dir} is empty or missing.")
        return None
        
    dataset = RPPGDataset(data_dir, clip_length=clip_length)
    if len(dataset) == 0:
        return None
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

if __name__ == "__main__":
    print("RPPGDataset and DataLoader configured.")
