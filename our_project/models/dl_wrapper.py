import os
import sys
import importlib.util
import torch
import numpy as np
import cv2

class DLWrapper:
    """
    Wrapper for loading deep-learning models from the external rPPG-Toolbox
    without modifying or directly copying their code.
    """
    def __init__(self, external_repo_path):
        self.repo_path = external_repo_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_module_from_file(self, module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def load_physnet(self, weights_path):
        """
        Dynamically loads PhysNet and its weights.
        """
        physnet_path = os.path.join(self.repo_path, "neural_methods", "model", "PhysNet.py")
        if not os.path.exists(physnet_path):
            raise FileNotFoundError(f"Cannot find PhysNet.py at {physnet_path}")
            
        physnet_module = self._load_module_from_file("PhysNet", physnet_path)
        
        # Instantiate model
        model = physnet_module.PhysNet_padding_Encoder_Decoder_MAX(frames=128)
        
        # Load weights
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Cannot find weights at {weights_path}")
            
        # Try to load state dict (handling potential DataParallel / 'module.' prefixes)
        state_dict = torch.load(weights_path, map_location=self.device)
        
        # Check if the state dict has 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            # Create a new state dict without the 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        model = model.to(self.device)
        model.eval()
        return model

    def preprocess_physnet_input(self, frames):
        """
        Preprocesses a list/array of frames (H, W, C) into the expected [1, 3, T, H, W] tensor.
        PhysNet expects difference normalized data or standard normalized, typically resized to 128x128.
        """
        # Ensure we have at least some frames
        if len(frames) == 0:
            return None
            
        resized_frames = []
        for frame in frames:
            # PhysNet expects 128x128
            resized = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
            resized = resized.astype(np.float32) / 255.0
            resized_frames.append(resized)
            
        # Convert to numpy array [T, H, W, C]
        video_tensor = np.array(resized_frames)
        
        # Transpose to [C, T, H, W]
        video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))
        
        # Convert to torch tensor and add batch dimension [1, C, T, H, W]
        tensor = torch.from_numpy(video_tensor).unsqueeze(0).float()
        return tensor

    @torch.no_grad()
    def predict_physnet(self, model, frames):
        """
        Runs inference using PhysNet.
        """
        tensor = self.preprocess_physnet_input(frames)
        if tensor is None:
            return np.array([])
            
        tensor = tensor.to(self.device)
        
        # PhysNet forward returns: rPPG, x_visual, x_visual3232, x_visual1616
        rppg, _, _, _ = model(tensor)
        
        # Convert to numpy and flatten
        rppg = rppg.cpu().numpy().flatten()
        return rppg

if __name__ == "__main__":
    import pathlib
    base_dir = pathlib.Path(__file__).parent.parent.parent
    external_repo = base_dir / "external" / "rPPG-Toolbox-main"
    weights = external_repo / "final_model_release" / "UBFC-rPPG_PhysNet_DiffNormalized.pth"
    
    wrapper = DLWrapper(str(external_repo))
    
    print(f"Loading PhysNet from {weights}...")
    try:
        model = wrapper.load_physnet(str(weights))
        print("PhysNet model loaded successfully!")
        
        # Generate dummy frames [T, H, W, C]
        print("Generating dummy frames...")
        dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(128)]
        
        print("Running inference...")
        rppg_signal = wrapper.predict_physnet(model, dummy_frames)
        print(f"Inference complete! Output signal shape: {rppg_signal.shape}")
    except Exception as e:
        print(f"Error loading/running PhysNet: {e}")
