import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from our_project.training.dataset import get_dataloader
from our_project.models.dl_wrapper import DLWrapper

class NegPearsonLoss(nn.Module):
    """
    Negative Pearson Correlation Loss, specifically effective for rPPG signals.
    """
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, labels):
        # preds: (Batch, T)
        # labels: (Batch, T)
        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_x2 = torch.sum(preds ** 2, dim=1)
        sum_y2 = torch.sum(labels ** 2, dim=1)
        sum_xy = torch.sum(preds * labels, dim=1)
        n = preds.size(1)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = torch.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
        
        # Add epsilon to prevent division by zero
        pearson = numerator / (denominator + 1e-7)
        loss = 1.0 - pearson
        return torch.mean(loss)

def train_model(data_dir, epochs=10, batch_size=4, lr=1e-4, save_dir="experiments/models"):
    """
    Standardized training loop for rPPG deep-learning models. (Phase 22)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup Dataloader
    dataloader = get_dataloader(data_dir, batch_size=batch_size, shuffle=True)
    if dataloader is None:
        print("Cannot train. Dataloader is empty.")
        return
        
    # Setup Model (PhysNet via wrapper)
    # We instantiate a fresh model for training/fine-tuning
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'rPPG-Toolbox-main'))
    wrapper = DLWrapper(repo_path)
    
    try:
        # We only load the architecture, not pre-trained weights, for fresh training
        physnet_path = os.path.join(repo_path, "neural_methods", "model", "PhysNet.py")
        physnet_module = wrapper._load_module_from_file("PhysNet", physnet_path)
        model = physnet_module.PhysNet_padding_Encoder_Decoder_MAX(frames=128)
        model = model.to(device)
    except Exception as e:
        print(f"Failed to load model architecture: {e}")
        return
        
    criterion = NegPearsonLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds, _, _, _ = model(videos)
            preds = preds.squeeze(2) # adjust dimension if needed based on model output [B, T, 1, 1]
            
            # Loss
            loss = criterion(preds, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"physnet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        
    print("Training complete.")

if __name__ == "__main__":
    print("Training Module Setup Complete.")
    # Usage: train_model("datasets/UBFC/processed")
