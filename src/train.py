import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from video_autoencoder import VideoAutoencoder
from stego_networks import (
    HiderNetwork,
    RevealerNetwork,
    format_secret_for_hiding,
    extract_secret_from_prediction,
)
from image_generator import ImageGenerator


class DummyVideoDataset(Dataset):
    def __init__(self, num_samples=100, frames=16, height=64, width=64):
        self.num_samples = num_samples
        self.frames = frames
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.rand(
            (3, self.frames, self.height, self.width), dtype=torch.float32
        )


import cv2
import glob
from torchvision import transforms
from PIL import Image

class RealVideoDataset(Dataset):
    """Loads actual .mp4 or .avi videos from a directory for autoencoder training."""
    def __init__(self, directory, frames=16, height=64, width=64):
        self.video_paths = glob.glob(os.path.join(directory, "**", "*.avi"), recursive=True) + \
                           glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True)
        self.frames = frames
        self.height = height
        self.width = width
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []
        while len(frames) < self.frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame) # shape (3, H, W)
            frames.append(frame_tensor)
        cap.release()
        
        # If video is too short, pad it with the last frame
        while len(frames) < self.frames and len(frames) > 0:
            frames.append(frames[-1])
            
        # If video couldn't be loaded at all, return zeros (edge case fallback)
        if len(frames) == 0:
            return torch.zeros((3, self.frames, self.height, self.width), dtype=torch.float32)
            
        # Stack into (C, F, H, W)
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor


def train_video_autoencoder(model, dataloader, epochs=5, device="cpu"):
    print("--- Training Video Autoencoder ---")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/video_autoencoder.pth")


def train_stego_networks(
    hider, revealer, image_generator, epochs=5, device="cpu", secret_dim=4096
):
    print("\\n--- Training Steganography Networks ---")
    hider.to(device)
    revealer.to(device)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    optimizer = optim.Adam(
        list(hider.parameters()) + list(revealer.parameters()), lr=1e-3
    )
    batch_size = 4
    iterations = 20
    for epoch in range(epochs):
        hider.train()
        revealer.train()
        img_loss = 0.0
        bit_loss = 0.0
        for _ in range(iterations):
            covers = torch.stack(
                [image_generator.generate_cover((256, 256)) for _ in range(batch_size)]
            ).to(device)
            secret_bits = (
                torch.randint(0, 2, (batch_size, secret_dim)).float().to(device)
            )
            spatial_secret = format_secret_for_hiding(
                secret_bits, (batch_size, 1, 256, 256)
            )
            stego = hider(covers, spatial_secret)
            secret_pred = extract_secret_from_prediction(revealer(stego), secret_dim)
            l_img = criterion_mse(stego, covers)
            l_bit = criterion_bce(secret_pred, secret_bits)
            loss = (10.0 * l_img) + l_bit
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            img_loss += l_img.item()
            bit_loss += l_bit.item()
        print(
            f"Epoch {epoch+1}/{epochs}, Img Loss: {img_loss/iterations:.4f}, Bit Loss: {bit_loss/iterations:.4f}"
        )
    torch.save(hider.state_dict(), "../models/hider.pth")
    torch.save(revealer.state_dict(), "../models/revealer.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # -------------------------------------------------------------
    # 1. SETUP: CHOOSE DUMMY DATA OR REAL DATA
    # -------------------------------------------------------------
    # Change this to True when you have downloaded the datasets!
    USE_REAL_DATA = False 
    
    # Define paths to your downloaded folders
    VIDEO_DATASET_PATH = "../data/UCF101/" # Folder containing .mp4 or .avi
    
    if USE_REAL_DATA:
        print("Loading REAL Video Dataset... (This might take a moment)")
        # Load your real video dataset and create a dataloader
        video_dataset = RealVideoDataset(directory=VIDEO_DATASET_PATH, frames=16)
        video_loader = DataLoader(video_dataset, batch_size=4, shuffle=True)
    else:
        print("Loading DUMMY Video Dataset... (For quick testing)")
        video_dataset = DummyVideoDataset(num_samples=20)
        video_loader = DataLoader(video_dataset, batch_size=4)

    # -------------------------------------------------------------
    # 2. RUN AUTOENCODER TRAINING
    # -------------------------------------------------------------
    ae = VideoAutoencoder(3, 256)
    train_video_autoencoder(ae, video_loader, epochs=2, device=device)

    # -------------------------------------------------------------
    # 3. RUN STEGANOGRAPHY NETWORKS TRAINING
    # -------------------------------------------------------------
    # For Stego networks, Stable Diffusion `ImageGenerator(use_dummy=not USE_REAL_DATA)` 
    # will handle real cover images if USE_REAL_DATA=True!
    hider = HiderNetwork(3, 1, 32)
    revealer = RevealerNetwork(3, 1, 32)
    img_gen = ImageGenerator(device, use_dummy=not USE_REAL_DATA)
    
    train_stego_networks(
        hider,
        revealer,
        img_gen,
        epochs=2,
        device=device,
        secret_dim=8416, 
    )
    print("Training Complete!")
