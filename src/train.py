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
    ae = VideoAutoencoder(3, 256)
    train_video_autoencoder(
        ae, DataLoader(DummyVideoDataset(20), batch_size=4), 2, device
    )
    train_stego_networks(
        HiderNetwork(3, 1, 32),
        RevealerNetwork(3, 1, 32),
        ImageGenerator(device, True),
        2,
        device,
        8416,
    )
    print("Training Complete!")
