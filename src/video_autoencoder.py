import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4096, latent_dim)

    def forward(self, x):
        return self.fc(self.flatten(self.encoder(x)))


class VideoDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 4096)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 3, 2, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, 2, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, out_channels, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.fc(x).view(-1, 256, 1, 4, 4))


class VideoAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = VideoEncoder(in_channels, latent_dim)
        self.decoder = VideoDecoder(in_channels, latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent), latent
