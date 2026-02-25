import torch
import torch.nn as nn


class HiderNetwork(nn.Module):
    def __init__(self, cover_channels=3, secret_channels=1, hidden_channels=64):
        super().__init__()
        in_channels = cover_channels + secret_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, cover_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cover, secret):
        return self.net(torch.cat([cover, secret], dim=1))


class RevealerNetwork(nn.Module):
    def __init__(self, stego_channels=3, secret_channels=1, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(stego_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, secret_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, stego):
        return self.net(stego)


def format_secret_for_hiding(secret_bits, target_shape):
    B, C, H, W = target_shape
    total_elements = C * H * W
    padded = torch.zeros(B, total_elements, device=secret_bits.device)
    for i in range(B):
        seq = secret_bits[i] if secret_bits.dim() > 1 else secret_bits
        length = min(len(seq), total_elements)
        padded[i, :length] = seq[:length]
    return padded.view(B, C, H, W)


def extract_secret_from_prediction(secret_pred_spatial, original_length):
    return secret_pred_spatial.view(secret_pred_spatial.shape[0], -1)[
        :, :original_length
    ]
