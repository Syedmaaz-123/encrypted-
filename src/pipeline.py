import torch
import os
import cv2
import numpy as np

from utils import extract_frames, compile_video
from video_autoencoder import VideoAutoencoder
from encryption import LatentEncryptor, ciphertext_to_bits, bits_to_ciphertext
from image_generator import ImageGenerator
from stego_networks import (
    HiderNetwork,
    RevealerNetwork,
    format_secret_for_hiding,
    extract_secret_from_prediction,
)


def save_image(tensor, path):
    import torchvision

    torchvision.utils.save_image(tensor, path)


class SteganoPipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.autoencoder = VideoAutoencoder(in_channels=3, latent_dim=256).to(device)
        self.encryptor = LatentEncryptor()
        self.generator = ImageGenerator(device=device, use_dummy=True)
        self.hider = HiderNetwork(cover_channels=3, secret_channels=1).to(device)
        self.revealer = RevealerNetwork(stego_channels=3, secret_channels=1).to(device)

    def hide_video(self, video_path, output_image_path):
        print(f"1. Extracting frames from {video_path}...")
        frames = (
            extract_frames(video_path, max_frames=16, resize_dim=(64, 64))
            .unsqueeze(0)
            .to(self.device)
        )
        print("2. Compressing video into latent vector...")
        with torch.no_grad():
            _, latent = self.autoencoder(frames)
        print("3. Encrypting latent vector using AES...")
        ciphertext, metadata = self.encryptor.encrypt(latent[0])
        print("4. Generating Cover Image (256x256)...")
        cover_image = (
            self.generator.generate_cover(size=(256, 256)).unsqueeze(0).to(self.device)
        )
        print("5. Packing encrypted data into spatial tensor...")
        spatial_secret = format_secret_for_hiding(
            ciphertext_to_bits(ciphertext, 65536).unsqueeze(0).to(self.device),
            (1, 1, 256, 256),
        )
        print("6. Embedding secret into Cover Image...")
        with torch.no_grad():
            stego_image = self.hider(cover_image, spatial_secret)
        print(f"7. Saving Stego Image to {output_image_path}...")
        save_image(stego_image[0], output_image_path)
        return metadata, len(ciphertext)

    def extract_video(self, stego_image_path, metadata, cipher_len, output_video_path):
        from torchvision.io import read_image

        print(f"1. Loading Stego Image from {stego_image_path}...")
        stego_image = (
            (read_image(stego_image_path).float() / 255.0).unsqueeze(0).to(self.device)
        )
        print("2. Extracting spatial data...")
        with torch.no_grad():
            secret_pred_spatial = self.revealer(stego_image)
        print("3. Reconstructing bit stream...")
        bit_tensor = extract_secret_from_prediction(secret_pred_spatial, cipher_len * 8)
        print("4. Repacking bits to ciphertext...")
        recovered_ciphertext = bits_to_ciphertext(bit_tensor[0], cipher_len)
        print("5. Decrypting latent vector using AES...")
        try:
            recovered_latent = self.encryptor.decrypt(
                recovered_ciphertext, metadata, self.device
            )
            print("Decryption successful!")
        except Exception:
            print(
                "Decryption failed! Networks untrained (Invalid AES Tag error expected during testing)."
            )
            print(
                "Using a random fallback latent vector to demonstrate the rest of the pipeline..."
            )
            recovered_latent = torch.rand(metadata["shape"]).to(self.device)

        print("6. Reconstructing video frames...")
        with torch.no_grad():
            reconstructed_frames = self.autoencoder.decoder(
                recovered_latent.unsqueeze(0)
            )
        print(f"7. Saving Reconstructed Video to {output_video_path}...")
        compile_video(reconstructed_frames[0], output_video_path, fps=15)
        print("--- Decode Complete ---")


if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../models", exist_ok=True)
    dummy_video_path = "../data/dummy_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dummy_video_path, fourcc, 15, (64, 64))
    for _ in range(16):
        out.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    out.release()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = SteganoPipeline(device=device)

    stego_img_path = "../data/stego_output.png"
    recon_video_path = "../data/reconstructed_video.mp4"

    metadata, cipher_length = pipeline.hide_video(dummy_video_path, stego_img_path)
    pipeline.extract_video(stego_img_path, metadata, cipher_length, recon_video_path)
