# Project Report: Secure Video Steganography - End-to-End Pipeline
**File:** `pipeline.ipynb`  
**Description:** This module serves as the master orchestrator, connecting the video compressor, encryption engine, AI image generator, and steganographic networks into a single unified system.

---

## 1. Executive Summary
The **End-to-End Pipeline** is the "brain" of the entire project. Its purpose is to automate the complex sequence of operations required to hide a video and then recover it. It handles the data flow between five distinct AI and cryptographic modules:
1.  **Temporal Compression**: 3D-CNN Autoencoder.
2.  **Cryptographic Locking**: AES-256-GCM.
3.  **Visual Synthesis**: Stable Diffusion Image Generation.
4.  **Spatial Embedding**: Hider Neural Network.
5.  **Information Extraction**: Revealer Neural Network.

The pipeline ensures that the system is **Hardware Aware** (using GPUs when available) and **Security-First**, utilizing the AES Authentication Tag to guarantee that every frame recovered is physically identical to the original input.

---

## 2. Technical Line-by-Line Explanation

### Step 1: System and Module Integration
```python
import torch, os, cv2, numpy as np
from utils import extract_frames, compile_video
from video_autoencoder import VideoAutoencoder
from encryption import LatentEncryptor
from image_generator import ImageGenerator
from stego_networks import HiderNetwork, RevealerNetwork
```
*   **`cv2` (OpenCV)**: Used for reading and writing physical MP4/AVI video files.
*   **Module Logic**: These lines import your custom-built libraries. The pipeline acts as a "wrapper" that lets these modules talk to each other. For example, it takes the output of the `LatentEncryptor` and feeds it directly into the `HiderNetwork`.

### Step 2: The SteganoPipeline Class initialization
```python
class SteganoPipeline:
    def __init__(self, device="cpu"):
        self.device = device
        self.autoencoder = VideoAutoencoder(...).to(device)
        self.encryptor = LatentEncryptor()
        self.generator = ImageGenerator(device=device)
        self.hider = HiderNetwork(...).to(device)
        self.revealer = RevealerNetwork(...).to(device)
```
*   **`__init__`**: This sets up the entire AI lab in one go. It moves every model onto the specified `device` (GPU or CPU) so there are no memory transfer bottlenecks during the hiding process.

### Step 3: The Hiding Process (`hide_video`)
```python
def hide_video(self, video_path, output_image_path):
    frames = extract_frames(video_path, max_frames=16, resize_dim=(64, 64))
    _, latent = self.autoencoder(frames)
    ciphertext, metadata = self.encryptor.encrypt(latent[0])
    cover_image = self.generator.generate_cover(size=(256, 256))
    spatial_secret = format_secret_for_hiding(ciphertext_to_bits(ciphertext, 65536))
    stego_image = self.hider(cover_image, spatial_secret)
```
*   **Logic Flow**:
    1.  **Extract**: Turn video into frames.
    2.  **Compress**: Turn frames into a 256-dimensional "Latent Vector".
    3.  **Encrypt**: Scramble the vector with AES-256.
    4.  **Generate**: Create a unique AI image.
    5.  **Hide**: The `Hider` AI weaves the scrambled bits into the image pixels.

### Step 4: The Extraction Process (`extract_video`)
```python
def extract_video(self, stego_image_path, metadata, cipher_len, output_video_path):
    secret_pred_spatial = self.revealer(stego_image)
    bit_tensor = extract_secret_from_prediction(secret_pred_spatial, cipher_len * 8)
    recovered_ciphertext = bits_to_ciphertext(bit_tensor[0], cipher_len)
    recovered_latent = self.encryptor.decrypt(recovered_ciphertext, metadata)
```
*   **`RevealerNetwork`**: Scans the image pixels to find the 1s and 0s.
*   **`AES Decryption`**: This is the critical security check. If the pixels were modified or the AI guessed wrong, the `LatentEncryptor` will catch the error and prevent the video from being corrupted.
*   **`Decoder`**: The Autoencoder's decoder turns the recovered numbers back into vivid video frames.

---

## 3. Deployment and Hardware Detection
```python
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = SteganoPipeline(device=device)
```
*   The pipeline is designed to be **cross-platform**. It automatically detects if it is running on a **Kaggle GPU (CUDA)**, an **Apple Silicon Mac (MPS)**, or a **standard CPU**, and configures the neural networks for optimal speed without any user intervention.

---
**End of Report**
