# Project Report: Secure Video Steganography - Video Autoencoder
**File:** `video_autoencoder.ipynb`  
**Description:** This module implements a 3D-Convolutional Neural Network (3D-CNN) designed to compress temporal video data into a compact latent representation.

---

## 1. Executive Summary
The **Video Autoencoder** is the dimensionality reduction engine of the project. Its primary goal is to take a sequence of video frames (motion data) and compress them into a tiny, high-density mathematical vector called the **Latent Space**. This compression is what makes the project possible; without it, the raw data from a 16-frame video would be far too large to fit inside the pixels of a single undercover image. The 3D-CNN layers are specifically chosen because they can "see" motion over time, not just static pixels.

---

## 2. Technical Component Breakdown

### Component 1: VideoEncoder (The Compressor)
```python
class VideoEncoder(nn.Module):
    self.encoder = nn.Sequential(
        nn.Conv3d(in_channels, 32, 3, 2, 1),
        nn.Conv3d(32, 64, 3, 2, 1),
        ...
    )
```
*   **3D Convolutions (`Conv3d`)**: Unlike standard photo AI, these filters have a third dimension for **Time**. They analyze how an object moves from Frame 1 to Frame 16.
*   **Stride & Downsampling**: The network uses jumps (strides) to physically shrink the video dimensions at every layer, discarding redundant background data while keeping the essential motion.
*   **Flatten & Fully Connected**: The final spatial grid is "crushed" into a list of 256 numbers. This list is the essence of the video.

### Component 2: VideoDecoder (The Reconstructor)
```python
class VideoDecoder(nn.Module):
    self.decoder = nn.Sequential(
        nn.ConvTranspose3d(...),
        nn.Sigmoid()
    )
```
*   **Transposed Convolutions**: These are "Up-sampling" layers. They take the 256 secret numbers and "blow them up" like a balloon to recreate the original height, width, and frame count of the video.
*   **Sigmoid Activation**: This final layer ensures that every pixel in the reconstructed video is between 0.0 and 1.0, making it ready to be displayed as a standard color frame.

---

## 3. Why 3D-CNNs are Essential
Traditional 2D-CNNs treat every frame like a separate photo. This project uses **3D-CNNs** because they understand **temporal correlation**. If a person is waving their hand, 3D-CNNs compress the *action* of the wave, rather than storing 16 separate copies of the hand. This drastically increases the security and efficiency of the steganographic hiding process.

---
**End of Report**
