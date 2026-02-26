# Project Report: Secure Video Steganography - Training Module
**File:** `train.ipynb`  
**Description:** This module contains the training loops, dataset loaders, and optimization logic required to teach the AI models how to compress video and hide it securely within images.

---

## 1. Executive Summary
The **Training Module** is the "classroom" where the project's AI models learn their jobs. Training is split into two distinct phases: 
1.  **Learning to Compress**: Training the Video Autoencoder to represent complex motion with a small set of numbers.
2.  **Learning to Hide**: Training the Hider and Revealer networks to embed and extract encrypted data with zero errors.
The module is built to be **High Performance**, utilizing GPU acceleration (CUDA) to process thousands of video frames and images.

---

## 2. Technical Line-by-Line Explanation

### Step 1: Data Preparation (`RealVideoDataset`)
```python
class RealVideoDataset(Dataset):
    def __init__(self, directory, frames=16, height=64, width=64):
        self.video_paths = glob.glob(...)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
```
*   **Dataset Loader**: This class recursively searches your dataset folder (like UCF101) for `.mp4` and `.avi` files.
*   **Frame Selection**: It pulls a sequence of exactly **16 frames** from each video.
*   **Standardization**: Every video is resized to $64 \times 64$ pixels and normalized to the `0.0 - 1.0` range. This consistency is required so the AI doesn't get confused by different video sizes.

### Step 2: Autoencoder Training Loop
```python
def train_video_autoencoder(model, dataloader, epochs=5, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # ...
    reconstructed, _ = model(batch)
    loss = criterion(reconstructed, batch)
```
*   **Goal**: The goal here is **Reconstruction Accuracy**.
*   **Loss Function (MSE)**: Mean Squared Error measures the "distance" between the original video and the AI's reconstructed version. The AI's job is to minimize this loss until the reconstructed video looks exactly like the original.
*   **Saving**: Once finished, it saves the weights as `video_autoencoder.pth`.

### Step 3: Steganography Training Loop (Adversarial)
```python
def train_stego_networks(hider, revealer, image_generator, epochs=5, ...):
    criterion_mse = nn.MSELoss() # Image quality loss
    criterion_bce = nn.BCELoss() # Bit accuracy loss
    loss = (10.0 * l_img) + l_bit
```
*   **Multi-Objective Loss**: This is the most complex part of the math. The AI has two bosses:
    1.  **Invisible Data (`l_img`)**: The Hider is penalized if the stego-image looks different from the cover image.
    2.  **Perfect Extraction (`l_bit`)**: The Revealer is penalized for every bit it gets wrong.
*   **The Weight (10.0)**: We multiply the image loss by 10 to force the AI to prioritize "Invisibility" over everything else.

---

## 3. Kaggle Setup and Configuration
```python
if __name__ == "__main__":
    USE_REAL_DATA = True 
    VIDEO_DATASET_PATH = "dataset" # Update this path on Kaggle!
```
*   **Generic Pathing**: The code uses a placeholder `"dataset"` string. This is designed for easy editing on the Kaggle platform after importing the UCF101 dataset.
*   **GPU Detection**: The script automatically detects the Kaggle P100/T4 GPUs using `torch.cuda.is_available()`, ensuring training runs 100x faster than on a standard CPU.

---
**End of Report**
