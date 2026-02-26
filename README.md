# 🛡️ AI-Driven Secure Video Steganography
> **A Deep Learning pipeline to hide encrypted video motion inside unique AI-generated cover images.**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Cryptography](https://img.shields.io/badge/AES--256--GCM-Secure-blue?style=flat&logo=lock)](https://cryptography.io/)

This project implements a state-of-the-art secure steganography system. It doesn't just hide data; it **compresses**, **encrypts**, and **synthesizes** a new visual reality to act as an undetectable undercover carrier.

---

## 🏗️ System Architecture

The pipeline consists of five major technological pillars:

1.  **Temporal Compression**: A 3D-CNN Autoencoder that shrinks 16 frames of video into a 256-dimensional "Latent Vector."
2.  **Cryptographic Locking**: AES-256-GCM encryption that turns the vector into authenticated cipher-bits.
3.  **Visual Synthesis**: Stable Diffusion v1.5 API generating unique cover images from text prompts.
4.  **Spatial Embedding**: A Deep Hider Network that adapts pixel changes to the image's texture.
5.  **Neural Extraction**: A Revealer Network that recovers bits with microscopic precision.

---

## 📁 Project Structure

All core logic is contained in the `src/` directory as interactive Jupyter Notebooks:

*   📂 **`src/`**
    *   `video_autoencoder.ipynb`: 3D-CNN architecture for video compression.
    *   `encryption.ipynb`: AES-256-GCM logic and bit-packing utilities.
    *   `image_generator.ipynb`: Stable Diffusion integration (Real & Dummy modes).
    *   `stego_networks.ipynb`: The Hider and Revealer AI models.
    *   `pipeline.ipynb`: **Master Orchestrator** for end-to-end hiding and extraction.
    *   `train.ipynb`: Training loops for GPU-accelerated learning.
*   📂 **`reports/`**
    *   *Contains 7 detailed technical manuals explaining every line of code in the project.*

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Syedmaaz-123/encrypted-.git
cd encrypted-

# Install dependencies
pip install -r requirements.txt
```

### 2. Training on Kaggle (Recommended)
Because this project uses 3D-CNNs and Stable Diffusion, a **GPU is strictly required** for training.
1.  Upload the `src/` folder to a Kaggle Notebook.
2.  Add the **UCF101** dataset via the Kaggle sidebar.
3.  In `train.ipynb`, update `VIDEO_DATASET_PATH` to point to the Kaggle input path.
4.  Enable **GPU P100** and run all cells.
5.  Download `video_autoencoder.pth`, `hider.pth`, and `revealer.pth` to your local `models/` folder.

### 3. Running the Pipeline
Once you have trained weights:
1.  Open `src/pipeline.ipynb`.
2.  Provide a path to an MP4 video.
3.  Run the cells to generate an AI image and hide your video inside it.

---

## 🛡️ Security Features
*   **Unique Carrier**: Every cover image is AI-generated; there is no "original" photo for an attacker to compare against.
*   **Authenticated Encryption**: Using AES-GCM means that if the Revealer extracts even one bit incorrectly, the system will detect the tampering and prevent decryption.
*   **Temporal Awareness**: 3D-CNNs ensure the video's motion is preserved through the latent space.

---

## 📖 Technical Documentation
For a deep dive into the mathematical and cryptographic implementation, please refer to the **`reports/`** directory. Each file therein provides a line-by-line explanation of the corresponding module.

---
**Author:** Syed Maaz  
**Repository:** [encrypted-](https://github.com/Syedmaaz-123/encrypted-)
