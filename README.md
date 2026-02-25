# Deep Learning-Based Secure Video Steganography

This project implements a secure video steganography system that hides a full video inside a single AI-generated image. It combines 3D-CNN Autoencoders, AES-256-GCM encryption, stable diffusion (AI image generation), and UNet/CNN-based Steganography networks.

## Project Structure

The codebase is split into modular components inside the `src/` folder:

1. **`video_autoencoder.py`**: Contains the 3D-CNN `VideoEncoder` and `VideoDecoder`. This compresses a video (e.g., 16 frames of 64x64) into a dense 256-dimensional numerical vector.
2. **`encryption.py`**: Handles AES-256-GCM encryption. It converts the vector from the autoencoder into encrypted bytes, and translates them into a mathematical bit format (0s and 1s) suitable for inserting into an image.
3. **`image_generator.py`**: A wrapper that connects to Stable Diffusion (via Hugging Face `diffusers`) to generate realistic "cover" images. It also has a dummy fast-mode for immediate local testing.
4. **`stego_networks.py`**: Contains two critical neural networks:
   - **`HiderNetwork`**: Takes the generated cover image and the hidden bits, and subtly changes the pixels to embed the data.
   - **`RevealerNetwork`**: Takes a stego (modified) image and mathematically extracts the hidden bits.
5. **`utils.py`**: Helper scripts like `extract_frames` and `compile_video` for loading MP4 videos into PyTorch tensors, and writing them back.
6. **`train.py`**: The main script to train the neural networks. 
7. **`pipeline.py`**: The End-to-End inference script that connects all the steps (Encode -> Encrypt -> Hide -> Reveal -> Decrypt -> Decode).

## Current Status and "Errors"

If you run the project right now using `python src/pipeline.py`, you will see an output ending with:
```
Decryption failed! Networks untrained.
```
**This is NOT a bug!** This is exactly how the system is designed to behave before it is fully trained. 

**Why does this happen?**
Because the `RevealerNetwork` is essentially initialized with random weights, it guesses random bits when trying to extract the payload from the image. Since the payload is encrypted with AES, if even a single bit is guessed wrong, AES's built-in security check (MAC check) fails, preventing the video from being corrupted or accessed by unauthorized users. 

To fix this, the networks must learn to hide and extract bits perfectly. This requires **training** the model on a GPU with real data.

## Next Steps: How to Complete the Project

I have built the entire mathematical architecture and foundation for you. To turn this into a fully functioning prototype, follow these next steps:

### Step 1: Download a Real Video Dataset
Currently, `train.py` uses a `DummyVideoDataset` (which just generates random noisy frames) to ensure the code runs without crashing.
- To train it properly, download a real video dataset like **UCF101** or **Kinetics-400**.
- Update `src/train.py` to load from a folder containing these MP4 videos. 

### Step 2: Train the Video Autoencoder
You need to train the video compressor so it learns how to compress and reconstruct realistic motion.
- Open your terminal.
- Activate your environment: `source venv/bin/activate`
- Run: `python src/train.py`
- Wait for the "Video Autoencoder" training to finish (this loss should get very close to 0).

### Step 3: Train the Steganography Networks
The script will then train the Hider and Revealer networks.
- It will heavily penalize the Revealer network for getting bits wrong.
- Let this train for many epochs (e.g., 50-100 epochs) on a powerful GPU.
- When the `Bit Loss (BCE)` gets to `0.000` (meaning perfect accuracy), the system will successfully decrypt data!

### Step 4: Run the Fully Trained Pipeline
Once the `.pth` weight files are saved in the `models/` directory, update `pipeline.py` to load these pre-trained weights (by uncommenting the `load_state_dict` lines you can add there).
- Run: `python src/pipeline.py`
- Now, AES decryption will succeed, and the hidden video will pop out and be saved in your `data/` folder!

## How to test/run what we have currently:
```bash
# Activate the python environment with all the installed AI libraries
source venv/bin/activate

# Move into the source folder
cd src

# Run the training script (runs on dummy data to show it works)
PYTHONPATH=. python train.py

# Run the full end-to-end pipeline
PYTHONPATH=. python pipeline.py
```
