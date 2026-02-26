# Deep Learning-Based Secure Video Steganography

This project implements a secure video steganography system that hides a full video inside a single AI-generated image. It combines 3D-CNN Autoencoders, AES-256-GCM## Installation & Dataset

Since video datasets are too large for GitHub, you must download the dataset separately before training:

1. Clone the repository:
   ```bash
   git clone https://github.com/Syedmaaz-123/encrypted-.git
   cd encrypted-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the UCF101 Dataset:**
   - Download the Action Recognition dataset from [UCF's public server](https://www.crcv.ucf.edu/data/UCF101.php).
   - Extract the `.rar` file.
   - Update `VIDEO_DATASET_PATH` in `src/train.ipynb` to point to your extracted folder.
   
## Project Structure

The codebase is split into modular components inside the `src/` folder:

1. **`video_autoencoder.ipynb`**: Contains the 3D-CNN `VideoEncoder` and `VideoDecoder`. This compresses a video (e.g., 16 frames of 64x64) into a dense 256-dimensional numerical vector.
2. **`encryption.ipynb`**: Handles AES-256-GCM encryption. It converts the vector from the autoencoder into encrypted bytes, and translates them into a mathematical bit format (0s and 1s) suitable for inserting into an image.
3. **`image_generator.ipynb`**: A wrapper that connects to Stable Diffusion (via Hugging Face `diffusers`) to generate realistic "cover" images. It also has a dummy fast-mode for immediate local testing.
4. **`stego_networks.ipynb`**: Contains two critical neural networks:
   - **`HiderNetwork`**: Takes the generated cover image and the hidden bits, and subtly changes the pixels to embed the data.
   - **`RevealerNetwork`**: Takes a stego (modified) image and mathematically extracts the hidden bits.
5. **`utils.ipynb`**: Helper scripts like `extract_frames` and `compile_video` for loading MP4 videos into PyTorch tensors, and writing them back.
6. **`train.ipynb`**: The main notebook to train the neural networks. 
7. **`pipeline.ipynb`**: The End-to-End inference notebook that connects all the steps (Encode -> Encrypt -> Hide -> Reveal -> Decrypt -> Decode).

## Current Status and "Errors"

If you run the pipeline notebook right now, you will see an output ending with:
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
Currently, `train.ipynb` uses a `DummyVideoDataset` (which just generates random noisy frames) to ensure the code runs without crashing.
- To train it properly, download a real video dataset like **UCF101** or **Kinetics-400**.
- Update `src/train.ipynb` to load from a folder containing these MP4 videos. 

### Step 2: Train the Video Autoencoder
You need to train the video compressor so it learns how to compress and reconstruct realistic motion.
- Open your environment: `source venv/bin/activate`
- Open `src/train.ipynb` in your preferred editor (like VS Code or Jupyter).
- Run the cells to train.
- Wait for the "Video Autoencoder" training to finish (this loss should get very close to 0).

### Step 3: Train the Steganography Networks
The script will then train the Hider and Revealer networks.
- It will heavily penalize the Revealer network for getting bits wrong.
- Let this train for many epochs (e.g., 50-100 epochs) on a powerful GPU.
- When the `Bit Loss (BCE)` gets to `0.000` (meaning perfect accuracy), the system will successfully decrypt data!

### Step 4: Run the Fully Trained Pipeline
Once the `.pth` weight files are saved in the `models/` directory, update `pipeline.ipynb` to load these pre-trained weights.
- Open and run the cells in `src/pipeline.ipynb`.
- Now, AES decryption will succeed, and the hidden video will pop out!

## How to test/run what we have currently:
```bash
# Activate the python environment with all the installed AI libraries
source venv/bin/activate

# Open the notebooks in VS Code or start Jupyter
jupyter notebook
```
