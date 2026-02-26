# Project Report: Secure Video Steganography - AI Image Generation
**File:** `image_generator.ipynb`  
**Description:** This module utilizes Generative AI (Stable Diffusion) to create unique, original "cover" images for hiding data, ensuring no original exist for comparison.

---

## 1. Executive Summary
The **Image Generation Module** is the creative heart of this steganography system. Its primary role is to provide a "cover" for the hidden video data. Unlike traditional steganography that uses existing photos, this module utilizes **Generative AI (Stable Diffusion)** to create brand-new, high-quality images from scratch. This uniqueness makes it mathematically impossible for an attacker to perform a "Comparative Analysis" to find the hidden data, as no original file exists in the world to compare against the stego-image.

---

## 2. Technical Line-by-Line Explanation

### Step 1: Core Framework Imports
```python
import torch
import numpy as np
```
*   **`import torch`**: The foundational framework for AI operations. Used to move images between CPU and GPU and handle tensors.
*   **`import numpy as np`**: Used for fast numerical pixel-level transformations that don't require the neural network.

### Step 2: The ImageGenerator Class & Initialization
```python
class ImageGenerator:
    def __init__(self, device="cpu", use_dummy=True):
        self.device = device
        self.use_dummy = use_dummy
```
*   **Purpose**: A wrapper that manages the complexity of the Stable Diffusion model.
*   **`device`**: Dictates if generation happens on the CPU or GPU (GPU is significantly faster).
*   **`use_dummy`**: A safety switch. If set to `True`, the code creates random noise instead of real AI images, allowing the system to run on computers without high-end graphics cards.

### Step 3: Loading the Stable Diffusion Brain
```python
if not self.use_dummy:
    from diffusers import StableDiffusionPipeline
    self.pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    ).to(self.device)
```
*   **`StableDiffusionPipeline`**: The industry-standard library for turning text prompts into pixels.
*   **`v1-5`**: Loads the pre-trained weights from the "Stable Diffusion v1.5" model.
*   **`float16`**: A memory-saving technique that uses half-precision math to fit the giant model into consumer GPUs.

### Step 4: Generating the Cover Image
```python
def generate_cover(self, prompt="...", size=(256, 256)):
```
*   **`prompt`**: The text-based description (e.g., "A realistic forest") that the AI uses to draw the cover image.
*   **`size`**: Controls the resolution. We default to 256x256 to keep processing speeds high.

### Step 5: Data Normalization and Permutation
```python
image_np = np.array(image).astype(np.float32) / 255.0
return torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
```
*   **`/ 255.0`**: Normalizes the pixel colors. Standard images use numbers up to 255, but AI models require decimals between `0.0` and `1.0`.
*   **`.permute(2, 0, 1)`**: Standard images are [Height, Width, Colors]. PyTorch models require [Colors, Height, Width]. This swap is critical so the `HiderNetwork` can "see" the image correctly.

---
**End of Report**
