# Project Report: Secure Video Steganography - AI Stego Networks
**File:** `stego_networks.ipynb`  
**Description:** This module contains the two primary "adversarial" neural networks that handle the physical embedding and extraction of encrypted bits into image pixels.

---

## 1. Executive Summary
The **Stego Networks** are the implementation of AI-driven data hiding. Unlike traditional steganography (like LSB), which follows fixed mathematical rules, these networks are **Generative and Adaptive**. 
*   The **Hider** learns how to change pixels in a way that is statistically invisible to the human eye. 
*   The **Revealer** learns how to scan an image and ignore the "noise" of the photo to extract only the secret bits. 

By training these two networks together, they develop a custom "language" for hiding data that is remarkably robust against standard detection tools.

---

## 2. Technical Architecture

### Step 1: HiderNetwork (The Invisibility Engine)
```python
class HiderNetwork(nn.Module):
    # Channels: Cover Image (3) + Secret Bits (1)
    in_channels = 4 
```
*   **Concatenation**: This model takes the 3-channel (RGB) Cover Image and the 1-channel Secret Bit Grid and "glues" them together.
*   **Deep Residual Convolution**: Use multiple layers of 2D convolutions to analyze the texture of the cover image. It learns to hide data in "complex" areas (like grass or trees) where changes are harder to see, and avoids "flat" areas (like clear skies) where changes would be obvious.
*   **Sigmoid**: Final layer ensures the modified "Stego Image" is a valid photo with pixel values between 0.0 and 1.0.

### Step 2: RevealerNetwork (The AI Detective)
```python
class RevealerNetwork(nn.Module):
```
*   **Input**: Takes the modified (stego) image.
*   **Pixel-Level Analysis**: It does not care about the "beauty" of the photo. It is looking for the microscopic residual differences added by the Hider. It learns to "subtract" the cover image content mentally to find the bit grid hidden underneath.
*   **Output**: Produces a probability map where each pixel represents the likelihood of a bit being a `1` or a `0`.

---

## 3. Data Formatting Helpers
Because Neural Networks process data in 2D grids (like images), we cannot give them a single long line of bits.
1.  **`format_secret_for_hiding`**: This function takes the flat list of thousands of encrypted bits and rearranges them into a **256x256 square grid**. This "spatially aware" format allows the Hider AI to map bits directly to specific pixel coordinates.
2.  **`extract_secret_from_prediction`**: Once the Revealer scans the square grid, this function "unrolls" the grid back into a single long line so the AES decryption engine can unlock the video.

---
**End of Report**
