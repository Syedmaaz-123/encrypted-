# Project Report: Secure Video Steganography - Encryption Module
**File:** `encryption.ipynb`  
**Description:** This module handles the AES-256-GCM encryption pipeline, data type transformations (Tensor to Bytes), and bit-packing for steganographic embedding.

---

## 1. Executive Summary
The Encryption Module acts as the security backbone of the video steganography system. It ensures that the compressed video data (latent vectors) produced by the AI models is mathematically scrambled before being hidden in an image. By utilizing **AES-256** in **Galois/Counter Mode (GCM)**, the system provides both **confidentiality** (data is unreadable) and **authenticity** (tampering is detected). If even a single bit is extracted incorrectly from the image, the decryption engine will identify the corruption and protect the integrity of the video.

---

## 2. Technical Line-by-Line Explanation

### Step 1: Library Imports
```python
import os
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
```
*   **`import os`**: Used for hardware-level, cryptographically secure random number generation (IVs and Keys).
*   **`import torch`**: The primary deep learning framework for handling the video data as mathematical Tensors.
*   **`Cipher, algorithms, modes`**: The core components of the `cryptography` library. `algorithms` defines the math (AES), `modes` defines the block handling (GCM), and `Cipher` ties them together.
*   **`default_backend`**: Instructs the library to use the fastest engine (like OpenSSL) available on the machine.

### Step 2: Key and IV Generation
```python
def generate_key():
    return os.urandom(32)

def generate_iv():
    return os.urandom(12)
```
*   **`generate_key`**: Creates a **256-bit (32-byte)** master key. This is a military-grade password that must be kept secret.
*   **`generate_iv`**: Generates a **96-bit (12-byte)** Initialization Vector. This changes every time you encrypt a video to ensure the output is always different, even if the video is the same.

### Step 3: Tensor to Byte Transformation
```python
def tensor_to_bytes(tensor):
    tensor_np = tensor.cpu().detach().numpy().astype("float32")
    return tensor_np.tobytes(), tensor_np.shape
```
*   **`tensor.cpu().detach()`**: Safely moves the AI data from the GPU to the CPU and stops the AI from tracking it for training.
*   **`.astype("float32")`**: Standardizes the decimals to 32 bits.
*   **`tobytes()`**: Flattens the complex video grid into a raw string of binary bytes ready for encryption.
*   **`tensor_np.shape`**: Records the original dimensions (like 1x256) so we can reconstruct the video later.

### Step 4: AES-256-GCM Encryption Logic
```python
def encrypt_data(data_bytes, key, iv):
    encryptor = Cipher(
        algorithms.AES(key), modes.GCM(iv), backend=default_backend()
    ).encryptor()
    return encryptor.update(data_bytes) + encryptor.finalize(), encryptor.tag
```
*   **`algorithms.AES(key)`**: Sets up the 256-bit AES lock.
*   **`modes.GCM(iv)`**: Enables Galois/Counter Mode for security and tampering detection.
*   **`encryptor.update() + encryptor.finalize()`**: Runs the video data through the math engine to create the "Ciphertext" (scrambled noise).
*   **`encryptor.tag`**: Produces a 16-byte signature. If the hidden data is changed by even one pixel in the image, this tag will no longer match.

### Step 5: Preparing Data for the Image (Bits)
```python
def ciphertext_to_bits(ciphertext, max_len=None):
    import numpy as np
    byte_array = np.frombuffer(ciphertext, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array).astype(np.float32)
    # ... padding logic ...
    return torch.from_numpy(bit_array)
```
*   **`np.unpackbits()`**: The "Explosion" step. It takes every 1 byte of encrypted data and explodes it into 8 individual binary bits (0s and 1s) so they can be hidden in pixels.
*   **`astype(np.float32)`**: Converts the bits to decimals (`0.0` or `1.0`) so the AI models can process them.
*   **`torch.from_numpy()`**: Sends the final list of bits to the Hider Network.

---

## 3. The Extraction Process (Decryption)
The module also contains the `bits_to_ciphertext`, `decrypt_data`, and `bytes_to_tensor` functions. These perform the **exact inverse** of the steps above:
1.  **Bit Packing**: Groups the extracted image bits back into bytes.
2.  **Tag Verification**: Checks the mathematical signature (`tag`) to ensure no one tampered with the image.
3.  **AES Unlocking**: Uses the secret key to turn the scrambled data back into video numbers.
4.  **Reshaping**: Folds the flat line of numbers back into a 3D video tensor for display.

---
**End of Report**
