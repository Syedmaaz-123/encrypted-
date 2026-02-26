# Project Report: Secure Video Steganography - Utility Module
**File:** `utils.ipynb`  
**Description:** This module provides the essential computer vision tools for handling video I/O, frame extraction, and temporal stitching.

---

## 1. Executive Summary
The **Utility Module** acts as the translator between the human world of video files and the AI world of mathematical tensors. It provides the preprocessing steps required before training (converting `.mp4` into tensors) and the post-processing steps required after extraction (converting tensors back into `.mp4`). These functions ensure that the video's motion, color, and resolution are preserved throughout the steganographic lifecycle.

---

## 2. Core Functional Logic

### Function 1: `extract_frames` (The Preprocessor)
```python
def extract_frames(video_path, max_frames=None, resize_dim=(128, 128)):
```
*   **Video Capture**: Uses OpenCV to open any video file (regardless of codec).
*   **Color Conversion**: Standardizes the video from BGR (camera format) to RGB (AI/Display format).
*   **Resizing**: Scales every frame to a fixed dimension (like 64x64 or 128x128). This is critical because Neural Networks cannot process varied sizes; they require a fixed mathematical grid.
*   **Normalization**: Divides all pixel values by 255.0 to bring them into the `0.0 - 1.0` decimal range required for AI processing.

### Function 2: `compile_video` (The Video Rebuilder)
```python
def compile_video(frames_tensor, output_path, fps=30):
```
*   **Detaching & Clipping**: It safely handles the AI-generated data, making sure it is detached from the AI memory and "clipping" any runaway math to stay between 0 and 255 (standard colors).
*   **FourCC Codec**: It uses the `mp4v` codec to optimize the video file size for standard media players.
*   **Stitching**: It iterates through the time dimension of the tensor, converting each numeric grid into a physical frame and writing it into a continuous stream to produce a reconstructed playable file.

---
**End of Report**
