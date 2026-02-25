import cv2
import numpy as np
import torch


def extract_frames(video_path, max_frames=None, resize_dim=(128, 128)):
    """
    Extracts frames from a video file, resizes them, and normalizes them.
    Returns a PyTorch tensor of shape (C, T, H, W) where T is the number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_dim is not None:
            frame = cv2.resize(frame, resize_dim)

        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    if not frames:
        raise ValueError(f"Could not extract any frames from {video_path}")

    frames_np = np.array(frames).astype(np.float32) / 255.0
    tensor_frames = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
    return tensor_frames


def compile_video(frames_tensor, output_path, fps=30):
    """
    Reconstructs a video from a PyTorch tensor of shape (C, T, H, W).
    """
    if frames_tensor.requires_grad:
        frames_tensor = frames_tensor.detach()
    frames_tensor = frames_tensor.cpu()

    frames_np = frames_tensor.permute(1, 2, 3, 0).numpy()
    frames_np = np.clip(frames_np * 255.0, 0, 255).astype(np.uint8)

    T, H, W, C = frames_np.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        frame = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
