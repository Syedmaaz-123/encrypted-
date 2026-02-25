import torch
import numpy as np


class ImageGenerator:
    """Wrapper for AI image generation."""

    def __init__(self, device="cpu", use_dummy=True):
        self.device = device
        self.use_dummy = use_dummy
        self.pipeline = None

        if not self.use_dummy:
            try:
                from diffusers import StableDiffusionPipeline

                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
                )
                self.pipeline = self.pipeline.to(self.device)
            except ImportError:
                print("Diffusers not installed. Falling back to dummy generator.")
                self.use_dummy = True

    def generate_cover(
        self,
        prompt="A beautiful realistic landscape photo, 4k resolution",
        size=(256, 256),
    ):
        if self.use_dummy or self.pipeline is None:
            img_tensor = (
                torch.rand((3, size[0], size[1]), dtype=torch.float32)
                .to(self.device)
                .unsqueeze(0)
            )
            import torch.nn.functional as F

            img_tensor = F.avg_pool2d(
                img_tensor, kernel_size=5, stride=1, padding=2
            ).squeeze(0)
            return (img_tensor - img_tensor.min()) / (
                img_tensor.max() - img_tensor.min()
            )
        else:
            image = self.pipeline(
                prompt, height=size[0], width=size[1], num_inference_steps=20
            ).images[0]
            image_np = np.array(image).astype(np.float32) / 255.0
            return torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
