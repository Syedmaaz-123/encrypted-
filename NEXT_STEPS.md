# Next Steps: Building the Final Production Model

This document outlines the exact exact 4 steps you need to take to convert this architectural prototype from using **dummy data** (which we used to prove the code compiles and the mathematics work) to **real video data**, and then train the networks on a GPU.

---

### Step 1: Download a Real Video Dataset
Right now, the code uses a `DummyVideoDataset` that generates random pixel noise. To compress real videos, the AI needs to be shown real videos.
- **Action**: Download the [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php) (a standard dataset of short YouTube action clips). 
- **Code Change Required**: We will need to update `src/train.py` to add a strong PyTorch `DataLoader` (like `torchvision.datasets.UCF101`) that reads `.mp4` or `.avi` files from your downloaded folder.

### Step 2: Set Up Real Cover Images
Currently, the pipeline creates "dummy" cover images (blurry noise) to save time and compute power. 
- **Action**: You can either download a real image dataset like **COCO / Div2K** to train the Stego networks, or we can use the **Stable Diffusion** code I included in `src/image_generator.py` by changing `use_dummy=True` to `use_dummy=False`.
- *Note: Generating Stable Diffusion images in real-time during training requires a very powerful GPU (like an Nvidia RTX 3090/4090).*

### Step 3: Train the Models on a GPU
Training these deep learning networks will require massive computational power. A Macbook CPU is not strong enough for this stage.
- **Action**: Upload this project to a cloud GPU provider like **Google Colab (Pro)**, **Kaggle**, or an **AWS / RunPod** instance.
- **Phase A (Video Autoencoder)**: Run `train.py` and let the `train_video_autoencoder` loop run until the MSE loss is very low (meaning it can compress and reconstruct video motion perfectly).
- **Phase B (Stego Networks)**: Run `train.py` and let the `train_stego_networks` loop run for many epochs. The `Revealer` network must learn to extract the hidden AES bits with **100% accuracy (`Bit Loss = 0.0`)**, otherwise the AES cryptographic lock will crash the decryption process to protect standard users.

### Step 4: Plug the Trained Weights into the Pipeline
Once training finishes, the scripts will save three files into your `project/models/` folder:
1. `video_autoencoder.pth`
2. `hider.pth`
3. `revealer.pth`
- **Action**: Update `src/pipeline.py` to load these `.pth` files. In `pipeline.py`'s `__init__`, you would add:
  ```python
  self.autoencoder.load_state_dict(torch.load('../models/video_autoencoder.pth'))
  self.hider.load_state_dict(torch.load('../models/hider.pth'))
  self.revealer.load_state_dict(torch.load('../models/revealer.pth'))
  ```
- Once these weights are loaded, the `Decryption failed!` testing message will disappear, because the networks are now highly trained and can extract the mathematical payload perfectly!

---

**Ready to start?**
If you want to begin Step 1 right now, I can rewrite the dataset loader in `src/train.py` to read real `.mp4` folders, so that as soon as your dataset finishes downloading, you are fully ready to train on a GPU.
