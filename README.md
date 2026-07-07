# Diffusion-based-Face-Image-Generation

## Table of Content
  * [Demo](#-demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Inference](#inference)
  * [Pretrained Weights](#pretrained-weights)
  * [Training](#training)
  * [Latent Encoding (VAE)](#latent-encoding-vae)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)

## 🎬 Demo

<a href="https://huggingface.co/spaces/keysun89/this_person_does_not_exist_ldm" target="_blank">
  <img align="right" src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.png" width="150" alt="Try it on Hugging Face">
</a>

Try the live demo on Hugging Face Spaces 👉 **[Launch Demo](https://huggingface.co/spaces/keysun89/this_person_does_not_exist_ldm)**

<p align="center">
  <img src="https://github.com/keysun8/Diffusion-based-Face-Image-Generation/raw/main/LDM_FACE_DEMO.png" alt="LDM Face Generation Demo" width="700">
</p>

<p align="center">
  <em>Latent Diffusion Model generating photorealistic, non-existent human faces.</em>
</p>

## Overview

A Latent Diffusion Model (LDM) that generates realistic human faces from pure noise — trained on CelebA latents and decoded through a VAE. Every face you see is fully synthetic and does not belong to a real person.

## Motivation

GANs made "This Person Does Not Exist" famous, but diffusion models have since become the dominant paradigm for generative image synthesis. This project revisits that idea through a Latent Diffusion Model, exploring how denoising in a compressed latent space (via a pretrained VAE) enables efficient, high-quality face generation. It served as a practical stepping stone toward more advanced generative modeling work, including conditional and cross-modal generation tasks.

## Technical Aspect

- **Framework:** Built entirely in PyTorch, with custom training loops (no high-level trainer abstractions) for full control over the diffusion process.
- **Architecture:** A UNet-based backbone with residual blocks and attention layers, conditioned on the diffusion timestep via sinusoidal embeddings.
- **Latent Space:** Images are encoded/decoded using a pretrained VAE (Stable Diffusion's `sd-vae-ft-mse`), reducing spatial dimensions before diffusion — significantly cutting compute vs. pixel-space models.
- **Training:** Trained on CelebA face latents with a standard DDPM noise schedule (forward diffusion + noise prediction objective, MSE loss).
- **Sampling:** Supports iterative denoising (ancestral sampling) to generate final latents, which are then decoded back to RGB images via the VAE decoder.

## Installation

The code is written in Python 3.10+. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/). Make sure you have the latest version of `pip` before installing dependencies.

Clone the repository and install the required packages:

```bash
git clone https://github.com/keysun8/Diffusion-based-Face-Image-Generation.git
cd Diffusion-based-Face-Image-Generation
pip install -r requirements.txt
```

A CUDA-enabled GPU is strongly recommended for both training and inference, though the scripts will fall back to CPU automatically if none is available.

## Inference

Download a trained checkpoint from the [pretrained weights repo](https://huggingface.co/keysun89/face_ldm_ckpt/tree/main) on Hugging Face, then generate a single face image:

```bash
python inference.py --ckpt checkpoints/ckpt_epoch_0095.pt --output_dir inference_outputs
```

| Argument | Description | Default |
|---|---|---|
| `--ckpt` | Path to a trained checkpoint (`.pt`) | *required* |
| `--output_dir` | Directory to save the generated image | `inference_outputs` |

The generated image is saved as a PNG inside `--output_dir`.

## Pretrained Weights

Trained checkpoints are hosted on Hugging Face Hub:
👉 **[keysun89/face_ldm_ckpt](https://huggingface.co/keysun89/face_ldm_ckpt/tree/main)**

Download a checkpoint and point `--ckpt` (inference) or `--resume_ckpt` (training) to the downloaded file:

```bash
huggingface-cli download keysun89/face_ldm_ckpt ckpt_epoch_0095.pt --local-dir checkpoints
```

## Training

Train the UNet-based latent diffusion model on pre-encoded CelebA latents:

```bash
python train.py \
    --data_root_pt data_256_latent \
    --data_root_jpg data_256 \
    --ckpt_dir checkpoints \
    --sample_dir samples \
    --epochs 100 --batch_size 32
```

### Resuming training from a checkpoint

If training is interrupted, resume exactly where you left off (model, optimizer, scheduler, AMP scaler, global step, and FID history are all restored):

```bash
python train.py \
    --data_root_pt data_256_latent \
    --data_root_jpg data_256 \
    --resume_ckpt checkpoints/ckpt_epoch_0050.pt
```

| Argument | Description | Default |
|---|---|---|
| `--data_root_pt` | Directory of pre-encoded VAE latents (`.pt` files) | *required* |
| `--data_root_jpg` | Directory of real RGB images, used for FID reference stats | *required* |
| `--ckpt_dir` | Where to save model checkpoints | `checkpoints` |
| `--resume_ckpt` | Path to a checkpoint to resume training from | `None` |
| `--sample_dir` | Where to save sample image grids during training | `samples` |
| `--epochs` | Number of training epochs | `100` |
| `--batch_size` | Training batch size | `32` |
| `--lr` | Learning rate | `1e-4` |
| `--seed` | Optional random seed for reproducibility | `None` |
| `--log_loss_every` | Log/plot the loss curve every N steps | `100` |
| `--save_img_every` | Save a sample image grid every N steps | `1500` |
| `--calc_fid_every` | Compute FID every N steps | `4500` |
| `--save_ckpt_every_epoch` | Save a checkpoint every N epochs | `5` |

Training checkpoints, loss curves (`loss_curve.png`, `epoch_loss_curve.png`), and FID curves (`fid_curve.png`) are saved automatically as training progresses.

## Latent Encoding (VAE)

Before training, images must be pre-encoded into VAE latents to speed up data loading. This step converts a folder of 256×256 RGB images into `(4, 32, 32)` latent tensors using Stable Diffusion's pretrained VAE (`CompVis/stable-diffusion-v1-4`).

```bash
python encode_latents.py \
    --img_dir  /path/to/images \
    --out_dir  /path/to/latents \
    --batch_size 64 \
    --num_workers 8
```

| Argument | Description | Default |
|---|---|---|
| `--img_dir` | Root folder containing images (searched recursively) | *required* |
| `--out_dir` | Output folder for `.pt` latent files (mirrors subfolder structure) | *required* |
| `--batch_size` | Images per VAE forward pass | `16` |
| `--num_workers` | DataLoader workers | `8` |
| `--model_id` | HuggingFace model id or local path for the VAE | `CompVis/stable-diffusion-v1-4` |
| `--no_fp16` | Save latents as float32 instead of float16 (doubles disk usage) | `False` |
| `--sample` | Store a sampled latent instead of the deterministic mean | `False` |

Already-encoded images are skipped automatically on re-runs, so the script can be safely resumed if interrupted.

## Technologies Used

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Diffusers-FFD21E?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge)

- **PyTorch** — core deep learning framework
- **🤗 Diffusers** — pretrained Stable Diffusion VAE (`AutoencoderKL`) for latent encoding/decoding
- **torchmetrics** — Fréchet Inception Distance (FID) evaluation during training
- **Gradio** — interactive demo hosted on Hugging Face Spaces
- **CelebA** — training dataset of aligned human face images

## Team

**Kishan Madlani (Keysun)**
GitHub: [@keysun8](https://github.com/keysun8) · Hugging Face: [@keysun89](https://huggingface.co/keysun89)

## License

This project is licensed under the [MIT License](LICENSE).

## Credits

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — foundational DDPM formulation used for the noise schedule and training objective
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — latent diffusion approach this project builds on
- [Stable Diffusion VAE](https://huggingface.co/CompVis/stable-diffusion-v1-4) — pretrained encoder/decoder used for latent-space compression
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) — training data for face generation
- Inspired by [This Person Does Not Exist](https://thispersondoesnotexist.com/)
