# Diffusion-based-Face-Image-Generation

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
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
[A Latent Diffusion Model (LDM) that generates realistic human faces from pure noise — trained on CelebA latents and decoded through a VAE. Every face you see is fully synthetic and does not belong to a real person.]

## Motivation
[GANs made "This Person Does Not Exist" famous, but diffusion models have since become the dominant paradigm for generative image synthesis. This project revisits that idea through a Latent Diffusion Model, exploring how denoising in a compressed latent space (via a pretrained VAE) enables efficient, high-quality face generation. It served as a practical stepping stone toward more advanced generative modeling work, including conditional and cross-modal generation tasks.]

## Technical Aspect
Framework: Built entirely in PyTorch, with custom training loops (no high-level trainer abstractions) for full control over the diffusion process.
Architecture: A UNet-based backbone with residual blocks and attention layers, conditioned on the diffusion timestep via sinusoidal embeddings.
Latent Space: Images are encoded/decoded using a pretrained VAE (Stable Diffusion's sd-vae-ft-mse), reducing spatial dimensions before diffusion — significantly cutting compute vs. pixel-space models.
Training: Trained on CelebA face latents with a standard DDPM noise schedule (forward diffusion + noise prediction objective, MSE loss).
Sampling: Supports iterative denoising (ancestral sampling) to generate final latents, which are then decoded back to RGB images via the VAE decoder.

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

```bash
pip install -r requirements.txt
