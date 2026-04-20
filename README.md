# DDPM on MNIST – Diffusion Model from Scratch

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate handwritten digits from the MNIST dataset.  
The notebook demonstrates the full pipeline of a diffusion model, including the **forward diffusion process, model training, and reverse diffusion for image generation**.

The implementation combines the dataset loading from TensorFlow with model training using PyTorch and provides an optional interactive interface for generation.

---

## Project Overview

Diffusion models are a class of generative models that learn to create data by gradually removing noise from random samples.

This notebook implements:

- Forward diffusion (adding noise to images)
- Reverse diffusion (denoising to generate images)
- A simplified **U-Net architecture**
- Training pipeline for DDPM
- Visualization of training results
- Interactive interface for generating digits

The model is trained on the **MNIST handwritten digits dataset**.

---

## Technologies Used

- PyTorch – model architecture and training  
- TensorFlow – loading the MNIST dataset  
- NumPy – numerical computations  
- Matplotlib – visualization of training and generated images  
- Gradio – interactive interface for digit generation  

---

## Project Structure
ddpm-mnist.ipynb
│
├── Install Dependencies
├── Imports and Device Setup
├── Load MNIST Dataset
├── Diffusion Schedule (Beta / Alpha)
├── Forward Diffusion Process
├── U-Net Architecture
├── Training Function
├── Model Training
├── Training Visualization
├── Reverse Diffusion (Sampling)
├── Image Generation
└── Gradio Interactive Interface


---

## Diffusion Process

### Forward Diffusion

Noise is gradually added to an image across multiple timesteps.

x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * ε


Where:

- `x₀` is the original image
- `x_t` is the noisy image at timestep `t`
- `ε` is Gaussian noise

---

### Reverse Diffusion

The neural network learns to **predict the noise added at each step**, allowing the model to progressively reconstruct an image from pure noise.

---

## Model Architecture

The model uses a **simplified U-Net architecture** adapted for **28×28 grayscale images**.

Inputs:
- Noisy image `x_t` → shape `(B, 1, 28, 28)`
- timestep `t`

Components:

- Sinusoidal timestep embedding
- Downsampling blocks
- Bottleneck layers
- Upsampling blocks
- Skip connections

Output:
- Predicted noise `ε`

---

## Training

Example configuration:
TIMESTEPS = 1000
epochs = 10
learning_rate = 1e-3
batch_size = 128


The model is trained to minimize the **Mean Squared Error between predicted and true noise**.

Loss function:
L = E || ε - ε_θ(x_t, t) ||²


---

## Generating Images

After training, the model generates digits by:

1. Starting with random noise
2. Iteratively denoising through the reverse diffusion process
3. Producing a final image resembling handwritten digits

Example output:

- Generated digits similar to MNIST
- Grid visualization of samples

---

## Interactive Interface

An optional interface built with **Gradio** allows users to generate digits interactively.

Features:

- Generate multiple images
- Visualize generated samples
- Run inference directly from the browser

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ddpm-mnist.git
cd ddpm-mnist

Install dependencies:
pip install torch torchvision tensorflow matplotlib numpy gradio

Run the notebook:
jupyter notebook ddpm-mnist.ipynb
