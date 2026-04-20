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
