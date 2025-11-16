# HeadShape Prediction with Cascade UNet and Diffusion Model

A cascaded framework combining UNet and Diffusion Model for fast and accurate head shape prediction.

## Overview
This project implements a cascade model that combines UNet and Diffusion Model to achieve both fast inference speed and high prediction accuracy for head shape prediction tasks. The UNet provides an initial prediction, which is then refined by the diffusion model to generate more accurate results.

## Features
- Cascaded architecture combining UNet and Diffusion Model
- Support for multiple loss functions (L1, L2, Smooth L1, Huber)
- Both DDPM and DDIM sampling methods
- Multi-GPU support
- Comprehensive logging and visualization tools

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- tqdm
- tensorboard

Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
Training
```bash
python main.py --mode train
```
Testing
```bash
python main.py --mode test
```
Visualization
```bash
python main.py --mode visualize_denoise_process
```
## Configuration
You can modify the configuration parameters in main.py to adjust:
- Device settings (CPU/GPU, multi-GPU)
- Model hyperparameters
- Training settings (batch size, epochs, learning rate)
- Loss function selection
- Checkpoint and logging settings
## Results
The model outputs predictions in .pth format, which include:
- Predicted head shapes
- Input features
- Ground truth targets
- Additional metadata for post-processing
