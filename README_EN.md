[**🇺🇸 English**](README_EN.md) | [**🇨🇳 简体中文**](README.md)
# Visual Neuron Encoding Model and MEI Generation

A deep learning-based visual neuron encoding model. This project fits neural responses to visual stimuli and uses backpropagation to generate Maximum Excitatory Images (MEIs).

---

## Project Overview

This project aims to build a deep learning encoding model for visual neurons. By training a neural network to fit the response patterns of neurons to visual stimuli, we utilize gradient optimization via backpropagation to synthesize images that maximally activate specific neurons (MEI). This technology is crucial for understanding receptive field characteristics and neural encoding mechanisms of the visual system.

**Core Features:**
- **Neural Encoding Modeling**: Fits response patterns of 50 visual neurons using deep learning
- **MEI Generation**: Synthesizes optimal stimulus images for specific neurons via gradient optimization
- **Behavioral Modulation**: Integrates behavioral data to modulate neural responses

---

## Installation

### Prerequisites
- CUDA 11.x or higher (RTX 4090 recommended)
- Conda package manager

### Step 1: Create and activate Conda environment

```bash
# Create environment from provided environment.yml
conda env create -f environment.yml

# Activate environment
conda activate chatgpt
```

### Step 2: Install deep learning dependencies

```bash
# Install PyTorch (select appropriate command based on CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install experiment tracking tool
pip install swanlab
```

### Main Dependencies
- Python 3.11.5
- PyTorch (Deep Learning Framework)
- staticnet (Custom core library)
- swanlab (Experiment tracking)
- scipy.io / h5py (Data loading)
- matplotlib (Visualization)

---

## Architecture

### Overall Data Flow

```
Input Image (1×36×64)
    ↓
PooledStacked2dCore (Feature Extraction)
    ├── Stacked2dCore: Stacked convolutional layers (2 layers, 32 channels)
    └── AvgPool2d(2): Spatial downsampling (36×64 → 18×32)
    ↓
Feature Map (32×18×32) → Flattened → 18,432-dimensional vector
    ↓
ProReadout (Readout Layer)
    ├── LayerNorm / BatchNorm: Normalization
    ├── MLP: Hidden layer (256-dim, GELU activation)
    └── Modulator: Behavioral modulation (2-dim → 50-dim)
    ↓
Neuron Response Prediction (50 neurons)
```

### Core Components

#### 1. Core (Feature Extraction Backbone)
- **Stacked2dCore**: Located in [`staticnet/cores.py`](inception_loop2019-master/staticnet/cores.py), a feature extraction network based on stacked convolutions
  - Supports skip connections and regularization
  - Default configuration: 2 convolutional layers, 32 channels, 5×3 kernel sizes

#### 2. PooledStacked2dCore (Spatial Compression Wrapper)
Custom wrapper class that:
- Outputs only the last layer's features (avoiding concatenation of all layers)
- Adds `AvgPool2d(2)` for spatial downsampling
- Output dimension: 32 × 18 × 32 = 18,432

#### 3. Base (Combination Base Class)
- **CorePlusReadout2d**: Located in [`staticnet/base.py`](inception_loop2019-master/staticnet/base.py)
  - Combines Core + Readout modules
  - Supports Behavioral Modulation
  - Includes built-in non-linear activation (softplus)

#### 4. Readout (Readout Layer)
- **ProReadout**: Maps features to neuron responses
  - First layer: LayerNorm/BatchNorm + Dropout(0.3)
  - Hidden layer: Linear(18,432 → 256) + LayerNorm + GELU
  - Output layer: Linear(256 → 50)
  - Behavioral modulation: Linear(2 → 16) + ReLU + Linear(16 → 50)

---

## Usage

### Phase 1: Training Model

Training script: [`train_pro_high_res_laynorm.py`](train_pro_high_res_laynorm.py)

```bash
python train_pro_high_res_laynorm.py
```

**Input Data:**
- File: `my_training_data.mat`
- Samples: 1,440
- Image Size: 36 × 64 (Single-channel grayscale)
- Behavior Dimension: 2
- Number of Neurons: 50

**Outputs and Configurations:**
- Weights: Saved as `best_pro_model_ln.pth`
- Logs: Automatically tracked via SwanLab
- Config: Batch Size 64 (train) / 64 (val), Adam optimizer (lr=3e-4, weight_decay=1e-4), CosineAnnealingLR (T_max=100), 100 epochs

---

### Phase 2: Generating MEIs

MEI generation script: [`validate_mei.py`](validate_mei.py)

```bash
python validate_mei.py
```

**Workflow:**
1. Loads trained model weights (`best_pro_model_ln.pth`)
2. Scans real dataset to find the highest-responding image as initial seed
3. Refines the seed image using gradient optimization (SGD)
4. Applies Gaussian blur and boundary constraints for smoother results

**Outputs and Configurations:**
- Output File: `final_validated_meis.png` (Visualization of generated MEIs)
- Config: 500 iterations, SGD optimizer (lr=0.001), Gaussian blur regularization (sigma=1.0), Pixel constraints [-1.0, 1.0]

---

## Current Status & Known Issues

### ✅ Phase 1: Forward Training
**Status: Stable Convergence**
- Training loss decreases normally
- Training and validation loss curves show good convergence
- Training stability has improved significantly after implementing LayerNorm architecture

### ⚠️ Phase 2: MEI Generation - Known Issue

**Description:**
During backpropagation optimization for MEI generation, generated images exhibit high-frequency snowflake noise (white noise). The model currently fails to extract clear receptive field features, and image quality does not meet biological expectations.

**Potential Causes:**
1. **Insufficient image regularization**: The current Gaussian blur constraint may not effectively suppress high-frequency noise
2. **LayerNorm gradient properties**: LayerNorm gradients during backpropagation may hinder image optimization
3. **Optimization strategy**: SGD + gradient sign clipping might need adjustments
4. **Network architecture**: Legacy BatchNorm layers inside the Core might introduce instability during optimization

**Current Attempted Solutions:**
- Applied real-time smoothing using Gaussian blur in [`validate_mei.py`](validate_mei.py)
- Used gradient sign instead of raw gradients for pixel updates
- Added strict pixel value boundary constraints

**Future Directions to Explore:**
- Introduce Total Variation (TV) Loss to enhance spatial smoothness
- Experiment with different regularization strategies (e.g., L2 penalty, spectral normalization)
- Replace SGD with Adam optimizer for image optimization
- Adjust learning rate scheduling strategy during synthesis

---

## Project Structure

```
MEI/
├── README.md                           # This document
├── README_EN.md                        # English version of README
├── environment.yml                      # Conda environment configuration
├── train_pro_high_res_laynorm.py       # Training script
├── validate_mei.py                     # MEI generation script
├── my_training_data.mat                # Training dataset
├── best_pro_model_ln.pth               # Trained model weights
├── final_validated_meis.png            # Generated MEI visualization
├── inception_loop2019-master/           # staticnet core library
│   └── staticnet/
│       ├── cores.py                    # Core definitions
│       ├── base.py                     # Base class definitions
│       ├── readouts.py                 # Readout definitions
│       └── ...
└── swanlog/                            # SwanLab training logs
```

---

## References

Part of the code in this project is based on the following work:
- Walker et al. 2019 *Nature Neuroscience*: [Inception Loops](https://www.nature.com/articles/s41593-019-0517-x)
- staticnet library: Adapted from [inception_loop2019](inception_loop2019-master/)

---

## License

This project is for academic research purposes only.
