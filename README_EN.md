# MEI-Visual-Neuron-Extraction

**A Low-Frequency Specialized CNN Framework for Retinal Ganglion Cell Maximally Exciting Images (MEI)**

English | [简体中文](README.md)

---

## Table of Contents

- [1. Background & Problem Definition](#1-background--problem-definition)
- [2. Dataset & Experimental Setup](#2-dataset--experimental-setup)
- [3. Our Solution](#3-our-solution)
- [4. Installation](#4-installation)
- [5. Usage & Execution](#5-usage--execution)
- [6. Outputs & Directory Structure](#6-outputs--directory-structure)
- [7. Key Findings & Results](#7-key-findings--results)
- [Acknowledgments](#acknowledgments)

---

## 1. Background & Problem Definition

### What is MEI?

**Maximally Exciting Images (MEI)** are visual stimulus patterns that elicit the strongest response from a specific neuron. Through MEI analysis, we can reverse-engineer a neuron's feature selectivity and understand how the visual system encodes external information.

### The Problem with Traditional Methods

In conventional MEI generation pipelines, gradient-based optimization methods often cause models to converge to **high-frequency local optima**. This manifests as:

| Artifact Type | Visual Characteristics | Biological Implausibility |
|---------------|------------------------|---------------------------|
| Checkerboard | Periodic light-dark alternation | RGC receptive fields lack periodic structures |
| Salt-and-pepper noise | Random high-frequency pixels | Exceeds neuronal spatial resolution |
| Edge aliasing | Discontinuous sharp boundaries | Contradicts Gabor filter smoothness |

These **high-frequency aliasing artifacts** not only degrade MEI visual quality but, more importantly, **violate the receptive field properties of biological retinal ganglion cells (RGCs)** — real RGC receptive fields typically exhibit smooth Gabor wave or Gaussian envelope structures.

> **Core Contradiction**: Models possess the ability to "cheat" (capturing arbitrary high frequencies) but lack the "discipline" to follow biological priors.

---

## 2. Dataset & Experimental Setup

The digital twin model in this project is built upon real neurobiological experimental data. Below is a detailed description of the experimental design and data acquisition.

### 2.1 Biological Recording Methodology

We employed **two-photon calcium imaging** to record neural activity in the **dorsal Lateral Geniculate Nucleus (dLGN)** of **awake mice**:

| Recording Parameter | Details |
|---------------------|---------|
| **Imaging Technique** | Two-photon excitation fluorescence microscopy |
| **Target Brain Region** | Dorsal LGN (dLGN) — the relay station for visual information from retina to cortex |
| **Cell Type** | Cart-positive retinal ganglion cell (RGC) axon terminals (boutons) |
| **Calcium Indicator** | GCaMP series genetically-encoded calcium indicator |
| **Recording Target** | Presynaptic boutons formed by RGC axons in dLGN |

> **Biological Significance**: By recording RGC axon terminals rather than cell bodies, we directly measure the visual signals transmitted to dLGN, providing a unique perspective for understanding retino-thalamic information transfer.

### 2.2 Visual Stimulation Paradigm

The experiment employed a carefully designed visual stimulus sequence:

```
Stimulus Design: 48 distinct visual stimulus patterns
Repetitions: Each stimulus presented 30 times (trials)
Total Stimuli: 48 × 30 = 1,440 presentations
```

This repeated-measures design enables us to:
- Calculate mean response characteristics for each neuron
- Assess response reliability and noise levels
- Construct a robust training dataset

### 2.3 Data Matrix Construction

From the extensive neural recordings, we rigorously selected **50 direction-selective (DS) boutons** to construct the core dataset for training the digital twin model. The dataset comprises three matrices:

#### Stimulus Matrix

```
Dimensions: [N_samples × Image_Height × Image_Width]
Content: 48 visual stimulus images × 30 repetitions
Format: Grayscale images, Z-score normalized
```

#### Response Matrix

```
Dimensions: [N_samples × N_neurons] = [1440 × 50]
Content: Calcium fluorescence responses from 50 DS boutons
Processing: ΔF/F₀ normalization, Z-score standardization
```

#### Behavior Matrix

```
Dimensions: [N_samples × 2]
Content: Synchronously recorded behavioral states of awake mice
Features:
  - Running: Locomotion speed (motor state)
  - Pupil: Pupil size (arousal level)
Purpose: Enable behavior modulation in the model
```

### 2.4 Data Matrix Relationship Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    Experimental Data Acquisition                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Stimulus       │  │  Response       │  │  Behavior       │
│  Matrix         │  │  Matrix         │  │  Matrix         │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ 48 stimuli      │  │ 50 DS boutons   │  │ Running speed   │
│ × 30 trials     │  │ × 1440 samples  │  │ Pupil size      │
│ = 1440 images   │  │                 │  │                 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
              ┌───────────────────────────────┐
              │    Digital Twin Training Data │
              │    my_training_data.mat       │
              └───────────────────────────────┘
```

### 2.5 Significance of Direction Selectivity (DS)

We specifically selected **direction-selective (DS) boutons** as our research subjects because:

1. **Clear Function**: DS neurons produce strongest responses to specific motion directions, serving as key units in visual motion processing
2. **Interpretable Features**: MEI analysis can clearly reveal their orientation and direction preferences
3. **Model Validation**: DS properties provide objective criteria for evaluating digital twin model accuracy

---

## 3. Our Solution

### Design Philosophy

We designed a **low-frequency specialized CNN architecture** that physically strips away the model's ability to capture high-frequency signals, forcing it to learn biologically plausible smooth features.

### Core Technologies

#### 3.1 Multi-Scale Physical Blurring (Anti-Aliasing)

```
Input Image → 3×3 AvgPool → 5×5 AvgPool → 7×7 AvgPool → Feature Extraction
```

Through three consecutive average pooling layers, we apply **physical low-pass filtering** before feature extraction, blocking high-frequency noise from entering the network at the source.

#### 3.2 Large-Scale Convolution Kernel Design

| Layer | Kernel Size | Function |
|-------|-------------|----------|
| Macro Stroking | **11×11** stride=2 | Capture large-scale spatial structures |
| Meso Outlining | **7×7** stride=2 | Extract medium-scale contours |
| Core Feature Region | **5×5** | Fine feature modeling (3×3 completely abandoned) |

Large convolution kernels naturally possess larger receptive fields and stronger smoothing effects, aligning with biological RGC receptive field properties.

#### 3.3 Circular Receptive Field Mask

Simulating the circular visual field of biological retina, we apply a circular mask at the input stage, filling regions outside the circle with background gray values to avoid edge effects from square boundaries.

#### 3.4 Parameter Efficiency

```
Total Parameters: ≈ 330,000 (330K)
Model File Size: ~1.4 MB
```

Achieving lightweight design while maintaining high performance, facilitating training and deployment.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image (100×100)                     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Circular Receptive Field Mask                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         Multi-Scale Blur (3×3 + 5×5 + 7×7 AvgPool)           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    Macro Stroking (11×11 Conv, stride=2) → 50×50             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    Meso Outlining (7×7 Conv, stride=2) → 25×25               │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    Stacked2dCore (5×5 Conv × 3 layers) → 25×25              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         Global Average Pooling → 7×7 = 1960-dim vector       │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    ProReadout (MLP + Behavior Modulation) → 50 Neurons       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Installation

### Requirements

- **OS**: Windows 10/11, Linux, macOS
- **GPU**: NVIDIA GPU (RTX 3080 or higher recommended), CUDA 11.8 support
- **Conda**: Miniconda or Anaconda

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/gorillaleap/dLGN-MEI.git
cd dLGN-MEI

# 2. Create Conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate chatgpt

# 4. Verify GPU support
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Core Dependencies

| Dependency | Purpose |
|------------|---------|
| `pytorch` | Deep learning framework |
| `torchvision` | Image transforms and augmentation |
| `numpy` | Numerical computing |
| `scipy` | Scientific computing (MAT file reading) |
| `h5py` | HDF5 file handling |
| `pandas` | Data analysis |
| `openpyxl` | Excel file export |
| `matplotlib` | Visualization |
| `swanlab` | Experiment tracking and logging |

### CUDA Version Compatibility

If you encounter CUDA version mismatch, modify `pytorch-cuda` in `environment.yml`:

```yaml
# RTX 30/40 Series
- pytorch-cuda=11.8  # or 12.1

# GTX 10/20 Series
- pytorch-cuda=11.7
```

---

## 5. Usage & Execution

### 5.0 Receptive Field Size & Mask Configuration

This project supports circular receptive field masks of different sizes. Users can select the appropriate configuration based on the biological receptive field size of their target neurons.

#### Available Mask Options

| Mask Name | Diameter (pixels) | Equivalent Original RF | Use Case |
|-----------|-------------------|------------------------|----------|
| **Default** | 100 px | 50 px | Standard configuration, suitable for most RGC neurons |
| **small** | 60 px | 30 px | Smaller receptive field, optimized for sparse stimuli |
| **small-55** | 55 px | 27.5 px | Extreme center focus, maximized signal-to-noise ratio |

> **Note**: Due to the 2x spatial scale remapping (512×640 → 1024×1280), dividing the crop size by 2 gives the equivalent receptive field size in the original image.

#### How to Switch Mask Configuration

Additional `small` and `small-55` mask configuration files are packaged in the accompanying archive. Follow these steps to switch configurations based on your biological expectations for target neuron receptive fields:

1. **Extract the required mask file**: Unzip the mask configuration of the corresponding size from the archive
2. **Replace default configuration**: Overwrite the default mask file in the project with the extracted file


### 5.1 Training the Digital Twin Model

**Script**: `train_circular_rf.py`

**Function**: Train a digital twin model capable of predicting responses from 50 retinal ganglion cells.

```bash
# Ensure data file exists
# my_training_data.mat (containing responses, images, behavior)

# Start training
python train_circular_rf.py
```

**Training Configuration**:
- Epochs: 150
- Batch Size: 32
- Optimizer: SAM (Sharpness-Aware Minimization) + Adam
- Learning Rate: 3e-4 (Cosine Annealing)
- Regularization: Laplacian penalty (1e-5) + L1 regularization (1e-5)

**Output**:
- `best_model_rf100_v9_330k.pth` - Trained model weights
- SwanLab logs (training curves viewable on web)

### 5.2 Batch MEI Generation & Analysis

**Script**: `batch_mei_analysis.py`

**Function**: Generate MEIs for all 50 neurons with two initialization strategies.

```bash
# Run batch analysis (requires trained model)
python batch_mei_analysis.py
```

**MEI Generation Strategies**:

| Strategy | Initialization | Characteristics |
|----------|---------------|-----------------|
| **Seeded MEI** | From real best-response image | Faster convergence, results closer to natural images |
| **Random MEI** | From random noise | Explores larger solution space, may discover better stimuli |

**MEI Optimization Parameters**:
- Iterations: 2000
- Learning Rate: 1e-3
- Regularization: TV Loss (1e-6 ~ 1e-5) + L2 (1e-4 ~ 1e-3)
- Spatial Jitter: Enabled for first 1500 iterations

### 5.3 Single Neuron Validation (Optional)

**Script**: `validate_mei_circular.py`

```bash
# Generate MEIs for specific neurons (default: 0 and 10)
python validate_mei_circular.py
```

---

## 6. Outputs & Directory Structure

### Project Directory

```
MEI-Visual-Neuron-Extraction/
│
├── train_circular_rf.py          # Training script
├── validate_mei_circular.py      # Validation script
├── batch_mei_analysis.py         # Batch analysis script
├── environment.yml               # Environment configuration
│
├── best_model_rf100_v9_330k.pth  # Model weights (~1.4 MB)
├── my_training_data.mat          # Training data (~73.5 MB)
│
├── MEI_Atlas_Cir/                # Batch analysis output ⭐
│   ├── Neuron_00_Comparison.png
│   ├── Neuron_01_Comparison.png
│   ├── ...
│   ├── Neuron_49_Comparison.png
│   ├── MEI_Response_Summary.xlsx
│   └── Improvement_Histograms.png
│
└── inception_loop2019-master/    # Core dependency library
    └── staticnet/
        ├── cores.py
        ├── base.py
        └── ...
```

### Output Files Explained

#### 6.1 Neuron Comparison Images (`Neuron_XX_Comparison.png`)

Each image contains three columns:

| Column | Content | Description |
|--------|---------|-------------|
| **Original Best** | Best-response image from real data | Baseline of natural stimuli |
| **Seeded MEI** | MEI initialized from real image | Optimized enhanced stimulus |
| **Random MEI** | MEI initialized from random noise | Model-discovered optimal pattern |

#### 6.2 Data Summary Table (`MEI_Response_Summary.xlsx`)

Excel spreadsheet containing:

| Field | Description |
|-------|-------------|
| `Neuron_ID` | Neuron index (0-49) |
| `Response_Real` | Response to real best image |
| `Response_Seeded` | Response to Seeded MEI |
| `Response_Random` | Response to Random MEI |
| `Ratio_Seeded` | Seeded enhancement = Response_Seeded / Response_Real |
| `Ratio_Random` | Random enhancement = Response_Random / Response_Real |

#### 6.3 Statistical Histograms (`Improvement_Histograms.png`)

Shows MEI enhancement ratio distribution across 50 neurons:
- Left: Seeded MEI enhancement distribution
- Right: Random MEI enhancement distribution
- Red dashed line: Baseline (1.0x)
- Green solid line: Mean enhancement ratio

---

## 7. Key Findings & Results

### 7.1 Complete Elimination of High-Frequency Artifacts

| Comparison | Traditional Methods | Our Method |
|------------|---------------------|------------|
| Checkerboard artifacts | Severe | **Completely eliminated** |
| Salt-and-pepper noise | Noticeable | **Completely eliminated** |
| Edge aliasing | Rough | **Smooth transitions** |
| Overall texture | Digital noise | **Silky smooth** |

### 7.2 Biologically Plausible MEI Structures

Generated MEIs exhibit features consistent with RGC receptive field properties:
- **Gabor waves**: Smooth sinusoidal modulation structures
- **Gaussian envelopes**: Spatial decay with strong center, weak periphery
- **Orientation selectivity**: Clear orientation preferences

### 7.3 Random Seed Validation Experiments

Through Random MEI experiments, we demonstrated:

> **The model can spontaneously discover optimal visual stimuli that elicit responses far exceeding natural images**

This proves that:
1. The digital twin model accurately learned neuronal feature selectivity
2. The low-frequency specialized architecture effectively constrains the search space
3. The MEI method can reveal intrinsic principles of neural coding

### 7.4 Quantitative Results Summary

```
Seeded MEI Average Enhancement: 1.5x ~ 2.0x
Random MEI Average Enhancement: 1.3x ~ 1.8x
Maximum Single Neuron Enhancement: > 3.0x
```

---

## Acknowledgments

This project builds upon the following open-source work:

- **inception_loop** - Walker et al. (2019) Nature Neuroscience
  - Paper: [Inception loops discover what excites neurons most](https://www.nature.com/articles/s41593-019-0517-x)
  - Code: [github.com/sacadena/monkey_mei](https://github.com/sacadena/monkey_mei)

Core dependencies:
- `staticnet/cores.py` - Stacked2dCore implementation
- `staticnet/base.py` - CorePlusReadout2d base class

---

## License

MIT License

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{mei_visual_neuron_extraction,
  title={MEI-Visual-Neuron-Extraction: Low-Frequency Specialized CNN for Retinal Ganglion Cell MEI Generation},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/gorillaleap/dLGN-MEI}}
}
```

---

<p align="center">
  <b>From High-Frequency Noise to Silky Smooth — A Biologically-Inspired MEI Architecture</b>
</p>
