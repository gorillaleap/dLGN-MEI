# MEI-Visual-Neuron-Extraction

**A Low-Frequency Specialized CNN Framework for Retinal Ganglion Cell Maximally Exciting Images (MEI)**

English | [简体中文](README.md)

---

## Table of Contents

- [1. Background & Problem Definition](#1-background--problem-definition)
- [2. Our Solution](#2-our-solution)
- [3. Installation](#3-installation)
- [4. Usage & Execution](#4-usage--execution)
- [5. Outputs & Directory Structure](#5-outputs--directory-structure)
- [6. Key Findings & Results](#6-key-findings--results)
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

## 2. Our Solution

### Design Philosophy

We designed a **low-frequency specialized CNN architecture** that physically strips away the model's ability to capture high-frequency signals, forcing it to learn biologically plausible smooth features.

### Core Technologies

#### 2.1 Multi-Scale Physical Blurring (Anti-Aliasing)

```
Input Image → 3×3 AvgPool → 5×5 AvgPool → 7×7 AvgPool → Feature Extraction
```

Through three consecutive average pooling layers, we apply **physical low-pass filtering** before feature extraction, blocking high-frequency noise from entering the network at the source.

#### 2.2 Large-Scale Convolution Kernel Design

| Layer | Kernel Size | Function |
|-------|-------------|----------|
| Macro Stroking | **11×11** stride=2 | Capture large-scale spatial structures |
| Meso Outlining | **7×7** stride=2 | Extract medium-scale contours |
| Core Feature Region | **5×5** | Fine feature modeling (3×3 completely abandoned) |

Large convolution kernels naturally possess larger receptive fields and stronger smoothing effects, aligning with biological RGC receptive field properties.

#### 2.3 Circular Receptive Field Mask

Simulating the circular visual field of biological retina, we apply a circular mask at the input stage, filling regions outside the circle with background gray values to avoid edge effects from square boundaries.

#### 2.4 Parameter Efficiency

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

## 3. Installation

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

## 4. Usage & Execution

### 4.1 Training the Digital Twin Model

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

### 4.2 Batch MEI Generation & Analysis

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

### 4.3 Single Neuron Validation (Optional)

**Script**: `validate_mei_circular.py`

```bash
# Generate MEIs for specific neurons (default: 0 and 10)
python validate_mei_circular.py
```

---

## 5. Outputs & Directory Structure

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

#### 5.1 Neuron Comparison Images (`Neuron_XX_Comparison.png`)

Each image contains three columns:

| Column | Content | Description |
|--------|---------|-------------|
| **Original Best** | Best-response image from real data | Baseline of natural stimuli |
| **Seeded MEI** | MEI initialized from real image | Optimized enhanced stimulus |
| **Random MEI** | MEI initialized from random noise | Model-discovered optimal pattern |

#### 5.2 Data Summary Table (`MEI_Response_Summary.xlsx`)

Excel spreadsheet containing:

| Field | Description |
|-------|-------------|
| `Neuron_ID` | Neuron index (0-49) |
| `Response_Real` | Response to real best image |
| `Response_Seeded` | Response to Seeded MEI |
| `Response_Random` | Response to Random MEI |
| `Ratio_Seeded` | Seeded enhancement = Response_Seeded / Response_Real |
| `Ratio_Random` | Random enhancement = Response_Random / Response_Real |

#### 5.3 Statistical Histograms (`Improvement_Histograms.png`)

Shows MEI enhancement ratio distribution across 50 neurons:
- Left: Seeded MEI enhancement distribution
- Right: Random MEI enhancement distribution
- Red dashed line: Baseline (1.0x)
- Green solid line: Mean enhancement ratio

---

## 6. Key Findings & Results

### 6.1 Complete Elimination of High-Frequency Artifacts

| Comparison | Traditional Methods | Our Method |
|------------|---------------------|------------|
| Checkerboard artifacts | Severe | **Completely eliminated** |
| Salt-and-pepper noise | Noticeable | **Completely eliminated** |
| Edge aliasing | Rough | **Smooth transitions** |
| Overall texture | Digital noise | **Silky smooth** |

### 6.2 Biologically Plausible MEI Structures

Generated MEIs exhibit features consistent with RGC receptive field properties:
- **Gabor waves**: Smooth sinusoidal modulation structures
- **Gaussian envelopes**: Spatial decay with strong center, weak periphery
- **Orientation selectivity**: Clear orientation preferences

### 6.3 Random Seed Validation Experiments

Through Random MEI experiments, we demonstrated:

> **The model can spontaneously discover optimal visual stimuli that elicit responses far exceeding natural images**

This proves that:
1. The digital twin model accurately learned neuronal feature selectivity
2. The low-frequency specialized architecture effectively constrains the search space
3. The MEI method can reveal intrinsic principles of neural coding

### 6.4 Quantitative Results Summary

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
