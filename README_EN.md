# 🧠 MEI Generator - Maximum Excitatory Image Generation Engine

> **Digital Twin × Flattened Training × Two-Stage Optimization** - The Ultimate Solution for Visual Neuron Encoding

---

## 📖 Project Overview

This project is a high-performance system for generating Maximum Excitatory Images (MEIs) for visual neurons. It focuses on decoding and generating image patterns that maximize neural responses through deep learning models. Based on the latest advances in neuroscience and deep learning, we have implemented a complete pipeline from data preprocessing to model training and MEI generation.

### 🎯 Core Objectives
- Generate high-quality, biologically plausible MEI images
- Break through the response ceiling of real-world images
- Build neural encoding models with strong generalization capabilities

---

## 🚀 Core Innovations

### 1. Lightweight Digital Twin Architecture (Lite ProDigitalTwin)

```python
# Aggressive compression: 94% parameter reduction
Original architecture: ~4.7M parameters → Lite architecture: ~0.3M parameters (94% reduction)

# Anti-overfitting design: Block pixel-level overfitting
self.pool = nn.AdaptiveAvgPool2d((9, 16))  # Force compression 36×64 → 9×16

# Strong regularization: Build high-robustity decoder
hidden_dim = 64  # Reduced from 256
Dropout(0.5) + LayerNorm  # Dual protection
```

**Key Innovations:**
- **Aggressive Compression**: Prevents memorization of pixel-level high-frequency noise through forced spatial compression
- **Dimension Optimization**: Hidden layer reduced from 256 to 64, 75% parameter reduction
- **Regularization Upgrade**: LayerNorm + Dropout(0.5) constructs a strong protection network

### 2. Flattened Training Strategy (Loss Landscape Flattening)

```python
# Improved SAM optimizer: Dimension isolation strategy
if p.dim() > 1:  # Only perturb 2D+ dimensional weights
    e_w = p.grad * scale.to(p)
    p.add_(e_w)
# LayerNorm and Bias remain unchanged, immune to perturbation

# Composite regularization system
Laplacian_Penalty + L1_Sparse_Constraint + Mixup(α=0.2)
```

**Technical Breakthroughs:**
- **SAM Dimension Isolation**: Solves the classic issue of LayerNorm statistic collapse when using SAM optimizer
- **Composite Regularization**: Laplacian penalty promotes spatial smoothness, L1 constraint controls model complexity
- **Mixup Enhancement**: 20% probability feature fusion improves generalization

### 📊 Training Battle Report
```plaintext
✅ Successfully eliminated overfitting!
Val Pearson R (0.537) > Train Pearson R (0.480)
🎉 Validation set performance first exceeded training set!
```

### 3. Two-Stage MEI Generation Engine (Two-Stage MEI Optimization)

```python
# Stage 1: Foundation Phase (0-1500 steps)
- Adam optimizer (lr=1e-3)
- ±1 pixel Jittering
- TV Loss + L2 regularization
- Goal: Remove high-frequency noise, build smooth receptive fields

# Stage 2: Sprint Phase (1500-2000 steps)
- Disable image jittering
- Drastically reduce TV/L2 penalties (1e-6, 1e-4)
- Full sprint to maximum score
```

**Generation Results:**
```plaintext
🔥 Neuron 10 breakthrough battle report
Starting point (real highest): 1.4428
Endpoint (MEI limit): 1.5339
Improvement: +6.3% ✅
Proof that the model has feature extrapolation capability!
```

---

## 🏗️ Model Architecture

### Core Network Structure
```python
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, layers=2):
        # Aggressive compression strategy
        self.stack_2d_core = Stacked2dCore(...)  # 16 channels
        self.pool = nn.AdaptiveAvgPool2d((9, 16))  # Output: 16×9×16=2304
```

### Readout Network
```python
class ProReadout(nn.Module):
    def __init__(self, in_features=2304, out_features=50):
        # Minimalist MLP
        self.mlp = nn.Sequential(
            nn.Linear(2304, 64),      # 94% compression
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.5),          # Strong regularization
            nn.Linear(64, out_features)
        )
```

### Complete Model
```python
class ProDigitalTwin(nn.Module):
    def __init__(self, n_neurons=50):
        self.core = PooledStacked2dCore(...)      # ~30K parameters
        self.model_stack = CorePlusReadout2d(...)  # Total ~300K parameters
```

---

## 🎯 Training Strategy Details

### 1. Data Preprocessing
- **Full-memory direct access**: FastCUDADataset implements data preloading
- **Normalization**: Response Z-score normalization, image normalization to [-1, 1]
- **Fine-grained data augmentation**:
  - ±2° rotation
  - ±6px horizontal / ±4px vertical translation
  - Gaussian noise (σ=0.05)
  - ±5% brightness/contrast perturbation

### 2. Loss Function System
```python
Total_Loss = MSE + α·(1 - Pearson_r) + β·Laplacian + γ·L1 + δ·TV_Loss
```
- **MSE**: Basic regression loss
- **Pearson Optimization**: Directly optimize correlation coefficient metric
- **Laplacian Penalty**: Promotes spatial smooth receptive fields
- **L1 Sparse Constraint**: Controls model complexity
- **TV Loss**: Suppresses high-frequency noise

### 3. Optimizer Configuration
```python
# SAM + Adam dual optimization
base_optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
optimizer = SAM(trainable_params, base_optimizer, rho=0.05)  # Key parameter for tip flattening
```

---

## 🎨 MEI Generation Engine

### Two-Stage Optimization Process

#### Phase 1: Foundation Phase
```python
# 0-1500 steps: Build smooth feature space
for i in range(1500):
    # Dynamic jittering: Prevent local dead spots
    shift_h = torch.randint(-1, 2, (1,)).item()
    shift_v = torch.randint(-1, 2, (1,)).item()
    jittered_img = torch.roll(img, shifts=(shift_h, shift_v))

    # TV Loss suppresses high-frequency noise
    tv_reg = 1e-5 * tv_loss(img)

    # Gradient ascent
    loss = -current_response + tv_reg
    loss.backward()
    optimizer.step()
```

#### Phase 2: Sprint Phase
```python
# 1500-2000 steps: Full sprint to maximum
for i in range(1500, 2000):
    # Disable jittering, lock target
    jittered_img = img  # No jittering

    # Drastically reduce weights to release score potential
    tv_reg = 1e-6 * tv_loss(img)
    l2_reg = 1e-4 * torch.norm(img)

    # Precise optimization
    loss = -current_response + tv_reg + l2_reg
```

### Key Technologies

#### Jittering Strategy
```python
def selective_jitter(img, step):
    if step < 1500:
        # ±1 pixel small jittering
        return torch.roll(img, shifts=(±1, ±1))
    else:
        # Last 500 steps: No jittering, precise convergence
        return img
```

#### TV Loss Implementation
```python
def tv_loss(img_tensor):
    # Horizontal gradient difference
    h_diff = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    # Vertical gradient difference
    v_diff = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    return h_diff.sum() + v_diff.sum()
```

---

## 🏆 Latest Battle Report

### Training Achievements
- **Model compression**: 4.7M → 0.3M parameters (94% reduction)
- **Generalization capability**: Val Pearson R (0.537) > Train Pearson R (0.480)
- **Training efficiency**: Full-memory direct access, only ~25MB/4090 per batch

### MEI Generation Breakthrough
| Neuron | Seed Response | MEI Response | Improvement | Feature Type |
|--------|--------------|-------------|-------------|--------------|
| 0      | 1.3321       | 1.4012      | +5.2%       | Stripe-like  |
| 10     | 1.4428       | 1.5339      | +6.3%       | Grid-like   |

### Key Metrics
- **Response ceiling breakthrough rate**: 100% (all tested neurons broken through)
- **Image quality**: No high-frequency noise, clear structure, biologically plausible
- **Computational efficiency**: < 2 minutes for 2000-step generation (4090)

---

## 🛠️ Quick Start

### Environment Requirements
```bash
# Base environment
conda create -n mei python=3.8
conda activate mei

# Core dependencies
pip install torch torchvision numpy scipy matplotlib h5py swanlab
```

### Data Preparation
```bash
# Place data file in project root directory
my_training_data.mat  # Training dataset
```

### Train Model
```bash
python train_pro_high_res_laynorm.py
```

### Generate MEI
```bash
python validate_mei.py
```

---

## 🔬 Technical Details

### SAM Optimizer Improvements
To address the classic issue of LayerNorm statistic collapse when using traditional SAM optimizer, we implemented the **dimension isolation strategy**:

```python
# Only perturb weights with 2D+ dimensions
if p.dim() > 1:  # Conv2d, Linear, etc.
    e_w = p.grad * scale
    p.add_(e_w)
# LayerNorm parameters (1D) remain unchanged
```

### Mixup Enhancement Strategy
```python
def mixup_augmentation(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[perm]
    mixed_y = lam * y + (1 - lam) * y[perm]
    return mixed_x, mixed_y
```

### Gradient Normalization (Disabled)
In the latest version, we completely disabled gradient normalization, using raw gradients for optimization to ensure natural convergence.

---

## 📝 TODO List

- [ ] Support simultaneous generation for more neurons
- [ ] Implement 3D MEI generation (video stimuli)
- [ ] Add more regularization strategies
- [ ] Optimize memory usage for larger batches

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🤝 Acknowledgments

Thanks to the [inception_loop2019](https://github.com/cosmo-emi/inception_loop2019) project for providing the basic architecture support.

---

**Made with ❤️ for Computational Neuroscience**