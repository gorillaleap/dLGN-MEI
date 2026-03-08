# MEI-Visual-Neuron-Extraction

**低频特化 CNN 架构的视网膜神经节细胞最大兴奋图像（MEI）提取框架**

[English](README_EN.md) | 简体中文

---

## 目录

- [1. 背景与问题](#1-背景与问题)
- [2. 解决方案](#2-解决方案)
- [3. 安装指南](#3-安装指南)
- [4. 运行说明](#4-运行说明)
- [5. 输出文件与目录结构](#5-输出文件与目录结构)
- [6. 主要研究成果](#6-主要研究成果)
- [致谢](#致谢)

---

## 1. 背景与问题

### 什么是 MEI？

**最大兴奋图像（Maximally Exciting Image, MEI）** 是指能够使特定神经元��生最大响应的视觉刺激模式。通过 MEI 分析，我们可以逆向解码神经元的特征选择性，理解视觉系统如何编码外界信息。

### 传统方法的痛点

在传统的 MEI 生成流程中，基于梯度优化的方法往往会导致模型陷入**高频局部最优解**。具体表现为：

| 伪影类型 | 视觉特征 | 生物学不合理性 |
|---------|---------|---------------|
| 棋盘格 | 周期性明暗交替 | RGC 感受野不具有周期性结构 |
| 麻点噪点 | 随机高频像素 | 超越了神经元的空间分辨率 |
| 边缘锯齿 | 不连续的锐利边界 | 与 Gabor 滤波器的平滑特性相悖 |

这些**高频锯齿伪影**不仅影响 MEI 的视觉质量，更重要的是**不符合生物视网膜神经节细胞（RGC）的感受野特性**——真实的 RGC 感受野通常呈现为平滑的 Gabor 波纹或高斯包络结构。

> 核心矛盾：模型拥有"作弊"的能力（捕捉任意高频），却缺乏"自律"的约束（遵循生物学先验）。

---

## 2. 解决方案

### 架构设计理念

我们设计了一款**低频特化版 CNN 架构**，从物理层面剥夺模型捕捉高频信号的能力，强制其学习符合生物学规律的平滑特征。

### 核心技术

#### 2.1 多尺度物理模糊（Anti-Aliasing）

```
输入图像 → 3×3 AvgPool → 5×5 AvgPool → 7×7 AvgPool → 特征提取
```

通过三层连续的平均池化叠加，在特征提取前对输入进行**物理低通滤波**，从源头阻断高频噪声进入网络。

#### 2.2 大尺度卷积核设计

| 层级 | 卷积核尺寸 | 功能 |
|-----|-----------|------|
| 宏观排刷层 | **11×11** stride=2 | 捕捉大范围空间结构 |
| 中观勾线层 | **7×7** stride=2 | 提取中等尺度轮廓 |
| 核心特征区 | **5×5** | 精细特征建模（全面弃用 3×3） |

大卷积核天然具有更大的感受野和更强的平滑效果，符合 RGC 感受野的生物学特性。

#### 2.3 圆形感受野遮罩

模拟生物视网膜的圆形视野，在输入端应用圆形遮罩，将圆外区域填充为背景灰度值，避免方形边界的边缘效应。

#### 2.4 参数效率

```
总参数量：≈ 330,000 (330K)
模型文件大小：~1.4 MB
```

在保持高性能的同时，实现了轻量化设计，便于训练和部署。

### 架构流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    输入图像 (100×100)                        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              圆形感受野遮罩 (Circular Mask)                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         多尺度模糊 (3×3 + 5×5 + 7×7 AvgPool)                 │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    宏观排刷 (11×11 Conv, stride=2) → 50×50                   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    中观勾线 (7×7 Conv, stride=2) → 25×25                     │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    Stacked2dCore (5×5 Conv × 3 layers) → 25×25              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│         全局平均池化 → 7×7 = 1960 维特征向量                  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│    ProReadout (MLP + 行为调制) → 50 神经元响应               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 安装指南

### 环境要求

- **操作系统**：Windows 10/11, Linux, macOS
- **GPU**：NVIDIA GPU（推荐 RTX 3080 或更高），支持 CUDA 11.8
- **Conda**：Miniconda 或 Anaconda

### 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/gorillaleap/dLGN-MEI.git
cd dLGN-MEI

# 2. 创建 Conda 环境
conda env create -f environment.yml

# 3. 激活环境
conda activate chatgpt

# 4. 验证 GPU 支持
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

### 核心依赖

| 依赖库 | 用途 |
|-------|------|
| `pytorch` | 深度学习框架 |
| `torchvision` | 图像变换与增强 |
| `numpy` | 数值计算 |
| `scipy` | 科学计算（MAT 文件读取） |
| `h5py` | HDF5 文件处理 |
| `pandas` | 数据分析 |
| `openpyxl` | Excel 文件导出 |
| `matplotlib` | 可视化绘图 |
| `swanlab` | 实验追踪与日志 |

### CUDA 版本适配

如遇 CUDA 版本不匹配，可修改 `environment.yml` 中的 `pytorch-cuda` 版本：

```yaml
# RTX 30/40 系列
- pytorch-cuda=11.8  # 或 12.1

# GTX 10/20 系列
- pytorch-cuda=11.7
```

---

## 4. 运行说明

### 4.1 训练数字孪生模型

**脚本**：`train_circular_rf.py`

**功能**：训练一个能够预测 50 个视网膜神经节细胞响应的数字孪生模型。

```bash
# 确保数据文件存在
# my_training_data.mat（包含 responses, images, behavior）

# 启动训练
python train_circular_rf.py
```

**训练配置**：
- Epochs: 150
- Batch Size: 32
- 优化器: SAM (Sharpness-Aware Minimization) + Adam
- 学习率: 3e-4 (Cosine Annealing)
- 正则化: Laplacian 惩罚 (1e-5) + L1 正则 (1e-5)

**输出**：
- `best_model_rf100_v9_330k.pth` - 训练完成的模型权重
- SwanLab 日志（可在网页端查看训练曲线）

### 4.2 批量 MEI 生成与分析

**脚本**：`batch_mei_analysis.py`

**功能**：对全部 50 个神经元进行 MEI 生成，支持两种初始化策略。

```bash
# 运行批量分析（需要先完成训练）
python batch_mei_analysis.py
```

**MEI 生成策略**：

| 策略 | 初始化方式 | 特点 |
|-----|-----------|------|
| **Seeded MEI** | 从真实最佳响应图像初始化 | 更快收敛，结果更接近自然图像 |
| **Random MEI** | 从随机噪声初始化 | 探索更大的解空间，可能发现更优刺激 |

**MEI 优化参数**：
- 迭代次数: 2000
- 学习率: 1e-3
- 正则化: TV Loss (1e-6 ~ 1e-5) + L2 (1e-4 ~ 1e-3)
- 空间抖动: 前 1500 次迭代启用

### 4.3 单神经元验证（可选）

**脚本**：`validate_mei_circular.py`

```bash
# 对指定神经元（默认 0 和 10）生成 MEI
python validate_mei_circular.py
```

---

## 5. 输出文件与目录结构

### 项目目录

```
MEI-Visual-Neuron-Extraction/
│
├── train_circular_rf.py          # 训练脚本
├── validate_mei_circular.py      # 验证脚本
├── batch_mei_analysis.py         # 批量分析脚本
├── environment.yml               # 环境配置
│
├── best_model_rf100_v9_330k.pth  # 模型权重 (~1.4 MB)
├── my_training_data.mat          # 训练数据 (~73.5 MB)
│
├── MEI_Atlas_Cir/                # 批量分析输出 ⭐
│   ├── Neuron_00_Comparison.png
│   ├── Neuron_01_Comparison.png
│   ├── ...
│   ├── Neuron_49_Comparison.png
│   ├── MEI_Response_Summary.xlsx
│   └── Improvement_Histograms.png
│
└── inception_loop2019-master/    # 核心依赖库
    └── staticnet/
        ├── cores.py
        ├── base.py
        └── ...
```

### 输出文件详解

#### 5.1 神经元对比图 (`Neuron_XX_Comparison.png`)

每张图包含三列对比：

| 列 | 内容 | 说明 |
|---|------|------|
| **Original Best** | 真实数据中的最佳响应图像 | 自然刺激的基线 |
| **Seeded MEI** | 从真实图像初始化生成的 MEI | 优化后的增强刺激 |
| **Random MEI** | 从随机噪声初始化生成的 MEI | 模型自发发现的最优模式 |

#### 5.2 数据汇总表 (`MEI_Response_Summary.xlsx`)

Excel 表格包含以下字段：

| 字段 | 说明 |
|------|------|
| `Neuron_ID` | 神经元编号 (0-49) |
| `Response_Real` | 真实最佳图像的响应值 |
| `Response_Seeded` | Seeded MEI 的响应值 |
| `Response_Random` | Random MEI 的响应值 |
| `Ratio_Seeded` | Seeded 提升率 = Response_Seeded / Response_Real |
| `Ratio_Random` | Random 提升率 = Response_Random / Response_Real |

#### 5.3 统计直方图 (`Improvement_Histograms.png`)

展示 50 个神经元的 MEI 提升率分布：
- 左图：Seeded MEI 提升率分布
- 右图：Random MEI 提升率分布
- 红色虚线：基线 (1.0x)
- 绿色实线：平均提升率

---

## 6. 主要研究成果

### 6.1 高频伪影的彻底消除

| 对比维度 | 传统方法 | 本项目方法 |
|---------|---------|-----------|
| 棋盘格伪影 | 严重 | **完全消除** |
| 麻点噪点 | 明显 | **完全消除** |
| 边缘锯齿 | 粗糙 | **平滑过渡** |
| 整体质感 | 数字噪声 | **丝绸般平滑** |

### 6.2 生物学合理的 MEI 结构

生成的 MEI 呈现出符合 RGC 感受野特性的特征：
- **Gabor 波纹**：平滑的正弦调制结构
- **高斯包络**：中心强、边缘弱的空间衰减
- **朝向选择性**：清晰的朝向偏好

### 6.3 Random Seed 验证实验

通过 Random MEI 实验，我们验证了：

> **模型能够自发地发现使神经元响应远超自然图像的最优视觉刺激模式**

这证明了：
1. 数字孪生模型准确学习了神经元的特征选择性
2. 低频特化架构有效约束了搜索空间
3. MEI 方法能够揭示神经元编码的内在规律

### 6.4 量化结果示例

```
Seeded MEI 平均提升率: 1.5x ~ 2.0x
Random MEI 平均提升率: 1.3x ~ 1.8x
最大单神经元提升: > 3.0x
```

---

## 致谢

本项目基于以下开源工作：

- **inception_loop** - Walker et al. (2019) Nature Neuroscience
  - 论文：[Inception loops discover what excites neurons most](https://www.nature.com/articles/s41593-019-0517-x)
  - 代码：[github.com/sacadena/monkey_mei](https://github.com/sacadena/monkey_mei)

核心依赖：
- `staticnet/cores.py` - Stacked2dCore 实现
- `staticnet/base.py` - CorePlusReadout2d 基类

---

## License

MIT License

---

## Citation

如果您在研究中使用了本项目，请引用：

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
  <b>从高频噪声到丝绸般平滑 —— 生物学启发的 MEI 生成架构</b>
</p>
