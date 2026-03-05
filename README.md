[**🇺🇸 English**](README_EN.md) | [**🇨🇳 简体中文**](README.md)
# 视觉神经元编码模型与 MEI 生成项目

基于深度学习的视觉神经元编码模型，通过拟合神经元响应并利用反向传播生成最大兴奋图像（MEI, Maximum Excitatory Image）。

---

## 项目简介

本项目旨在构建视觉神经元的深度学习编码模型，通过训练神经网络拟合神经元对视觉刺激的响应规律，并通过反向传播优化生成能够最大化激活特定神经元的图像（MEI）。该技术对于理解视觉系统的感受野特征和神经编码机制具有重要意义。

**核心功能：**
- **神经编码建模**：使用深度学习拟合 50 个视觉神经元的响应模式
- **MEI 生成**：通过梯度优化合成对特定神经元最优刺激的图像
- **行为调制**：整合行为数据对神经元响应进行调制

---

## 环境安装 (Installation)

### 前置要求
- CUDA 11.x 或更高版本（推荐 RTX 4090）
- Conda 包管理器

### 安装步骤

1. **创建并激活 Conda 环境**

```bash
# 从提供的 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate chatgpt
```

2. **补充必要的深度学习依赖**

```bash
# 安装 PyTorch (根据 CUDA 版本选择相应命令)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装实验跟踪工具
pip install swanlab
```

### 主要依赖
- Python 3.11.5
- PyTorch (深度学习框架)
- staticnet (自定义核心库)
- swanlab (实验追踪)
- scipy.io / h5py (数据加载)
- matplotlib (可视化)

---

## 代码架构与逻辑 (Architecture)

### 整体数据流向

```
输入图像 (1×36×64)
    ↓
PooledStacked2dCore (特征提取)
    ├── Stacked2dCore: 堆叠卷积层 (2层, 32通道)
    └── AvgPool2d(2): 空间下采样 (36×64 → 18×32)
    ↓
特征图 (32×18×32) → 展平 → 18,432 维向量
    ↓
ProReadout (读出层)
    ├── LayerNorm / BatchNorm: 归一化
    ├── MLP: 隐藏层 (256维, GELU激活)
    └── Modulator: 行为调制 (2维 → 50维)
    ↓
神经元响应预测 (50 个神经元)
```

### 核心组件说明

#### 1. Core（特征提取骨干）
- **Stacked2dCore**: 位于 [`staticnet/cores.py`](inception_loop2019-master/staticnet/cores.py)，基于堆叠卷积的特征提取网络
  - 支持 skip 连接和正则化
  - 默认配置：2 层卷积，32 通道，5×3 卷积核

#### 2. PooledStacked2dCore（空间压缩包装器）
自定义包装类，实现：
- 只输出最后一层特征（避免所有层拼接）
- 添加 `AvgPool2d(2)` 进行空间下采样
- 输出维度：32 × 18 × 32 = 18,432

#### 3. Base（组合基类）
- **CorePlusReadout2d**: 位于 [`staticnet/base.py`](inception_loop2019-master/staticnet/base.py)
  - 组合 Core + Readout
  - 支持行为调制（Behavior Modulation）
  - 内置非线性激活（softplus）

#### 4. Readout（读出层）
- **ProReadout**: 将特征映射到神经元响应
  - 第一层：LayerNorm/BatchNorm + Dropout(0.3)
  - 隐藏层：Linear(18,432 → 256) + LayerNorm + GELU
  - 输出层：Linear(256 → 50)
  - 行为调制：Linear(2 → 16) + ReLU + Linear(16 → 50)

---

## 运行指南 (Usage)

### Phase 1: 训练模型

训练脚本：[`train_pro_high_res_laynorm.py`](train_pro_high_res_laynorm.py)

```bash
python train_pro_high_res_laynorm.py
```

**输入数据：**
- 文件：`my_training_data.mat`
- 样本数：1,440 个
- 图像尺寸：36 × 64 (单通道灰度)
- 行为维度：2 维
- 神经元数量：50 个

**输出模型：**
- 权重文件：`best_pro_model_ln.pth`
- 训练日志：通过 SwanLab 自动记录

**训练配置：**
- Batch Size: 64 (训练集) / 64 (验证集)
- 优化器：Adam (lr=3e-4, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR (T_max=100)
- 训练轮数：100 epochs

---

### Phase 2: 生成 MEI

MEI 生成脚本：[`validate_mei.py`](validate_mei.py)

```bash
python validate_mei.py
```

**工作流程：**
1. 加载训练好的模型权重 (`best_pro_model_ln.pth`)
2. 扫描真实数据集，找到对目标神经元响应最高的图像作为种子
3. 使用梯度优化（SGD）精修种子图像
4. 应用高斯模糊和边界约束以获得平滑结果

**输出文件：**
- `final_validated_meis.png`: 可视化生成的 MEI 图像

**生成配置：**
- 迭代次数：500
- 优化器：SGD (lr=0.001)
- 正则化：高斯模糊 (sigma=1.0)
- 约束：像素值 [-1.0, 1.0]

---

## 当前状态与已知问题 (Current Status & Known Issues)

### ✅ 正向训练（Phase 1）
**状态：稳定收敛**
- Loss 能够正常下降
- 训练和验证 Loss 曲线收敛良好
- 使用 LayerNorm 架构后训练稳定性提升

### ⚠️ MEI 生成（Phase 2）- 已知问题

**问题描述：**
在 MEI 生成阶段，对输入图像的反向传播优化存在以下问题：
- 生成的图像呈现**高频雪花状/白噪声**（Snowflake noise）
- 未能提取出清晰的感受野特征
- 图像质量不符合生物学预期

**可能原因分析：**
1. **图像正则化不足**：当前的高斯模糊约束可能不足以抑制高频噪声
2. **LayerNorm 梯度特性**：LayerNorm 在反向传播时的梯度可能不利于图像优化
3. **优化策略**：SGD + 梯度符号裁剪可能需要调整
4. **网络架构**：Core 的 BatchNorm 可能在优化过程中引入不稳定性

**当前解决方案尝试：**
- 在 [`validate_mei.py`](validate_mei.py) 中已应用高斯模糊进行实时平滑
- 使用梯度符号（sign）而非原始梯度进行更新
- 添加像素值边界约束

**待探索方向：**
- 引入 TV Loss（Total Variation Loss）增强空间平滑性
- 尝试不同的正则化策略（如 L2 正则化、谱归一化）
- 使用 Adam 替代 SGD 进行图像优化
- 调整学习率调度策略

---

## 项目结构

```
MEI/
├── README.md                           # 本文档
├── environment.yml                      # Conda 环境配置
├── train_pro_high_res_laynorm.py        # 训练脚本
├── validate_mei.py                      # MEI 生成脚本
├── my_training_data.mat                 # 训练数据
├── best_pro_model_ln.pth                # 训练好的模型权重
├── final_validated_meis.png             # 生成的 MEI 图像
├── inception_loop2019-master/           # staticnet 核心库
│   └── staticnet/
│       ├── cores.py                     # Core 定义
│       ├── base.py                      # 基类定义
│       ├── readouts.py                  # Readout 定义
│       └── ...
└── swanlog/                            # SwanLab 训练日志
```

---

## 引用

本项目部分代码基于以下工作：
- Walker et al. 2019 *Nature Neuroscience*: [Inception Loops](https://www.nature.com/articles/s41593-019-0517-x)
- staticnet 库：来自 [inception_loop2019](inception_loop2019-master/)

---

## 许可证

本项目仅供学术研究使用。
