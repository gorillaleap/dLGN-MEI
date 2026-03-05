[English](README_EN.md) | **简体中文**
# 🧠 MEI Generator - 最大兴奋图像生成引擎

> **数字孪生 × 平原化训练 × 两阶段优化** - 视觉神经元编码的终极解决方案

---

## 📖 项目简介

本项目是一个高性能的视觉神经元 MEI（最大兴奋图像）生成系统，专注于通过深度学习模型解码并生成能够最大化神经元响应的图像模式。基于最新的神经科学发现和深度学习技术，我们实现了从数据预处理到模型训练、再到 MEI 生成的完整流水线。

### 🎯 核心目标
- 生成高质量、具有生物学合理性的 MEI 图像
- 突破真实世界图像的响应天花板
- 构建泛化能力强的神经编码模型

---

## 🚀 核心创新点

### 1. 极简数字孪生架构 (Lite ProDigitalTwin)

```python
# 激进压缩：94% 参数量缩减
原始架构: ~470万参数 → Lite架构: ~30万参数 (缩减94%)

# 抗高频设计：阻断像素级过拟合
self.pool = nn.AdaptiveAvgPool2d((9, 16))  # 强制压缩 36×64 → 9×16

# 强力正则 Readout：构建高鲁棒性解码器
hidden_dim = 64  # 从 256 降至 64
Dropout(0.5) + LayerNorm  # 双重防护
```

**创新亮点：**
- **激进压缩**：通过 AdaptiveAvgPool2d 强制空间压缩，防止模型死记硬背像素级高频噪声
- **维度优化**：隐藏层从 256 降至 64，参数量减少 75%
- **正则化升级**：LayerNorm + Dropout(0.5) 构建强力防护网

### 2. 平原化训练策略 (Loss Landscape Flattening)

```python
# 改良版 SAM 优化器：维度隔离策略
if p.dim() > 1:  # 仅扰动 2D+ 维度权重
    e_w = p.grad * scale.to(p)
    p.add_(e_w)
# LayerNorm 和 Bias 保持不变，免受扰动影响

# 复合正则化体系
Laplacian_Penalty + L1_Sparse_Constraint + Mixup(α=0.2)
```

**技术突破：**
- **SAM 维度隔离**：解决 SAM 优化器导致 LayerNorm 统计量崩溃的经典问题
- **复合正则化**：拉普拉斯惩罚促进空间平滑，L1 约束控制模型复杂度
- **Mixup 增强**：20% 概率特征融合，提升泛化能力

### 📊 训练战报
```plaintext
✅ 成功消除过拟合！
Val Pearson R (0.537) > Train Pearson R (0.480)
🎉 验证集表现首次超越训练集！
```

### 3. 两阶段 MEI 生成引擎 (Two-Stage MEI Optimization)

```python
# 阶段一：筑基期 (0-1500步)
- Adam 优化器 (lr=1e-3)
- ±1 像素 Jittering
- TV Loss + L2 正则化
- 目标：洗去高频噪点，构建平滑感受野

# 阶段二：冲刺期 (1500-2000步)
- 关闭图像抖动
- TV/L2 惩罚大幅降权 (1e-6, 1e-4)
- 全速冲刺最高分
```

**生成成果：**
```plaintext
🔥 10号神经元突破战报
起点 (真实最高): 1.4428
终点 (MEI极限): 1.5339
提升: +6.3% ✅
证明模型具备特征外推能力！
```

---

## 🏗️ 模型架构

### Core 网络结构
```python
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=16, layers=2):
        # 激进压缩策略
        self.stack_2d_core = Stacked2dCore(...)  # 16通道
        self.pool = nn.AdaptiveAvgPool2d((9, 16))  # 输出: 16×9×16=2304
```

### Readout 网络
```python
class ProReadout(nn.Module):
    def __init__(self, in_features=2304, out_features=50):
        # 极简 MLP
        self.mlp = nn.Sequential(
            nn.Linear(2304, 64),      # 94% 压缩
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.5),          # 强力正则
            nn.Linear(64, out_features)
        )
```

### 完整模型
```python
class ProDigitalTwin(nn.Module):
    def __init__(self, n_neurons=50):
        self.core = PooledStacked2dCore(...)      # ~3万参数
        self.model_stack = CorePlusReadout2d(...)  # 总计 ~30万参数
```

---

## 🎯 训练策略详解

### 1. 数据预处理
- **全显存直达**：FastCUDADataset 实现数据预加载
- **标准化**：响应 Z-score 标准化，图像归一化到 [-1, 1]
- **精细化数据增广**：
  - ±2° 旋转
  - ±6px 水平/±4px 垂直平移
  - 高斯噪声 (σ=0.05)
  - ±5% 亮度/对比度扰动

### 2. 损失函数体系
```python
Total_Loss = MSE + α·(1 - Pearson_r) + β·Laplacian + γ·L1 + δ·TV_Loss
```
- **MSE**：基础回归损失
- **Pearson 优化**：直接优化相关系数指标
- **Laplacian 惩罚**：促进空间平滑感受野
- **L1 稀疏约束**：控制模型复杂度
- **TV Loss**：抑制高频噪声

### 3. 优化器配置
```python
# SAM + Adam 双层优化
base_optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
optimizer = SAM(trainable_params, base_optimizer, rho=0.05)  # 针尖磨平关键参数
```

---

## 🎨 MEI 生成引擎

### 两阶段优化流程

#### Phase 1: 筑基期 (Foundation Phase)
```python
# 0-1500步：构建平滑特征空间
for i in range(1500):
    # 动态抖动：防止局部死点
    shift_h = torch.randint(-1, 2, (1,)).item()
    shift_v = torch.randint(-1, 2, (1,)).item()
    jittered_img = torch.roll(img, shifts=(shift_h, shift_v))

    # TV Loss 抑制高频噪声
    tv_reg = 1e-5 * tv_loss(img)

    # 梯度上升
    loss = -current_response + tv_reg
    loss.backward()
    optimizer.step()
```

#### Phase 2: 冲刺期 (Sprint Phase)
```python
# 1500-2000步：全速冲刺最高分
for i in range(1500, 2000):
    # 关闭抖动，锁定目标
    jittered_img = img  # 无抖动

    # 大幅降权，释放分数潜力
    tv_reg = 1e-6 * tv_loss(img)
    l2_reg = 1e-4 * torch.norm(img)

    # 精密优化
    loss = -current_response + tv_reg + l2_reg
```

### 关键技术

#### Jittering 策略
```python
def selective_jitter(img, step):
    if step < 1500:
        # ±1 像素微小抖动
        return torch.roll(img, shifts=(±1, ±1))
    else:
        # 最后 500 步：无抖动，精密收敛
        return img
```

#### TV Loss 实现
```python
def tv_loss(img_tensor):
    # 水平梯度差
    h_diff = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1])
    # 垂直梯度差
    v_diff = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :])
    return h_diff.sum() + v_diff.sum()
```

---

## 🏆 最新战报

### 训练成果
- **模型压缩**：470万 → 30万参数（94%缩减）
- **泛化能力**：Val Pearson R (0.537) > Train Pearson R (0.480)
- **训练效率**：全显存直达，每批次仅占用 ~25MB/4090

### MEI 生成突破
| 神经元 | 种子响应 | MEI响应 | 提升幅度 | 特征类型 |
|--------|----------|---------|----------|----------|
| 0号    | 1.3321   | 1.4012  | +5.2%    | 条纹状   |
| 10号   | 1.4428   | 1.5339  | +6.3%    | 网格状   |

### 关键指标
- **响应天花板突破率**：100% (测试神经元全部突破)
- **图像质量**：无高频噪声，结构清晰，具有生物学合理性
- **计算效率**：2000步生成耗时 < 2分钟 (4090)

---

## 🛠️ 快速开始

### 环境要求
```bash
# 基础环境
conda create -n mei python=3.8
conda activate mei

# 核心依赖
pip install torch torchvision numpy scipy matplotlib h5py swanlab
```

### 数据准备
```bash
# 将数据文件放入项目根目录
my_training_data.mat  # 训练数据集
```

### 训练模型
```bash
python train_pro_high_res_laynorm.py
```

### 生成 MEI
```bash
python validate_mei.py
```

---

## 🔬 技术细节

### SAM 优化器改进
针对传统 SAM 在应用 LayerNorm 时出现统计量崩溃的问题，我们实现了**维度隔离策略**：

```python
# 仅扰动 2D 及以上维度的权重
if p.dim() > 1:  # Conv2d, Linear 等
    e_w = p.grad * scale
    p.add_(e_w)
# LayerNorm 的参数 (1D) 保持不变
```

### Mixup 增强策略
```python
def mixup_augmentation(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x + (1 - lam) * x[perm]
    mixed_y = lam * y + (1 - lam) * y[perm]
    return mixed_x, mixed_y
```

### 梯度标准化（禁用）
在最新版本中，我们彻底禁用了梯度标准化，采用原始梯度进行优化，确保优化的自然收敛。

---

## 📝 待办事项

- [ ] 支持更多神经元的同时生成
- [ ] 实现 3D MEI 生成（视频刺激）
- [ ] 添加更多的正则化策略
- [ ] 优化内存占用，支持更大批量

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 🤝 致谢

感谢 [inception_loop2019](https://github.com/cosmo-emi/inception_loop2019) 项目提供的基础架构支持。

---

**Made with ❤️ for Computational Neuroscience**