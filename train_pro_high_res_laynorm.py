import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as sio
import h5py
import numpy as np
from pathlib import Path
import swanlab
import torchvision.transforms.functional as TF

# ==========================================
# 1. 环境与 Base 类锁定
# ==========================================
current_dir = Path(__file__).resolve().parent
found_path = Path(r'C:\Users\vipuser\Documents\MEI\inception_loop2019-master')
if not (found_path / 'staticnet').exists():
    found_path = current_dir
sys.path.append(str(found_path))

from staticnet.cores import Stacked2dCore
from staticnet.base import CorePlusReadout2d

# ==========================================
# 2. 全显存直达数据加载类 (FastCUDADataset)
# ==========================================
class FastCUDADataset(Dataset):
    def __init__(self, mat_path, device, target_h=36, target_w=64):
        print(f"🚀 启动全显存直达模式，目标设备: {device} ...")
        try:
            with h5py.File(mat_path, 'r') as f:
                responses = f['responses'][:].astype(np.float32)
                images = f['images'][:].astype(np.float32)
                behavior = f['behavior'][:].astype(np.float32)
        except:
            data = sio.loadmat(mat_path)
            responses = data['responses'].astype(np.float32)
            images = data['images'].astype(np.float32)
            behavior = data['behavior'].astype(np.float32)

        responses = responses.T
        images = np.transpose(images, (2, 1, 0))
        behavior = behavior.T

        self.images = torch.from_numpy(images).unsqueeze(1).to(device)
        self.responses = torch.from_numpy(responses).to(device)
        self.behavior = torch.from_numpy(behavior).to(device)

        r_mean = self.responses.mean(dim=0)
        r_std = self.responses.std(dim=0)
        self.responses = (self.responses - r_mean) / (r_std + 1e-8)

        self.images = F.interpolate(self.images, size=(target_h, target_w), mode='bilinear', align_corners=False)
        self.images = (self.images - 0.5) / 0.5

        mb_size = (self.images.nelement() + self.responses.nelement() + self.behavior.nelement()) * 4 / (1024 * 1024)
        print(f"✅ 数据全部进驻显存完毕！仅占用: {mb_size:.2f} MB / 24576 MB (4090)")

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.responses[idx], self.behavior[idx]

# ==========================================
# 3. 辅助函数定义
# ==========================================
def laplacian_penalty(weight):
    """
    计算卷积核权重的空间二阶导数平方和（拉普拉斯惩罚）

    Args:
        weight: 卷积核权重，形状为 (out_channels, in_channels, kH, kW)

    Returns:
        拉普拉斯惩罚值
    """
    if weight.dim() != 4:
        return torch.tensor(0.0, device=weight.device, requires_grad=True)

    # 拉普拉斯算子
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=weight.dtype, device=weight.device)

    # 为每个输入通道创建拉普拉斯算子
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(weight.size(1), 1, 1, 1)

    # 对每个输入通道计算拉普拉斯
    laplacian_maps = F.conv2d(
        weight.view(-1, weight.size(2), weight.size(3)).unsqueeze(1),
        laplacian_kernel,
        padding=1
    )

    # 返回所有拉普拉斯映射的平方和
    return torch.sum(laplacian_maps ** 2)


def calculate_pearson_r(preds, targets):
    """
    计算预测值与真实响应之间的 Pearson 相关系数

    Args:
        preds: 预测值，形状为 (batch_size, n_neurons)
        targets: 真实响应，形状为 (batch_size, n_neurons)

    Returns:
        平均 Pearson 相关系数
    """
    # 确保输入是 2D 张量
    if preds.dim() == 1:
        preds = preds.unsqueeze(1)
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    # 计算均值
    pred_mean = torch.mean(preds, dim=0, keepdim=True)
    target_mean = torch.mean(targets, dim=0, keepdim=True)

    # 计算协方差
    cov = torch.mean((preds - pred_mean) * (targets - target_mean), dim=0)

    # 计算标准差
    pred_std = torch.std(preds, dim=0)
    target_std = torch.std(targets, dim=0)

    # 计算 Pearson 相关系数
    pearson_r = cov / (pred_std * target_std + 1e-8)

    # 返回所有神经元的平均 Pearson 相关系数
    return torch.mean(pearson_r)


def apply_augmentation(imgs, is_training=True):
    """
    应用精细化的数据增强

    Args:
        imgs: 输入图像，形状为 (batch_size, 1, H, W)
        is_training: 是否在训练模式

    Returns:
        增强后的图像
    """
    if not is_training:
        return imgs

    augmented_imgs = imgs.clone()

    # 1. 微小旋转：±2 度以内
    angle = torch.randn(1).item() * 2  # -2 到 2 度
    augmented_imgs = TF.rotate(augmented_imgs, angle)

    # 2. 多尺度平移：水平 ±6 像素，垂直 ±4 像素
    shift_y = torch.randint(-4, 5, (1,)).item()
    shift_x = torch.randint(-6, 7, (1,)).item()
    augmented_imgs = torch.roll(augmented_imgs, shifts=(shift_y, shift_x), dims=(2, 3))

    # 3. 高斯底噪：std=0.05
    augmented_imgs = augmented_imgs + torch.randn_like(augmented_imgs) * 0.05

    # 4. 亮度与对比度：±5% 范围内的极轻微扰动
    # 亮度调整
    brightness_factor = 1.0 + torch.randn(1).item() * 0.05
    augmented_imgs = augmented_imgs * brightness_factor

    # 对比度调整
    contrast_factor = 1.0 + torch.randn(1).item() * 0.05
    mean = torch.mean(augmented_imgs)
    augmented_imgs = (augmented_imgs - mean) * contrast_factor + mean

    return augmented_imgs


def mixup_augmentation(x, y, alpha=0.2):
    """
    Mixup 数据增强
    随机线性插值两个样本及其标签

    Args:
        x: 输入图像，形状为 (batch_size, C, H, W)
        y: 标签，形状为 (batch_size, n_neurons)
        alpha: mixup 参数，控制插值的强度

    Returns:
        mixed_x: 混合后的图像
        mixed_y: 混合后的标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]

    return mixed_x, mixed_y, lam

# ==========================================
# 4. SAM 优化器实现 (Sharpness-Aware Minimization)
# ==========================================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) 优化器
    论文: https://arxiv.org/abs/2010.01412
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho value: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.defaults = base_optimizer.defaults

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        第一步：计算梯度并找到扰动后的最小点
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 🌟 护盾：只对 2D/3D/4D 权重（Conv, Linear）做扰动
                if p.dim() > 1:
                    e_w = p.grad * scale.to(p)
                    self.state[p]['e_w'] = e_w
                    p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        第二步：使用扰动后的权重计算梯度并更新
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 🌟 终极防卫：安全获取状态字典，只有真正被扰动过的参数才恢复
                state = self.state[p]
                if 'e_w' in state:
                    p.sub_(state['e_w'])
                    del state['e_w']  # 用完即焚，保持显存和状态干净

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        兼容标准优化器接口
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def _grad_norm(self):
        """
        计算梯度的 L2 范数
        """
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.zeros([], device=shared_device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.data.norm(2).to(shared_device) ** 2
        return norm.sqrt()

    def load_state_dict(self, state_dict):
        """
        加载状态字典
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])

    def state_dict(self):
        """
        保存状态字典
        """
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        return state_dict

# ==========================================
# 5. 包装类：激进空间压缩 + 只取最后一层
# ==========================================
class PooledStacked2dCore(nn.Module):
    """
    精简版 Core：
    - hidden_channels: 32 → 16 (减少 50%)
    - 空间池化: AvgPool2d(2) → AdaptiveAvgPool2d((9, 16))
      强制压缩从 36×64 → 9×16，不让模型记忆高频噪声细节
    - 输出维度: 16 × 9 × 16 = 2,304 (原始 18,432)
    """
    def __init__(self, input_channels, hidden_channels=16, layers=2, **kwargs):
        super().__init__()
        self.stack_2d_core = Stacked2dCore(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            layers=layers,
            **kwargs
        )
        self.layers = layers
        self.hidden_channels = hidden_channels
        # 🌟 激进压缩：强制输出到 9×16，不让模型死记硬背像素噪声
        self.pool = nn.AdaptiveAvgPool2d((9, 16))

    def forward(self, x):
        full_output = self.stack_2d_core(x)
        last_layer_output = full_output[:, -self.hidden_channels:, :, :]
        return self.pool(last_layer_output)

    @property
    def outchannels(self):
        return self.hidden_channels

    @property
    def multiple_outputs(self):
        return False

# ==========================================
# 6. 精简版模型定义 (强力正则化)
# ==========================================
class ProReadout(nn.Module):
    """
    精简版 Readout：
    - in_features: 18432 → 2304 (16通道 × 9 × 16)
    - hidden_dim: 256 → 64 (减少 75%)
    - Dropout: 0.3 → 0.5 (强力正则化防过拟合)
    """
    def __init__(self, in_features, out_features, behavior_dim=2):
        super().__init__()
        # 🌟 1. 输入归一化与强力失活
        self.ln_in = nn.LayerNorm(in_features)
        self.dropout_in = nn.Dropout(0.5)  # 从 0.3 提高到 0.5

        # 🌟 2. 精简 MLP：隐藏层从 256 砍到 64
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),  # 从 0.3 提高到 0.5
            nn.Linear(hidden_dim, out_features)
        )

        # 3. 行为调制器保持不变
        self.modulator = nn.Sequential(
            nn.Linear(behavior_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )

        self._init_weights()

    def _init_weights(self):
        """专门针对 GELU/ReLU 架构的 Kaiming 初始化"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.modulator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, behavior=None, **kwargs):
        x = x.reshape(x.size(0), -1)
        x = self.ln_in(x)
        x = self.dropout_in(x)
        x = self.mlp(x)
        if behavior is not None:
            x = x + self.modulator(behavior)
        return x

class ProDigitalTwin(nn.Module):
    """
    精简版完整模型：
    - Core: hidden_channels=16, AdaptiveAvgPool2d((9,16))
    - Readout: in_features=2304, hidden_dim=64, Dropout=0.5
    - 预估总参数量: ~30万 (原始 ~470万，缩减约 94%)
    """
    def __init__(self, n_neurons=50):
        super().__init__()
        # 🌟 Core 精简：通道数 32→16
        self.core = PooledStacked2dCore(
            input_channels=1,
            hidden_channels=16,  # 从 32 降到 16
            layers=2,
            input_kern=5,
            hidden_kern=3
        )

        # 冻结/微调策略：冻结前一半，解冻后一半
        for param in self.core.parameters():
            param.requires_grad = False
        params = list(self.core.parameters())
        for p in params[len(params)//2:]:
            p.requires_grad = True

        # 🌟 Readout 精简：输入维度 16×9×16=2304
        readout_dict = nn.ModuleDict({
            'default': ProReadout(in_features=2304, out_features=n_neurons)  # 从 18432 降到 2304
        })
        self.model_stack = CorePlusReadout2d(self.core, readout_dict)

    def forward(self, x, behavior=None):
        return self.model_stack(x, behavior=behavior)

# ==========================================
# 7. 训练主引擎（平原化训练策略）
# ==========================================
def run_training():
    swanlab.init(project="MEI-Final", experiment_name="4090-LiteModel-Flattening")
    device = torch.device('cuda')

    dataset = FastCUDADataset('my_training_data.mat', device)
    train_split = int(0.8 * len(dataset))

    train_loader = DataLoader(Subset(dataset, range(0, train_split)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(train_split, len(dataset))), batch_size=64, shuffle=False)

    model = ProDigitalTwin().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    base_optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
    optimizer = SAM(trainable_params, base_optimizer=base_optimizer, rho=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"🔥 4090 就绪！SAM 磨平策略启动。")

    laplacian_weight = 1e-3
    l1_weight = 1e-4

    def get_first_conv_weight(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                return m.weight
        return None

    for epoch in range(100):
        model.train()
        train_loss, lap_total, l1_total = 0.0, 0.0, 0.0
        train_rs = []

        for batch_idx, (imgs, resps, behs) in enumerate(train_loader):
            # 1. 增强与 Mixup
            if torch.rand(1).item() < 0.2:
                imgs, resps, _ = mixup_augmentation(imgs, resps, alpha=0.2)
            augmented_imgs = apply_augmentation(imgs, is_training=True)

            # --- SAM 第一步：寻找针尖 ---
            preds = model(augmented_imgs, behavior=behs)
            loss = criterion(preds, resps)
            
            w = get_first_conv_weight(model.core)
            lap_t = laplacian_weight * laplacian_penalty(w) if w is not None else torch.tensor(0.0, device=device)
            l1_t = l1_weight * torch.sum(torch.abs(model.model_stack.readout['default'].mlp[0].weight))

            (loss + lap_t + l1_t).backward()
            optimizer.first_step(zero_grad=True)

            # --- SAM 第二步：磨平针尖 ---
            preds_adv = model(augmented_imgs, behavior=behs)
            loss_adv = criterion(preds_adv, resps)
            
            w_adv = get_first_conv_weight(model.core)
            lap_t_adv = laplacian_weight * laplacian_penalty(w_adv) if w_adv is not None else torch.tensor(0.0, device=device)
            l1_t_adv = l1_weight * torch.sum(torch.abs(model.model_stack.readout['default'].mlp[0].weight))

            (loss_adv + lap_t_adv + l1_t_adv).backward()
            optimizer.second_step(zero_grad=True)

            # 统计
            with torch.no_grad():
                train_loss += loss.item()
                lap_total += lap_t.item()
                l1_total += l1_t.item()
                r = calculate_pearson_r(preds, resps)
                train_rs.append(r.item())
            
            # 🌟 原本这里的 Batch 打印已经彻底删除了，不留任何痕迹

        # --- 验证环节 ---
        model.eval()
        val_loss, val_r = 0.0, 0.0
        with torch.no_grad():
            for vi, vr, vb in val_loader:
                vp = model(vi, behavior=vb)
                val_loss += criterion(vp, vr).item()
                val_r += calculate_pearson_r(vp, vr).item()

        # 计算指标
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_r": np.mean(train_rs),
            "val_r": val_r / len(val_loader),
            "lap": lap_total / len(train_loader),
            "l1": l1_total / len(train_loader),
            "lr": optimizer.base_optimizer.param_groups[0]['lr']
        }

        scheduler.step()
        
        # 🌟 SwanLab 依然每轮记录，方便网页端查看，不影响终端
        swanlab.log(metrics)

        # 🌟 核心修改：每 10 个 Epoch 打印一次
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(f"🚀 [Epoch {epoch+1:3d}/100] Summary | Train R: {metrics['train_r']:.4f} | Val R: {metrics['val_r']:.4f} | Loss: {metrics['val_loss']:.4f}")

    torch.save(model.state_dict(), 'best_pro_model_lite_flattened.pth')
    print("\n🏁 100个Epoch磨平工程圆满结束，最终模型已封存。")

if __name__ == "__main__":
    run_training()