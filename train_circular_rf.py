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
# 2. 物理视网膜遮罩层 (Batch 级 GPU 加速版)
# ==========================================
class CircularMask(nn.Module):
    def __init__(self, size=100, radius=50):
        super().__init__()
        Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist_from_center = torch.sqrt((X - size/2)**2 + (Y - size/2)**2)
        mask = (dist_from_center <= radius).float()
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0)) 

    def forward(self, x, bg_value=0.0):
        # 🌟 利用 GPU 对整个 Batch 进行极速遮挡计算
        return torch.where(self.mask.bool(), x, torch.ones_like(x) * bg_value)

# ==========================================
# 3. 数据集与辅助函数（CPU 极简版，释放算力）
# ==========================================
def crop_center(img, crop_size):
    _, _, h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]

class CircularRFDataset(Dataset):
    def __init__(self, mat_path, device, rf_diameter=100, is_training=True):
        print(f"[启动] 圆形感受野 V4 (Batch极速版)，目标设备: {device} ...")
        self.rf_diameter = rf_diameter 
        self.rf_radius = rf_diameter // 2 
        self.is_training = is_training
        self.device = device

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

        # Z-score
        self.global_mean = images.mean()
        self.global_std = images.std()
        images_normalized = (images - self.global_mean) / (self.global_std + 1e-8)
        self.images_original = torch.from_numpy(images_normalized).unsqueeze(1).to(device)

        # 临时计算一下背景均值
        y, x = torch.meshgrid(torch.arange(rf_diameter), torch.arange(rf_diameter), indexing='ij')
        center = rf_diameter // 2
        circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= self.rf_radius ** 2
        center_cropped = crop_center(self.images_original, rf_diameter)
        self.bg_value = center_cropped[:, 0][:, ~circular_mask].mean().item()

        self.responses = torch.from_numpy(responses).to(device)
        self.behavior = torch.from_numpy(behavior).to(device)

        r_mean = self.responses.mean(dim=0)
        r_std = self.responses.std(dim=0)
        self.responses = (self.responses - r_mean) / (r_std + 1e-8)

    def __len__(self):
        return len(self.images_original)

    def __getitem__(self, idx):
        img = self.images_original[idx:idx+1]
        
        # 默认居中裁剪的起始坐标
        start_h = (512 - self.rf_diameter) // 2
        start_w = (640 - self.rf_diameter) // 2

        # 🌟 O(1) 极速平移：只改动裁剪框的坐标，绝对不搬运像素！
        if self.is_training:
            h_max = (640 - self.rf_diameter) // 2 - 10 
            v_max = (512 - self.rf_diameter) // 2 - 10 
            
            # 注意符号反转：图片往下 roll，等价于裁剪框往上挪 (-)
            shift_v = torch.randint(-v_max, v_max + 1, (1,)).item()
            shift_h = torch.randint(-h_max, h_max + 1, (1,)).item()
            start_h -= shift_v
            start_w -= shift_h

        # 零拷贝切片，速度起飞
        cropped = img[:, :, start_h:start_h+self.rf_diameter, start_w:start_w+self.rf_diameter]
        return cropped.squeeze(0), self.responses[idx], self.behavior[idx]

def apply_augmentation_v2(imgs, bg_value, is_training=True):
    if not is_training: return imgs
    augmented_imgs = imgs.clone()
    
    # 🌟 1. 旋转：保留！(±2度) 必须带有 fill 参数防止边缘高频黑三角！
    angle = torch.randn(1).item() * 2
    augmented_imgs = TF.rotate(augmented_imgs, angle, fill=[bg_value])
    
    # 🌟 2. 亮度调整：保留！(±5% 波动)
    # brightness_factor = 1.0 + torch.randn(1).item() * 0.05
    # augmented_imgs = augmented_imgs * brightness_factor
    
    # 🌟 3. 对比度调整：保留！(±5% 波动)
    contrast_factor = 1.0 + torch.randn(1).item() * 0.05
    mean = torch.mean(augmented_imgs)
    augmented_imgs = (augmented_imgs - mean) * contrast_factor + mean

    # 4. 极弱高斯底噪：降到 0.01 (主要靠上面的物理扰动防过拟合)
    augmented_imgs = augmented_imgs + torch.randn_like(augmented_imgs) * 0.01
    
    return augmented_imgs

def calculate_pearson_r(preds, targets):
    if preds.dim() == 1: preds = preds.unsqueeze(1)
    if targets.dim() == 1: targets = targets.unsqueeze(1)
    pred_mean = torch.mean(preds, dim=0, keepdim=True)
    target_mean = torch.mean(targets, dim=0, keepdim=True)
    cov = torch.mean((preds - pred_mean) * (targets - target_mean), dim=0)
    pred_std = torch.std(preds, dim=0)
    target_std = torch.std(targets, dim=0)
    denominator = torch.clamp(pred_std * target_std, min=1e-6)
    pearson_r = cov / denominator
    return torch.mean(torch.clamp(pearson_r, -1.0, 1.0))

def laplacian_penalty(weight):
    if weight is None or weight.dim() != 4:
        return torch.tensor(0.0, device=weight.device if weight is not None else 'cuda', requires_grad=True)
    lap_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=weight.dtype, device=weight.device)
    lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(weight.size(1), 1, 1, 1)
    lap_maps = F.conv2d(weight.view(-1, weight.size(2), weight.size(3)).unsqueeze(1), lap_kernel, padding=1)
    return torch.sum(lap_maps ** 2)

# ==========================================
# 4. SAM 优化器实现
# ==========================================
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None: continue
                if p.dim() > 1:
                    e_w = p.grad * scale.to(p)
                    self.state[p]['e_w'] = e_w
                    p.add_(e_w)
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'e_w' in state:
                    p.sub_(state['e_w'])
                    del state['e_w'] 
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.zeros([], device=shared_device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norm += p.grad.data.norm(2).to(shared_device) ** 2
        return norm.sqrt()

# ==========================================
# 5. 模型定义（恢复高分能力的平衡架构 ~300k 参数）
# ==========================================
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels, hidden_channels=40, layers=3, **kwargs):
        super().__init__()
        # 1. 前置物理低通滤波（毛玻璃），不降维，纯模糊
        self.anti_alias_blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # 2. 宏观大核 (100x100 -> 50x50)
        self.stem_macro = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # 3. 中观大核 (50x50 -> 25x25)
        self.stem_meso = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        # 4. 微观核心
        self.stack_2d_core = Stacked2dCore(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            layers=layers,
            **kwargs
        )
        self.layers = layers
        self.hidden_channels = hidden_channels
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.anti_alias_blur(x)
        x = self.stem_macro(x)
        x = self.stem_meso(x)
        full_output = self.stack_2d_core(x)
        last_layer_output = full_output[:, -self.hidden_channels:, :, :]
        return self.pool(last_layer_output)

    @property
    def outchannels(self): return self.hidden_channels
    @property
    def multiple_outputs(self): return False

class ProReadout(nn.Module):
    def __init__(self, in_features, out_features, behavior_dim=2):
        super().__init__()
        self.ln_in = nn.LayerNorm(in_features)
        self.dropout_in = nn.Dropout(0.1)
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_features)
        )
        self.modulator = nn.Sequential(nn.Linear(behavior_dim, 16), nn.ReLU(), nn.Linear(16, out_features))
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        for m in self.modulator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, behavior=None, **kwargs):
        x = x.reshape(x.size(0), -1)
        x = self.ln_in(x)
        x = self.dropout_in(x)
        x = self.mlp(x)
        if behavior is not None: x = x + self.modulator(behavior)
        return x

class ProDigitalTwin(nn.Module):
    def __init__(self, n_neurons=50, bg_value=0.0):
        super().__init__()
        self.bg_value = bg_value
        self.retinal_mask = CircularMask(size=100, radius=50)

        self.core = PooledStacked2dCore(
            input_channels=1,
            hidden_channels=40,  # 设定为 40 通道
            layers=3,
            input_kern=5,
            hidden_kern=5  # 彻底封杀 3x3，核心区全部使用 5x5 的平滑大核
        )

        # 维度精确对齐：40通道 × 7 × 7 = 1960
        readout_dict = nn.ModuleDict({
            'default': ProReadout(in_features=1960, out_features=n_neurons)
        })
        self.model_stack = CorePlusReadout2d(self.core, readout_dict)

    def forward(self, x, behavior=None):
        try:
            x_masked = self.retinal_mask(x, self.bg_value)
        except TypeError:
            x_masked = self.retinal_mask(x)
        return self.model_stack(x_masked, behavior=behavior)

# ==========================================
# 6. 训练主引擎 (修复统计Bug & 释放算力版)
# ==========================================
def run_training():
    swanlab.init(project="MEI-Circular-RF", experiment_name="3080-Unleashed-GlobalR")
    device = torch.device('cuda')

    full_dataset = CircularRFDataset('my_training_data.mat', device, is_training=True)
    train_split = int(0.8 * len(full_dataset))
    
    val_subset = Subset(full_dataset, range(train_split, len(full_dataset)))
    
    train_loader = DataLoader(Subset(full_dataset, range(0, train_split)), batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    model = ProDigitalTwin(bg_value=full_dataset.bg_value).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    base_optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
    optimizer = SAM(trainable_params, base_optimizer=base_optimizer, rho=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.MSELoss()

    # 🌟 修复点 1：彻底松开手刹！把惩罚项降低 100 倍！
    laplacian_weight = 1e-5
    l1_weight = 1e-5

    def get_first_conv_weight(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d): return m.weight
        return None

    print(f"[火箭] 3080 算力全开！全局 R 值统计模式启动！")

    for epoch in range(150):
        model.train()
        full_dataset.is_training = True 
        train_loss = 0.0
        train_rs = []

        for batch_idx, (imgs, resps, behs) in enumerate(train_loader):
            augmented_imgs = apply_augmentation_v2(imgs, bg_value=full_dataset.bg_value, is_training=True)

            # --- SAM Step 1 ---
            preds = model(augmented_imgs, behavior=behs)
            loss = criterion(preds, resps)
            
            w = get_first_conv_weight(model.core)
            lap_t = laplacian_weight * laplacian_penalty(w) if w is not None else torch.tensor(0.0, device=device)
            l1_t = l1_weight * torch.sum(torch.abs(model.model_stack.readout['default'].mlp[0].weight))
            
            (loss + lap_t + l1_t).backward()
            optimizer.first_step(zero_grad=True)

            # --- SAM Step 2 ---
            preds_adv = model(augmented_imgs, behavior=behs)
            loss_adv = criterion(preds_adv, resps)
            
            w_adv = get_first_conv_weight(model.core)
            lap_t_adv = laplacian_weight * laplacian_penalty(w_adv) if w_adv is not None else torch.tensor(0.0, device=device)
            l1_t_adv = l1_weight * torch.sum(torch.abs(model.model_stack.readout['default'].mlp[0].weight))

            (loss_adv + lap_t_adv + l1_t_adv).backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                train_loss += loss.item()
                train_rs.append(calculate_pearson_r(preds, resps).item())

        # --- Validation (全局统计修复版) ---
        model.eval()
        full_dataset.is_training = False 
        val_loss = 0.0
        
        all_val_preds = []
        all_val_resps = []
        
        with torch.no_grad():
            for vi, vr, vb in val_loader:
                vp = model(vi, behavior=vb)
                val_loss += criterion(vp, vr).item()
                
                # 🌟 修复点 2：把所有批次的结果收集起来
                all_val_preds.append(vp)
                all_val_resps.append(vr)

        # 🌟 修复点 3：在整个验证集上统一计算真正的 Global R
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_resps = torch.cat(all_val_resps, dim=0)
        global_val_r = calculate_pearson_r(all_val_preds, all_val_resps).item()

        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_r": np.mean(train_rs),
            "val_r": global_val_r,  # 👈 现在这里是真实的高分了！
            "lr": optimizer.base_optimizer.param_groups[0]['lr']
        }
        scheduler.step()
        swanlab.log(metrics)

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1:3d}/150] Train R: {metrics['train_r']:.4f} | Val R (Global): {metrics['val_r']:.4f}")

    torch.save(model.state_dict(), 'best_model_rf100_v9_330k.pth')
    print("\n[庆祝] 低频特化架构（~330k参数）训练结束，权重已封存为 v9。")

if __name__ == "__main__":
    run_training()