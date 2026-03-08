import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import h5py
import numpy as np
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
# 2. 圆形感受野数据加载类（V2 纯净版）
# ==========================================
def crop_center(img, crop_size):
    _, _, h, w = img.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]

class CircularRFDataset(torch.utils.data.Dataset):
    def __init__(self, mat_path, device, rf_diameter=100, is_training=False):
        print(f"加载圆形感受野数据集V2: {mat_path} ...")
        self.rf_diameter = rf_diameter 
        self.rf_radius = rf_diameter // 2 
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

        # Z-score 标准化
        self.global_mean = images.mean()
        self.global_std = images.std()
        images_normalized = (images - self.global_mean) / (self.global_std + 1e-8)
        self.images_original = torch.from_numpy(images_normalized).unsqueeze(1).to(device)

        # 圆形遮罩
        y, x = torch.meshgrid(torch.arange(rf_diameter), torch.arange(rf_diameter), indexing='ij')
        center = rf_diameter // 2
        self.circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= self.rf_radius ** 2
        self.circular_mask = self.circular_mask.to(device)

        # 背景均值
        center_cropped = crop_center(self.images_original, rf_diameter)
        bg_mask = ~self.circular_mask
        bg_values = center_cropped[:, 0][:, bg_mask]
        self.bg_value = bg_values.mean().item()

        self.responses = torch.from_numpy(responses).to(device)
        self.behavior = torch.from_numpy(behavior).to(device)

        r_mean = self.responses.mean(dim=0)
        r_std = self.responses.std(dim=0)
        self.responses = (self.responses - r_mean) / (r_std + 1e-8)

        # 🌟 核心修正：直接保留 260x260 分辨率，绝不降采样到 36x64！
        cropped = crop_center(self.images_original, self.rf_diameter)
        self.images = torch.where(
            self.circular_mask.unsqueeze(0).unsqueeze(0),
            cropped,
            torch.ones_like(cropped) * self.bg_value
        )
        print(f"[参数] 直径={self.rf_diameter}, 最终送入模型尺寸={self.images.shape[2:]}")

    def __len__(self): return len(self.images_original)
    def __getitem__(self, idx): return self.images[idx], self.responses[idx], self.behavior[idx]

# ==========================================
# 物理视网膜遮罩层 (Circular Mask)
# ==========================================
class CircularMask(nn.Module):
    def __init__(self, size=100, radius=50):
        super().__init__()
        Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist_from_center = torch.sqrt((X - size/2)**2 + (Y - size/2)**2)
        mask = (dist_from_center <= radius).float()
        # 注册为 buffer，自动跟随模型到 GPU，不参与梯度更新
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0)) 

    def forward(self, x, bg_value=0.0):
        # 极速遮挡计算
        return torch.where(self.mask.bool(), x, torch.ones_like(x) * bg_value)

# ==========================================
# 同步更新：验证脚本里的 Retina-LGN-V1 仿生架构
# ==========================================
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels, hidden_channels=40, layers=3, **kwargs):
        super().__init__()

        # 1. 毛玻璃抗锯齿
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
        # 4. 微观核心区
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
        self.dropout_in = nn.Dropout(0.5) 
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_features)
        )
        self.modulator = nn.Sequential(
            nn.Linear(behavior_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )
        self._init_weights()

    def _init_weights(self):
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
            hidden_channels=40,  # 强制对齐 40 通道
            layers=3,
            input_kern=5,
            hidden_kern=5    # 彻底封杀 3x3，核心区全用 5x5
        )

        # 强制对齐 Readout 维度: 40通道 × 7 × 7 = 1960
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
# 4. 圆形感受野 MEI 生成器（V2 纯净版）
# ==========================================
def generate_mei(model, neuron_idx=0, iterations=2000, lr=0.1, device='cpu',
                 seed_img=None, seed_behavior=None, circular_mask=None,
                 rf_diameter=100, bg_value=0.0):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    rf_radius = rf_diameter // 2 
    if circular_mask is None:
        y, x = torch.meshgrid(torch.arange(rf_diameter), torch.arange(rf_diameter), indexing='ij')
        center = rf_diameter // 2
        circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= rf_radius ** 2
        circular_mask = circular_mask.to(device)

    # 核心修正：所有的优化和前向传播都在 100x100 下进行
    if seed_img is not None:
    #    if seed_img.shape[-1] != rf_diameter:
    #        img = F.interpolate(seed_img, size=(rf_diameter, rf_diameter), mode='bilinear', align_corners=False)
    #    else:
    #        img = seed_img.clone()
    #    img = img.to(device).detach()
    #    img = TF.gaussian_blur(img, kernel_size=5, sigma=[1.5, 1.5])
    #else:
        # 如果没有种子，从背景灰度开始，加一点点随机扰动
        img = (torch.randn(1, 1, rf_diameter, rf_diameter, device=device) * 0.1 + bg_value)
    
    img.requires_grad_(True)
    neutral_behavior = seed_behavior.clone().to(device) if seed_behavior is not None else None
    optimizer = torch.optim.Adam([img], lr=lr)

    def tv_loss(img_tensor):
        masked_img = img_tensor * circular_mask
        horizontal_diff = torch.abs(masked_img[:, :, :, 1:] - masked_img[:, :, :, :-1]).sum()
        vertical_diff = torch.abs(masked_img[:, :, 1:, :] - masked_img[:, :, :-1, :]).sum()
        return horizontal_diff + vertical_diff

    print(f"\n[启动] V2版MEI生成器！目标神经元 {neuron_idx}...")
    
    with torch.no_grad():
        masked_init = torch.where(circular_mask, img, torch.ones_like(img) * bg_value)
        initial_resp = model(masked_init, behavior=neutral_behavior)[0, neuron_idx].item()
    print(f"[种子] 初始种子响应: {initial_resp:.4f}")

    for i in range(iterations):
        optimizer.zero_grad()
        is_sprint_phase = (i > 1500) 

        if not is_sprint_phase:
            shift_h = torch.randint(-5, 6, (1,), device=device).item()
            shift_v = torch.randint(-5, 6, (1,), device=device).item()
            jittered_img = torch.roll(img, shifts=(shift_v, shift_h), dims=(2, 3))
        else:
            jittered_img = img

        jittered_masked = torch.where(circular_mask, jittered_img, torch.ones_like(jittered_img) * bg_value)

        # 核心修正：直接把 100x100 的图送进去，不缩放！
        outputs = model(jittered_masked, behavior=neutral_behavior)
        current_val = outputs[0, neuron_idx]

        loss = -current_val
        tv_weight = 1e-6 if is_sprint_phase else 1e-5
        l2_weight = 1e-4 if is_sprint_phase else 1e-3

        tv_reg = tv_weight * tv_loss(img)
        l2_reg = l2_weight * torch.norm(img * circular_mask)

        total_loss = loss + tv_reg + l2_reg
        total_loss.backward()

        if img.grad is not None:
            img.grad = img.grad * circular_mask.unsqueeze(0).unsqueeze(0).float()

        optimizer.step()

        # 🌟 7. 后处理：绝对物理截断 + 保持背景
        with torch.no_grad():
            # 【新增核心修复】限制在 Z-score 的合理物理范围内 (例如 -3.5 到 3.5)
            # 这样优化器就无法通过无限加深黑点来作弊了
            img.clamp_(-3.5, 3.5) 
            
            # 圆外强制保持背景灰度
            img.data = torch.where(circular_mask, img.data, torch.ones_like(img.data) * bg_value)

        if i < 5 or (i + 1) % 100 == 0:
            phase_name = "冲刺" if is_sprint_phase else "筑基"
            print(f"迭代 [{i+1:4d}/{iterations}] {phase_name} | 响应: {current_val.item():.4f} | TV: {tv_reg.item():.4f}")

    with torch.no_grad():
        final_masked = torch.where(circular_mask, img, torch.ones_like(img) * bg_value)
        final_resp = model(final_masked, behavior=neutral_behavior)[0, neuron_idx].item()

    print(f"[完成] 最终响应: {final_resp:.4f} (相比初始 {final_resp - initial_resp:+.4f})")
    return img.cpu().detach()

# ==========================================
# 5. MEI 图集生成主流程
# ==========================================
def run_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"验证设备就绪: {device}")

    model = ProDigitalTwin(n_neurons=50).to(device)
    # 根据你刚才训练生成的权重名字进行修改
    weight_path = 'best_model_rf100_v9_330k.pth'

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"成功加载权重: {weight_path}")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return

    dataset = CircularRFDataset('my_training_data.mat', device, is_training=False)
    model.eval()

    with torch.no_grad():
        neutral_behs = torch.zeros(len(dataset), 2).to(device)
        all_real_preds = model(dataset.images, behavior=neutral_behs)

    neurons_to_test = [0, 10]
    plt.figure(figsize=(12, 6))

    for idx, n_idx in enumerate(neurons_to_test):
        best_real_idx = all_real_preds[:, n_idx].argmax().item()
        best_real_img = dataset.images[best_real_idx:best_real_idx+1]
        max_real_response = all_real_preds[best_real_idx, n_idx].item()
        best_real_behavior = dataset.behavior[best_real_idx:best_real_idx+1]

        mei_tensor = generate_mei(
            model,
            neuron_idx=n_idx,
            iterations=2000,
            lr=1e-3,  # 在 Z-score 空间下，可能需要根据情况微调学习率
            device=device,
            seed_img=best_real_img,
            seed_behavior=best_real_behavior,
            circular_mask=dataset.circular_mask,
            rf_diameter=dataset.rf_diameter,
            bg_value=dataset.bg_value
        )

        model.eval()
        with torch.no_grad():
            test_img = mei_tensor.to(device)
            if test_img.ndim == 3: test_img = test_img.unsqueeze(0)
            masked_test = torch.where(
                dataset.circular_mask,
                test_img,
                torch.ones_like(test_img) * dataset.bg_value
            )
            final_resp = model(masked_test, behavior=best_real_behavior)[0, n_idx].item()

        print(f"\n[{n_idx} 号神经元 - V2圆形感受野优化战报]")
        print(f"起点 (真实最高): {max_real_response:.4f}")
        print(f"终点 (MEI 极限): {final_resp:.4f}")

        plt.subplot(1, 2, idx + 1)
        display_img = mei_tensor[0, 0].numpy()
        plt.imshow(display_img, cmap='gray')
        plt.title(f"Neuron {n_idx}\nSeed: {max_real_response:.2f} -> MEI: {final_resp:.2f}")
        plt.axis('off')

    output_filename = "final_validated_meis_circular_rf_v9_330k.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nV9 低频特化版（~330k参数) MEI 图像已保存为 {output_filename}")

if __name__ == "__main__":
    run_validation()