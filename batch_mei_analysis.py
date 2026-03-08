"""
批量MEI分析脚本 - 终极验证版
对全部50个神经元进行深度MEI分析，生成独立对比图和统计报告
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.io as sio
import h5py
import pandas as pd

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
# 2. 模型架构定义（多重低通滤波架构）
# ==========================================
class CircularMask(nn.Module):
    def __init__(self, size=100, radius=50):
        super().__init__()
        Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist_from_center = torch.sqrt((X - size/2)**2 + (Y - size/2)**2)
        mask = (dist_from_center <= radius).float()
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x, bg_value=0.0):
        return torch.where(self.mask.bool(), x, torch.ones_like(x) * bg_value)


class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels, hidden_channels=40, layers=3, **kwargs):
        super().__init__()

        # 核心改进：三重多尺度模糊，彻底封杀高频
        self.super_blur = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        )

        self.stem_macro = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.stem_meso = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

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
        x = self.super_blur(x)
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
            hidden_channels=40,
            layers=3,
            input_kern=5,
            hidden_kern=5
        )

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
# 3. 数据加载
# ==========================================
def load_data(mat_path, device, rf_diameter=100):
    """加载数据并预处理"""
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

    # Z-score标准化
    global_mean = images.mean()
    global_std = images.std()
    images_normalized = (images - global_mean) / (global_std + 1e-8)
    images_tensor = torch.from_numpy(images_normalized).unsqueeze(1).to(device)

    # 居中裁剪
    _, _, h, w = images_tensor.shape
    start_h = (h - rf_diameter) // 2
    start_w = (w - rf_diameter) // 2
    cropped = images_tensor[:, :, start_h:start_h+rf_diameter, start_w:start_w+rf_diameter]

    # 圆形遮罩
    y, x = torch.meshgrid(torch.arange(rf_diameter), torch.arange(rf_diameter), indexing='ij')
    center = rf_diameter // 2
    radius = rf_diameter // 2
    circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    circular_mask = circular_mask.to(device)

    # 背景值
    bg_values = cropped[:, 0][:, ~circular_mask.cpu()]
    bg_value = bg_values.mean().item()

    # 应用遮罩
    images_masked = torch.where(
        circular_mask.unsqueeze(0).unsqueeze(0),
        cropped,
        torch.ones_like(cropped) * bg_value
    )

    responses_tensor = torch.from_numpy(responses).to(device)
    behavior_tensor = torch.from_numpy(behavior).to(device)

    # 响应值Z-score
    r_mean = responses_tensor.mean(dim=0)
    r_std = responses_tensor.std(dim=0)
    responses_normalized = (responses_tensor - r_mean) / (r_std + 1e-8)

    return images_masked, responses_normalized, behavior_tensor, circular_mask, bg_value


# ==========================================
# 4. MEI生成器
# ==========================================
def generate_mei(model, neuron_idx, iterations=2000, lr=0.1, device='cpu',
                 seed_img=None, seed_behavior=None, circular_mask=None,
                 rf_diameter=100, bg_value=0.0):
    """生成MEI"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    rf_radius = rf_diameter // 2
    if circular_mask is None:
        y, x = torch.meshgrid(torch.arange(rf_diameter), torch.arange(rf_diameter), indexing='ij')
        center = rf_diameter // 2
        circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= rf_radius ** 2
        circular_mask = circular_mask.to(device)

    # 初始化图像
    if seed_img is not None:
        img = seed_img.clone().to(device).detach()
    else:
        img = torch.randn(1, 1, rf_diameter, rf_diameter, device=device) * 0.1 + bg_value

    img.requires_grad_(True)
    neutral_behavior = seed_behavior.clone().to(device) if seed_behavior is not None else None
    optimizer = torch.optim.Adam([img], lr=lr)

    def tv_loss(img_tensor):
        masked_img = img_tensor * circular_mask
        horizontal_diff = torch.abs(masked_img[:, :, :, 1:] - masked_img[:, :, :, :-1]).sum()
        vertical_diff = torch.abs(masked_img[:, :, 1:, :] - masked_img[:, :, :-1, :]).sum()
        return horizontal_diff + vertical_diff

    for i in range(iterations):
        optimizer.zero_grad()
        is_sprint_phase = (i > 1500)

        # Jitter
        if not is_sprint_phase:
            shift_h = torch.randint(-5, 6, (1,), device=device).item()
            shift_v = torch.randint(-5, 6, (1,), device=device).item()
            jittered_img = torch.roll(img, shifts=(shift_v, shift_h), dims=(2, 3))
        else:
            jittered_img = img

        jittered_masked = torch.where(circular_mask, jittered_img, torch.ones_like(jittered_img) * bg_value)

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

        with torch.no_grad():
            img.clamp_(-3.5, 3.5)
            img.data = torch.where(circular_mask, img.data, torch.ones_like(img.data) * bg_value)

        if (i + 1) % 500 == 0:
            print(f"  迭代 [{i+1:4d}/{iterations}] 响应: {current_val.item():.4f}")

    return img.cpu().detach()


# ==========================================
# 5. 主分析流程
# ==========================================
def run_batch_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 创建输出目录
    output_dir = Path('MEI_Atlas_Cir')
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir.resolve()}")

    # 加载模型
    model = ProDigitalTwin(n_neurons=50).to(device)
    weight_path = 'best_model_rf100_v9_330k.pth'

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"成功加载权重: {weight_path}")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return

    # 加载数据
    print("加载数据...")
    images, responses, behaviors, circular_mask, bg_value = load_data('my_training_data.mat', device)

    # 存储结果
    results = {
        'resp_real': [],
        'resp_seeded': [],
        'resp_random': [],
        'best_real_imgs': [],
        'mei_seeded': [],
        'mei_random': []
    }

    model.eval()
    n_neurons = 50

    # 找出每个神经元的最佳原始刺激
    print("\n查找各神经元的最佳原始刺激...")
    with torch.no_grad():
        neutral_behs = torch.zeros(len(images), 2).to(device)
        all_preds = model(images, behavior=neutral_behs)

    best_indices = all_preds.argmax(dim=0).cpu().numpy()

    # 批量处理每个神经元
    for neuron_idx in range(n_neurons):
        print(f"\n{'='*50}")
        print(f"处理神经元 {neuron_idx}/{n_neurons-1}")
        print(f"{'='*50}")

        best_idx = best_indices[neuron_idx]
        best_real_img = images[best_idx:best_idx+1]
        best_behavior = behaviors[best_idx:best_idx+1]

        # 计算原始响应
        with torch.no_grad():
            resp_real = model(best_real_img, behavior=best_behavior)[0, neuron_idx].item()
        print(f"原始最佳响应: {resp_real:.4f}")

        # 生成 Seeded MEI
        print(f"生成 Seeded MEI...")
        mei_seeded = generate_mei(
            model, neuron_idx, iterations=2000, lr=1e-3, device=device,
            seed_img=best_real_img, seed_behavior=best_behavior,
            circular_mask=circular_mask, rf_diameter=100, bg_value=bg_value
        )

        with torch.no_grad():
            test_seeded = mei_seeded.to(device)
            if test_seeded.ndim == 3: test_seeded = test_seeded.unsqueeze(0)
            masked_seeded = torch.where(circular_mask, test_seeded, torch.ones_like(test_seeded) * bg_value)
            resp_seeded = model(masked_seeded, behavior=best_behavior)[0, neuron_idx].item()
        print(f"Seeded MEI 响应: {resp_seeded:.4f}")

        # 生成 Random MEI
        print(f"生成 Random MEI...")
        mei_random = generate_mei(
            model, neuron_idx, iterations=2000, lr=1e-3, device=device,
            seed_img=None, seed_behavior=best_behavior,
            circular_mask=circular_mask, rf_diameter=100, bg_value=bg_value
        )

        with torch.no_grad():
            test_random = mei_random.to(device)
            if test_random.ndim == 3: test_random = test_random.unsqueeze(0)
            masked_random = torch.where(circular_mask, test_random, torch.ones_like(test_random) * bg_value)
            resp_random = model(masked_random, behavior=best_behavior)[0, neuron_idx].item()
        print(f"Random MEI 响应: {resp_random:.4f}")

        # 存储结果
        results['resp_real'].append(resp_real)
        results['resp_seeded'].append(resp_seeded)
        results['resp_random'].append(resp_random)
        results['best_real_imgs'].append(best_real_img.cpu())
        results['mei_seeded'].append(mei_seeded)
        results['mei_random'].append(mei_random)

    # ==========================================
    # 6. 可视化输出 - 独立图片
    # ==========================================
    print("\n" + "="*50)
    print("生成独立对比图...")
    print("="*50)

    for i in range(n_neurons):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 原始最佳图像
        axes[0].imshow(results['best_real_imgs'][i][0, 0].numpy(), cmap='gray')
        axes[0].set_title(f"Original Best\nResp: {results['resp_real'][i]:.2f}", fontsize=10)
        axes[0].axis('off')

        # Seeded MEI
        axes[1].imshow(results['mei_seeded'][i][0, 0].numpy(), cmap='gray')
        ratio_seeded = results['resp_seeded'][i] / results['resp_real'][i] if results['resp_real'][i] != 0 else 0
        axes[1].set_title(f"Seeded MEI\nResp: {results['resp_seeded'][i]:.2f} ({ratio_seeded:.2f}x)", fontsize=10)
        axes[1].axis('off')

        # Random MEI
        axes[2].imshow(results['mei_random'][i][0, 0].numpy(), cmap='gray')
        ratio_random = results['resp_random'][i] / results['resp_real'][i] if results['resp_real'][i] != 0 else 0
        axes[2].set_title(f"Random MEI\nResp: {results['resp_random'][i]:.2f} ({ratio_random:.2f}x)", fontsize=10)
        axes[2].axis('off')

        # 总标题
        fig.suptitle(f"Neuron {i}: Real: {results['resp_real'][i]:.2f} -> Seeded: {results['resp_seeded'][i]:.2f} -> Random: {results['resp_random'][i]:.2f}",
                     fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / f'Neuron_{i:02d}_Comparison.png', dpi=150, bbox_inches='tight')
        plt.close()  # 释放内存

        if (i + 1) % 10 == 0:
            print(f"  已生成 {i+1}/{n_neurons} 张对比图")

    print(f"所有对比图已保存至: {output_dir}")

    # ==========================================
    # 7. 数据导出 - Excel
    # ==========================================
    print("\n" + "="*50)
    print("导出数据汇总...")
    print("="*50)

    ratios_seeded = np.array([results['resp_seeded'][i] / results['resp_real'][i]
                              if results['resp_real'][i] != 0 else 0 for i in range(n_neurons)])
    ratios_random = np.array([results['resp_random'][i] / results['resp_real'][i]
                              if results['resp_real'][i] != 0 else 0 for i in range(n_neurons)])

    df = pd.DataFrame({
        'Neuron_ID': list(range(n_neurons)),
        'Response_Real': results['resp_real'],
        'Response_Seeded': results['resp_seeded'],
        'Response_Random': results['resp_random'],
        'Ratio_Seeded': ratios_seeded,
        'Ratio_Random': ratios_random
    })

    excel_path = output_dir / 'MEI_Response_Summary.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Excel文件已保存: {excel_path}")

    # ==========================================
    # 8. 统计报表 - 直方图
    # ==========================================
    print("\n" + "="*50)
    print("生成统计直方图...")
    print("="*50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Seeded 提升率分布
    ax1.hist(ratios_seeded, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax1.axvline(x=np.mean(ratios_seeded), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(ratios_seeded):.2f}x')
    ax1.set_xlabel(r'$Ratio_{Seeded} = \frac{Resp_{Seeded}}{Resp_{Real}}$', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Seeded MEI Enhancement Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Random 提升率分布
    ax2.hist(ratios_random, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax2.axvline(x=np.mean(ratios_random), color='green', linestyle='-', linewidth=2,
                label=f'Mean: {np.mean(ratios_random):.2f}x')
    ax2.set_xlabel(r'$Ratio_{Random} = \frac{Resp_{Random}}{Resp_{Real}}$', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Random MEI Enhancement Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    hist_path = output_dir / 'Improvement_Histograms.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"直方图已保存: {hist_path}")

    # ==========================================
    # 9. 统计摘要
    # ==========================================
    print("\n" + "="*50)
    print("统计摘要")
    print("="*50)
    print(f"Seeded MEI 平均提升率: {np.mean(ratios_seeded):.2f}x (±{np.std(ratios_seeded):.2f})")
    print(f"Random MEI 平均提升率: {np.mean(ratios_random):.2f}x (±{np.std(ratios_random):.2f})")
    print(f"Seeded MEI 最大提升率: {np.max(ratios_seeded):.2f}x")
    print(f"Random MEI 最大提升率: {np.max(ratios_random):.2f}x")

    print("\n分析完成！")
    print(f"所有输出文件位于: {output_dir.resolve()}")


if __name__ == "__main__":
    run_batch_analysis()
