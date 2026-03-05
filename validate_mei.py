import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

import scipy.io as sio
import h5py
import numpy as np

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
# 1.5 补上全显存数据加载类 (用于对比真实天花板)
# ==========================================
class FastCUDADataset(torch.utils.data.Dataset):
    def __init__(self, mat_path, device, target_h=36, target_w=64):
        print(f"加载真实数据进行基准测试: {mat_path} ...")
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

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.responses[idx], self.behavior[idx]

# ==========================================
# 2. 精简版模型定义（与训练端完全一致）
# ==========================================
class PooledStacked2dCore(nn.Module):
    """
    精简版 Core：
    - hidden_channels: 16 (与训练端一致)
    - 空间池化: AdaptiveAvgPool2d((9, 16))
    - 输出维度: 16 × 9 × 16 = 2,304
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
        # 激进压缩：强制输出到 9×16
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

class ProReadout(nn.Module):
    """
    精简版 Readout：
    - in_features: 2304 (16通道 × 9 × 16)
    - hidden_dim: 64
    - Dropout: 0.5 (强力正则化)
    - 使用 LayerNorm 替代 BatchNorm
    """
    def __init__(self, in_features, out_features, behavior_dim=2):
        super().__init__()
        # 1. 输入归一化与强力失活（LayerNorm）
        self.ln_in = nn.LayerNorm(in_features)
        self.dropout_in = nn.Dropout(0.5)  # 从 0.3 提高到 0.5

        # 2. 精简 MLP：隐藏层从 256 降到 64
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
    精简版完整模型（与训练端完全一致）：
    - Core: hidden_channels=16, AdaptiveAvgPool2d((9,16))
    - Readout: in_features=2304, hidden_dim=64, Dropout=0.5
    """
    def __init__(self, n_neurons=50):
        super().__init__()
        # Core 精简：通道数 16
        self.core = PooledStacked2dCore(
            input_channels=1,
            hidden_channels=16,  # 从 32 降到 16
            layers=2,
            input_kern=5,
            hidden_kern=3
        )

        # Readout 精简：输入维度 16×9×16=2304
        readout_dict = nn.ModuleDict({
            'default': ProReadout(in_features=2304, out_features=n_neurons)  # 从 18432 降到 2304
        })
        self.model_stack = CorePlusReadout2d(self.core, readout_dict)

    def forward(self, x, behavior=None):
        # 绝对不能有 F.softplus，保持线性输出！
        return self.model_stack(x, behavior=behavior)

# ==========================================
# 3. "显微镜级" MEI 生成策略
# ==========================================
def generate_mei(model, neuron_idx=0, iterations=2000, lr=0.1, device='cpu', seed_img=None, seed_behavior=None):
    """
    终极洗图版 MEI 生成器：去雪花，防爆炸，找回真实的生物学特征
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 1. 变量初始化（你的报错就是因为丢了这行）
    img = seed_img.clone().to(device).detach()
    img.requires_grad_(True)
    
    neutral_behavior = seed_behavior.clone().to(device) if seed_behavior is not None else None

    # 2. 优化器与物理边界
    optimizer = torch.optim.Adam([img], lr=lr) 
    V_MAX = 510.0  
    V_MIN = -1.0   

    # 3. TV Loss 辅助函数（局部定义，防止丢失）
    def tv_loss(img_tensor):
        horizontal_diff = torch.abs(img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :-1]).sum()
        vertical_diff = torch.abs(img_tensor[:, :, 1:, :] - img_tensor[:, :, :-1, :]).sum()
        return horizontal_diff + vertical_diff

    print(f"\n🚀 启动洗图模式！目标神经元 {neuron_idx}...")
    
    # 记录初始战力
    with torch.no_grad():
        initial_resp = model(img, behavior=neutral_behavior)[0, neuron_idx].item()
    print(f"🌱 初始种子响应: {initial_resp:.4f}")

    for i in range(iterations):
        optimizer.zero_grad()

        # ==========================================
        # 🚀 阶段切分策略
        # ==========================================
        is_sprint_phase = (i > 1500) # 最后 500 步进入冲刺期

        # 1. 动态抖动 (冲刺期关闭抖动，锁定目标)
        if not is_sprint_phase:
            # 缩小抖动范围，±1 像素就够了
            shift_h = torch.randint(-1, 2, (1,), device=device).item()
            shift_v = torch.randint(-1, 2, (1,), device=device).item()
            jittered_img = torch.roll(img, shifts=(shift_h, shift_v), dims=(2, 3))
        else:
            jittered_img = img

        # 2. 前向传播
        outputs = model(jittered_img, behavior=neutral_behavior)
        current_val = outputs[0, neuron_idx]

        # 3. 动态计算 Loss
        loss = -current_val 
        
        # 冲刺期大幅降低惩罚，释放分数潜力
        tv_weight = 1e-6 if is_sprint_phase else 1e-5
        l2_weight = 1e-4 if is_sprint_phase else 1e-3

        tv_reg = tv_weight * tv_loss(img)
        l2_reg = l2_weight * torch.norm(img)

        total_loss = loss + tv_reg + l2_reg
        total_loss.backward()

        optimizer.step()

        # 4. 软性数值截断
        with torch.no_grad():
            img.clamp_(V_MIN, V_MAX)

        # 5. 打印进度
        if i < 5 or (i + 1) % 100 == 0:
            phase_name = "冲刺" if is_sprint_phase else "筑基"
            print(f"迭代 [{i+1:4d}/{iterations}] {phase_name} | 响应: {current_val.item():.4f} | TV: {tv_reg.item():.4f}")

    # 获取最后无抖动的真实响应
    with torch.no_grad():
        final_resp = model(img, behavior=neutral_behavior)[0, neuron_idx].item()
    
    print(f"🏁 最终响应: {final_resp:.4f} (相比初始 {final_resp - initial_resp:+.4f})")
    
    return img.cpu().detach()

# ==========================================
# 4. 主控流程（自动寻找最优图作为种子）
# ==========================================
def run_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"验证设备就绪: {device}")

    # 加载平原化后的新模型权重
    model = ProDigitalTwin(n_neurons=50).to(device)
    weight_path = 'best_pro_model_lite_flattened.pth' # 使用 flattened 版本

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 成功加载平原化权重: {weight_path}")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件 {weight_path}")
        return

    dataset = FastCUDADataset('my_training_data.mat', device)
    model.eval()

    with torch.no_grad():
        # 获取所有样本的行为数据
        neutral_behs = torch.zeros(len(dataset), 2).to(device)
        all_real_preds = model(dataset.images, behavior=neutral_behs)

    neurons_to_test = [0, 10]
    plt.figure(figsize=(12, 6))

    for idx, n_idx in enumerate(neurons_to_test):
        # 找出真实最优
        best_real_idx = all_real_preds[:, n_idx].argmax().item()
        best_real_img = dataset.images[best_real_idx:best_real_idx+1]
        max_real_response = all_real_preds[best_real_idx, n_idx].item()

        # 保持行为数据一致
        best_real_behavior = dataset.behavior[best_real_idx:best_real_idx+1]

        # 🌟 修正参数名，并使用 1e-3 的精修学习率
        mei_tensor = generate_mei(
            model, neuron_idx=n_idx, iterations=2000, lr=1e-3, 
            device=device, seed_img=best_real_img, seed_behavior=best_real_behavior
        )

        model.eval()
        with torch.no_grad():
            test_img = mei_tensor.to(device)
            if test_img.ndim == 3: test_img = test_img.unsqueeze(0)

            # 使用相同的行为数据进行评估
            final_resp = model(test_img, behavior=best_real_behavior)[0, n_idx].item()

        print(f"\n[{n_idx} 号神经元 - 显微镜级优化战报]")
        print(f"起点 (真实最高): {max_real_response:.4f}")
        print(f"终点 (MEI 极限): {final_resp:.4f}")

        if final_resp > max_real_response:
            improvement = (final_resp - max_real_response) / max_real_response * 100
            print(f"🔥 绝杀！突破 {improvement:.1f}%，成功击穿天花板！")
            if final_resp > 1.5:
                print("🎉 恭喜！成功突破 1.5 大关！")
        else:
            print("还在原地踏步？请检查代码！")

        plt.subplot(1, 2, idx + 1)
        # 🌟 极简画图，直接丢给 matplotlib 去自适应亮度
        display_img = mei_tensor[0, 0].numpy()
        plt.imshow(display_img, cmap='gray') 
        # 删掉 plt.imshow 里的 vmin=0, vmax=1
        plt.title(f"Neuron {n_idx}\nSeed: {max_real_response:.2f} -> MEI: {final_resp:.2f}")
        plt.axis('off')

    output_filename = "final_validated_meis_microscope_optimization.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n显微镜级优化的 MEI 图像已保存为 {output_filename}")

if __name__ == "__main__":
    run_validation()