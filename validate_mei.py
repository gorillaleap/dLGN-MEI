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
        print(f"📦 加载真实数据进行基准测试: {mat_path} ...")
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
# 2. 必须和训练时完全一致的架构！(一字不差)
# ==========================================
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels, hidden_channels=32, layers=2, **kwargs):
        super().__init__()
        self.stack_2d_core = Stacked2dCore(
            input_channels=input_channels, hidden_channels=hidden_channels, layers=layers, **kwargs
        )
        self.hidden_channels = hidden_channels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
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
        self.bn_in = nn.BatchNorm1d(in_features)
        self.dropout_in = nn.Dropout(0.3) 
        
        hidden_dim = 256 
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_features)
        )
        
        self.modulator = nn.Sequential(
            nn.Linear(behavior_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_features)
        )

    def forward(self, x, behavior=None, **kwargs):
        x = x.reshape(x.size(0), -1)
        x = self.bn_in(x)
        x = self.dropout_in(x)
        x = self.mlp(x)
        if behavior is not None:
            x = x + self.modulator(behavior)
        return x

class ProDigitalTwin(nn.Module):
    def __init__(self, n_neurons=50):
        super().__init__()
        self.core = PooledStacked2dCore(input_channels=1, hidden_channels=32, layers=2, input_kern=5, hidden_kern=3)
        readout_dict = nn.ModuleDict({
            'default': ProReadout(in_features=18432, out_features=n_neurons)
        })
        self.model_stack = CorePlusReadout2d(self.core, readout_dict)

    def forward(self, x, behavior=None):
        # 🌟 绝对不能有 F.softplus，保持线性输出！
        return self.model_stack(x, behavior=behavior)

# ==========================================
# 3. 殿堂级 MEI 生成（引入 TV Loss 生物学平滑约束）
# ==========================================
import torch.nn.functional as F

def generate_mei(model, neuron_idx=0, iterations=300, lr=0.0001, device='cpu', seed_img=None):
    model.eval() 
    
    # 🌟 1. 种子图起步，绝对不动原始数据
    img = seed_img.clone().to(device).detach()
    img.requires_grad_(True) 

    # 放弃 Adam，使用最原始的 SGD（步长更可控）
    optimizer = torch.optim.SGD([img], lr=lr)
    neutral_behavior = torch.zeros(1, 2).to(device)

    # 强力高斯模糊（用于维持图像的“自然感”）
    def gaussian_blur(t, sigma=1.0):
        kernel_size = 3
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).to(device)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        return F.conv2d(t, kernel, padding=1)

    print(f"\n🔬 [极限维稳模式] 正在为神经元 {neuron_idx} 寻找微小增量...")
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # 🌟 2. 核心：在喂给模型前，先强行平滑图像
        # 这样 BatchNorm 看到的一直都是温和的信号
        blurred_img = gaussian_blur(img)
        
        outputs = model(blurred_img, behavior=neutral_behavior)
        current_val = outputs[0, neuron_idx]
        
        # 诊断：看看前 5 步是怎么跌的
        if i < 5:
            print(f"   Step {i}: Score = {current_val.item():.4f}")

        loss = -current_val
        loss.backward()

        # 🌟 3. 梯度更新：使用符号梯度（Sign），只看方向，不看大小
        with torch.no_grad():
            if img.grad is not None:
                img.data.add_(img.grad.sign(), alpha=lr)
                
            # 🌟 4. 实时修剪：每一步都再次平滑图像，不给噪点留任何机会
            img.data = gaussian_blur(img.data)
            img.data.clamp_(-1.0, 1.0)

        if (i + 1) % 100 == 0:
            print(f"迭代 [{i+1:3d}/{iterations}] | 响应: {current_val.item():.4f}")

    return img.detach().cpu()

# ==========================================
# 4. 主控流程 (自动寻找最优图作为种子)
# ==========================================
def run_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"验证设备就绪: {device}")

    model = ProDigitalTwin(n_neurons=50).to(device)
    weight_path = 'best_pro_model.pth' 
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 成功加载训练权重: {weight_path}")
    except FileNotFoundError:
        print(f"❌ 找不到权重，退出。")
        return

    print("正在扫描真实数据集寻找“最强基因”...")
    dataset = FastCUDADataset('my_training_data.mat', device)
    
    model.eval()
    with torch.no_grad():
        neutral_behs = torch.zeros(len(dataset), 2).to(device)
        all_real_preds = model(dataset.images, behavior=neutral_behs)

    neurons_to_test = [0, 10]
    plt.figure(figsize=(12, 6))
    
    for idx, n_idx in enumerate(neurons_to_test):
        # 🌟 1. 找出真实世界里最让它兴奋的那张图的索引
        best_real_idx = all_real_preds[:, n_idx].argmax().item()
        best_real_img = dataset.images[best_real_idx:best_real_idx+1] # 保持 (1, 1, 36, 64) 形状
        max_real_response = all_real_preds[best_real_idx, n_idx].item()
        
        # 🌟 2. 把这张“最强原图”作为种子，喂给优化器精修！
        mei_tensor = generate_mei(
            model, neuron_idx=n_idx, iterations=500, lr=0.001, 
            device=device, seed_img=best_real_img
        )

        model.eval() # 🌟 再次强力确保是评估模式
        with torch.no_grad():
            # 确保输入是 (1, 1, 36, 64) 这种带 Batch 维度的
            test_img = mei_tensor.to(device)
            if test_img.ndim == 3: test_img = test_img.unsqueeze(0)
            
            mei_response = model(test_img, behavior=torch.zeros(1, 2).to(device))[0, n_idx].item()

        print(f"\n[{n_idx} 号神经元 - 终极战报 🏆]")
        print(f"🥈 起点 (真实最高): {max_real_response:.4f}")
        print(f"🥇 终点 (MEI 极限): {mei_response:.4f}")
        
        if mei_response > max_real_response:
            print("✅ 绝杀！从真实图像中破茧而出，成功击穿天花板！")
        else:
            print("⚠️ 还在原地踏步？请检查代码！")

        plt.subplot(1, 2, idx + 1)
        display_img = (mei_tensor[0, 0].numpy() + 1.0) / 2.0 
        plt.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Neuron {n_idx}\nSeed: {max_real_response:.2f} ➡️ MEI: {mei_response:.2f}")
        plt.axis('off')

    output_filename = "final_validated_meis.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n🎉 突破极限的 MEI 图像已保存为 {output_filename}")

if __name__ == "__main__":
    run_validation()