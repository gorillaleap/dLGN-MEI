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
# 3. 包装类：空间压缩 + 只取最后一层
# ==========================================
class PooledStacked2dCore(nn.Module):
    def __init__(self, input_channels, hidden_channels=32, layers=2, **kwargs):
        super().__init__()
        self.stack_2d_core = Stacked2dCore(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            layers=layers,
            **kwargs
        )
        self.layers = layers
        self.hidden_channels = hidden_channels
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

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
# 4. 模型定义 (已修改为 LayerNorm)
# ==========================================
class ProReadout(nn.Module):
    def __init__(self, in_features, out_features, behavior_dim=2):
        super().__init__()
        # 🌟 修改点 1: BatchNorm1d -> LayerNorm
        self.ln_in = nn.LayerNorm(in_features)
        self.dropout_in = nn.Dropout(0.3) 
        
        hidden_dim = 256 
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            # 🌟 修改点 2: BatchNorm1d -> LayerNorm
            nn.LayerNorm(hidden_dim), 
            nn.GELU(), 
            nn.Dropout(0.3),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        for m in self.modulator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, behavior=None, **kwargs):
        # 展平特征以适配 LayerNorm 和 Linear
        x = x.reshape(x.size(0), -1)
        x = self.ln_in(x)
        x = self.dropout_in(x)
        x = self.mlp(x)
        if behavior is not None:
            x = x + self.modulator(behavior)
        return x

class ProDigitalTwin(nn.Module):
    def __init__(self, n_neurons=50):
        super().__init__()
        self.core = PooledStacked2dCore(
            input_channels=1,
            hidden_channels=32,
            layers=2,
            input_kern=5,
            hidden_kern=3
        )

        for param in self.core.parameters():
            param.requires_grad = False
        params = list(self.core.parameters())
        for p in params[len(params)//2:]:
            p.requires_grad = True

        readout_dict = nn.ModuleDict({
            'default': ProReadout(in_features=18432, out_features=n_neurons)
        })
        self.model_stack = CorePlusReadout2d(self.core, readout_dict)

    def forward(self, x, behavior=None):
        return self.model_stack(x, behavior=behavior)

# ==========================================
# 5. 训练主引擎
# ==========================================
def run_training():
    swanlab.init(project="MEI-Final", experiment_name="4090-LayerNorm-Upgrade")
    device = torch.device('cuda')

    dataset = FastCUDADataset('my_training_data.mat', device)
    train_split = int(0.8 * len(dataset))
    
    train_loader = DataLoader(Subset(dataset, range(0, train_split)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(train_split, len(dataset))), batch_size=64, shuffle=False)

    model = ProDigitalTwin().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"🔥 4090 就绪！使用 LayerNorm 架构。可训练参数: {sum(p.numel() for p in trainable_params)}")
        
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for imgs, resps, behs in train_loader:
            optimizer.zero_grad()
            preds = model(imgs, behavior=behs)
            loss = criterion(preds, resps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vi, vr, vb in val_loader:
                vp = model(vi, behavior=vb)
                val_loss += criterion(vp, vr).item()

        avg_t = train_loss / len(train_loader)
        avg_v = val_loss / len(val_loader)
        
        scheduler.step()

        swanlab.log({
            "train_loss": avg_t,
            "val_loss": avg_v,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch + 1
        })
        print(f"Epoch [{epoch+1:3d}] Train: {avg_t:.4f} | Val: {avg_v:.4f}")
        
    torch.save(model.state_dict(), 'best_pro_model_ln.pth')
    print("✅ 模型已保存为 best_pro_model_ln.pth")

if __name__ == "__main__":
    run_training()