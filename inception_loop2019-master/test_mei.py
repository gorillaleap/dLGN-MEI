import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from mei_optim import deepdraw

# 从之前的测试中复用模型创建逻辑
from staticnet.cores import Stacked2dCore
from staticnet.base import CorePlusReadout2d
from torch.nn import ModuleDict

# 创建一个简单的读取输出层
class SimpleReadout(nn.Module):
    def __init__(self, in_channels, out_neurons):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_neurons)

    def forward(self, x):
        return self.linear(x.mean(dim=(2, 3)))  # 平均池化后通过线性层

# 创建目标模型（只返回第0个神经元的激活值）
class TargetModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output[0, 0]  # 只返回第0个神经元的激活值

# 创建测试模型
def create_test_model():
    # 创建核心
    core = Stacked2dCore(
        input_channels=1,
        hidden_channels=16,
        input_kern=5,
        hidden_kern=3,
        layers=2
    )

    # 创建读取输出层并包装在 ModuleDict 中
    readout_module = SimpleReadout(32, 50)  # 假设核心输出32个通道，50个神经元
    readout = ModuleDict({'default': readout_module})

    # 创建模型
    model = CorePlusReadout2d(core, readout)
    return model

# 测试 MEI 生成
def test_mei_generation():
    print("Starting MEI generation test...")

    # 创建模型
    model = create_test_model()
    model.eval()

    # 创建目标模型（只优化第0个神经元）
    target_model = TargetModel(model)

    # 创建种子图像：灰色背景 + 微弱噪声
    seed_image = np.ones((36, 64, 1)) * 0.5  # 灰色背景
    seed_image += np.random.randn(36, 64, 1) * 0.01  # 微弱高斯噪声

    # 定义优化配置
    octaves = [
        {
            'iter_n': 20,  # 总共20步迭代
            'start_sigma': 1.5,
            'end_sigma': 0.01,
            'start_step_size': 12.0 * 0.25,
            'end_step_size': 0.5 * 0.25,
        },
    ]

    print("Generating MEI for neuron 0...")

    # 执行梯度上升
    mei = deepdraw(
        target_model,
        seed_image,
        octaves=octaves,
        bias=0.4,
        scale=0.224,
        device='cpu',
        step_gain=1.0,
        blur=True,
        jitter=0
    )

    # 转换为 numpy 数组并保存
    mei_array = np.squeeze(mei)  # 移除通道维度
    plt.imshow(mei_array, cmap='gray')
    plt.title("MEI for Neuron 0")
    plt.axis('off')
    plt.savefig('test_mei_output.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    print("MEI generation completed!")
    print("Result saved as test_mei_output.png")

if __name__ == "__main__":
    test_mei_generation()