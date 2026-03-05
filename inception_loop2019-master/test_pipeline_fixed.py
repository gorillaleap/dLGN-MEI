import torch
import torch.nn as nn
from staticnet.cores import Stacked2dCore
from staticnet.base import CorePlusReadout2d
from staticnet.readouts import PooledReadout

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

    # 创建读取输出层
    readout = PooledReadout()
    readout.neurons = 50  # 设置神经元数量
    readout.in_channels = 32  # 假设核心输出32个通道

    # 创建模型
    model = CorePlusReadout2d(core, readout)
    return model

# 测试函数
def test_pipeline():
    print("Starting test for refactored interface...")

    # 创建模型
    model = create_test_model()
    model.eval()

    # 创建测试输入 (Batch=2, Channels=1, Height=36, Width=64)
    x = torch.randn(2, 1, 36, 64)

    print(f"Input tensor shape: {x.shape}")

    # 前向传播
    y = model(x)  # 使用新接口，behavior=None

    print(f"Output tensor shape: {y.shape}")
    print(f"Output type: {type(y)}")

    # 验证
    assert isinstance(y, torch.Tensor), f"输出类型错误: {type(y)}"
    assert y.shape == (2, 50), f"输出形状错误: 预期 (2, 50)，实际 {y.shape}"

    print("Test passed: Input matrix successfully converted to neuron vector!")

if __name__ == "__main__":
    test_pipeline()