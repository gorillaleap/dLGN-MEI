import numpy as np

# 创建一个简单的模拟模型
class SimpleTestModel:
    def __init__(self, output_neurons=50):
        self.output_neurons = output_neurons

    def forward(self, x, behavior=None):
        # 模拟前向传播
        # x: (batch, channels, height, width)
        batch_size = x.shape[0]

        # 模拟特征提取和读出
        # 这里简化处理：将输入展平并通过线性变换
        flattened = x.reshape(batch_size, -1)
        output = np.random.randn(batch_size, self.output_neurons)  # 模拟输出

        return output

# 测试函数
def test_pipeline():
    print("Starting test for refactored interface...")

    # 创建模型
    model = SimpleTestModel(output_neurons=50)

    # 创建测试输入 (Batch=2, Channels=1, Height=36, Width=64)
    x = np.random.randn(2, 1, 36, 64)

    print(f"Input tensor shape: {x.shape}")

    # 前向传播
    y = model.forward(x)  # 使用新接口，behavior=None

    print(f"Output tensor shape: {y.shape}")
    print(f"Output type: {type(y)}")

    # 验证
    assert isinstance(y, np.ndarray), f"输出类型错误: {type(y)}"
    assert y.shape == (2, 50), f"输出形状错误: 预期 (2, 50)，实际 {y.shape}"

    print("Test passed: Input matrix successfully converted to neuron vector!")

if __name__ == "__main__":
    test_pipeline()