"""
CIFAR10 Image Process
-> Input (3@32x32)
-> Convolution (5x5 kernel)
    <- Feature maps (32@32x32)
-> Max-Pooling (2x2 kernel)
    <-(32@16x16)
-> Convolution (5x5 kernel)
    <- Feature maps (32@16x16)
-> Max-Pooling (2x2 kernel)
    <- Feature maps (32@8x8)
-> Convolution (5x5 kernel)
    <- Feature maps (64@8x8)
-> Max-Pooling (2x2 kernel)
    <- Feature maps (64@4x4)
-> Flatten
-> Hidden Units (64)
    <- Outputs 10 (10 Targets)
Max-Pooling can not change the number of channels
Conv can change the number of channels
"""
import torch.nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # """
        # To calc padding: use formula in doc.
        # dilation[0]: 0 is default.
        # 32 + 2 * padding - 4 - 1 = 31 * stride = 31
        # 2 * padding = 31 - 27 = 4
        # padding = 2, stride = 1: stride = 1 is suited.
        # """
        # self.conv1 = Conv2d(3, 32, 5, 1, 2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, 1, 2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, 1, 2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # # Hidden Units
        # """
        # # 1024 = 64x4x4
        # -> Linear Layers
        #     <- Output: Hidden Units (64)
        # -> Linear Layers
        #     <- Output: (10)
        # """
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        # Equals to ↓
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        # Equals to ↓
        x = self.model1(x)
        return x

mod = Model()
print(mod)
input = torch.ones((64, 3, 32, 32))
output = mod(input)
print(output.shape)

# TensorBoard Use
writer = SummaryWriter("logs2")
writer.add_graph(mod, input)
writer.close()