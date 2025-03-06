"""
max_pool_2d

Usage of Pooling
-> Kernel
    Size: 3x3
    Get the max
-> if overflowed
    | | | | |
        | | | | -> save ? ceil_mode
    -> ceil_mode | True: save

Apply of Pooling
-> save image feature
-> reduce data size
"""
import torch
import torch.nn
import torchvision.datasets
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./4_dataset", train=False, download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

input = torch.tensor(data=[
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]], dtype=torch.float32)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

md = Model()
output = md(input)

#--
step = 0
writer = SummaryWriter("logs1")
for data in dataloader:
    imgs, targets = data
    writer.add_images("8 input", imgs, step)
    output = md(imgs)
    writer.add_images("8 output", output, step)
    step += 1
writer.close()