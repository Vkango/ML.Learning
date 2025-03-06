import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
import torch.nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./4_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = Conv2d(3, 6, 3, 1, 0)
    def forward(self, x):
        x = self.conv1(x)
        return x

md = Module()
writer = SummaryWriter("./logs1")
step = 0
for data in dataloader:
    imgs, target = data
    output = md(imgs)
    print(imgs.shape, output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1