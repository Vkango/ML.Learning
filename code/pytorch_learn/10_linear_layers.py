"""
Usage of linear layers
"""
import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        output = self.linear1(input)
        return output
mod = model()
dataset = torchvision.datasets.CIFAR10("4_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)
for data in dataloader:
    imgs, targets = data
    # 64x3x32x32
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # 1x1x1x196608 (64*3*32*32=196608)
    output = torch.flatten(imgs) # ↑ equal
    output = mod(output)
    print(output.shape)
