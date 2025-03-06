import torchvision.datasets
from torch.nn import CrossEntropyLoss
import torch.nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        x = self.model1(x)
        return x
dataset = torchvision.datasets.CIFAR10("4_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=1)
mod = Model()
loss = CrossEntropyLoss()
optim = torch.optim.SGD(mod.parameters(), lr = 0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = mod(imgs)
        res = loss(outputs, targets)
        optim.zero_grad()
        res.backward()
        optim.step()
        running_loss = running_loss + res
    print(running_loss)
    # 呃呃越训练越傻