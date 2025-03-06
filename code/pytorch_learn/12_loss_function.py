"""
Targets: answers
Loss = sum(questions_num - correct_answers_num)
Aim: reduce loss
Method: Train
X: 1, 2, 3
Y: 1, 2, 5
# L1loss = (|0| + |0| + |2|) / 3 = 0.6
    L1loss = sum/element_nums

# MSELoss = (0 + 0 + 2^2) / 3 = 1.3333

# Cross-entropy Loss:
Suits for classify problems
eg: classify images
0 - Person; 1 - dog; 2 - cat
-> In: an image
-> Neutral Network
-> output: [0.1, 0.2, 0.3]
! Target: 1 (class = 1, dog)
loss(x, class) = -0.2 + log(exp(0.1) + exp(0.2) + exp(0.3))
               = -x[class] + log(sum(exp(x[j])))
                 ↑ -0.2  +   log(exp(0.1) + exp(0.2) + exp(0.3))
! log: ln
"""
import torch
import torchvision.datasets
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
import torch.nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets,(1, 1, 1, 3))
loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)

# usage
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
for data in dataloader:
    imgs, targets = data
    outputs = mod(imgs)
    res = loss(outputs, targets)

    # provide grad for updating parameters
    # gradient descent
    res.backward()