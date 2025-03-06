import torch
import torch.nn
import torchvision.datasets
from torch.nn import ReLU6, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from urllib3.filepost import writer

# eg ReLU
input = torch.tensor([
    [1, -0.5],
    [-1, 3]
])
output = torch.reshape(input, (-1, 1, 2,2))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU = torch.nn.ReLU(inplace=False)
        self.sigmoid = Sigmoid()
        """
        input = -1
        ReLU(input,inplace = True)
            input = 0
        ReLU(input,inplace = False)
            input = -1
            NEW output = 0
        Difference: save the input or not
        """
    def forward(self, input):
        output = self.sigmoid(input)
        return output

mod = Model()
print(mod(input))

# sigmoid
dataset = torchvision.datasets.CIFAR10("./4_dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)
writer = SummaryWriter("./logs1")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("sigmoid input", imgs, global_step=step)
    output = mod(imgs)
    writer.add_images("sigmoid output", output, global_step=step)
    step += 1