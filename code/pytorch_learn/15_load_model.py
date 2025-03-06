"""
Loading methods:
1. torch.load()
2. torchvision.models.vgg16()
   vgg16.load_state_dict()
"""
import torch, torchvision
# method 1
torch.load("vgg16.pth")
# method 2
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_.pth"))
# vgg16 is ready to use.

# [!] TRAPS:
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
    def forward(self, x):
        xx = self.conv1(x)
        return x
# Must define model-structure in use.
# Solution: put models in single Python modules.
