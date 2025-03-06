"""
Saving methods:
1. torch.save(vgg16, "vgg16_method1.pth")
    <- save model structure & params to a binary file
2. torch.save(vgg16.state_dict()) (Recommended)
    <- save ONLY model params to a Python dictionary

[!] TRAPS:
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
    def forward(self, x):
        xx = self.conv1(x)
        return x

"""
import torch
import torchvision
# method 1
vgg16 = torchvision.models.vgg16()
torch.save(vgg16, "vgg16.pth")

# method 2
torch.save(vgg16.state_dict("vgg16_.pth"))

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
    def forward(self, x):
        xx = self.conv1(x)
        return x

mod = model()
