"""
Dataset: ImageNet
Model: vgg16 (pretrained=False)
"""
import torchvision.datasets
import torch.nn
# train_data = torchvision.datasets.ImageNet("./data_imgnet", split="train", download=True, transform=torchvision.transforms.ToTensor())
    # Cannot download: 100+GB
vgg16_false = torchvision.models.vgg16()
# vgg16_true = torchvision.models.vgg16(pretrained=True)

vgg16_false.add_module('add_linear', torch.nn.Linear(1000, 10)) # Add new module down all the modules
vgg16_false.classifier.add_module(torch.nn.Linear(1000, 10)) # Add new module down the classifier (or name)
vgg16_false.classifier[0] = torch.nn.Linear(1000, 10) # Modify the model
print(vgg16_false)