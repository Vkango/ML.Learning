# load data to net from dataset
# how to load? -> dataloader
import torchvision.datasets
from torch.utils.data import DataLoader

# prepare test dataset
test_data = torchvision.datasets.CIFAR10("./4_dataset", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
# batch_size = 4
img, target = test_data[0]
print(img.shape)
print(target)
# get_item() -> img, target
# dataloader(batch_size=4)
