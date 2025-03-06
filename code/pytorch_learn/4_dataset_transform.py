import torchvision
import tensorboard
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)
train_set = torchvision.datasets.CIFAR10(root="./4_dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./4_dataset", train=False, transform=dataset_transform, download=True)
# img, target = test_set[0]
# print(test_set.classes)
# print(img)
# print(target)
# print(img.show())
writer = SummaryWriter("logs/logs")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()
print(test_set[0])