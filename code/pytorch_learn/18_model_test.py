import torch.nn
import torchvision.transforms
from PIL import Image
image_path = "./image/airplane.png"
image = Image.open(image_path)
image.show()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
image = transform(image)
# MAKE IMAGE TO 32x32
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            # 64x4x4
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("mod_9.pth", weights_only=False)
image = torch.reshape(image, (1, 3, 32, 32)).cuda()
model.eval()
with torch.no_grad():
    output = model(image)
tag = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
print("预测结果:", tag[output.argmax(1)])