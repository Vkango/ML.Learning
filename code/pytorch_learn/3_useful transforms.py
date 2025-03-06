"""
# methods opening image:
## PIL: Image.open()
## tensor: ToTensor()
## OpenCV: cv.imread()

"""
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")
img = Image.open("image/image.png")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)
writer.close()

# Normalize
trans_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normalize = trans_normalize(img_tensor)
