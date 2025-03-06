from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
_image_path = ""
img_PIL = Image.open(_image_path)
img_array = np.array(img_PIL)
writer = SummaryWriter('logs')
writer.add_image('.//', img_array, 2, dataformats="HWC")
# step = 2

for i in range(100):
    writer.add_scalar('y = 2x', 2 * i, i)
writer.close()