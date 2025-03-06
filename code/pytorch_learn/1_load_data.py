# data_set / data_label (answer) -> count...
# data_loader -> make batch, provide suited data-set for next nets.
from torch.utils.data import Dataset
from PIL import Image
import os
class my_data(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path) # all img path

    def __getitem__(self, idx):
        # read data by id -> input(image path) + label
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    def __len__(self):
        return len(self.img_path)
root_dir = 'dataset/train'
ants_label_dir = 'ants'
ants_dataset = my_data(root_dir, ants_label_dir)
