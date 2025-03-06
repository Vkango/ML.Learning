from PIL import Image
from torchvision import transforms
"""
transform.py - A TOOLKIT
toolkit:
    to_tensor
    resize
    ...
    some images  --> TOOLKIT --> result
    
tensor:
    use transforms.ToTensor solve problems
        1. how to use transforms?
        2. understand tensor type
"""
img_path = ""
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(img)