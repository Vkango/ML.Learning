"""
Understandings of torch.reshape
1. The number of elements is NOT changed
    e.g. Before: (2,3) -> 6 elements
         after: (6,) / (3,2) -> keep 6 elements
2. Usage
    Flatten a tensor
        x = torch.tensor([[1,2,3],[4,5,6]])
            shape: (2,3)
        flattened_x = torch.reshape(x, (-1,))
            shape: (6, )
            print: tensor([1,2,3,4,5,6])
    Reorganize a tensor
        x = torch.tensor([1,2,3,4,5,6])
            shape: (6, )
        reshaped_x = torch.reshape(x,(2,3))
            shape: (2,3)
            print: tensor([[1,2,3],
                           [4,5,6]
                          ])
3. auto interfere
    x = torch.tensor([1,2,3,4,5,6])
    reshaped_x = torch.reshape(x,(3,-1))
                -> shape: (3,2) 2 is interfered from the tensor
                -> print: tensor([[1,2],
                                  [3,4],
                                  [5,6]
                                )
"""
import torch
import torch.nn.functional as F
input = torch.tensor(data=[
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]])
# tensor: 5x5 (5,5)
kernel = torch.tensor(data=[
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])
# tensor: 3x3 (3,3)
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(kernel.shape)
output = F.conv2d(input, kernel)
print(output)

# stride
output = F.conv2d(input, kernel, stride=2)
print(output)

# padding -> Use 0 to make a border, width = padding value
output = F.conv2d(input, kernel, padding=1)
print(output)