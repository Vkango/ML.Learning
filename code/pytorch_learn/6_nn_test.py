import torch
import torch.nn as nn
class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return  output

mod = Module()
x = torch.tensor(1.0)
output = mod(x)
print(output)
"""
# print: 2.0
# Process:
  -> init -> forward (performed by PyTorch)
"""