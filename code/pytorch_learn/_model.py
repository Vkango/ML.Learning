import torch.nn
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
# if __name__ == '__main__':
#     mod = Module()
#     input = torch.ones((64, 3, 32, 32))
#     output = mod(input)
#     print(output.shape)
