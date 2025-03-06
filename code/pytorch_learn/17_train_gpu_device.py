import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from _model_gpu import Module

"""
Here
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda

train_data = torchvision.datasets.CIFAR10(root="4_dataset", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="4_dataset", train=False, transform=torchvision.transforms.ToTensor())
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"train_data_size: {train_data_size}")
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# build up net
mod = Module()
mod = mod.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 1e-2
optimizer = torch.optim.SGD(mod.parameters(), lr=learning_rate)

# set up parameters
# record training count
total_train_step = 0
# record testing count
total_test_step = 0
# epoch
epoch = 10
# tensorboard
writer = SummaryWriter("./log_train_16")

for i in range(epoch):
    print('---------------------------------------')
    print(f"epoch {i + 1} training is beginning...")
    mod.train()
    for data in train_dataloader:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = mod(img)
        loss = loss_fn(outputs, target)
        # clear grad!! -> Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_test_step)
            print(f"train {total_train_step}, loss: {loss.item()}")
    # eval & test
    mod.eval()
    total_test_loss = 0
    total_test_step += 1
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            outputs = mod(img)
            loss = loss_fn(outputs, target)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy += accuracy
    print(f"total_accuracy: {total_accuracy / test_data_size}")
    print(f"total_test_loss: {total_test_loss}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    torch.save(mod, f"mod_{i}.pth")
    # torch.save(mod.state_dict(), f"mod_{i}.pth")

    print(f"Model saved | mod_{i}.pth")
writer.close()