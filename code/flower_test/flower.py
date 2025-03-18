import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.checkpoint import checkpoint
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image
"""Prepare dataset"""
data_dir = './dataset'
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"

"""
PreProcess dataset for train / valid
"""
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize([96, 96]),
            transforms.RandomRotation(45), # from -45 to 45
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.1,
                saturation=0.1,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.025),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize([96, 96]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

"""
load dataset
"""
batch_size = 128
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

"""
category to name
"""
with open("./dataset/cat_to_name.json", 'r') as f:
    cat_to_name = json.load(f)

"""
load_model
"""
model_name = 'resnet'
feature_extract = True # freeze all layers except output-layer. (avoid influencing parameters)
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("Training on CPU")
else:
    print("Training on GPU:CUDA")
device = torch.device('cuda:0' if train_on_gpu else 'cpu')

def set_parameters_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.require_grad = False
            # freeze all layers, do not update parameters

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet50(pretrained=True)
    set_parameters_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    # override fc layer, make out feature fit dataset
    model_ft.fc = nn.Linear(num_ftrs, 102)
    input_size = 64
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

filename = "model.pth"
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)

"""
Optimizer
"""
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1) # Learning rate will reduce as train deeps.
criterion = nn.CrossEntropyLoss()

"""
train model -> output layer
"""
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='model_output.pth'):
    since = time.time()
    best_acc = 0
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"epoch { epoch } / { num_epochs - 1 }")
        print("-" * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # update weights
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        time_elapsed = time.time() - since
        print('time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(f'{ phase } Loss: {epoch_loss} Acc: {epoch_acc}')
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state_dice': model.state_dict(),
                'best_acc': best_model_wts,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, filename)
        if phase == 'valid':
            val_acc_history.append(epoch_acc)
            valid_losses.append(epoch_loss)
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_losses.append(epoch_loss)

        print('Learning Rate: {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        scheduler.step() # lr reduce 10% every 10 epoches
    time_elapsed = time.time() - since
    print("Training completed | {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy | {.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs
train_model(model_ft, dataloaders, criterion, optimizer_ft, 25, "opt.pth")
"""
train front layers
"""
for param in model_ft.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

checkpoint = torch.load(filename)
best_ac = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
train_model(model_ft, dataloaders, criterion, optimizer_ft, 25, "opt.pth")
