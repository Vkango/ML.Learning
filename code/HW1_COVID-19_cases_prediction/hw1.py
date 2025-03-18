import math
import numpy as np
import pandas as pd
import os
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
def same_seed(seed):
    """Keep the train procedure same, because we use the same random number."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Do not let torch use the best model when training.
    np.random.seed(seed) # use same seed to generate random number
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # if cuda available, use GPU to train our model.
def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set.
    Aim: use part of train dataset to evaluate train quality."""
    valid_set_size = int(valid_ratio * len(data_set)) # valid_ratio is less than 1.
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set,
                                        [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
def predict(test_loader, model, device):
    """Make predicton on a test dataset.
    :parameter
    `test_loader`
    """
    model.eval() # set model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets. if none, predict it!
    """
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
    def __getitem__(self, item):
        if self.y is None:
            return self.x[item]
        else:
            return self.x[item], self.y[item]
    def __len__(self):
        return len(self.x)

class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x
def select_feature(train_data, valid_data, test_data, select_all=True):
    """
    Select useful features to perform regression.
    You are given different datas that may influence the prediction,
    but may all of them make difference to the prediction result, or even make the accuracy decline.
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1] # last column is the number of patients.
    raw_x_train, raw_x_valid, raw_x_test = test_data[:, :-1], valid_data[:, :-1], test_data # why does the value not test_data[:, :-1] ?
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]
        # TODO: select suitable feature columns. Try!
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss() # DO NOT modify the loss function.

    # define optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # use TensorBoard and create directory to save model.
    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

same_seed(config['seed'])

train_data, test_data = pd.read_csv('./dataset/covid.train_new.csv').values, pd.read_csv('./dataset/covid.test_un.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}
test_data size: {test_data.shape}""")

x_train, x_valid, x_test, y_train, y_valid = select_feature(train_data, valid_data, test_data, config['select_all'])

print(f"number of features: {x_train.shape[1]}")

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), COVID19Dataset(x_valid, y_valid), COVID19Dataset(x_test)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')