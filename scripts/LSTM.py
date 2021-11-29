"""
Preamble for most code and jupyter notebooks
@author: tobinsouth
@notebook date: 28 Oct 2021
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
import math, string, re, pickle, json, os, sys, datetime, itertools
from collections import Counter
from tqdm import tqdm


from torch.utils.data import Dataset, DataLoader
import torch

class StaysDataset(Dataset):
    """Loads in stayz dataset as zipped csv files"""

    def __init__(self, root_dir):
        import pickle
        from glob import glob
        self.root_dir = root_dir
        self.all_csvs = glob(root_dir+'.each_traj/traj_*.csv')
        with open(self.root_dir +'geoid_mapping.pickle', "rb") as f:
            self.geoid_mapping = pickle.load(f)

    def __len__(self):
        return len(self.all_csvs)

    def __getitem__(self, idx):
        """Get item from grouped frame"""
        file = self.all_csvs[idx]
        data = pd.read_csv(file)
        data['GEOID'] = data['GEOID'].map(self.geoid_mapping).fillna(0).astype(int)
        return torch.as_tensor(data['GEOID'].values)


root_dir = '../data/'
staysDataset = StaysDataset(root_dir)

train_count = int(0.7*len(staysDataset)//1)
train_set, val_set = torch.utils.data.random_split(staysDataset, [train_count, len(staysDataset) - train_count])


from torch.nn.utils.rnn import pad_sequence
collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=0)


# Model definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentEnc(nn.Module):
    def __init__(self, num_locations, hidden_size, dropout=0):
        super(SentEnc, self).__init__()
        self.embedding = nn.Embedding(num_locations, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, dropout=dropout, batch_first=True)
        self.linear =  nn.Linear(hidden_size, num_locations)      

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        loc_space = self.linear(lstm_out)
        return loc_space


# Training
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")
# device = 'cpu'

HIDDEN_SIZE = 256
batch_size = 16
num_epochs = 10
dropout = 0

lstm = SentEnc(len(staysDataset.geoid_mapping)+1, HIDDEN_SIZE, dropout).to(device)
optimizer = torch.optim.Adam(lstm.parameters())
criterion = nn.CrossEntropyLoss()
trainStays = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testStays = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Training LSTM next step prediction on sequences
detailed_training_loss, training_losses, test_losses= [], [], []
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for seq_batch in tqdm(trainStays):
        seq_batch = seq_batch.to(device)
        optimizer.zero_grad()
        lstm_out = lstm(seq_batch)
        loss = criterion(lstm_out.transpose(1,-1), seq_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        detailed_training_loss.append(loss.item())
    test_losses.append(running_loss / len(trainStays))

    # Get validation accuracy
    running_loss = 0.0
    for seq_batch in testStays:
        seq_batch = seq_batch.to(device)
        lstm_out = lstm(seq_batch)
        running_loss += criterion(lstm_out.transpose(1,-1), seq_batch).item()
    test_losses.append(running_loss / len(testStays))


# # Debug
# exit()

# x = lstm.embedding(seq_batch)
# lstm_out = lstm.lstm(x)
# loc_space = lstm.linear(lstm_out)

# lstm_out.shape

