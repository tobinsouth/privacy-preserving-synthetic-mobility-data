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
        from glob import glob
        self.root_dir = root_dir
        self.all_csvs = glob(root_dir+'/*.csv.gz')
        stays = pd.concat([pd.read_csv(csv) for csv in self.all_csvs])

        self.all_users = list(stays['user'].unique())
        self.grouped_users = stays.groupby('user')
        self.user_homes = dict(self.grouped_users['GEOID_home'].unique())
        self.grouped_stays = self.grouped_users['GEOID']
        self.all_geoid = list(set(list(stays['GEOID'].unique()) + [l.item() for l in self.user_homes.values()]))
        self.all_geoid_mapping = dict(zip(self.all_geoid, range(1,len(self.all_geoid)+1)))

        # We could also truncate each time they leave home?

    def __len__(self):
        return len(self.all_users)

    def __getitem__(self, idx):
        """Get item from grouped frame"""
        user = self.all_users[idx]
        user_stays_seq = self.grouped_stays.get_group(user).to_list()
        user_home = self.user_homes[user].item()
        user_stays_seq = [self.all_geoid_mapping[user_home]] + [self.all_geoid_mapping[geoid] for geoid in user_stays_seq]
        user_stays_seq = torch.tensor(user_stays_seq, dtype=torch.long)
        return user_stays_seq


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
device = torch.device("cuda" if use_cuda else "cpu")
# device = 'cpu'

HIDDEN_SIZE = 64
batch_size = 8
num_epochs = 20
dropout = 0

lstm = SentEnc(len(staysDataset.all_geoid)+1, HIDDEN_SIZE, dropout).to(device)
optimizer = torch.optim.Adam(lstm.parameters())
criterion = nn.CrossEntropyLoss()
trainStays = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
testStays = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# Training LSTM next step prediction on sequences
training_losses, test_losses= [], []
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0
    for i, seq_batch in enumerate(trainStays):
        seq_batch = seq_batch.to(device)
        optimizer.zero_grad()
        lstm_out = lstm(seq_batch)
        loss = criterion(lstm_out.transpose(1,-1), seq_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
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

