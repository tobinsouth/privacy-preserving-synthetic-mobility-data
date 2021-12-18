"""This file contains the DataLoader class that can be used for training. It has an optional flag to load in either the propriety cubqiue data or the benchmark foresqaure data. Note that currently the cubqiue data is only a portion of the full (and very large) dataset.
"""


import pickle

from torch.utils.data import Dataset, DataLoader
import torch

class MobilitySeqDataset(Dataset):
    """Loads in stayz dataset as zipped csv files"""

    def __init__(self, root_dir, dataset='cuebiq'):

        if dataset == 'cuebiq':
            with open(root_dir+'cuebiq.pickle', 'rb') as f:
                self.all_sequences = pickle.load(f)
            with open(root_dir+'geoid_mapping.pickle', 'rb') as f:
                self.geoid_mapping = pickle.load(f)
    
        if dataset == 'foresquare':
            with open(root_dir+'foursquare.pickle', 'rb') as f:
                self.all_sequences = pickle.load(f)
            all_geoids = sorted([g for seq in all_sequences for g,t in seq])
            self.geoid_mapping = dict(zip(all_geoids, range(1,len(all_geoids)+1)))


    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        """Get item from grouped frame"""
        seq = self.all_sequences[idx]
        user_stays_seq = [self.geoid_mapping[s[0]] for s in seq]
        user_stays_seq = torch.tensor(user_stays_seq, dtype=torch.long)
        times = [s[1] for s in seq]
        times = torch.tensor(times, dtype=torch.float)
        return user_stays_seq, times


def get_train_test(train_size=0.7, batch_size=1, shuffle=True, data_directory='/mas/projects/privacy-pres-mobility/data/processed_data/'):
    staysDataset = MobilitySeqDataset(root_dir = data_directory)
    train_count = int(0.8*len(staysDataset)//1)
    train_set, val_set = torch.utils.data.random_split(staysDataset, [train_count, len(staysDataset) - train_count])

    from torch.nn.utils.rnn import pad_sequence
    collate_fn=lambda batch: pad_sequence(batch, batch_first=True, padding_value=0)
    trainStays = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    testStays = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return trainStays, testStays
