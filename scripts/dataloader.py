"""This file contains the DataLoader class that can be used for training. It has an optional flag to load in either the propriety cubqiue data or the benchmark foresqaure data. Note that currently the cubqiue data is only a portion of the full (and very large) dataset.
"""

import pickle

from torch.utils.data import Dataset, DataLoader
import torch

NUMBER_OF_SPECIAL_INDICES = 2
MAP_TO_INDEX = False

class MobilitySeqDataset(Dataset):
    """Loads in processed cuebiq data from the pickle files"""

    def __init__(self, data_directory, dataset):

        if 'cuebiq' in dataset: # Use cuebiq in the name of the dataset
            # This is the main usecase that has been tested
            with open(data_directory+dataset, 'rb') as f:
                self.all_sequences = pickle.load(f)
            
        elif dataset == 'foresquare':
            with open(data_directory+'foursquare.pickle', 'rb') as f:
                self.all_sequences = pickle.load(f)
        else:
            raise ValueError('Dataset must include "cuebiq" or "foresquare" in the name')

        self.all_locations = sorted(set([l for seq in self.all_sequences for l in seq]))
        self._vocab_size = len(self.all_locations)
        self._max_seq_len = max([len(seq) for seq in self.all_sequences])

        if MAP_TO_INDEX:
            self.location_mapping = dict(zip(self.all_locations, range(NUMBER_OF_SPECIAL_INDICES,len(self.all_locations)+NUMBER_OF_SPECIAL_INDICES)))
            self._vocab_size = len(self.location_mapping)

    if MAP_TO_INDEX:
         # If we want to map the locations to indices (0-|V|) (as is often standard with tokens)
        def __getitem__(self, idx):
            seq = self.all_sequences[idx]
            user_stays_seq = [self.location_mapping[s] for s in seq] # We use location_mapping to map the locations to indices
            return torch.tensor(user_stays_seq)
    else:
        # Simpler and faster
        def __getitem__(self, idx):
            return torch.tensor(self.all_sequences[idx])

    def __len__(self):
        return len(self.all_sequences)


def get_train_test(train_size=0.7, batch_size=1, shuffle=True, data_directory='/mas/projects/privacy-pres-mobility/data/', dataset='24hr_cuebiq_trajectories.pickle', padding=False):
    staysDataset = MobilitySeqDataset(data_directory = data_directory, dataset=dataset)
    train_count = int(train_size*len(staysDataset))
    train_set, val_set = torch.utils.data.random_split(staysDataset, [train_count, len(staysDataset) - train_count])

    if padding: 
        # Padding shouldn't be needed for the 24hr long sequence out of a data chunking pipeline.
        from torch.nn.utils.rnn import pad_sequence
        collate_fn = lambda batch: pad_sequence(batch, batch_first=True, padding_value=0)
    else:
        collate_fn = None
        
    trainStays = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    testStays = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


    return trainStays, testStays
