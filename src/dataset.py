'''
This file contains the dataset class for the MoA competition.
'''

import torch
from torch.utils.data import Dataset


class MoADataset(Dataset):
    def __init__(self, feats, targets):
        self.feats = feats
        self.targets = targets

    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.feats)

    def __getitem__(self, idx):
        '''
        Returns the idx-th item of the dataset as Xs and Ys.
        '''
        return {
            'x': torch.tensor(self.feats[idx, :], dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
