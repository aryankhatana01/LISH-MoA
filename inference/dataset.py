'''
This file contains the dataset class for the inference module.
This dataset won't be having any targets.
So it would just return the feature tensors.
'''

import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, feats):
        '''
        Input:
            feats: pd.DataFrame of shape (num_samples, num_features).

        Returns:
            None
        '''
        self.feats = feats

    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.feats)

    def __getitem__(self, idx):
        '''
        Returns the idx-th item of the dataset as Xs.
        '''
        return {
            'x': torch.tensor(self.feats[idx, :], dtype=torch.float),
        }
