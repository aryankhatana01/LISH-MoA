'''
This is the config file for the inference module.
It wouldn't contain any training related parameters.
'''

import torch


class CFG:
    batch_size = 32
    model_dir = '../models/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_components = 50
    hidden_size = 256
    n_splits = 5
    