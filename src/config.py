import torch

class CFG:
    n_splits = 5
    seed = 42
    n_epochs = 10
    batch_size = 128
    lr = 1e-3
    num_workers = 4
    input_dir = "../input/"
    output_dir = "../models/"
    device = "cuda" if torch.cuda.is_available() else "cpu"