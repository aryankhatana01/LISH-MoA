'''
This file contains the model definition.
'''

# import torch
import torch.nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        '''
        Input:
            num_features (int): Number of features in the input.
            num_targets (int): Number of targets in the output.
            hidden_size (int): Number of neurons in the hidden layers.

        Returns:
            None
        '''
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        '''
        Input:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            x (torch.Tensor): Output tensor of shape (batch_size, num_targets).
        '''
        x = self.batch_norm1(x)  # Shape --> (batch_size, num_features)
        x = self.dropout1(x)  # Shape --> (batch_size, num_features)
        x = F.relu(self.dense1(x))  # Shape --> (batch_size, hidden_size)
        print(x.shape)

        x = self.batch_norm2(x)  # Shape --> (batch_size, hidden_size)
        x = self.dropout2(x)  # Shape --> (batch_size, hidden_size)
        x = F.relu(self.dense2(x))  # Shape --> (batch_size, hidden_size)
        print(x.shape)

        x = self.batch_norm3(x)  # Shape --> (batch_size, hidden_size)
        x = self.dropout3(x)  # Shape --> (batch_size, hidden_size)
        x = self.dense3(x)  # Shape --> (batch_size, num_targets)
        print(x.shape)

        return x


'''
if __name__ == '__main__':
    num_features = 800
    num_targets = 206
    batch_size = 32
    hidden_size = 1024
    x = torch.randn(batch_size, num_features)  # Input tensor
    model = Model(num_features, num_targets, hidden_size)
    y = model(x)  # Output tensor
'''
