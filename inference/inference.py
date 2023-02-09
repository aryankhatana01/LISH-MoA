'''
This file contains the inference code for the model.
The inference code is very similar to the validation code in the training file.

We need to first apply the same feature engineering on the test data as we did on the training data.
'''

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from config import CFG
from model import Model
from dataset import TestDataset
from tqdm import tqdm

# To import utils.py from the src folder
import sys
sys.path.append("../")
from src import utils


# Reading the test data
test_features = pd.read_csv('../input/test_features.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

# Applying the feature engineering on the testing data.
test_features = utils.PCA_g(test_features, n_components=CFG.n_components)  # PCA for genes
test_features = utils.PCA_c(test_features, n_components=CFG.n_components)  # PCA for cells

test = test_features.drop('cp_type', axis=1)  # Dropping the cp_type column because it is not needed anymore
test['cp_time'] = test['cp_time'].astype(str)  # To fix a warning
test['cp_dose'] = test['cp_dose'].astype(str)  # To fix a warning
test = pd.get_dummies(test, columns=['cp_time', 'cp_dose'])  # One hot encoding the cp_time and cp_dose columns

feature_cols = [c for c in test.columns if c not in ['sig_id']]  # Getting the feature columns

x_test = test[feature_cols].values  # The numpy array of the features


def inference_fn(model, data_loader, device):
    """
    Infers on the model.

    Input:
        model: nn.Module object.
        data_loader: torch.utils.data.DataLoader object.
        device: torch.device object.

    Returns:
        preds: np.ndarray of shape (num_samples, num_targets).
    """
    model.eval()  # Set the model to evaluation mode.
    preds = []  # Initialize the list of predictions.

    with torch.no_grad():  # Disable gradient calculation for computation efficiency.
        for data in tqdm(data_loader, total=len(data_loader)):
            inputs = data["x"].to(device)
            outputs = model(inputs)  # Pass the inputs through the model.
            preds.append(
                outputs.sigmoid().detach().cpu().numpy()
            )  # Append the predictions to the list.

    preds = np.concatenate(preds)  # Concatenate the predictions.

    return preds


def infer_single_fold(fold):
    '''
    Infers on a single fold.

    Input:
        fold: int. The fold number.

    Returns:
        preds: np.ndarray of shape (num_samples, num_targets).
    '''
    # Create the model object.
    model = Model(
        num_features=len(feature_cols),
        num_targets=206,
        hidden_size=CFG.hidden_size
    )
    model.to(CFG.device)  # Move the model to the device.

    # Load the model's state dictionary.
    model.load_state_dict(torch.load(CFG.model_dir + f'model_fold_{fold}.pth'))

    # Create the test dataset.
    test_dataset = TestDataset(x_test)

    # Create the test data loader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
    )

    # Infer on the model.
    preds = inference_fn(model, test_loader, CFG.device)

    return preds


def infer_k_folds():
    '''
    Infers on all the folds.

    Returns:
        preds: np.ndarray of shape (num_samples, num_targets).
    '''
    predictions = np.zeros((len(test), 206))  # Initialize the list of predictions.

    # Infer on each fold.
    for fold in range(CFG.n_splits):
        print(f'Inferring on fold {fold}...')
        predictions += (infer_single_fold(fold) / CFG.n_splits)

    return predictions


if __name__ == '__main__':
    predictions = infer_k_folds()  # Infer on all the folds.
    print(predictions.shape)

    # Create the submission file.
    print("Generating submission file...")
    submission = pd.DataFrame(columns=sample_submission.columns)  # Create a submission dataframe with columns.
    submission['sig_id'] = test['sig_id']  # filling the sig_id column.
    submission.iloc[:, 1:] = predictions  # filling the predictions columns.
    submission = submission.fillna(0.0)  # filling the NaN values with 0.0.
    submission.to_csv('../submission.csv', index=False)  # Saving the submission file.
