"""
This file contains the model training code.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import pandas as pd
import numpy as np
from config import CFG
from model import Model
from dataset import MoADataset
import utils
from tqdm import tqdm


# Reading the training data (I won't be using the non_scored targets)
train_features = pd.read_csv('../input/train_features.csv')
train_targets_scored = pd.read_csv('../input/train_folds.csv')


# Applying the feature engineering on the training data.
train_features = utils.PCA_g(train_features, n_components=CFG.n_components)  # PCA for genes
train_features = utils.PCA_c(train_features, n_components=CFG.n_components)  # PCA for cells


train = train_features.merge(train_targets_scored, on='sig_id')  # Merging the features and targets
train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)  # Removing the control samples

target = train[train_targets_scored.columns[:-1]]  # Selecting the target columns except kfold column
train = train.drop('cp_type', axis=1)  # Dropping the cp_type column because it is not needed anymore


train['cp_time'] = train['cp_time'].astype(str)  # To fix a warning
train['cp_dose'] = train['cp_dose'].astype(str)  # To fix a warning
train = pd.get_dummies(train, columns=['cp_time', 'cp_dose'])  # One hot encoding the cp_time and cp_dose columns


# Getting all the columns except the target columns and sig_id
target_cols = target.drop('sig_id', axis=1).columns.values.tolist()  # Getting the target columns in list except sig_id
feature_cols = [c for c in train.columns if c not in target_cols]  # Getting the feature columns
feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]  # Removing kfold and sig_id from columns


def train_fn(model, data_loader, optimizer, loss_fn, device, scheduler=None):
    """
    Trains the model for one epoch.

    Input:
        model: nn.Module object.
        data_loader: torch.utils.data.DataLoader object.
        optimizer: torch.optim.Optimizer object.
        device: torch.device object.
        scheduler: torch.optim.lr_scheduler._LRScheduler object.

    Returns:
        train_loss: float (Mean of all the batch losses)
    """
    model.train()  # Set the model to training mode.
    train_loss = 0  # Initialize the total training loss.

    for data in tqdm(data_loader, total=len(data_loader)):
        optimizer.zero_grad()  # Zero the gradients.
        inputs, targets = data["x"].to(device), data["y"].to(
            device
        )  # Get the inputs and targets.
        outputs = model(inputs)  # Pass the inputs through the model.
        loss = loss_fn(outputs, targets)  # Compute the loss for this batch.
        loss.backward()  # Compute the gradients.
        optimizer.step()  # Update the model parameters.
        train_loss += (
            loss.item()
        )  # Update the total training loss by adding the batch loss.

    if scheduler is not None:
        scheduler.step()

    return train_loss / len(data_loader)


def eval_fn(model, data_loader, loss_fn, device):
    """
    Evaluates the model for one epoch.

    Input:
        model: nn.Module object.
        data_loader: torch.utils.data.DataLoader object.
        device: torch.device object.

    Returns:
        valid_loss: float.
        valid_preds: np.ndarray of shape (num_samples, num_targets).
    """
    model.eval()  # Set the model to evaluation mode.
    valid_loss = 0  # Initialize the total validation loss.
    valid_preds = []  # Initialize the list of predictions.

    with torch.no_grad():  # Disable gradient calculation for computation efficiency.
        for data in tqdm(data_loader, total=len(data_loader)):
            inputs, targets = data["x"].to(device), data["y"].to(
                device
            )  # Get the inputs and targets.
            outputs = model(inputs)  # Pass the inputs through the model.
            loss = loss_fn(outputs, targets)  # Compute the loss for this batch.
            valid_loss += (
                loss.item()
            )  # Update the total validation loss by adding the batch loss.
            valid_preds.append(
                outputs.sigmoid().detach().cpu().numpy()
            )  # Append the predictions to the list.

    valid_preds = np.concatenate(valid_preds)  # Concatenate the predictions.

    return valid_loss / len(data_loader), valid_preds


def run_single_fold(fold):
    """
    Trains the model for one fold.

    Input:
        fold: int.

    Returns:
        oof: np.ndarray of shape (len(train), num_targets).
    """
    # Getting validation indices for this fold.
    val_idx = train[train['kfold'] == fold].index

    # Get the training and validation dataframes.
    train_df = train[train["kfold"] != fold].reset_index(drop=True)  # train_df is the df with all rows except the fold
    valid_df = train[train["kfold"] == fold].reset_index(drop=True)  # val_df is the df with all rows with the fold No.

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values  # Getting the training data
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values  # Getting the validation data

    # Creating the dataset using the training data and val_data.
    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)

    # Creating the dataloaders using the datasets.
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    # Initializing the model
    model = Model(
        num_features=len(feature_cols),
        num_targets=len(target_cols),
        hidden_size=CFG.hidden_size,
    )
    model.to(CFG.device)  # Moving the model to the device.

    # Initializing the optimizer and scheduler.
    optimizer = Adam(model.parameters(), lr=CFG.lr)
    scheduler = OneCycleLR(
        optimizer=optimizer,
        pct_start=0.1,
        div_factor=1e3,
        max_lr=1e-2,
        epochs=CFG.n_epochs,
        steps_per_epoch=len(train_loader)
    )

    # Initializing the loss function.
    loss_fn = nn.BCEWithLogitsLoss()

    # Initializing the best validation loss to infinity.
    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))  # Initializing the out of fold predictions.
    # oof is a matrix of zeros with the same shape as the target matrix
    # and it means we're predicting on all the exmaples in the train set except the fold we're currently on.
    best_loss = np.inf

    for epoch in range(CFG.n_epochs):
        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            loss_fn,
            CFG.device,
            scheduler
        )  # Training the model for one epoch.
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}")

        valid_loss, valid_preds = eval_fn(
            model,
            valid_loader,
            loss_fn,
            CFG.device,
        )  # Evaluating the model for one epoch.
        print(f"FOLD: {fold}, EPOCH: {epoch}, valid_loss: {valid_loss}")

        if valid_loss < best_loss:
            print(f'Loss decreased from {best_loss} to {valid_loss}. Saving the model...')
            best_loss = valid_loss
            oof[val_idx] = valid_preds  # Updating the out of fold predictions.
            torch.save(model.state_dict(), f"{CFG.output_dir}/model_fold_{fold}.pth")  # Saving the model.
    return oof


def run_k_fold():
    """
    Trains the model for k folds.

    Input:
        None

    Returns:
        None
    """
    oof = np.zeros((len(train), len(target_cols)))  # Initializing the out of fold predictions.
    for fold in range(CFG.n_splits):
        print(f"========== Training for FOLD {fold} ==========")
        oof += run_single_fold(fold)  # Training the model for one fold.
    return oof


if __name__ == "__main__":
    oof = run_k_fold()  # Training the model for k folds.
