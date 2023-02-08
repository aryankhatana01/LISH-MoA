"""
This file conatins all the utility functions and feature engineering we might need.
"""

import random
import os
import numpy as np
import torch
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def seed_everything(seed=42):
    """
    This function seeds everything and makes the results reproducible/deterministic.
    Input:
        seed: int, seed to be used

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def PCA_g(data, n_components=50):
    """
    This function performs PCA on the given data for the genes.
    i.e. it reduces the dimensionality of the data.
    Input:
        data: pandas dataframe, data on which PCA is to be performed
        n_components: int, number of components to be used for PCA

    Returns:
        transformed_data: pandas dataframe, transformed data
    """
    GENES = [
        col for col in data.columns if col.startswith("g-")
    ]  # All the genes columns
    data_ = pd.DataFrame(data[GENES])  # Selecting the data of only the genes columns
    data_ = PCA(n_components=n_components, random_state=42).fit_transform(
        data_[GENES]
    )  # Performing PCA
    feats = pd.DataFrame(data_, columns=[f"pca_G-{i}" for i in range(n_components)])
    final_features = pd.concat((data, feats), axis=1)
    return final_features


def PCA_c(data, n_components=50):
    """
    This function performs PCA on the given data for the cells.
    i.e. it reduces the dimensionality of the data.
    Input:
        data: pandas dataframe, data on which PCA is to be performed
        n_components: int, number of components to be used for PCA

    Returns:
        transformed_data: pandas dataframe, transformed data
    """
    CELLS = [
        col for col in data.columns if col.startswith("c-")
    ]  # All the cells columns
    data_ = pd.DataFrame(
        data[CELLS]
    )  # Selecting the data of only the cells columns and creating a dataframe
    data_ = PCA(n_components=n_components, random_state=42).fit_transform(
        data_[CELLS]
    )  # Performing PCA
    feats = pd.DataFrame(
        data_, columns=[f"pca_C-{i}" for i in range(n_components)]
    )  # Creating a dataframe of the transformed data
    final_features = pd.concat((data, feats), axis=1)  # Concatenating the transformed data with the original data
    return final_features


def feature_selection(data, threshold=0.8):
    """
    This function performs feature selection using variance thresholding on the given data.
    Input:
        data: pandas dataframe, data on which feature selection is to be performed
        threshold: float, threshold value for feature selection

    Returns:
        transformed_data: pandas dataframe, transformed data
    """
    variance_thresh = VarianceThreshold(threshold=threshold)  # Creating the object
    data_transformed = variance_thresh.fit_transform(data.iloc[:, 4:])  # Performing feature selection
    feats = pd.DataFrame(
        data[["sig_id", "cp_type", "cp_time", "cp_dose"]].values.reshape(-1, 4),
        columns=["sig_id", "cp_type", "cp_time", "cp_dose"],
    )  # Creating a dataframe of the features that are not transformed
    final_feats = pd.concat(
        [feats, pd.DataFrame(data_transformed)],
        axis=1
    )  # Concatenating the transformed data with the features that are not transformed
    return final_feats
