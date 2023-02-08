"""
Create folds for cross validation using the Multi-Label
Stratified K-Folds from the Iterstrat Module.

We do this for every project do evaluate the model
on unseen data and make sure it isn't overfitting the training data.
"""

import pandas as pd  # To read the data
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
)  # To create stratified folds
from src.config import CFG  # To get the number of folds from the config file


# Function to create folds
def create_fold(df, save_path="input/train_folds.csv"):
    """
    Create folds for cross validation using the Multi-Label Stratified K-Fold.
    Input:
        df: Pandas DataFrame containing the data
        save_path: Path to save the folds

    Output:
        None(Saves the folds in a csv file)
    """
    df[
        "kfold"
    ] = -1  # Create a new column in the DataFrame called kfold to store the folds
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data

    y = df.drop(
        ["sig_id"], axis=1
    ).values  # Get the labels by dropping the sig_id column beacuse it is not a target

    mskf = MultilabelStratifiedKFold(
        n_splits=CFG.n_splits
    )  # Create the Multi-Label Stratified K-Fold object by passing in the number of folds from the config file

    for fold, (train_idx, val_idx) in enumerate(
        mskf.split(df, y)
    ):  # Loop through the folds and get trn_idx and val_idx for each fold
        print(len(train_idx), len(val_idx))
        df.loc[
            val_idx, "kfold"
        ] = fold  # Store the fold number in the kfold column for the validation indices of that particular fold
        # The above line is the crux of the fold creation process. Eg. for fold 0, We get the val_idx from mskf.
        # Then in the kfold column of all those indices we put in 0 as the fold number. This process repeats for all the folds.
    df.to_csv(save_path, index=False)  # Save the folds in a csv file


if __name__ == "__main__":
    csv_path = "input/train_targets_scored.csv"  # Get the path to the csv file containing the data
    df = pd.read_csv(csv_path)  # Read the csv file
    print("Creating folds...")
    create_fold(df)  # Create the folds by passing in the DataFrame
    print("Folds created!")
