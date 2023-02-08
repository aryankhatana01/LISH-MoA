# Mechansisms of Action - Kaggle

This is a re-implementation of my solution in the LISH-MoA Kaggle Competition. I'm trying to create this to be really easy to understand and intuitive for a beginner and learn how to deal with tabular data using NN.

Link to the competition - [Click Here](https://www.kaggle.com/competitions/lish-moa)

Download the Data and place it in the ```input``` folder.

## A little intro to the problem

This is a multi-label problem and for each ```sig_id``` we need to predict the 207 targets in the ```train_targets_scored.csv``` file.

### Input Files (Same can be found on the competition page)
- ```train_features.csv``` - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; ```cp_time``` and ```cp_dose``` indicate treatment duration (24, 48, 72 hours) and dose (high or low).
- ```train_drug.csv``` - This file contains an anonymous ```drug_id``` for the training set only.
- ```train_targets_scored.csv``` - The binary MoA targets that are scored.
- ```train_targets_nonscored.csv``` - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
- ```test_features.csv``` - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
- ```sample_submission.csv``` - A submission file in the correct format.

## How I'll be approaching the problem
I'll be creating simple Linear Neural Network with a little feature engineering in the beginning. This solution won't be state of the art or even enough to get a good score but it'll be a good starting point for beginners to understand how to deal with tabular data using NN.

First, I'll be creating the Dataset class to take input data and retrieve the features and the targets. The features would be in ```train_features.csv``` and the targets would be in ```train_targets_scored.csv```.