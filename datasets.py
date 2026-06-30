"""
Dataset utilities for loading and preprocessing regression datasets.

Provides functions to:
- Load datasets from CSV files (train/test splits)
- Scale data to [0, 1] range
- Display dataset statistics
"""

import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# List of available datasets
NAMES = [
    "abalone",
    "auto_mpg",
    "bank32nh",
    "compactive",
    "concrete",
    "dee",
    "ele_2",
    "elevators",
    "kinematics32nh",
    "kinematics8nm",
    "laser",
    "machineCPU",
    "pumadyn32nh",
    "pyramidines",
    "stock",
    "TF1",
    "TF2",
    "TF3",
    "TF4",
    "TF5-5",
    "TF5",
    "triazines",
    "treasury",
    "wizmir"    
]


def load_dataset(name):
    """
    Load a dataset by name from CSV files.
    
    Args:
        name: Dataset name (must be in NAMES list).
    
    Returns:
        dict: Contains 'train_input', 'train_label', 'test_input', 'test_label'.
    """
    # Locate data folder
    base_path = os.path.dirname(__file__)
    data_path = Path(base_path) / "data"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder {data_path} not found.")
    
    if name not in NAMES:
        raise ValueError(f"Dataset '{name}' not found. Available: {NAMES}")

    # Load train and test CSV files
    train_file = data_path / f"{name}_trn.csv"
    test_file = data_path / f"{name}_tst.csv"
    
    train_data = np.genfromtxt(train_file, delimiter=",")
    test_data = np.genfromtxt(test_file, delimiter=",")
    
    # Split features (all columns except last) and labels (last column)
    dataset = {
        "train_input": train_data[:, :-1],
        "train_label": train_data[:, [-1]].ravel(),
        "test_input": test_data[:, :-1],
        "test_label": test_data[:, [-1]].ravel(),
    }
    return dataset


def min_max_scale_dataset(dataset):
    """
    Scale inputs and labels to [0, 1] range using training statistics.
    
    Uses min-max scaling fitted on training data, then applied to test data.
    
    Args:
        dataset: Dict with 'train_input', 'train_label', 'test_input', 'test_label'.
    
    Returns:
        tuple: (scaled_dict, scaler_dict) for scaled data and reusable scalers.
    """
    X_train = dataset["train_input"]
    y_train = dataset["train_label"]
    X_test = dataset["test_input"]
    y_test = dataset["test_label"]

    # Create min-max scalers for inputs and labels
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scalers on training data
    x_scaler.fit(X_train)
    y_scaler.fit(y_train.reshape(-1, 1))

    # Scale all data using training statistics
    scaled = {
        "train_input": x_scaler.transform(X_train),
        "test_input": x_scaler.transform(X_test),
        "train_label": y_scaler.transform(y_train.reshape(-1, 1)).ravel(),
        "test_label": y_scaler.transform(y_test.reshape(-1, 1)).ravel(),
    }
    
    # Return scalers for inverse transform if needed
    scaler = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }
    return scaled, scaler


def show_datasets():
    """Display statistics for all available datasets."""
    # Print header
    print(f"{'Nb.':<3}|{'Name':<20}|{'n_vars':<10}|{'n_samples_train':<20}|{'n_samples_test':<20}")
    print("-" * 80)
    
    # Print info for each dataset
    for i, name in enumerate(NAMES, 1):
        dataset = load_dataset(name)
        n_vars = dataset["train_input"].shape[1]
        n_samples_train = dataset["train_input"].shape[0]
        n_samples_test = dataset["test_input"].shape[0]
        print(f"{i:<3}|{name:<20}|{n_vars:<10}|{n_samples_train:<20}|{n_samples_test:<20}")
        print("-" * 80)


# CLI: Display all datasets when run directly
if __name__ == "__main__":
    show_datasets()