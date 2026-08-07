"""
Utility functions for model training and prediction.
"""
import os
import sys
import time
import torch
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Add pykan and sgkan to path using relative imports
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_path, 'pykan'))
sys.path.insert(0, os.path.join(base_path, 'sgkan'))

import evaluation_metrics as em
from kan import KAN  # type: ignore
from hkan.hkan import (
    make_hkan_layer, extend_hkan, set_tqdm_disable,
)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ─── HKAN Utilities ────────────────────────────────────────────
# Functions for building, training, and predicting with HKAN models


def build_hkan_model_from_configs(model_configs, tqdm_disable=False):
    """
    Build an HKAN model from a list of layer-parameter dictionaries.
    
    Stacks layers sequentially, starting with layer 0 and extending upward.
    Each layer can have different basis functions, number of basis, centers, and regressors.

    Returns:
        HKAN model: Constructed multi-layer HKAN model ready for training
    """
    # Sort parameters by layer index to ensure correct order
    params = sorted(model_configs, key=lambda p: int(p["layer"]))
    model = None
    set_tqdm_disable(tqdm_disable)

    for p in params:
        layer_idx = int(p["layer"])
        n_vars_out = p["n_vars_out"]
        basis_fn = p["basis_fn"]
        n_basis = p.get("n_basis", 10)
        centers = p.get("centers", "random")
        expanding_base_regressor = p.get("expanding_base_regressor", None)
        connecting_base_regressor = p.get("connecting_base_regressor", None)


        if layer_idx == 0:
            # Create first layer with specified parameters
            model = make_hkan_layer(
                layer_idx=layer_idx,
                n_vars_out=n_vars_out,
                basis_fn=basis_fn,
                n_basis=n_basis,
                centers=centers,
                expanding_base_regressor=expanding_base_regressor,
                connecting_base_regressor=connecting_base_regressor
            )
        else:
            # Stack additional layers on top of existing model
            if model is None:
                raise ValueError("First layer (layer 0) must be provided before extending.")
            model = extend_hkan(
                model,
                layer_idx=layer_idx,
                n_vars_out=n_vars_out,
                basis_fn=basis_fn,
                n_basis=n_basis,
                centers=centers,
                expanding_base_regressor=expanding_base_regressor,
                connecting_base_regressor=connecting_base_regressor
            )
    return model


def fit_hkan(model, X_train, y_train):
    """
    Train an HKAN model on training data.
    
    Measures and reports training time.

    Args:
        model: HKAN model instance (from build_hkan_from_params)
        X_train (numpy array): Training input features
        y_train (numpy array): Training target labels

    Returns:
        None (model is trained in-place)
    """
    if model is not None:
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        print(f"\n✓ HKAN training completed in {elapsed_time:.4f} seconds")


def predict_hkan(model, X_test):
    """
    Make predictions with a trained HKAN model.
    
    Measures and reports prediction time.

    Args:
        model: Trained HKAN model instance
        X_test (numpy array): Test input features

    Returns:
        numpy array: Predictions on test data
    """
    if model is not None:
        start_time = time.time()
        predictions = model.predict(X_test)
        elapsed_time = time.time() - start_time
        print(f"✓ HKAN prediction completed in {elapsed_time:.4f} seconds")
        return predictions


# ─── KAN Utilities ─────────────────────────────────────────────
# Functions for building, training, and predicting with KAN models


def build_kan(width, grid=3, k=3, seed=42):
    """
    Build a KAN model with a given architecture.

    Args:
        width (list): Layer widths, for example [2, 5, 1]
        grid (int): Grid size for spline basis
        k (int): Spline degree
        seed (int): Random seed

    Returns:
        KAN: A KAN model instance
    """
    return KAN(width=width, grid=grid, k=k, seed=seed, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


def fit_kan(model, X_train, y_train, X_test, y_test, steps=20, opt="LBFGS", lamb=0):
    """
    Train a KAN model using train and test data.

    Args:
        model: KAN model instance
        X_train, y_train: Training data
        X_test, y_test: Validation or test data
        steps (int): Number of optimization steps
        opt (str): Optimizer name
        lamb (float): Regularization strength

    Returns:
        KAN: Trained model
    """
    start_time = time.time()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=model.device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=model.device).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=model.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=model.device).reshape(-1, 1)

    dataset = {
        'train_input': X_train_tensor,
        'train_label': y_train_tensor,
        'test_input': X_test_tensor,
        'test_label': y_test_tensor,
    }

    model.fit(dataset, opt=opt, steps=steps, lamb=lamb)

    elapsed_time = time.time() - start_time
    print(f"\n✓ KAN training completed in {elapsed_time:.4f} seconds")
    return model

def fit_kan_with_grid_extension(model, X_train, y_train, X_test, y_test,
                                 grids=(5, 10, 20), steps_per_grid=20,
                                 opt="LBFGS", lamb=0):
    """
    Train a KAN model with coarse-to-fine grid extension.

    Trains on the model's current (coarse) grid first, then refines to
    each grid size in `grids` in turn, training again after each refinement.

    Args:
        model: KAN model instance (already built with the first grid size)
        X_train, y_train, X_test, y_test: data
        grids: grid sizes to refine to, in increasing order (should not
            repeat the model's initial grid)
        steps_per_grid (int): training steps at each grid stage
        opt (str): optimizer, "LBFGS" recommended
        lamb (float): regularization strength

    Returns:
        KAN: the trained model, refined through all grid stages
    """
    start_time = time.time()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float64, device=model.device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=model.device).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=model.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=model.device).reshape(-1, 1)

    dataset = {
        'train_input': X_train_tensor,
        'train_label': y_train_tensor,
        'test_input': X_test_tensor,
        'test_label': y_test_tensor,
    }

    # Train at the initial grid first, so refine() has cached data to work from
    model.fit(dataset, opt=opt, steps=steps_per_grid, lamb=lamb)

    for grid in grids:
        model = model.refine(grid)
        model.fit(dataset, opt=opt, steps=steps_per_grid, lamb=lamb)

    elapsed_time = time.time() - start_time
    print(f"\n✓ KAN training with grid extension completed in {elapsed_time:.4f} seconds")
    return model


def predict_kan(model, X_test):
    """
    Make predictions with a trained KAN model.

    Args:
        model: Trained KAN model instance
        X_test (numpy array): Test input features

    Returns:
        numpy array: Predictions on test data
    """
    start_time = time.time()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=model.device)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    elapsed_time = time.time() - start_time
    print(f"✓ KAN prediction completed in {elapsed_time:.4f} seconds")
    return predictions


# Wrapper to add predict method for em.evaluate compatibility
class KANModel:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float64, device=device)
        return self.model(X_tensor).detach().cpu().numpy()


def objective_kan(trial, X_train, y_train, X_test, y_test, val_size=0.2):
    """
    Optuna objective: pick the best KAN width (from the HKAN paper's search
    space W) via a single train/validation split. Each candidate is trained
    once, with coarse-to-fine grid extension (LBFGS at each stage). Returns
    validation RMSE, the sole criterion used for architecture selection.
    The test set is not touched during the search.
    """
    n = X_train.shape[1]

    width_options = [
        [n, 1], [n, 2, 1],
        [n, n + 1, 1], [n, 2 * n + 1, 1], [n, 2, 2, 1],
        [n, n + 1, 2, 1], [n, 2 * n + 1, 2, 1], [n, n + 1, n + 1, 1],
        [n, 2 * n + 1, n + 1, 1], [n, 2 * n + 1, 2 * n + 1, 1],
    ]
    width_idx = trial.suggest_categorical("width_idx", list(range(len(width_options))))
    width = width_options[width_idx]
    architecture_str = str(width) 

    n_samples = X_train.shape[0]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    n_val = int(round(n_samples * val_size))
    val_idx, sub_train_idx = perm[:n_val], perm[n_val:]
    X_sub_train, X_val = X_train[sub_train_idx], X_train[val_idx]
    y_sub_train, y_val = y_train[sub_train_idx], y_train[val_idx]

    grids = (5, 10, 20)
    steps_per_grid = 20
    opt = "LBFGS"
    lamb = 0

    model = build_kan(width=width, grid=3, k=3, seed=42)
    model = fit_kan_with_grid_extension(
        model, X_sub_train, y_sub_train, X_val, y_val,
        grids=grids, steps_per_grid=steps_per_grid, opt=opt, lamb=lamb,
    )

    val_results = em.evaluate(KANModel(model), X_sub_train, y_sub_train, X_val, y_val)
    val_rmse = float(val_results['test']['rmse'])

    trial.set_user_attr("architecture", architecture_str)
    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("grids", str(grids))
    trial.set_user_attr("steps_per_grid", steps_per_grid)
    trial.set_user_attr("opt", opt)
    trial.set_user_attr("lamb", lamb)

    return val_rmse


def study_optuna_kan(dataset_name, X_train, y_train, X_test, y_test):
    """
    Run an exhaustive Optuna grid search over the KAN architecture search
    space W defined by the HKAN paper, using GridSampler so every
    architecture is evaluated exactly once. Each trial's grid extension
    configuration, optimizer, validation RMSE, and duration are recorded in
    the returned trials dataframe. The test set is not used during the search.

    Args:
        dataset_name (str): Dataset name for output file and logging
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        dict: Best parameters with keys 'width_idx' and 'architecture'
    """
    csv_path = f"data/{dataset_name.lower()}_kan_optuna_search.csv"
    n = X_train.shape[1]
    width_options = [
        [n, 1], [n, 2, 1], 
        [n, n + 1, 1], [n, 2 * n + 1, 1], [n, 2, 2, 1],
        [n, n + 1, 2, 1], [n, 2 * n + 1, 2, 1], [n, n + 1, n + 1, 1],
        [n, 2 * n + 1, n + 1, 1], [n, 2 * n + 1, 2 * n + 1, 1],
    ]

    print_cols = ['number', 'params_width_idx', 'user_attrs_architecture',
                  'user_attrs_val_rmse', 'user_attrs_grids',
                  'user_attrs_steps_per_grid', 'user_attrs_opt',
                  'user_attrs_lamb', 'duration']

    if os.path.exists(csv_path):
        print("=" * 70)
        print(f"Results found for {dataset_name}. Loading from {csv_path}")
        print("=" * 70)
        trials_df = pd.read_csv(csv_path)
        best_idx = trials_df['user_attrs_val_rmse'].idxmin()
        best_row = trials_df.loc[best_idx]
        width_idx = int(best_row['params_width_idx']) # type: ignore
        best_arch = width_options[width_idx]

        print(f"Best Validation RMSE:   {best_row['user_attrs_val_rmse']:.6f}")
        print(f"Best architecture:       {best_row['user_attrs_architecture']}")
        print(f"\nAll {len(trials_df)} trials:")
        print(trials_df[print_cols].to_string())

        return {"width_idx": width_idx, "architecture": best_arch}

    print("=" * 70)
    print(f"Optimizing KAN architecture (exhaustive grid search) on {dataset_name}")
    print("=" * 70)

    search_space = {"width_idx": list(range(len(width_options)))}
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.GridSampler(search_space, seed=42),
    )
    study.optimize(
        lambda trial: objective_kan(trial, X_train, y_train, X_test, y_test, val_size=0.2),
        n_trials=len(width_options),
        show_progress_bar=True,
    )

    best_idx = study.best_params['width_idx']
    best_arch = width_options[best_idx]
    print(f"\nBest Validation RMSE: {study.best_value:.6f}")
    print(f"Best architecture:     {best_arch}")

    trials_df = study.trials_dataframe()
    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    trials_df.to_csv(csv_path, index=False)

    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[print_cols].to_string())

    return {"width_idx": best_idx, "architecture": best_arch}