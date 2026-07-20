"""
Utility functions for model training and prediction.
"""
import os
import sys
import time
import torch
import gpytorch
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

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


def fit_kan(model, X_train, y_train, X_test, y_test, steps=50, opt="Adam", lamb=0.01):
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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=model.device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=model.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=model.device)

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
                                 grids=(5, 10, 20), steps_per_grid=50,
                                 opt="LBFGS", lamb=1e-3):
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
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64, device=model.device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64, device=model.device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float64, device=model.device)

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


def objective_kan(trial, X_train, y_train, X_test, y_test, n_splits=5):
    """
    Optuna objective: pick the best KAN width (from the HKAN paper's search
    space W) via 5-fold CV. Each candidate is trained with coarse-to-fine
    grid extension (LBFGS at each stage). Returns mean validation RMSE
    across folds; also stores test RMSE as a trial attribute.
    """
    n = X_train.shape[1]

    # Architecture search space W from the HKAN paper
    width_options = [
        [n, 1],
        [n, 2, 1],
        [n, n + 1, 1],
        [n, 2 * n + 1, 1],
        [n, 2, 2, 1],
        [n, n + 1, 2, 1],
        [n, 2 * n + 1, 2, 1],
        [n, n + 1, n + 1, 1],
        [n, 2 * n + 1, n + 1, 1],
        [n, 2 * n + 1, 2 * n + 1, 1],
    ]

    # Suggest architecture
    width_idx = trial.suggest_categorical("width_idx", list(range(len(width_options))))
    width = width_options[width_idx]

    # K-Fold Cross-Validation on training data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_rmses = []
    
    for _, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # Split into fold train/validation
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Build and train model with suggested architecture
        model = build_kan(width=width, grid=3, k=3, seed=42)
        model = fit_kan_with_grid_extension(model, X_fold_train, y_fold_train, 
                X_fold_val, y_fold_val,
                opt="LBFGS", lamb=1e-3)
        
        # Evaluate on this fold's validation set
        results = em.evaluate(
            KANModel(model),
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val
        )
        fold_val_rmses.append(results['test']['rmse'])
    
    # Average validation RMSE across all folds (what Optuna minimizes)
    avg_val_rmse = float(np.mean(fold_val_rmses))
    
    # Also evaluate on the actual test set for reference (only once, not per fold)
    final_model = build_kan(width=width, grid=3, k=3, seed=42)
    final_model = fit_kan_with_grid_extension(final_model, X_train, y_train, X_test, y_test, opt="LBFGS", lamb=1e-3)
    test_results = em.evaluate(KANModel(final_model), X_train, y_train, X_test, y_test)
    test_rmse = float(test_results['test']['rmse'])
    
    # Store test RMSE as user attribute (will appear in trials dataframe)
    trial.set_user_attr("test_rmse", test_rmse)
    trial.set_user_attr("val_rmse", avg_val_rmse)
    
    # Return validation RMSE for optimization
    return avg_val_rmse


def study_optuna_kan(dataset_name, X_train, y_train, X_test, y_test, n_trials=10):
    """
    Run Optuna parameter search for KAN architecture on a dataset.

    Uses 5-Fold Cross-Validation to evaluate architectures from the HKAN paper
    over exhaustive grid of all 10 predefined width options.

    Args:
        dataset_name (str): Dataset name for output file and logging
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_trials (int): Number of trials to run (default: 10 for grid exhaustion)

    Returns:
        dict: Best parameters with keys 'width_idx' and 'architecture'
    """
    csv_path = f"data/{dataset_name.lower()}_kan_optuna_search.csv"
    n = X_train.shape[1]

    # Architecture search space W from the HKAN paper (parameterized by n input features)
    width_options = [
        [n, 1], [n, 2, 1], [n, n+1, 1], [n, 2*n+1, 1], [n, 2, 2, 1],
        [n, n+1, 2, 1], [n, 2*n+1, 2, 1], [n, n+1, n+1, 1],
        [n, 2*n+1, n+1, 1], [n, 2*n+1, 2*n+1, 1],
    ]

    # Load cached results if they exist
    if os.path.exists(csv_path):
        print("=" * 70)
        print(f"Results found for {dataset_name}. Loading from {csv_path}")
        print("=" * 70)

        trials_df = pd.read_csv(csv_path)
        best_idx = trials_df['value'].idxmin()
        best_row = trials_df.loc[best_idx]

        width_idx = int(best_row['params_width_idx']) # type: ignore
        best_arch = width_options[width_idx]

        print(f"Best CV Validation RMSE: {best_row['value']:.6f}")
        print(f"Best Test RMSE:          {best_row['test_rmse']:.6f}")
        print(f"Best width_idx:          {width_idx}")
        print(f"Best architecture:       {best_arch}")

        print(f"\nAll {len(trials_df)} trials:")
        print(trials_df[['number', 'value', 'params_width_idx', 'val_rmse', 'test_rmse']].to_string())

        return {"width_idx": width_idx, "architecture": best_arch}

    # Run optimization if results don't exist
    print("=" * 70)
    print(f"Optimizing KAN architecture with 5-Fold CV on {dataset_name}")
    print("=" * 70)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.GridSampler(
            {"width_idx": list(range(10))}
        ),
    )

    study.optimize(
        lambda trial: objective_kan(
            trial, X_train, y_train, X_test, y_test, n_splits=5
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print and save optimization results
    print("\n" + "=" * 70)
    print("Best trial results:")
    print("=" * 70)
    print(f"Best CV Validation RMSE: {study.best_value:.6f}")
    best_test_rmse = study.best_trial.user_attrs.get("test_rmse", "N/A")
    test_rmse_str = f"{best_test_rmse:.6f}" if isinstance(best_test_rmse, float) else "N/A"
    print(f"Best Test RMSE:          {test_rmse_str}")
    print(f"Best width_idx:          {study.best_params['width_idx']}")
    print(f"Best architecture:       {width_options[study.best_params['width_idx']]}")

    trials_df = study.trials_dataframe()
    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[['number', 'value', 'params_width_idx', 'user_attrs_val_rmse', 'user_attrs_test_rmse']].to_string())

    # Rename columns and save to CSV for future reference
    trials_df = trials_df.rename(columns={
        "user_attrs_test_rmse": "test_rmse",
        "user_attrs_val_rmse": "val_rmse",
    })
    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    trials_df.to_csv(csv_path, index=False)

    best_idx = study.best_params['width_idx']
    return {"width_idx": best_idx, "architecture": width_options[best_idx]}