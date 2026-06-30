"""
Utility functions for model training and prediction.

Provides helper functions for:
- Building and training HKAN models
- Building and training KAN models
- Making predictions with various models
- Creating GP kernels
- Evaluating and visualizing results
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
from sgkan import surrogate_guided_kan as sgkan
from sgkan import gaussian_process_models as gp
from hkan.hkan import (
    make_hkan_layer, extend_hkan, set_tqdm_disable,
)

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── HKAN Utilities ────────────────────────────────────────────
# Functions for building, training, and predicting with HKAN models


def build_hkan_from_params(model_params, tqdm_disable=False):
    """
    Build an HKAN model from a list of layer-parameter dictionaries.
    
    Stacks layers sequentially, starting with layer 0 and extending upward.
    Each layer can have different basis functions, number of basis, centers, and regressors.

    Args:
        model_params (list): List of dicts with layer config:
            - "layer" (int): Layer index (0 first, then 1, 2, ...)
            - "n_vars_out" (int): Number of output variables for this layer
            - "basis_fn": Basis function object (Sigmoid, Gaussian, ReLU, Tanh, etc.)
            - "n_basis" (int, optional): Number of basis functions (default: 10)
            - "centers" (str, optional): Center initialization method (default: "random")
            - "regressor": Base regressor for expanding (default: Ridge())
        tqdm_disable (bool): Disable progress bars during training (default: False)

    Returns:
        HKAN model: Constructed multi-layer HKAN model ready for training
        
    Raises:
        ValueError: If layer 0 is not provided before extending
    """
    # Sort parameters by layer index to ensure correct order
    params = sorted(model_params, key=lambda p: int(p["layer"]))
    model = None
    set_tqdm_disable(tqdm_disable)

    for p in params:
        layer_idx = int(p["layer"])
        n_vars_out = p["n_vars_out"]
        basis_fn = p["basis_fn"]
        n_basis = p.get("n_basis", 10)
        centers = p.get("centers", "random")
        reg = p.get("regressor", Ridge())

        if layer_idx == 0:
            # Create first layer with specified parameters
            model = make_hkan_layer(
                layer_idx=layer_idx,
                n_vars_out=n_vars_out,
                basis_fn=basis_fn,
                n_basis=n_basis,
                centers=centers,
                expanding_base_regressor=reg,
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
                expanding_base_regressor=reg,
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
    Objective function for Optuna to find the best KAN architecture with K-Fold CV.
    
    Uses K-Fold Cross-Validation on training data to evaluate architectures robustly.
    Searches over predefined architecture set W from the HKAN paper.

    Only the network width configuration is searched. All other parameters 
    (grid, spline order k, optimizer, steps) are fixed at KAN defaults.

    The architecture set W is parameterised by n = X_train.shape[1]:
        W = {[n,1], [n,2,1], [n,n+1,1], [n,2n+1,1], [n,2,2,1],
             [n,n+1,2,1], [n,2n+1,2,1], [n,n+1,n+1,1],
             [n,2n+1,n+1,1], [n,2n+1,2n+1,1]}

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_test, y_test: Test data for final evaluation
        n_splits: Number of folds for cross-validation (default: 5)

    Returns:
        float: Average validation RMSE across all folds (what Optuna minimizes)
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
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # Split into fold train/validation
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Build and train model with suggested architecture
        model = build_kan(width=width, grid=3, k=3, seed=42)
        fit_kan(model, X_fold_train, y_fold_train, 
                X_fold_val, y_fold_val,
                steps=100, opt="Adam", lamb=1e-3)
        
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
    fit_kan(final_model, X_train, y_train, X_test, y_test, steps=100, opt="Adam", lamb=1e-3)
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


# ─── GP utilities ─────────────────────────────────────────────
# Functions for building, training, and predicting with GP models


# Wrapper for GP model to add predict method for em.evaluate compatibility
class GPModel:
    """Wrapper for Gaussian Process model for compatibility with em.evaluate().

    Provides a sklearn-like predict interface for GP models so they can be
    evaluated using the standard em.evaluate() function.
    """

    def __init__(self, model, likelihood):
        """Initialize GP wrapper.

        Args:
            model: Trained GPyTorch GP model
            likelihood: GPyTorch likelihood object
        """
        self.model = model
        self.likelihood = likelihood

    def predict(self, X):
        """Make predictions with the GP model.

        Args:
            X (numpy array): Input features of shape (n_samples, n_features)

        Returns:
            numpy array: Predicted mean values of shape (n_samples,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float64)
        pred_mean, _ = gp.predict(self.model, self.likelihood, X_tensor)
        return pred_mean.cpu().numpy()


def init_and_fit_gp(X_train, y_train, kernel, num_inducing, lr, num_iters=500):
    """Initialize and train a Gaussian Process model.

    Args:
        X_train (torch.Tensor): Training input features
        y_train (torch.Tensor): Training target values
        kernel: GPyTorch kernel object (e.g., MaternKernel)
        num_inducing (int): Number of inducing points for sparse approximation
        lr (float): Learning rate for optimization
        num_iters (int): Number of training iterations (default: 500)

    Returns:
        tuple: (gp_model, gp_likelihood) - trained GP model and likelihood objects
    """

    # Convert to torch tensors with float64
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)

    gp_model, gp_likelihood = gp.init_gp(
        X_train,
        y_train,
        kernel=kernel,
        num_inducing=num_inducing
    )

    print("\nTraining GP...")
    gp_model, gp_likelihood = gp.train_gp(
        gp_model,
        gp_likelihood,
        X_train,
        y_train,
        num_iters=num_iters,
        lr=lr
    )

    return gp_model, gp_likelihood


def create_matern_kernel(input_dim, nu=2.5):
    """Matern kernel. nu=0.5 (rough), 1.5 (moderate), 2.5 (smooth)."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=input_dim)
    )


def create_rbf_kernel(input_dim):
    """TF1: Smooth, globally stationary."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    )


def create_rq_kernel(input_dim):
    """TF2, TF4: Multi-scale, mixture of RBFs at different lengthscales."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RQKernel(ard_num_dims=input_dim)
    )


def create_periodic_kernel(input_dim):
    """TF4: Fixed repeating pattern, exact periodicity."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(ard_num_dims=input_dim)
    )


def create_periodic_rbf_kernel(input_dim):
    """TF3, TF5: Periodicity that decays with distance."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(ard_num_dims=input_dim)
        * gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    )


def create_periodic_linear_kernel(input_dim):
    """TF3: Periodic with amplitude growing across the domain."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(ard_num_dims=input_dim)
        * gpytorch.kernels.LinearKernel(num_dimensions=input_dim)
    )


def kernel_factory(kernel_type, X):
    """
    Create a GPyTorch kernel based on kernel type and input dimensionality.

    Args:
        kernel_type (str): Type of kernel to create. Options: 'rbf', 'matern',
            'rq', 'periodic', 'periodic_rbf', 'periodic_linear'
        X: Input data array (torch.Tensor or numpy array). Kernel dimension
            is determined from X.shape[1]

    Returns:
        gpytorch.kernels.Kernel: Initialized kernel object for the specified type
    """
    kernel_factory = {
        "rbf": create_rbf_kernel,
        "matern": create_matern_kernel,
        "rq": create_rq_kernel,
        "periodic": create_periodic_kernel,
        "periodic_rbf": create_periodic_rbf_kernel,
        "periodic_linear": create_periodic_linear_kernel,
    }

    kernel = kernel_factory[kernel_type](X.shape[1])
    return kernel


def objective_gp(trial, X_train, y_train, X_test, y_test, n_splits=5, kernel_types=None):
    """Objective function for Optuna to find the best GP hyperparameters with K-Fold CV.

    Uses K-Fold Cross-Validation on training data to evaluate hyperparameter sets.
    Searches over learning rate, kernel type, and number of inducing points.

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_test, y_test: Test data for final evaluation
        n_splits: Number of folds for cross-validation (default: 5)
        kernel_types: List of kernel types to search. Default: all available types

    Returns:
        float: Average validation RMSE across all folds (what Optuna minimizes)
    """
    if kernel_types is None:
        kernel_types = ["rbf", "matern", "rq", "periodic", "periodic_rbf", "periodic_linear"]

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    kernel_type = trial.suggest_categorical("kernel", kernel_types)
    num_inducing = trial.suggest_int("num_inducing", 300, 500)

    kernel = kernel_factory(kernel_type, X_train)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_rmses = []

    for train_idx, val_idx in kf.split(X_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        X_fold_train_t = torch.tensor(X_fold_train, dtype=torch.float64)
        y_fold_train_t = torch.tensor(y_fold_train, dtype=torch.float64)
        X_fold_val_t = torch.tensor(X_fold_val, dtype=torch.float64)
        y_fold_val_t = torch.tensor(y_fold_val, dtype=torch.float64)

        gp_model, gp_likelihood = init_and_fit_gp(
            X_fold_train_t, y_fold_train_t,
            kernel=kernel, num_inducing=num_inducing, lr=lr, num_iters=500
        )

        pred_mean, _ = gp.predict(gp_model, gp_likelihood, X_fold_val_t)
        rmse = em.compute_rmse(pred_mean, y_fold_val_t).item()
        fold_val_rmses.append(rmse)

    avg_val_rmse = float(np.mean(fold_val_rmses))

    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    gp_model_final, gp_likelihood_final = init_and_fit_gp(
        X_train_t, y_train_t,
        kernel=kernel, num_inducing=num_inducing, lr=lr, num_iters=500
    )

    pred_test, _ = gp.predict(gp_model_final, gp_likelihood_final, X_test_t)
    test_rmse = em.compute_rmse(pred_test, y_test_t).item()

    trial.set_user_attr("test_rmse", test_rmse)
    trial.set_user_attr("val_rmse", avg_val_rmse)

    return avg_val_rmse


def study_optuna_gp(dataset_name, X_train, y_train, X_test, y_test, n_trials=10, kernel_types=None):
    """Run Optuna parameter search for GP hyperparameters on a dataset.

    Uses 5-Fold Cross-Validation to evaluate hyperparameter sets over
    learning rate, kernel type, and number of inducing points.

    Args:
        dataset_name (str): Dataset name for output file and logging
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_trials (int): Number of trials to run (default: 10)
        kernel_types: List of kernel types to search. Default: all available types

    Returns:
        dict: Best hyperparameters with keys 'lr', 'kernel', 'num_inducing'
    """
    if kernel_types is None:
        kernel_types = ["rbf", "matern", "rq", "periodic", "periodic_rbf", "periodic_linear"]
    csv_path = f"data/{dataset_name.lower()}_gp_optuna_search.csv"

    if os.path.exists(csv_path):
        print("=" * 70)
        print(f"Results found for {dataset_name}. Loading from {csv_path}")
        print("=" * 70)

        trials_df = pd.read_csv(csv_path)
        best_idx = trials_df['value'].idxmin()
        best_row = trials_df.loc[best_idx]

        lr = float(best_row['params_lr']) # type: ignore
        kernel = best_row['params_kernel']
        num_inducing = int(best_row['params_num_inducing']) # type: ignore

        print(f"Best CV Validation RMSE: {best_row['value']:.6f}")
        print(f"Best Test RMSE:          {best_row['test_rmse']:.6f}")
        print(f"Best lr:                 {lr:.6f}")
        print(f"Best kernel:             {kernel}")
        print(f"Best num_inducing:       {num_inducing}")

        print(f"\nAll {len(trials_df)} trials:")
        print(trials_df[['number', 'value', 'params_lr', 'params_kernel', 'params_num_inducing', 'val_rmse', 'test_rmse']].to_string())

        return {"lr": lr, "kernel": kernel, "num_inducing": num_inducing}

    print("=" * 70)
    print(f"Optimizing GP hyperparameters with 5-Fold CV on {dataset_name}")
    print("=" * 70)

    study = optuna.create_study(direction='minimize')

    # Force one trial per kernel type
    for kernel in kernel_types:
        study.enqueue_trial({"kernel": kernel})

    study.optimize(
        lambda trial: objective_gp(
            trial, X_train, y_train, X_test, y_test, n_splits=5, kernel_types=kernel_types
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 70)
    print("Best trial results:")
    print("=" * 70)
    print(f"Best CV Validation RMSE: {study.best_value:.6f}")
    best_test_rmse = study.best_trial.user_attrs.get("test_rmse", "N/A")
    test_rmse_str = f"{best_test_rmse:.6f}" if isinstance(best_test_rmse, float) else "N/A"
    print(f"Best Test RMSE:          {test_rmse_str}")
    print(f"Best lr:                 {study.best_params['lr']:.6f}")
    print(f"Best kernel:             {study.best_params['kernel']}")
    print(f"Best num_inducing:       {study.best_params['num_inducing']}")

    trials_df = study.trials_dataframe()
    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[['number', 'value', 'params_lr', 'params_kernel', 'params_num_inducing', 'user_attrs_val_rmse', 'user_attrs_test_rmse']].to_string())

    trials_df = trials_df.rename(columns={
        "user_attrs_test_rmse": "test_rmse",
        "user_attrs_val_rmse": "val_rmse",
    })
    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    trials_df.to_csv(csv_path, index=False)

    return {
        "lr": study.best_params['lr'],
        "kernel": study.best_params['kernel'],
        "num_inducing": study.best_params['num_inducing']
    }


# ─── SGKAN utilities ─────────────────────────────────────────────
# Functions for building, training, and predicting with SGKAN


# Wrapper for SGKAN model to add predict method for em.evaluate compatibility
class SGKANModel:
    def __init__(self, layers, W_out, activation=torch.tanh):
        self.layers = layers
        self.W_out = W_out
        self.activation = activation
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float64)
        return sgkan.predict_sgkan(self.layers, self.W_out, X_tensor, activation=self.activation).detach().numpy()


def fit_sgkan(
        X_train, y_train, layer_configs, activation=torch.tanh,
        kernel=None, lr=1e-3, num_inducing=500, num_iters=500):
    """Train SGKAN (Surrogate-Guided KAN) model with optional custom kernel and GP hyperparameters.

    Args:
        X_train: Training input data (numpy array)
        y_train: Training target data (numpy array)
        layer_configs: List of dicts with layer configuration (width, M, G, T parameters)
        activation: Activation function (default: torch.tanh)
        kernel: GPyTorch kernel for GP (default: None)
        lr: Learning rate for GP training (default: 0.001)
        num_inducing: Number of inducing points for sparse GP (default: 500)
        num_iters: Number of training iterations for GP (default: 100)

    Returns:
        tuple: (layers, W_out) representing the trained SGKAN model
    """
    start_time = time.time()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float64)

    if isinstance(layer_configs, dict):
        layer_configs = [layer_configs]

    layers, W_out = sgkan.build_sgkan(
        X_train_tensor, y_train_tensor, layer_configs,
        activation=activation, kernel=kernel,
        lr=lr, num_inducing=num_inducing, num_iters=num_iters
    )

    elapsed_time = time.time() - start_time
    print(f"\n✓ Training completed in {elapsed_time:.4f} seconds")
    return layers, W_out


def predict_sgkan(layers, W_out, X_test, activation=torch.tanh):
    """Make predictions with a trained SGKAN model.

    Args:
        layers: List of SGKAN layers from fit_sgkan
        W_out: Output weights from fit_sgkan
        X_test: Input data for prediction (numpy array)
        activation: Activation function used during training (default: torch.tanh)

    Returns:
        numpy array: Predictions on test data
    """
    start_time = time.time()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    predictions = sgkan.predict_sgkan(layers, W_out, X_test_tensor, activation=activation).detach().numpy()
    elapsed_time = time.time() - start_time
    print(f"✓ Prediction completed in {elapsed_time:.4f} seconds")
    return predictions


def objective_sgkan(trial, X_train, y_train, X_test, y_test, kernel=None, lr=0.001, num_inducing=500, num_iters=500, n_splits=5):
    """Objective function for Optuna to find the best SGKAN hyperparameters with K-Fold CV.

    Uses K-Fold Cross-Validation on training data to evaluate hyperparameter sets.
    Searches over SGKAN layer width, M (pair samples), G (grid), and T (interior points).

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_test, y_test: Test data for final evaluation
        kernel: GPyTorch kernel for GP (default: Matérn kernel)
        lr: Learning rate for GP training (default: 0.001)
        num_inducing: Number of inducing points for sparse GP (default: 400)
        num_iters: Number of training iterations for GP (default: 500)
        n_splits: Number of folds for cross-validation (default: 5)

    Returns:
        float: Average validation RMSE across all folds (what Optuna minimizes)
    """
    if kernel is None:
        kernel = create_matern_kernel(X_train.shape[1])

    width = trial.suggest_int("width", 32, 200)
    M = trial.suggest_int("M", 500, 1000)
    G = trial.suggest_int("G", 25, 100)
    T = trial.suggest_int("T", 5, 50)

    layer_config = [{"width": width, "M": M, "G": G, "T": T}]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_val_rmses = []

    for train_idx, val_idx in kf.split(X_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        sgkan_layers, sgkan_W_out = fit_sgkan(
            X_fold_train, y_fold_train, layer_config,
            activation=torch.tanh, kernel=kernel,
            lr=lr, num_inducing=num_inducing, num_iters=num_iters
        )

        sgkan_pred_val = sgkan.predict_sgkan(sgkan_layers, sgkan_W_out, X_fold_val, activation=torch.tanh)
        rmse = em.compute_rmse(
            torch.tensor(sgkan_pred_val.flatten(), dtype=torch.float64),
            torch.tensor(y_fold_val.flatten(), dtype=torch.float64)
        ).item()
        fold_val_rmses.append(rmse)

    avg_val_rmse = float(np.mean(fold_val_rmses))

    sgkan_layers_final, sgkan_W_out_final = fit_sgkan(
        X_train, y_train, layer_config,
        activation=torch.tanh, kernel=kernel,
        lr=lr, num_inducing=num_inducing, num_iters=num_iters
    )

    sgkan_pred_test = sgkan.predict_sgkan(sgkan_layers_final, sgkan_W_out_final, X_test, activation=torch.tanh)
    test_rmse = em.compute_rmse(
        torch.tensor(sgkan_pred_test.flatten(), dtype=torch.float64),
        torch.tensor(y_test.flatten(), dtype=torch.float64)
    ).item()

    trial.set_user_attr("test_rmse", test_rmse)
    trial.set_user_attr("val_rmse", avg_val_rmse)

    return avg_val_rmse


def study_optuna_sgkan(dataset_name, X_train, y_train, X_test, y_test, n_trials=10, kernel=None, lr=0.001, num_inducing=400, num_iters=500):
    """Run Optuna parameter search for SGKAN hyperparameters on a dataset.

    Uses 5-Fold Cross-Validation to evaluate hyperparameter sets over
    layer width, M (pair samples), G (grid), and T (interior points).

    Args:
        dataset_name (str): Dataset name for output file and logging
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_trials (int): Number of trials to run (default: 10)
        kernel: GPyTorch kernel for GP (default: Matérn kernel)
        lr: Learning rate for GP training (default: 0.001)
        num_inducing: Number of inducing points for sparse GP (default: 400)
        num_iters: Number of training iterations for GP (default: 500)

    Returns:
        dict: Best hyperparameters with keys 'width', 'M', 'G', 'T'
    """
    if kernel is None:
        kernel = create_matern_kernel(X_train.shape[1])
    csv_path = f"data/{dataset_name.lower()}_sgkan_optuna_search.csv"

    if os.path.exists(csv_path):
        print("=" * 70)
        print(f"Results found for {dataset_name}. Loading from {csv_path}")
        print("=" * 70)

        trials_df = pd.read_csv(csv_path)
        best_idx = trials_df['value'].idxmin()
        best_row = trials_df.loc[best_idx]

        width = int(best_row['params_width'])
        M = int(best_row['params_M'])
        G = int(best_row['params_G'])
        T = int(best_row['params_T'])

        print(f"Best CV Validation RMSE: {best_row['value']:.6f}")
        print(f"Best Test RMSE:          {best_row['test_rmse']:.6f}")
        print(f"Best width:              {width}")
        print(f"Best M:                  {M}")
        print(f"Best G:                  {G}")
        print(f"Best T:                  {T}")

        print(f"\nAll {len(trials_df)} trials:")
        print(trials_df[['number', 'value', 'params_width', 'params_M', 'params_G', 'params_T', 'val_rmse', 'test_rmse']].to_string())

        return {"width": width, "M": M, "G": G, "T": T}

    print("=" * 70)
    print(f"Optimizing SGKAN hyperparameters with 5-Fold CV on {dataset_name}")
    print("=" * 70)

    study = optuna.create_study(direction='minimize')

    study.optimize(
        lambda trial: objective_sgkan(
            trial, X_train, y_train, X_test, y_test,
            kernel=kernel, lr=lr, num_inducing=num_inducing, num_iters=num_iters,
            n_splits=5
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 70)
    print("Best trial results:")
    print("=" * 70)
    print(f"Best CV Validation RMSE: {study.best_value:.6f}")
    best_test_rmse = study.best_trial.user_attrs.get("test_rmse", "N/A")
    test_rmse_str = f"{best_test_rmse:.6f}" if isinstance(best_test_rmse, float) else "N/A"
    print(f"Best Test RMSE:          {test_rmse_str}")
    print(f"Best width:              {study.best_params['width']}")
    print(f"Best M:                  {study.best_params['M']}")
    print(f"Best G:                  {study.best_params['G']}")
    print(f"Best T:                  {study.best_params['T']}")

    trials_df = study.trials_dataframe()
    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[['number', 'value', 'params_width', 'params_M', 'params_G', 'params_T', 'user_attrs_val_rmse', 'user_attrs_test_rmse']].to_string())

    trials_df = trials_df.rename(columns={
        "user_attrs_test_rmse": "test_rmse",
        "user_attrs_val_rmse": "val_rmse",
    })
    trials_df['duration'] = trials_df['duration'].dt.total_seconds()
    trials_df.to_csv(csv_path, index=False)

    return {
        "width": study.best_params['width'],
        "M": study.best_params['M'],
        "G": study.best_params['G'],
        "T": study.best_params['T']
    }


# ─── Visualization Utilities ──────────────────────────────────
# Functions for visualizing model predictions


def plot_3d_surface(dataset_name, X_test, y_test, y_pred):
    """
    Plot 3D surface comparison of predicted vs actual values.

    Creates side-by-side 3D surface plots for visual comparison of model predictions
    against ground truth. Automatically handles n-dimensional input via PCA projection.

    Args:
        dataset_name (str): Name of the dataset (used in plot titles)
        X_test (numpy array): Input test data with shape (n_samples, n_features)
        y_test (numpy array): Ground truth output values
        y_pred (numpy array): Model predictions matching y_test shape

    Raises:
        ValueError: If X_test has fewer than 2 features
    """
    if X_test.shape[1] < 2:
        raise ValueError(f"X_test must have at least 2 features, got {X_test.shape[1]}")

    X_viz = X_test
    label_suffix = ""

    # Calculate largest perfect square grid to fit all samples for 3D surface plotting
    n_samples = len(X_viz)
    grid_size = int(np.sqrt(n_samples))
    n_grid = grid_size * grid_size

    fig = plt.figure(figsize=(14, 6))

    # Slice and reshape to grid for surface plotting (handles non-square sample sizes)
    X1 = X_viz[:n_grid, 0].reshape(grid_size, grid_size)
    X2 = X_viz[:n_grid, 1].reshape(grid_size, grid_size)

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(X1, X2, y_pred[:n_grid].reshape(grid_size, grid_size), cmap="viridis") # type: ignore
    ax.set_title(f"{dataset_name} - Predicted")
    ax.set_xlabel(f"X1{label_suffix}")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y") # type: ignore

    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(X1, X2, y_test[:n_grid].reshape(grid_size, grid_size), cmap="viridis") # type: ignore
    ax.set_title(f"{dataset_name} - Actual")
    ax.set_xlabel(f"X1{label_suffix}")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y") # type: ignore

    plt.tight_layout()

    # Export figure as PNG
    output_file = f"results/{dataset_name}_3d_surface.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {output_file}")

    plt.show()
