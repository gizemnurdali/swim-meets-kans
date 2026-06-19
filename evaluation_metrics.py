import torch
import numpy as np


def to_tensor(arr, dtype=torch.float64):
    """
    Convert array to torch tensor, handling both numpy and torch inputs.

    Flattens the array and converts to specified dtype. If input is already
    a torch tensor, ensures it has the correct dtype.

    Args:
        arr: Input array (numpy array or torch tensor)
        dtype: Target torch dtype (default: torch.float64)

    Returns:
        torch.Tensor: Flattened tensor with specified dtype
    """
    if isinstance(arr, torch.Tensor):
        # Already a tensor, just ensure correct dtype
        return arr.flatten().to(dtype=dtype)
    else:
        # Assume numpy array or similar, convert and flatten
        return torch.tensor(np.asarray(arr).flatten(), dtype=dtype)


# Compute mean absolute error between predictions and targets.
def compute_mae(y_pred, y_true):
    """Compute Mean Absolute Error.

    Args:
        y_pred: Predicted values (numpy array or torch tensor), shape (N,)
        y_true: Ground truth values (numpy array or torch tensor), shape (N,)

    Returns:
        mae: MAE value (scalar)
    """
    y_pred_t = to_tensor(y_pred)
    y_true_t = to_tensor(y_true)
    mae = torch.abs(y_pred_t - y_true_t).mean()
    return mae


# Compute relative L2 error normalized by target magnitude.
def compute_relative_l2(y_pred, y_true):
    """Compute relative L2 error: ||y_pred - y_true||_2 / ||y_true||_2

    Args:
        y_pred: Predicted values (numpy array or torch tensor), shape (N,)
        y_true: Ground truth values (numpy array or torch tensor), shape (N,)

    Returns:
        rel_l2: Relative L2 error (scalar)
    """
    y_pred_t = to_tensor(y_pred)
    y_true_t = to_tensor(y_true)
    rel_l2 = torch.norm(y_pred_t - y_true_t) / torch.norm(y_true_t)
    return rel_l2


# Compute root mean squared error between predictions and targets.
def compute_rmse(y_pred, y_true):
    """Compute Root Mean Squared Error.

    Args:
        y_pred: Predicted values (numpy array or torch tensor), shape (N,)
        y_true: Ground truth values (numpy array or torch tensor), shape (N,)

    Returns:
        rmse: RMSE value (scalar)
    """
    y_pred_t = to_tensor(y_pred)
    y_true_t = to_tensor(y_true)
    rmse = torch.sqrt(((y_pred_t - y_true_t) ** 2).mean())
    return rmse


def evaluate(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train and test sets.

    Args:
        model: Model with a predict method
        X_train: Training input data
        y_train: Training target data
        X_test: Test input data
        y_test: Test target data

    Returns:
        dict: Results with structure {'train': {'mae': float, 'rmse': float},
                                     'test': {'mae': float, 'rmse': float}}
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Each metric function handles type conversion internally
    results = {
        "train": {
            "mae": compute_mae(y_train_pred, y_train).item(),
            "rmse": compute_rmse(y_train_pred, y_train).item(),
        },
        "test": {
            "mae": compute_mae(y_test_pred, y_test).item(),
            "rmse": compute_rmse(y_test_pred, y_test).item(),
        }
    }

    return results


def print_results(results):
    """
    Pretty-print evaluation results in a small table.

    Args:
        results: Dict with structure {'train': {'mae': float, 'rmse': float},
                                     'test': {'mae': float, 'rmse': float}}
    """
    print(f"{'Metric':<15} {'Train':<20} {'Test':<20}")
    print("-" * 55)

    for metric in ["mae", "rmse"]:
        train_val = results["train"][metric]
        test_val = results["test"][metric]
        print(f"{metric.upper():<15} {train_val:<20.6e} {test_val:<20.6e}")

