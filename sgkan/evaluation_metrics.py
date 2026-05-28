import torch


# Compute mean squared error between predictions and targets.
def compute_mse(y_pred, y_true):
    """Compute Mean Squared Error.
    
    Args:
        y_pred: Predicted values, shape (N,)
        y_true: Ground truth values, shape (N,)
    
    Returns:
        mse: MSE value (scalar)
    """
    mse = ((y_pred - y_true) ** 2).mean()
    return mse


# Compute relative L2 error normalized by target magnitude.
def compute_relative_l2(y_pred, y_true):
    """Compute relative L2 error: ||y_pred - y_true||_2 / ||y_true||_2
    
    Args:
        y_pred: Predicted values, shape (N,)
        y_true: Ground truth values, shape (N,)
    
    Returns:
        rel_l2: Relative L2 error (scalar)
    """
    rel_l2 = torch.norm(y_pred - y_true) / torch.norm(y_true)
    return rel_l2