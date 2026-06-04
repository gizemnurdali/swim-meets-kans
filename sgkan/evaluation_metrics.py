import torch


# Compute mean absolute error between predictions and targets.
def compute_mae(y_pred, y_true):
    """Compute Mean Absolute Error.

    Args:
        y_pred: Predicted values, shape (N,)
        y_true: Ground truth values, shape (N,)

    Returns:
        mae: MAE value (scalar)
    """
    mae = torch.abs(y_pred - y_true).mean()
    return mae


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


# Compute root mean squared error between predictions and targets.
def compute_rmse(y_pred, y_true):
    """Compute Root Mean Squared Error.
    
    Args:
        y_pred: Predicted values, shape (N,)
        y_true: Ground truth values, shape (N,)
    
    Returns:
        rmse: RMSE value (scalar)
    """
    rmse = torch.sqrt(((y_pred - y_true) ** 2).mean())
    return rmse


def evaluate(model, X_train, y_train, X_test, y_test):
    
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # convert to torch tensors
    def to_tensor(arr):
        return torch.tensor(arr.flatten(), dtype=torch.float64)
    
    y_train_true_t = to_tensor(y_train)
    y_train_pred_t = to_tensor(y_train_pred)
    y_test_true_t  = to_tensor(y_test)
    y_test_pred_t  = to_tensor(y_test_pred)

    results = {
        "train": {
            "mae":        compute_mae(y_train_pred_t, y_train_true_t).item(),
            "rmse":       compute_rmse(y_train_pred_t, y_train_true_t).item(),
        },
        "test": {
            "mae":        compute_mae(y_test_pred_t, y_test_true_t).item(),
            "rmse":       compute_rmse(y_test_pred_t, y_test_true_t).item(),
        }
    }

    return results

