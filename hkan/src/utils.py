import torch


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.tensor(x, dtype=dtype)


def make_centers(n_vars_out, n_vars_in, n_basis, centers, X=None):

    if centers == "random":
        return torch.rand((n_vars_out, n_vars_in, n_basis), dtype=torch.float32)

    elif centers == "equally_spaced":
        lin = torch.linspace(0.0, 1.0, n_basis, dtype=torch.float32)
        return lin.unsqueeze(0).unsqueeze(0).repeat(n_vars_out, n_vars_in, 1)
    
    elif centers == "random_data_points":
        if X is not None:
            X_t = _to_tensor(X)
            C = torch.empty((n_vars_out, n_vars_in, n_basis), dtype=torch.float32)
            n_samples = X_t.shape[0]
            for q in range(n_vars_out):
                for p in range(n_vars_in):
                    idx = torch.randint(0, n_samples, (n_basis,))
                    C[q, p, :] = X_t[idx, p]
            return C
        else:
            raise ValueError("X cannot be None for creating centers using random_data_points method.")
    else:
        raise ValueError("Possible values for 'centers' are 'random', 'equally_spaced', or 'random_data_points'.")


def apply_basis_fn(X, centers_arr, basis_fn, q, p):
    # X: (n_samples, n_vars_in), centers_arr: (n_vars_out, n_vars_in, n_basis)
    X_t = _to_tensor(X)
    n_samples = X_t.shape[0]
    n_basis = centers_arr.shape[2]
    # compute differences: (n_samples, n_basis)
    dif = X_t[:, p].unsqueeze(1) - centers_arr[q, p, :].unsqueeze(0)
    return basis_fn(dif)