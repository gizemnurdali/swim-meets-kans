import itertools
import tqdm

import numpy as np
import torch
import torch.nn as nn


MININTERVAL = 1
TQDM_DISABLE = True


def set_mininterval(mininterval):
    global MININTERVAL
    MININTERVAL = mininterval


def set_tqdm_disable(disable):
    global TQDM_DISABLE
    TQDM_DISABLE = disable


class Sigmoid:
    def __init__(self, s=1.0):
        self.s = s

    def __call__(self, x: torch.Tensor):
        return 1.0 / (1.0 + torch.exp(-self.s * x))


class Gaussian:
    def __init__(self, s=1.0):
        self.s = s

    def __call__(self, x: torch.Tensor):
        return torch.exp(-((self.s * x) ** 2))


class ReLU:
    def __init__(self, s=1.0):
        self.s = s

    def __call__(self, x: torch.Tensor):
        return torch.maximum(torch.zeros_like(x), self.s * x)


class Tanh:
    def __init__(self, s=1.0):
        self.s = s

    def __call__(self, x: torch.Tensor):
        return torch.tanh(self.s * x)


class Softplus:
    def __init__(self, s=1.0):
        self.s = s

    def __call__(self, x: torch.Tensor):
        return torch.log1p(torch.exp(self.s * x))


class Identity:
    def __call__(self, x: torch.Tensor):
        return x


def _to_tensor(x, dtype=torch.float32):
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


class ExpandingLayerTorch(nn.Module):
    """Expanding layer implemented as an nn.Module.

    - weights are stored as a Parameter when `learnable=True`, otherwise as a buffer.
    - `fit` performs closed-form least-squares initialization of weights.
    - `transform` / `forward` computes per-(q,p) responses producing
      a tensor shaped (n_vars_out, n_vars_in, n_samples).
    """

    def __init__(self, n_vars_out, n_basis=10, centers="random", basis_fn=Sigmoid(), learnable=False):
        super().__init__()
        self.n_vars_out = n_vars_out
        self.n_basis = n_basis
        self.centers = centers
        self.basis_fn = basis_fn
        self.learnable = bool(learnable)

        # created in fit
        self.n_vars_in_ = None
        self.register_buffer("centers_arr_", None)
        # weights either a Parameter or a buffer named 'weights'
        self.weights = None

    def fit(self, X, y=None):
        X_t = _to_tensor(X)
        if y is None:
            raise ValueError("y must be provided to fit ExpandingLayerTorch")
        y_t = _to_tensor(y).view(-1)

        self.n_samples_, self.n_vars_in_ = X_t.shape

        centers_arr = make_centers(self.n_vars_out, self.n_vars_in_, self.n_basis, self.centers, X)
        assert centers_arr.shape == (self.n_vars_out, self.n_vars_in_, self.n_basis)
        self.register_buffer("centers_arr_", centers_arr)

        W = torch.empty((self.n_vars_out, self.n_vars_in_, self.n_basis), dtype=torch.float32)

        for q, p in tqdm.tqdm(itertools.product(range(self.n_vars_out), range(self.n_vars_in_)),
                               desc="Fitting 1D regressors",
                               total=self.n_vars_out * self.n_vars_in_,
                               mininterval=MININTERVAL,
                               disable=TQDM_DISABLE):
            Phi = apply_basis_fn(X_t, self.centers_arr_, self.basis_fn, q, p)
            try:
                sol = torch.linalg.lstsq(Phi, y_t.unsqueeze(1))
                w = sol.solution.squeeze(1)
            except Exception:
                pinv = torch.linalg.pinv(Phi)
                w = (pinv @ y_t.unsqueeze(1)).squeeze(1)
            W[q, p, :] = w

        if self.learnable:
            self.weights = nn.Parameter(W)
        else:
            # register as buffer for non-learnable closed-form weights
            self.register_buffer("weights", W)

        return self

    def transform(self, X):
        return self.forward(X)

    def forward(self, X):
        X_t = _to_tensor(X)
        assert self.n_vars_in_ == X_t.shape[1]

        W = self.weights if isinstance(self.weights, torch.Tensor) else self.weights
        if W is None:
            # if weights stored as buffer (registered as 'weights')
            W = getattr(self, "weights")

        out = torch.empty((self.n_vars_out, self.n_vars_in_, X_t.shape[0]), dtype=torch.float32)

        for q in range(self.n_vars_out):
            for p in range(self.n_vars_in_):
                Phi = apply_basis_fn(X_t, self.centers_arr_, self.basis_fn, q, p)
                w = W[q, p, :]
                preds = Phi @ w
                out[q, p, :] = preds

        return out


class ConnectingLayerTorch(nn.Module):
    """Connecting layer as nn.Module.

    - holds weights of shape (n_vars_out, n_vars_in + 1) including intercept
    - `fit` computes closed-form least-squares initialization
    - `forward` for multi-output returns shape (n_vars_out, n_samples)
    - `predict` returns numpy array for single-output case to match previous API
    """

    def __init__(self, base_regressor=None, learnable=False):
        super().__init__()
        self.base_regressor = base_regressor
        self.learnable = bool(learnable)
        self.register_buffer("weights", None)
        self.n_vars_out_ = None
        self.n_vars_in_ = None

    def fit(self, X, y=None):
        X_t = _to_tensor(X)
        if y is None:
            raise ValueError("y must be provided to fit ConnectingLayerTorch")

        y_t = _to_tensor(y).view(-1)
        self.n_vars_out_, self.n_vars_in_, self.n_samples_ = X_t.shape

        W = torch.empty((self.n_vars_out_, self.n_vars_in_ + 1), dtype=torch.float32)

        for q in tqdm.tqdm(range(self.n_vars_out_),
                           desc="Fitting connecting regressors",
                           total=self.n_vars_out_,
                           mininterval=MININTERVAL,
                           disable=TQDM_DISABLE):
            features = X_t[q, :, :].T
            ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
            features_i = torch.cat([features, ones], dim=1)
            try:
                sol = torch.linalg.lstsq(features_i, y_t.unsqueeze(1))
                w = sol.solution.squeeze(1)
            except Exception:
                pinv = torch.linalg.pinv(features_i)
                w = (pinv @ y_t.unsqueeze(1)).squeeze(1)
            W[q, :] = w

        if self.learnable:
            self.weights = nn.Parameter(W)
        else:
            self.register_buffer("weights", W)

        return self

    def transform(self, X):
        # return (n_samples, n_vars_out) like previous transform returned out.T
        X_t = _to_tensor(X)
        W = self.weights if isinstance(self.weights, torch.Tensor) else getattr(self, "weights")
        assert W is not None
        assert self.n_vars_out_ > 1, "Unable to transform data with only one output variable. Use predict instead."

        out = torch.empty((self.n_vars_out_, X_t.shape[2]), dtype=torch.float32)
        for q in range(self.n_vars_out_):
            features = X_t[q, :, :].T
            ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
            features_i = torch.cat([features, ones], dim=1)
            w = W[q, :]
            preds = features_i @ w
            out[q, :] = preds

        return out.T

    def forward(self, X):
        # for consistency with transform
        return self.transform(X)

    def predict(self, X):
        X_t = _to_tensor(X)
        W = self.weights if isinstance(self.weights, torch.Tensor) else getattr(self, "weights")
        assert W is not None
        assert self.n_vars_out_ == 1, "Unable to predict data with more than one output variable. Use transform instead."
        w = W[0, :]
        features = X_t[0, :, :].T
        ones = torch.ones((features.shape[0], 1), dtype=features.dtype)
        features_i = torch.cat([features, ones], dim=1)
        return (features_i @ w).numpy()


class TorchPipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        data = X
        for name, step in self.steps:
            # call fit
            if hasattr(step, "fit"):
                step.fit(data, y)
            # then transform for the next step if available
            if hasattr(step, "transform"):
                # expanding layer expects the original X for transforming into basis responses
                if isinstance(step, ExpandingLayerTorch):
                    data = step.transform(X)
                else:
                    data = step.transform(data)
        return self

    def transform(self, X):
        data = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                data = step.transform(data)
        return data

    def predict(self, X):
        data = X
        for name, step in self.steps:
            if hasattr(step, "transform"):
                data = step.transform(data)
            if hasattr(step, "predict"):
                return step.predict(data)
        raise RuntimeError("No predict method found in pipeline steps")


def make_hkan_layer(*, layer_idx, n_vars_out, n_basis=10, centers="random", basis_fn=Sigmoid(), expanding_base_regressor=None, connecting_base_regressor=None):
    steps = [
        (f"expanding_layer_{layer_idx}", ExpandingLayerTorch(n_vars_out=n_vars_out, n_basis=n_basis, centers=centers, basis_fn=basis_fn, base_regressor=expanding_base_regressor)),
        (f"connecting_layer_{layer_idx}", ConnectingLayerTorch(base_regressor=connecting_base_regressor)),
    ]
    return TorchPipeline(steps)


def extend_hkan(model, *, layer_idx=None, n_vars_out=1, n_basis=10, centers="random", basis_fn=Sigmoid(), expanding_base_regressor=None, connecting_base_regressor=None):
    if layer_idx is None:
        layer_idx = len(model.steps) // 2

    new_layer = make_hkan_layer(layer_idx=layer_idx, n_vars_out=n_vars_out, n_basis=n_basis, centers=centers, basis_fn=basis_fn, expanding_base_regressor=expanding_base_regressor, connecting_base_regressor=connecting_base_regressor)
    return TorchPipeline(model.steps + new_layer.steps)


if __name__ == "__main__":
    # small smoke test
    import numpy as np

    N = 100
    D = 3
    X = np.random.rand(N, D).astype(np.float32)
    # simple linear target
    true_w = np.array([1.2, -0.7, 0.3], dtype=np.float32)
    y = X @ true_w + 0.5 * np.random.randn(N).astype(np.float32)

    pipe = make_hkan_layer(layer_idx=0, n_vars_out=1, n_basis=8, centers="random", basis_fn=Sigmoid())
    # fit expanding then connecting manually to match sklearn pipeline flow
    exp = pipe.steps[0][1]
    conn = pipe.steps[1][1]
    exp.fit(X, y)
    transformed = exp.transform(X)
    conn.fit(transformed, y)
    if conn.n_vars_out_ > 1:
        out = conn.transform(transformed)
    else:
        out = conn.predict(transformed)
    print("Smoke test output shape:", getattr(out, 'shape', np.shape(out)))
