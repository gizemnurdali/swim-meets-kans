import tqdm
import itertools
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import hkan.swim_sampling as ss
import hkan.hkan_swim as hsc


MININTERVAL = 1
TQDM_DISABLE = True 
random_seed = 42

def set_mininterval(mininterval):
    """Set the minimum interval for tqdm progress bar updates."""
    global MININTERVAL
    MININTERVAL = mininterval

def set_tqdm_disable(disable):
    """Enable or disable tqdm progress bars."""
    global TQDM_DISABLE
    TQDM_DISABLE = disable

def set_random_seed(seed):
    """Set the random seed for SWIM sampling and center generation."""
    global random_seed
    random_seed = seed

class Sigmoid:
    """Sigmoid activation function."""
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return 1 / (1 + np.exp(-self.s * x))

class Gaussian:
    """Gaussian (RBF) activation function."""
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.exp(-((self.s * x) ** 2))

class ReLU:
    """Rectified Linear Unit (ReLU) activation function."""
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.maximum(0, self.s * x)

class Tanh:
    """Hyperbolic tangent (Tanh) activation function."""
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.tanh(self.s * x)

class Softplus:
    """Softplus activation function."""
    def __init__(self, s=1):
        self.s = s

    def __call__(self, x):
        return np.log(1 + np.exp(self.s * x))
    
class Identity:
    """Identity activation function (no transformation)."""
    def __call__(self, x):
        return x



def make_centers(n_vars_out, n_vars_in, n_basis, centers, X=None, y=None, M=None,
                  random_seed=None):
    """
    Generate basis function centers using various strategies.

    Parameters:
    - n_vars_out: Number of output variables.
    - n_vars_in: Number of input variables.
    - n_basis: Number of basis functions per (output, input) edge.
    - centers: Center generation method. Options:
        - 'random': Uniform random in [0, 1]
        - 'equally_spaced': Linearly spaced in [0, 1]
        - 'random_data_points': Random samples from input data
        - 'neuron_shared_swim_centers': One candidate pool per neuron q, distinct pairs per edge
        - 'edge_isolated_swim_centers': Distinct pairs per edge (q,p), 1D distance per column
    - X: Input training data (required for 'random_data_points' and all SWIM variants).
    - y: Target training data (required for all SWIM variants).
    - M: Number of candidate pairs to generate (used by SWIM variants).
    - random_seed: Seed for SWIM variants. Falls back to the module-level
      `random_seed` (see set_random_seed) if None.

    Returns:
    - Centers array of shape (n_vars_out, n_vars_in, n_basis).
    """
    if random_seed is None:
        random_seed = globals()["random_seed"]

    if centers == "random":
        return np.random.uniform(0, 1, (n_vars_out, n_vars_in, n_basis))
    elif centers == "equally_spaced":
        return np.tile(np.linspace(0, 1, n_basis), (n_vars_out, n_vars_in, 1))
    elif centers == "random_data_points":
        if X is not None:
            C = np.empty((n_vars_out, n_vars_in, n_basis))
            for q in range(n_vars_out):
                for p in range(n_vars_in):
                    C[q, p, :] = np.random.choice(X[:, p], n_basis)
            return C
        else:
            raise ValueError(
                "X cannot be None for creating centers using random_data_points method."
            )
    # SWIM centers: see hkan_swim.py for edge_isolated vs neuron_shared trade-offs.
    # Only the centers half is used here; sigmas (if requested) are computed
    # together with centers in one call, inside ExpandingLayer.fit().
    elif centers == "neuron_shared_swim_centers":
        centers_arr, _ = hsc.neuron_shared_swim(
            X, y, M, n_vars_out, n_vars_in, n_basis,
            random_seed=random_seed, use_swim_sigma=False,
        )
        return centers_arr
    elif centers == "edge_isolated_swim_centers":
        centers_arr, _ = hsc.edge_isolated_swim(
            X, y, M, n_vars_out, n_vars_in, n_basis,
            random_seed=random_seed, use_swim_sigma=False,
        )
        return centers_arr

    else:
        raise ValueError(
            "Possible values for 'centers' are: 'random', 'equally_spaced', 'random_data_points', "
            "'neuron_shared_swim_centers', 'edge_isolated_swim_centers'."
        )


def apply_basis_fn(X, centers_arr, basis_fn, q, p, sigmas_arr=None):
    """
    Apply the basis function to the difference between the input data and the centers.

    Parameters:
    - X: Input data. Column vector of shape (n_samples, n_vars_in).
    - centers_arr: Centers of the basis functions. ndarray of shape (n_vars_out, n_vars_in, n_basis).
    - basis_fn: Basis function to apply.
    - q: Index of the output variable.
    - p: Index of the input variable.
    - sigmas_arr: Optional per-(q,p,basis) sigma array, same shape as centers_arr.
      When provided, the (X - center) difference is scaled elementwise by
      sigmas_arr[q, p, :] BEFORE basis_fn is applied. Note this multiplies
      with whatever `basis_fn.s` already does internally (e.g. Sigmoid's
      1/(1+exp(-s*x))) -- pass basis_fn with s=1 if you want sigmas_arr to be
      the sole source of sharpness, rather than a further multiplier on top
      of a separately-tuned global s.

    Returns:
    - The result of applying the basis function to the (optionally scaled)
      difference between the input data and the centers.
    """
    n_samples = X.shape[0]
    n_basis = centers_arr.shape[2]
    diff = X[:, p].reshape(n_samples, 1) - centers_arr[q, p, :].reshape(1, n_basis)
    if sigmas_arr is not None:
        diff = sigmas_arr[q, p, :].reshape(1, n_basis) * diff
    return basis_fn(diff)


class ExpandingLayer(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        n_vars_out,
        n_basis=10,
        centers="random",
        basis_fn=Sigmoid(),
        base_regressor=None,
        sigmas="none",
        sigma_scale=1.0,
        fixed_sigma=1.0,
        random_seed=42,
    ):
        """
        Initialize the ExpandingLayer.

        Parameters:
        - n_vars_out: Number of output variables.
        - n_basis: Number of basis functions.
        - centers: Method to determine the centers of the basis functions.
        - basis_fn: Basis function to use, default is sigmoid.
        - base_regressor: Base regressor to use, default is LinearRegression.
        - sigmas: Method to determine per-edge sigmas. One of:
            - 'none': no per-edge sigma array; sharpness comes only from
              basis_fn's own scalar `s`. Works with any centers method.
            - 'fixed': every basis function gets the constant `fixed_sigma`.
              Works with any centers method (no pair information needed).
            - 'edge_isolated_swim_sigmas' / 'neuron_shared_swim_sigmas':
              sigma derived from each selected pair's own distance. Requires
              centers to be the matching SWIM method
              ('edge_isolated_swim_centers' / 'neuron_shared_swim_centers') --
              centers and sigmas are computed together in one call, so they
              are guaranteed to come from the same selected pairs.
        - sigma_scale: Scale factor for sigma = sigma_scale / |x_a - x_b|,
          used by the two SWIM sigma variants. sigma_min/sigma_max are
          derived internally from the data's pair-distance distribution.
        - fixed_sigma: Constant sigma value used when sigmas == 'fixed'.
        """
        self.n_vars_out = n_vars_out
        self.n_basis = n_basis
        self.centers = centers
        self.basis_fn = basis_fn
        self.base_regressor = base_regressor
        self.sigmas = sigmas
        self.sigma_scale = sigma_scale
        self.fixed_sigma = fixed_sigma
        self.random_seed = random_seed

    def fit(self, X, y=None):
        """
        Fit the model using the input data X and target y.

        Parameters:
        - X: Input data of shape (n_samples, n_vars_in).
        - y: Target data of shape (n_samples,).

        Returns:
        - self: Fitted estimator.
        """

        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=False)

        self.n_samples_, self.n_vars_in_ = X.shape

        swim_fns = {
            "edge_isolated_swim_centers": hsc.edge_isolated_swim,
            "neuron_shared_swim_centers": hsc.neuron_shared_swim,
        }

        if self.sigmas in ("none", "fixed"):
            # Both cases use the same centers computation -- they only
            # differ in what sigmas_arr_ ends up being.
            self.centers_arr_ = make_centers(
                self.n_vars_out, self.n_vars_in_, self.n_basis, self.centers, X, y,
                random_seed=self.random_seed,
            )
            if self.sigmas == "none":
                self.sigmas_arr_ = None
            else:
                self.sigmas_arr_ = np.full(
                    (self.n_vars_out, self.n_vars_in_, self.n_basis), self.fixed_sigma
                )

        elif self.sigmas in ("edge_isolated_swim_sigmas", "neuron_shared_swim_sigmas"):
            # Real SWIM-derived sigma needs the selected pairs, so centers
            # MUST be the matching SWIM method. Centers + sigmas are computed
            # together in one call, guaranteeing they come from the same pairs.
            if self.centers not in swim_fns:
                raise ValueError(
                    f"sigmas={self.sigmas!r} requires centers to be "
                    "'edge_isolated_swim_centers' or 'neuron_shared_swim_centers'; "
                    f"got centers={self.centers!r}."
                )
            self.centers_arr_, self.sigmas_arr_ = swim_fns[self.centers](
                X, y, None, self.n_vars_out, self.n_vars_in_, self.n_basis,
                random_seed=self.random_seed, use_swim_sigma=True,
                sigma_scale=self.sigma_scale,
            )

        else:
            raise ValueError(
                f"Unrecognized sigmas={self.sigmas!r}. Possible values are "
                "'none', 'fixed', 'edge_isolated_swim_sigmas', "
                "'neuron_shared_swim_sigmas'."
            )

        assert self.centers_arr_.shape == (
            self.n_vars_out,
            self.n_vars_in_,
            self.n_basis,
        ), (
            f"Centers shape is {self.centers_arr_.shape}, "
            f"expected {(self.n_vars_out, self.n_vars_in_, self.n_basis)}"
        )

        if self.sigmas_arr_ is not None:
            assert self.sigmas_arr_.shape == (
                self.n_vars_out,
                self.n_vars_in_,
                self.n_basis,
            ), (
                f"Sigmas shape is {self.sigmas_arr_.shape}, "
                f"expected {(self.n_vars_out, self.n_vars_in_, self.n_basis)}"
            )

        self.models_ = []

        for q, p in tqdm.tqdm(
            itertools.product(range(self.n_vars_out), range(self.n_vars_in_)),
            desc="Fitting 1D regressors",
            total=self.n_vars_out * self.n_vars_in_,
            mininterval=MININTERVAL,
            disable=TQDM_DISABLE,
        ):
            transformed_features = apply_basis_fn(
                X, self.centers_arr_, self.basis_fn, q, p, sigmas_arr=self.sigmas_arr_
            )
            reg = clone(self.base_regressor).fit(transformed_features, y)
            self.models_.append((q, p, reg))

        return self

    def transform(self, X):
        """
        Transform the input data X.

        Parameters:
        - X: Input data to transform.

        Returns:
        - Transformed data.
        """
        assert (
            self.n_vars_in_ == X.shape[1]
        ), f"Input data has {X.shape[1]} features but expected {self.n_vars_in_}"

        out = np.empty((self.n_vars_out, self.n_vars_in_, X.shape[0]))

        for q, p, reg in self.models_:
            transformed_features = apply_basis_fn(
                X, self.centers_arr_, self.basis_fn, q, p, sigmas_arr=self.sigmas_arr_
            )
            out[q, p, :] = reg.predict(transformed_features)

        return out


class ConnectingLayer(TransformerMixin, RegressorMixin, BaseEstimator):
    """Connect outputs from ExpandingLayer across input dimensions for each output variable."""

    def __init__(self, base_regressor=None):
        """
        Initialize the ConnectingLayer.

        Parameters:
        - base_regressor: Base regressor to use, default is LinearRegression with intercept.
        """
        self.base_regressor = base_regressor

    def fit(self, X, y=None):
        """
        Fit the connecting layer regressors.

        Parameters:
        - X: Transformed features of shape (n_vars_out, n_vars_in, n_samples) from ExpandingLayer.
        - y: Target data of shape (n_samples,).

        Returns:
        - self: Fitted estimator.
        """
        if self.base_regressor is None:
            self.base_regressor = LinearRegression(fit_intercept=True)

        self.n_vars_out_, self.n_vars_in_, self.n_samples_ = X.shape
        self.models_ = []

        for q in tqdm.tqdm(
            range(self.n_vars_out_),
            desc="Fitting connecting regressors",
            total=self.n_vars_out_,
            mininterval=MININTERVAL,
            disable=TQDM_DISABLE,
        ):
            reg = clone(self.base_regressor).fit(X[q, :, :].T, y)
            self.models_.append((q, reg))

        return self

    def transform(self, X):
        """
        Transform data with multiple output variables.

        Parameters:
        - X: Transformed features of shape (n_vars_out, n_vars_in, n_samples).

        Returns:
        - Predictions of shape (n_samples, n_vars_out).
        """
        assert (
            self.n_vars_out_ > 1
        ), "Unable to transform data with only one output variable. Use predict instead."

        out = np.empty((self.n_vars_out_, X.shape[2]))

        for q, reg in self.models_:
            out[q, :] = reg.predict(X[q, :, :].T)

        return out.T

    def predict(self, X):
        """
        Make predictions for single output variable.

        Parameters:
        - X: Transformed features of shape (1, n_vars_in, n_samples).

        Returns:
        - Predictions of shape (n_samples,).
        """
        assert (
            self.n_vars_out_ == 1
        ), "Unable to predict data with more than one output variable. Use transform instead."

        _, reg = self.models_[0]
        return reg.predict(X[0, :, :].T)


def make_hkan_layer(
    *,
    layer_idx,
    n_vars_out,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
    sigmas="none",
    sigma_scale=1.0,
    fixed_sigma=1.0,
    random_seed=None,
):
    """
    Create a single HKAN layer combining ExpandingLayer and ConnectingLayer.

    Parameters:
    - layer_idx: Index of the layer (used for naming pipeline steps).
    - n_vars_out: Number of output variables.
    - n_basis: Number of basis functions.
    - centers: Method for center selection.
    - basis_fn: Basis function to use.
    - expanding_base_regressor: Base regressor for ExpandingLayer.
    - connecting_base_regressor: Base regressor for ConnectingLayer.
    - sigmas: Method for per-edge sigma selection ('none', 'fixed',
      'edge_isolated_swim_sigmas', 'neuron_shared_swim_sigmas'). Whenever this
      is not 'none', centers must be the matching SWIM method
      ('edge_isolated_swim_centers' or 'neuron_shared_swim_centers') -- the
      merged hsc function is called once internally, so centers/sigmas are
      guaranteed paired from the same selected pairs.
    - sigma_scale: Scale factor for sigma = sigma_scale / |x_a - x_b|, used
      by the two SWIM sigma variants. sigma_min/sigma_max are derived
      internally from the data.
    - fixed_sigma: Constant sigma value used when sigmas == 'fixed'.
    - random_seed: Seed shared by centers and sigmas for this layer.

    Returns:
    - A Pipeline containing ExpandingLayer and ConnectingLayer.
    """
    steps = [
        (
            f"expanding_layer_{layer_idx}",
            ExpandingLayer(
                n_vars_out=n_vars_out,
                n_basis=n_basis,
                centers=centers,
                basis_fn=basis_fn,
                base_regressor=expanding_base_regressor,
                sigmas=sigmas,
                sigma_scale=sigma_scale,
                fixed_sigma=fixed_sigma,
                random_seed=random_seed,
            ),
        ),
        (
            f"connecting_layer_{layer_idx}",
            ConnectingLayer(base_regressor=connecting_base_regressor),
        ),
    ]
    return Pipeline(steps)


def extend_hkan(
    model,
    *,
    layer_idx=None,
    n_vars_out=1,
    n_basis=10,
    centers="random",
    basis_fn=Sigmoid(),
    expanding_base_regressor=None,
    connecting_base_regressor=None,
    sigmas="none",
    sigma_scale=1.0,
    fixed_sigma=1.0,
    random_seed=None,
):
    """
    Extend the HKAN model with an additional HKAN layer.

    Parameters:
    - model: The HKAN Pipeline model to extend.
    - layer_idx: Index for the new layer (defaults to len(model.steps) // 2).
    - n_vars_out: Number of output variables.
    - n_basis: Number of basis functions.
    - centers: Method to determine the centers of the basis functions.
    - basis_fn: Basis function to use, default is sigmoid.
    - expanding_base_regressor: Base regressor for the expanding layer.
    - connecting_base_regressor: Base regressor for the connecting layer.
    - sigmas: Method for per-edge sigma selection ('none', 'fixed',
      'edge_isolated_swim_sigmas', 'neuron_shared_swim_sigmas'). See
      ExpandingLayer/make_hkan_layer for the centers-pairing requirement.
    - sigma_scale: Scale factor for sigma = sigma_scale / |x_a - x_b|.
      sigma_min and sigma_max are derived internally from the data.
    - fixed_sigma: Constant sigma value used when sigmas == 'fixed'.
    - random_seed: Seed shared by centers and sigmas for this layer.

    Returns:
    - A new Pipeline with the additional HKAN layer appended.
    """
    if layer_idx is None:
        layer_idx = len(model.steps) // 2

    new_layer = make_hkan_layer(
        layer_idx=layer_idx,
        n_vars_out=n_vars_out,
        n_basis=n_basis,
        centers=centers,
        basis_fn=basis_fn,
        expanding_base_regressor=expanding_base_regressor,
        connecting_base_regressor=connecting_base_regressor,
        sigmas=sigmas,
        sigma_scale=sigma_scale,
        fixed_sigma=fixed_sigma,
        random_seed=random_seed,
    )

    return Pipeline(model.steps + new_layer.steps)