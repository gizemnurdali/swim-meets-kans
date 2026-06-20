import torch
import gpytorch
import numpy as np
torch.manual_seed(42)

# ─── Exact GP (for small datasets, exact inference) ──────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process for small-to-medium datasets.

    Uses exact inference with RBF kernel and ARD (automatic relevance determination).
    """
    def __init__(self, X_train, y_train, likelihood, kernel=None):
        """Initialize ExactGPModel.

        Args:
            X_train: Training input tensor, shape (N, D)
            y_train: Training output tensor, shape (N,)
            likelihood: GPyTorch likelihood object
            kernel: Optional kernel module. If None, uses RBFKernel with ARD.
                   If provided, wraps it in ScaleKernel if not already wrapped.
        """
        super().__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel is None:
            # Default: RBF kernel with ARD (separate lengthscale per input dimension)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1])
            )
        elif isinstance(kernel, gpytorch.kernels.ScaleKernel):
            # Already wrapped in ScaleKernel
            self.covar_module = kernel
        else:
            # Wrap kernel in ScaleKernel
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        """Forward pass: compute mean and covariance."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar) # type: ignore
    

# ─── Sparse GP (for large datasets, inducing point approximation) ──────────────────────
class SparseGPModel(gpytorch.models.ApproximateGP):
    """Sparse Gaussian Process using variational inference.

    Uses inducing point approximation for scalability to large datasets.
    Inducing points are initialized randomly and learned during training.
    """
    def __init__(self, X_train, num_inducing=100, kernel=None):
        """Initialize SparseGPModel.

        Args:
            X_train: Training input tensor, shape (N, D)
            num_inducing: Number of inducing points (default: 100)
            kernel: Optional kernel module. If None, uses RBFKernel with ARD.
                   If provided, wraps it in ScaleKernel if not already wrapped.
        """
        # Select random subset of training data as inducing points
        inducing_idx = torch.randperm(X_train.shape[0])[:num_inducing]
        inducing_points = X_train[inducing_idx]

        # Variational distribution q(u) over inducing points
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        # Variational strategy: defines how GP approximation is constructed
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True  # inducing points are optimized during training
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        if kernel is None:
            # Default: RBF kernel with ARD (separate lengthscale per input dimension)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1])
            )
        elif isinstance(kernel, gpytorch.kernels.ScaleKernel):
            # Already wrapped in ScaleKernel
            self.covar_module = kernel
        else:
            # Wrap kernel in ScaleKernel
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        """Forward pass: compute mean and covariance."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar) # type: ignore
    

# Initialize GP model and likelihood based on dataset size.
def init_gp(X_train, y_train, num_inducing=None, kernel=None):
    """Initialize a Gaussian Process model, auto-selected by training set size.

    Uses ExactGP for N <= 5000, SparseGP for N > 5000.
    Inducing points are scaled automatically: min(500, max(200, N // 20)),
    unless overridden via num_inducing.

    Args:
        X_train: Training input tensor, shape (N, D)
        y_train: Training output tensor, shape (N,)
        num_inducing: Optional number of inducing points for SparseGP
        kernel: Optional kernel module. If None, uses RBFKernel with ARD.

    Returns:
        model: GPyTorch model (ExactGPModel or SparseGPModel)
        likelihood: GPyTorch likelihood object
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    N = X_train.shape[0]

    if N < 2000:
        model = ExactGPModel(X_train, y_train, likelihood, kernel=kernel)
        print(f"[DEBUG] Initialized ExactGPModel: X_train.shape={X_train.shape}")
    else:
        # Use 5% of N, clamped between 200 (minimum quality) and 500 (cost capacity)
        if num_inducing is None:
            num_inducing = min(500, max(200, N // 20))
        model = SparseGPModel(X_train, num_inducing, kernel=kernel)
        print(f"[DEBUG] Initialized SparseGPModel: X_train.shape={X_train.shape}, num_inducing={num_inducing}")

    return model, likelihood


# Train GP hyperparameters using exact or variational objective.
def train_gp(model, likelihood, X_train, y_train, num_iters=100, lr=0.1):
    """Train a Gaussian Process model.
    
    Args:
        model: GPyTorch model (ExactGPModel or SparseGPModel)
        likelihood: GPyTorch likelihood object
        X_train: Training input tensor, shape (N, D)
        y_train: Training output tensor, shape (N,)
        num_iters: Number of training iterations (default: 100)
        lr: Learning rate for Adam optimizer (default: 0.1)
    
    Returns:
        model: Trained model in eval mode
        likelihood: Trained likelihood in eval mode
    """

    # Set to train mode - required by parameters for optimization
    unfreeze_gp(model, likelihood)

    # Combine model and likelihood parameters for optimization
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )

    # Choose loss function based on GP type
    if isinstance(model, ExactGPModel):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        loss_name = "MLL"
    else:
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.shape[0])
        loss_name = "ELBO"

    print(f"[DEBUG] Training {type(model).__name__}: {num_iters} iters, lr={lr}, loss={loss_name}")

    # Training loop
    for i in range(num_iters):
        optimizer.zero_grad()
        # Only consider latent f(X_train)
        output = model(X_train) 
        # ← Noise estimated HERE
        loss = -mll(output, y_train) # type: ignore
        loss.backward()
        optimizer.step()
        
        if (i + 1) % max(1, num_iters // 5) == 0:
            print(f"  Iter {i+1}/{num_iters}, Loss: {loss.item():.6f}")
    
    print(f"[DEBUG] Training complete.")

    try:
        print(f"  Lengthscale: {model.covar_module.base_kernel.lengthscale.detach()}")
    except (AttributeError, RuntimeError):
        print(f"  Lengthscale: Not available for this kernel")

    try:
        print(f"  Kernel type: {type(model.covar_module.base_kernel).__name__}")
    except (AttributeError, RuntimeError):
        print(f"  Kernel type: Unknown")

    try:
        print(f"  Outputscale: {model.covar_module.outputscale.item():.4f}")
    except (AttributeError, RuntimeError):
        print(f"  Outputscale: Not available for this kernel")

    try:
        print(f"  Noise: {likelihood.noise.item():.6f}")
    except (AttributeError, RuntimeError):
        print(f"  Noise: Not available")

    try:
        print(f"  Mean const: {model.mean_module.constant.item():.4f}")  # type: ignore
    except (AttributeError, RuntimeError):
        print(f"  Mean const: Not available")

    # Switch to eval mode after training - disables gradient tracking for prediction
    freeze_gp(model, likelihood)    

    return model, likelihood


# Predict posterior mean and std on input points.
def predict(model, likelihood, X):
    """Make predictions using trained GP model.
    
    Args:
        model: Trained GPyTorch model
        likelihood: Trained likelihood object
        X: Input tensor, shape (N, D)
    
    Returns:
        mean: Predicted mean, shape (N,)
        std: Predicted standard deviation, shape (N,)
    """
    # Ensure eval mode - safe to call even if already set
    freeze_gp(model, likelihood)

    with torch.no_grad():
        # Query model and likelihood
        output = likelihood(model(X))
        mean = output.mean
        std = output.stddev
    
    return mean, std


# Switch model and likelihood to eval mode.
def freeze_gp(model, likelihood):
    """Freeze (switch to eval mode) GP model and likelihood.
    
    Sets model and likelihood to evaluation mode, disabling gradient computation
    and dropout. Use this before making predictions or after training is complete.
    
    Args:
        model: GPyTorch model
        likelihood: GPyTorch likelihood object
    """
    model.eval()
    likelihood.eval()
    print(f"[DEBUG] {type(model).__name__} and likelihood frozen (eval mode)")


# Switch model and likelihood to train mode.
def unfreeze_gp(model, likelihood):
    """Unfreeze (switch to train mode) GP model and likelihood.
    
    Sets model and likelihood to training mode, enabling gradient computation.
    Use this before training.
    
    Args:
        model: GPyTorch model
        likelihood: GPyTorch likelihood object
    """
    model.train()
    likelihood.train()
    print(f"[DEBUG] {type(model).__name__} and likelihood unfrozen (train mode)")