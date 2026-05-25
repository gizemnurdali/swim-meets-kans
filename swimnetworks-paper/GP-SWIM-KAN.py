import torch
import gpytorch
import numpy as np

M = 100
random_seed = 42
torch.manual_seed(random_seed)


# STAGE 1: FIT Gaussian Process in training set
# ─── 1. Create dataset ───────────────────────────────────
N_train = 100
N_test  = 300

# Train set: uniform in [-3, 3]
X_train = torch.linspace(-3, 3, N_train).unsqueeze(1)  # shape (100, 1)
y_train = torch.sin(X_train.squeeze()) + 0.1 * torch.randn(N_train)

# Test set: uniform in [-4, 4]
X_test  = torch.linspace(-4, 4, N_test).unsqueeze(1)   # shape (300, 1)
y_test  = torch.sin(X_test.squeeze())                   # noiseless ground truth

print(f"X_train: {X_train.shape},  y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},   y_test:  {y_test.shape}")

# ─── 2. Define GP model ──────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar) # type: ignore
    
# ─── 3. Initialize ───────────────────────────────────────
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model      = ExactGPModel(X_train, y_train, likelihood)

# ─── 4. Train GP ────────────────────────────────────────────
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll       = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

num_iters = 100 
for i in range(num_iters):
    optimizer.zero_grad()
    loss = -mll(model(X_train), y_train) # type: ignore
    loss.backward()
    optimizer.step()

print(f"\nGP fitted successfully.")
print(f"  Length scale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
print(f"  Output scale: {model.covar_module.outputscale.item():.4f}")
print(f"  Noise:        {likelihood.noise.item():.4f}")
print(f"  Mean const:   {model.mean_module.constant.item():.4f}") # type: ignore

# ─── 5. Freeze GP ────────────────────────────────────────
model.eval()
likelihood.eval()
print(f"\nGP frozen. Ready for Stage 2 — pair sampling.")


# STAGE 2: GP Driven SWIM Scores
# ── Step 1: Sample M candidate pairs ─────────────────
# Same logic as SWIM — delta trick guarantees idx_from != idx_to
M = 100 # Update later
rng = np.random.default_rng(random_seed)
idx_from = rng.integers(low=0, high=N_train, size=M)
delta    = rng.integers(low=1, high=N_train-1, size=M)
idx_to   = (idx_from + delta) % N_train

# Select corresponding values using the indices list
x_a = X_train[idx_from]   # shape (M, d)
x_b = X_train[idx_to]     # shape (M, d)
y_a = y_train[idx_from]   # shape (M,)
y_b = y_train[idx_to]   # shape (M,)

# ── Step 2: Create T interior points per pair ────────
# t in {1/(T+1), 2/(T+1), ..., T/(T+1)} — avoids endpoints
T = 3
t_values = torch.linspace(0, 1, T+2)[1:-1]  # shape (T,)
# x_t shape: (M, T, d)
# x_a[:, None, :] broadcasts to (M, 1, d)
x_interior = (
    x_a.unsqueeze(1) +
    t_values.view(1, T, 1) * (x_b - x_a).unsqueeze(1)
)  # (M, T, d)
# Flatten to (M*T, d) for single GP query
x_interior_flat = x_interior.reshape(M * T, -1)

# ── Step 3: Query frozen GP at interior points ───────
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred         = likelihood(model(x_interior_flat))
    mu_interior  = pred.mean.reshape(M, T)      # (M, T)
    std_interior = pred.variance.sqrt().reshape(M, T)  # (M, T)

# ── Step 4: SWIM style scores using posterior GP ───────
# ── Endpoint gradients: need grad through mu ──
x_a_g = x_a.detach().requires_grad_(True)  # (M, d)
x_b_g = x_b.detach().requires_grad_(True)  # (M, d)

with gpytorch.settings.fast_pred_var():
    pred_a = likelihood(model(x_a_g))
    pred_b = likelihood(model(x_b_g))
    
    mu_a  = pred_a.mean          # (M,)
    std_a = pred_a.variance.sqrt()  # (M,)
    
    mu_b  = pred_b.mean          # (M,)
    std_b = pred_b.variance.sqrt()  # (M,)

# ── Numerator: L-inf norm of gradient difference ──
grad_a = torch.autograd.grad(mu_a.sum(), x_a_g)[0]  # (M, d)
grad_b = torch.autograd.grad(mu_b.sum(), x_b_g)[0]  # (M, d)

numerator = (grad_a - grad_b).abs().max(dim=1).values  # (M,)

# ── Denominator: uncertainty at endpoints + along segment ──
epsilon = 1e-6
denominator = std_a + std_interior.sum(dim=1) + std_b + epsilon  # (M,)

# ── Scores and probabilities ──
scores      = numerator / denominator          # (M,)
probs       = scores / scores.sum()            # (M,)  sums to 1

# ── Step 6: Sample winning pairs ──
layer_width = 5
probs_np = probs.detach().cpu().numpy()  # multinomial needs numpy for rng.choice

selected_idx = rng.choice(
    M,                        # sample from M candidates
    size=layer_width,         # pick layer_width winners
    replace=True,             # same pair can be selected multiple times
    p=probs_np
)

# Index into your pair tensors
x_a_selected = x_a[selected_idx]  # (layer_width, d)
x_b_selected = x_b[selected_idx]  # (layer_width, d)

# Index into your pair tensors
y_a_selected = y_a[selected_idx]  # (layer_width, d)
y_b_selected = y_b[selected_idx]  # (layer_width, d)

# ── Step 7: Sample GP posterior functions over selected segments ──

# Create dense interior points for each selected pair (for smooth function)
T_sample = 50  # more points for a smooth curve equivalent to 50
t_dense  = torch.linspace(0, 1, T_sample)  # (T_sample,)

# Interior points for selected pairs only
x_segments = (
    x_a_selected.unsqueeze(1) +
    t_dense.view(1, T_sample, 1) * (x_b_selected - x_a_selected).unsqueeze(1)
)  # (layer_width, T_sample, d)

# Flatten for GP query
x_segments_flat = x_segments.reshape(layer_width * T_sample, -1)  # (layer_width*T_sample, d)

# Get posterior distribution over these points
with gpytorch.settings.fast_pred_var():
    pred_segments = likelihood(model(x_segments_flat))

# Reshape mean and covariance for sampling
# We need to sample per segment separately
sampled_functions = []

for i in range(layer_width):
    # Points for this segment
    x_seg_i = x_segments[i]  # (T_sample, d)
    
    with gpytorch.settings.fast_pred_var():
        pred_i = likelihood(model(x_seg_i))
    
    # Sample one function from the posterior
    f_sample = pred_i.rsample()  # (T_sample,)
    sampled_functions.append(f_sample)

sampled_functions = torch.stack(sampled_functions)  # (layer_width, T_sample)