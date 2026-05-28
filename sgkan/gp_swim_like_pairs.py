import torch
import gpytorch
import numpy as np


# Sample SWIM-style candidate pairs from training data.
def sample_candidate_pairs(X_train, y_train, M=None, random_seed=42):
    """
    Sample M candidate pairs from training data using the SWIM delta trick.
    Guarantees idx_from != idx_to for all pairs.
    
    Args:
        X_train:     (N, D) tensor
        y_train:     (N,)   tensor
        M:           number of candidate pairs (defaults to N if None)
        random_seed: for reproducibility
    
    Returns:
        x_a: (M, D) — start points of pairs
        x_b: (M, D) — end points of pairs
        y_a: (M,)   — target values at x_a
        y_b: (M,)   — target values at x_b
    """
    N = X_train.shape[0]
    # TODO: update with ceiling later
    M = M if M is not None else N 

    rng      = np.random.default_rng(random_seed)
    idx_from = rng.integers(low=0,   high=N,   size=M)
    # delta >= 1 guarantees no self-pairs
    delta    = rng.integers(low=1,   high=N-1, size=M)  
    idx_to   = (idx_from + delta) % N

    x_a = X_train[idx_from]  # (M, D)
    x_b = X_train[idx_to]    # (M, D)
    y_a = y_train[idx_from]  # (M,)
    y_b = y_train[idx_to]    # (M,)

    return x_a, x_b, y_a, y_b


# Create evenly spaced interior points along each pair segment.
def create_interior_points(x_a, x_b, T=30):
    """
    Create T interior points along each pair segment (x_a, x_b).
    Points are evenly spaced, excluding endpoints.
    
    Args:
        x_a: (M, D) — start points
        x_b: (M, D) — end points
        T:   number of interior points per pair
    
    Returns:
        x_interior:      (M, T, D) — interior points per pair
        x_interior_flat: (M*T, D)  — flattened for GP query
    """
    M = x_a.shape[0]

    # t in {1/(T+1), 2/(T+1), ..., T/(T+1)} — avoids endpoints
    t_values = torch.linspace(0, 1, T + 2)[1:-1]  # (T,)

    # x_a[:, None, :] broadcasts (M,1,D) + (1,T,1)*(M,1,D) → (M,T,D)
    x_interior = (
        x_a.unsqueeze(1) +
        t_values.view(1, T, 1) * (x_b - x_a).unsqueeze(1)
    )  # (M, T, D)

    x_interior_flat = x_interior.reshape(M * T, -1)  # (M*T, D)

    return x_interior, x_interior_flat


# Compute GP-SWIM scores from posterior gradients and uncertainty.
def compute_score_g(model, x_a, x_b, T=30, epsilon=1e-6):
    """
    Compute GP-SWIM pair scores using GP posterior gradients and uncertainty.

    The score for each pair (x_a, x_b) is:
        score = ||grad_mu(x_a) - grad_mu(x_b)||_inf  /  (std(x_a) + sum(std(interior)) + std(x_b))

    High score = large gradient difference (function varies a lot along this pair)
                 relative to low uncertainty (GP is confident about this region)

    Args:
        model:      frozen GP model (in eval mode)
        x_a:        (M, D) — start points of candidate pairs
        x_b:        (M, D) — end points of candidate pairs
        T:          number of interior points per pair (default: 30)
        epsilon:    numerical stability constant (default: 1e-6)

    Returns:
        scores: (M,) — raw unnormalized scores per pair
        probs:  (M,) — normalized probabilities (sums to 1)
    """
    M = x_a.shape[0]

    # ── Step 1: Interior points for uncertainty along segment ──────────────────
    _, x_interior_flat = create_interior_points(x_a, x_b, T=T)

    # Query GP at interior points (no grad needed — only for std)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_interior = model(x_interior_flat)
        std_interior  = pred_interior.stddev.reshape(M, T)  # (M, T)

    # ── Step 2: Endpoint gradients through GP posterior mean ──────────────────
    # requires_grad=True so autograd can flow through GP mean
    x_a_g = x_a.detach().requires_grad_(True)  # (M, D)
    x_b_g = x_b.detach().requires_grad_(True)  # (M, D)

    with gpytorch.settings.fast_pred_var():
        pred_a = model(x_a_g)
        pred_b = model(x_b_g)

        mu_a  = pred_a.mean    # (M,)
        std_a = pred_a.stddev  # (M,) — latent uncertainty, no observation noise
        mu_b  = pred_b.mean    # (M,)
        std_b = pred_b.stddev  # (M,) — latent uncertainty, no observation noise

    # retain_graph=True: keeps computation graph alive for std_a/std_b after grad computation
    grad_a = torch.autograd.grad(mu_a.sum(), x_a_g, retain_graph=True)[0]  # (M, D)
    grad_b = torch.autograd.grad(mu_b.sum(), x_b_g, retain_graph=True)[0]  # (M, D)

    # ── Step 3: Score = gradient difference / uncertainty ─────────────────────
    numerator   = (grad_a - grad_b).abs().max(dim=1).values          # (M,) L-inf norm
    denominator = std_a + std_interior.sum(dim=1) + std_b + epsilon  # (M,)

    scores = numerator / denominator  # (M,)
    probs  = scores / scores.sum()    # (M,) sums to 1

    print(f"[DEBUG] Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    print(f"[DEBUG] Probs:  min={probs.min():.6f},  max={probs.max():.6f},  sum={probs.sum():.6f}")

    return scores, probs


# Sample informative pairs based on GP-SWIM probabilities.
def select_pairs(x_a, x_b, probs, layer_width, random_seed=42):
    """
    Select layer_width winning pairs from candidates using GP-SWIM probabilities.
    
    Args:
        x_a:         (M, D) — candidate start points
        x_b:         (M, D) — candidate end points
        probs:       (M,)   — sampling probabilities (must sum to 1)
        layer_width: number of pairs to select (= number of neurons/edges)
        random_seed: for reproducibility
    
    Returns:
        x_a_selected: (layer_width, D) — selected start points
        x_b_selected: (layer_width, D) — selected end points
        selected_idx: (layer_width,)   — indices into original candidate arrays
    """
    M = x_a.shape[0]

    rng      = np.random.default_rng(random_seed)
    probs_np = probs.detach().cpu().numpy()

    selected_idx = rng.choice(
        M,                   # sample from M candidates
        size=layer_width,    # pick layer_width winners
        replace=True,        # same pair can be selected multiple times
        p=probs_np
    )

    x_a_selected = x_a[selected_idx]  # (layer_width, D)
    x_b_selected = x_b[selected_idx]  # (layer_width, D)

    print(f"[DEBUG] Selected {layer_width} pairs from {M} candidates")
    print(f"[DEBUG] Unique pairs selected: {len(set(selected_idx))} / {layer_width}")

    return x_a_selected, x_b_selected, selected_idx