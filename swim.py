import numpy as np


def sample_candidate_pairs(X_train, y_train, M=None, random_seed=42):
    """
    Sample M candidate pairs from training data using the SWIM delta trick.

    Creates M pairs (x_a, x_b) by sampling random starting indices and adding
    a random offset (delta). Guarantees idx_from != idx_to for all pairs,
    avoiding self-pairs.

    These pairs are later scored by gradient magnitude to identify informative
    samples (high output change relative to input distance).

    Args:
        X_train:     Input data of shape (N, D).
        y_train:     Target data of shape (N,) or (N, D_y).
        M:           Number of candidate pairs to generate. Defaults to N.
        random_seed: For reproducibility.

    Returns:
        x_a: Start points of pairs, shape (M, D).
        x_b: End points of pairs, shape (M, D).
        y_a: Target values at x_a, shape (M,) or (M, D_y).
        y_b: Target values at x_b, shape (M,) or (M, D_y).
    """
    N = X_train.shape[0]
    M = M if M is not None else N

    rng = np.random.default_rng(random_seed)
    idx_from = rng.integers(low=0, high=N, size=M)
    # delta >= 1 ensures idx_from != idx_to (no self-pairs)
    delta = rng.integers(low=1, high=N-1, size=M)
    idx_to = (idx_from + delta) % N

    x_a = X_train[idx_from]
    x_b = X_train[idx_to]
    y_a = y_train[idx_from]
    y_b = y_train[idx_to]

    return x_a, x_b, y_a, y_b


def create_swim_probabilities(x_a, x_b, y_a, y_b, dist_min=1e-10):
    """
    Compute SWIM-style sampling probabilities for candidate pairs.

    Scores each pair by the ratio of output change to input distance:

        p ∝ ||y_b - y_a||_∞ / ||x_b - x_a||_2

    Higher scores are given to pairs with large output changes relative to
    input distance (informative pairs). Probabilities are normalized to sum to 1.

    **Edge case:** If all gradients are near-zero, falls back to uniform sampling.

    Args:
        x_a: Start points of candidate pairs, shape (M, D).
        x_b: End points of candidate pairs, shape (M, D).
        y_a: Target values at x_a, shape (M,) or (M, D_y).
        y_b: Target values at x_b, shape (M,) or (M, D_y).
        dist_min: Minimum threshold to avoid division by zero and detect
            near-zero gradients.

    Returns:
        probs: Normalized sampling probabilities, shape (M,), summing to 1.
    """
    # Adjust shape
    y_a = y_a.reshape(-1, 1)
    y_b = y_b.reshape(-1, 1)

    # Compute output space differences (gradient components)
    dy = y_b - y_a

    # # Compute input space distances and clip to avoid division by zero
    dists = np.linalg.norm(x_b - x_a, axis=-1, keepdims=True)
    # dists = np.clip(dists, a_min=dist_min, a_max=None)
        
    # Data-driven floor: since inputs are min-max scaled to [0,1],
    # a percentile-based floor is comparable across all dimensions.
    dist_percentile = 5
    floor = np.percentile(dists, dist_percentile)
    dists = np.clip(dists, a_min=floor, a_max=None)

    # Score: max output change / input distance
    # Use max across all outputs to penalize pairs with low change in ANY output
    gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()

    # Fallback to uniform if all gradients are near-zero
    if np.sum(gradients) < dist_min:
        probs = np.ones(gradients.shape[0]) / len(gradients)
    else:
        # Normalize to probability distribution
        probs = gradients / np.sum(gradients)

    return probs


def select_pairs(x_a, x_b, y_a, y_b, probs, layer_width, random_seed=42):
    """
    Select layer_width winning pairs from candidates using SWIM probabilities.

    Samples pairs without replacement according to their SWIM scores (higher-scored
    pairs are more likely to be selected). Allows the same pair to be selected
    multiple times.

    Args:
        x_a: Start points of candidate pairs, shape (M, D).
        x_b: End points of candidate pairs, shape (M, D).
        y_a: Target values at x_a, shape (M,) or (M, D_y).
        y_b: Target values at x_b, shape (M,) or (M, D_y).
        probs: SWIM probabilities for each candidate, shape (M,), summing to 1.
        layer_width: Number of pairs to select (e.g., number of basis functions).
        random_seed: For reproducibility.

    Returns:
        x_a_selected: Selected start points, shape (layer_width, D).
        x_b_selected: Selected end points, shape (layer_width, D).
        y_a_selected: Target values at selected x_a, shape (layer_width,) or (layer_width, D_y).
        y_b_selected: Target values at selected x_b, shape (layer_width,) or (layer_width, D_y).
        selected_idx: Indices into original candidate arrays, shape (layer_width,).
    """
    M = x_a.shape[0]
    rng = np.random.default_rng(random_seed)

    # Sample layer_width pairs WITH REPLACEMENT according to probabilities
    selected_idx = rng.choice(
        M,
        size=layer_width,
        replace=False,
        p=probs
    )

    # Index into candidate arrays using selected indices
    x_a_selected = x_a[selected_idx]
    x_b_selected = x_b[selected_idx]
    y_a_selected = y_a[selected_idx]
    y_b_selected = y_b[selected_idx]

    return x_a_selected, x_b_selected, y_a_selected, y_b_selected, selected_idx