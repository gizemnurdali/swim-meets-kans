"""SWIM Guided KAN (SGKAN) Algorithm"""
import numpy as np
from sklearn.linear_model import Ridge
from collections import Counter
import time
import swim as ss


# SWIM pair sampling
def select_swim_pairs_gen(X, y, layer_width, random_seed):
    # 1. Generate M candidate pairs using full-D distance (multi-dimensional)
    x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X, y, random_seed=random_seed)
    # 2) score each candidate pair: |dy| / ||dx|| (full-D distance)
    probs = ss.create_swim_probabilities(x_a, x_b, y_a, y_b)
    # 3) Sample pairs for each neuron (q) in the hidden layer
    x_a_sel, x_b_sel, y_a_sel, y_b_sel, _ = ss.select_pairs(
        x_a, x_b, y_a, y_b, probs, layer_width=layer_width, random_seed=random_seed
    )
    return x_a_sel, x_b_sel, y_a_sel, y_b_sel

# Project sampled SWIM pair to 1D and collect local points
def collect_local_gen(X, y, p, lo, hi, min_points=5, max_points=1000, k_fallback=15, cap_seed=0):
    # Take the values of feature p
    x_p = X[:, p]
    # Collect the points that fall inside the interval
    mask = (x_p >= lo) & (x_p <= hi)
    n_inside = int(mask.sum())
    
    # print(f"Number of inside points: {n_inside}, train set size: {X.shape[0]}")
    
    # Center of the interval
    midpoint = (lo + hi) / 2
    # If there are enough points inside the interval
    if n_inside >= min_points:
        x_inside, y_inside = x_p[mask], y[mask]
        # If there are too many points inside, keep only the ones closest to the midpoint
        if n_inside <= max_points:
            return x_inside, y_inside, "interval"
        # Take random max_points from the interval
        rng = np.random.default_rng(cap_seed)
        cap_idx = rng.choice(n_inside, size=max_points, replace=False)
        return x_inside[cap_idx], y_inside[cap_idx], "interval_capped"
    # Otherwise there are too few points inside the interval
    else:
        # Take the nearest points around the midpoint instead
        order = np.argsort(np.abs(x_p - midpoint))
        knn_idx = order[:k_fallback]
        return x_p[knn_idx], y[knn_idx], "knn_fallback"
    
# GP - Like kernel functions
def rbf_kernel(x1, x2, l_p):
    """RBF function"""
    r2 = (x1[:, None] - x2[None, :]) ** 2
    return np.exp(-r2 / (2 * l_p ** 2))

def matern_kernel(x1, x2, l_p):
    """Matern nu=5/2"""
    r = np.abs(x1[:, None] - x2[None, :]) / l_p
    sqrt5_r = np.sqrt(5) * r
    return (1 + sqrt5_r + (5 / 3) * r ** 2) * np.exp(-sqrt5_r)

def periodic_kernel(x1, x2, l_p, period=1.0):
    """ExpSineSquared"""
    r = np.abs(x1[:, None] - x2[None, :])
    return np.exp(-2 * np.sin(np.pi * r / period) ** 2 / (l_p ** 2))

def apply_kernel(x1, x2, l_p, kernel_type="rbf", period=1.0):
    # Routes to the selected kernel function
    if kernel_type == "rbf":
        return rbf_kernel(x1, x2, l_p)
    elif kernel_type == "matern":
        return matern_kernel(x1, x2, l_p)
    elif kernel_type == "periodic":
        return periodic_kernel(x1, x2, l_p, period=period)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
# MODEL SETUP
# Model wrapper
class SGKANModel:
    """Thin wrapper of sgkan functions."""
    def __init__(self, layer_configs):
        self.layer_configs = layer_configs
        self.neurons_per_layer = []

    def fit(self, X, y):
        X_current = X
        self.neurons_per_layer = []
        fit_start = time.perf_counter()
        for cfg in self.layer_configs:
            H, neurons = fit_layer(X_current, y, **cfg)
            self.neurons_per_layer.append(neurons)
            X_current = H
        fit_duration = time.perf_counter() - fit_start
        print(f"[SGKANModel.fit] Total fit duration: {fit_duration:.4f}s")
        return self

    def predict(self, X):
        X_current = X
        for neurons in self.neurons_per_layer:
            X_current = transform_layer(X_current, neurons)
        return X_current.ravel()

# Kernel matrix creation
def build_edge_features(local_x_p, lo, hi, num_inducing=15, sigma_scale=1.0, seed=0,
    kernel_type="rbf", period=1.0, lengthscale_method="range"):

    n = local_x_p.shape[0]
    n_ind = min(num_inducing, n)

    # Randomly select a fixed number of inducing (representative) points from the local data
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_ind, replace=False)
    z_p = np.sort(local_x_p[idx])

    if lengthscale_method == "median_trick":
        # Median heuristic: set lengthscale from the typical pairwise distance in the local data
        diffs = np.abs(local_x_p[:, None] - local_x_p[None, :])
        mask_diag = ~np.eye(n, dtype=bool)
        l_p = sigma_scale * np.median(diffs[mask_diag]) if n > 1 else sigma_scale
        l_p = max(l_p, 1e-6) # fallback for degenerated case
    elif lengthscale_method == "range":
        # Lengthscale derived directly from the SWIM-selected interval width,
        # NOT from the median pairwise distance of the (possibly capped) local subset.
        l_p = sigma_scale * (hi - lo)
        l_p = max(l_p, 1e-6)  # guard against degenerate (near-zero-width) intervals
    else: 
        raise ValueError("please select lengthscale_method as median_trick or range")
    # Evaluate the kernel between every training point and the inducing points
    k_local = apply_kernel(local_x_p, z_p, l_p, kernel_type=kernel_type, period=period)

    return k_local, z_p, l_p

# Fit SGKAN layer: one Ridge regression for per edge (p), one for neuron (q)
def fit_layer(
    X, y, layer_width, pair_selection_strategy="swim", num_inducing=15,
    lengthscale_method="range", sigma_scale=1.0, alpha_edge=1e-3, alpha_neuron=1e-3,
    seed=0, kernel_type="rbf", period=1.0, max_local_points=1000):

    # Number of data points and features
    N, D_in = X.shape[0], X.shape[1]
    # Select data pairs using swim or random
    if pair_selection_strategy == "swim":
        x_a, x_b, y_a, y_b = select_swim_pairs_gen(X, y, layer_width, random_seed=seed)
    elif pair_selection_strategy == "random":
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X, y, layer_width, random_seed=seed)
    else:
        raise ValueError("please select pair_selection method as swim or random")
    print(f"[fit_layer] pair_selection_strategy = '{pair_selection_strategy}' "
          f"(layer_width={layer_width}, D_in={D_in})")
    
    # Create an empty hidden layer output matrix
    H = np.zeros((N, layer_width))
    neurons = []
    method_counts = Counter()
    eps = 1e-8
    # Fit a Ridge model for each neuron, combining its incoming edge signals
    for q in range(layer_width):
        edge_outputs, edge_params = [], []
        for p in range(D_in):
            lo = min(x_a[q, p], x_b[q, p])
            hi = max(x_a[q, p], x_b[q, p])
            local_x_p, local_y, method = collect_local_gen(
                X, y, p, lo, hi, max_points=max_local_points,
                cap_seed=seed + q * D_in + p
            )
            method_counts[method] += 1

            # Stage 1: pick inducing points z_p from the local subset (x-only, no y).
            k_local, z_p, l_p = build_edge_features(
                local_x_p, lo, hi, num_inducing=num_inducing,
                sigma_scale=sigma_scale, seed=seed + q, kernel_type=kernel_type,
                period=period, lengthscale_method=lengthscale_method
            )
            if pair_selection_strategy == "swim":
                # Dimension-specific SWIM score for Ridge Regressor penalization
                dx = abs(x_b[q, p] - x_a[q, p])
                dy = abs(y_b[q] - y_a[q])
                dim_score = dy / (dx + eps)
                # If dim_score is small then penalize more
                alpha_edge_p = max(alpha_edge / (1 + dim_score), 1e-8)
            else: 
                alpha_edge_p = alpha_edge

            # Fit ridge k_local and l_p are now consistent
            edge_model = Ridge(alpha=alpha_edge_p, fit_intercept=False)
            edge_model.fit(k_local, local_y)
            w_p = edge_model.coef_

            # Evaluate this edge on the full training set, using the SAME l_p as fit
            k_train = apply_kernel(X[:, p], z_p, l_p, kernel_type=kernel_type, period=period)
            edge_outputs.append(k_train @ w_p)
            edge_params.append({"z_p": z_p, "l_p": l_p, "w_p": w_p})
        
        # (N, D_in) -- one column per edge output, each column denotes one edge result
        edge_outputs = np.column_stack(edge_outputs)

        neuron_model = Ridge(alpha=alpha_neuron, fit_intercept=True)
        neuron_model.fit(edge_outputs, y)
        H[:, q] = neuron_model.predict(edge_outputs)

        neurons.append({
            "edge_params": edge_params,
            "neuron_w": neuron_model.coef_, "bias": neuron_model.intercept_,
            "kernel_type": kernel_type, "period": period,
        })

    total_edges = layer_width * D_in
    print(f"[fit_layer] local-window method counts over {total_edges} edges: "
          f"{dict(method_counts)}")
    return H, neurons

# Transform SGKAN
def edge_function(x_p_query, z_p, l_p, w_p, kernel_type="rbf", period=1.0):
    # Kernel similarity between query points and the FIXED inducing points from training
    k = apply_kernel(x_p_query, z_p, l_p, kernel_type=kernel_type, period=period)
    # Weighted sum -> this edge's output for each query point
    return k @ w_p

def transform_layer(X_new, neurons):

    # Number of new samples and number of neurons in this layer
    N_new = X_new.shape[0]
    layer_width = len(neurons)
    H_new = np.zeros((N_new, layer_width))

    # Apply each already-fitted neuron to the new data (no refitting here)
    for q, neuron in enumerate(neurons):
        edge_outputs = []
        for p, ep in enumerate(neuron["edge_params"]):
            # Recompute this edge's output on X_new, using the stored (fixed) z_p, l_p, w_p
            k = apply_kernel(X_new[:, p], ep["z_p"], ep["l_p"],
                              kernel_type=neuron["kernel_type"], period=neuron["period"])
            edge_outputs.append(k @ ep["w_p"])
        edge_outputs = np.column_stack(edge_outputs)

        # Apply the stored neuron-level (stage 2) weights and bias
        H_new[:, q] = edge_outputs @ neuron["neuron_w"] + neuron["bias"]

    return H_new

# Utility function for visualization
def sgkan_layer_step(
        X, y, layer_width, pair_selection_strategy="swim", num_inducing=15, lengthscale_method="range",
        sigma_scale=1.0, alpha_edge=1e-3, alpha_neuron=1e-3, seed=0,
        kernel_type="rbf", period=1.0, max_local_points=1000):

    # Fit layer
    H, neurons = fit_layer(
        X, y, layer_width, pair_selection_strategy=pair_selection_strategy,
        num_inducing=num_inducing, lengthscale_method=lengthscale_method, sigma_scale=sigma_scale,
        alpha_edge=alpha_edge, alpha_neuron=alpha_neuron, seed=seed,
        kernel_type=kernel_type, period=period, max_local_points=max_local_points)

    D_in = X.shape[1]
    N = X.shape[0]
    phi = np.zeros((layer_width, D_in, N))

    for q, neuron in enumerate(neurons):
        for p, edge_p in enumerate(neuron["edge_params"]):
            k = apply_kernel(X[:, p], edge_p["z_p"], edge_p["l_p"],
                              kernel_type=neuron["kernel_type"], period=neuron["period"])
            phi[q, p, :] = k @ edge_p["w_p"]

    return phi, H, neurons

def build_sgkan_stack(X_train, y_train, layer_configs):
    """For stacking layers for sgkan"""
    X_current = X_train
    stages = []
    layer_neurons = []
    for cfg in layer_configs:
        layer_idx = len(stages) + 1
        print(f"[build_sgkan_stack] Layer {layer_idx}: "
              f"pair_selection_strategy = '{cfg.get('pair_selection_strategy', 'swim')}'")
        phi, h, neurons = sgkan_layer_step(X_current, y_train, **cfg)
        stages.append((layer_idx, phi, h))
        X_current = h
        layer_neurons.append(neurons)
    # last stage phi values are equivalent to the predicted y
    yhat = stages[-1][2].ravel()
    return stages, yhat, layer_neurons