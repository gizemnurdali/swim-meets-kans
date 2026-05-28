import torch
import gpytorch
import numpy as np
import evaluation_metrics as em
import gaussian_process_models as gp
import gp_swim_like_pairs as gs


# Sample GP posterior mean along each selected segment to build edge functions.
def sample_edge_functions(model, x_a_selected, x_b_selected, G_sample=200):
    """
    Sample GP posterior mean functions over each selected pair segment.
    These become the edge activation functions in the KAN-style network.

    Uses GP posterior MEAN to avoid noise and ensure
    smooth, reproducible edge functions across all edges.

    Args:
        model:          frozen GP model (eval mode)
        x_a_selected:   (layer_width, D) — selected start points
        x_b_selected:   (layer_width, D) — selected end points
        G_sample:       number of grid points per segment (default: 200)

    Returns:
        x_segments:       (layer_width, G_sample, D) — segment input points
        edge_functions:   (layer_width, G_sample)    — GP mean values along each segment
    """
    layer_width = x_a_selected.shape[0]

    # ── Create dense points along each selected segment ────────────────────────
    # Reuse create_interior_points but with G_sample points including endpoints
    g_dense = torch.linspace(0, 1, G_sample)  # (G_sample,) — includes 0 and 1

    x_segments = (
        x_a_selected.unsqueeze(1) +
        g_dense.view(1, G_sample, 1) * (x_b_selected - x_a_selected).unsqueeze(1)
    )  # (layer_width, G_sample, D)

    # ── Query GP posterior mean per segment ────────────────────────────────────
    edge_functions = []

    for i in range(layer_width):
        x_seg_i = x_segments[i]  # (G_sample, D)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_i = model(x_seg_i)  # latent posterior — no observation noise

        # Use posterior mean instead of rsample() to avoid:
        # 1. Stochastic noise in the lookup table
        # 2. Independent sampling inconsistency across edges
        # 3. Compounding errors in interpolation step
        f_mean = pred_i.mean  # (G_sample,)
        edge_functions.append(f_mean)

    edge_functions = torch.stack(edge_functions)  # (layer_width, G_sample)

    print(f"[DEBUG] x_segments shape:     {x_segments.shape}")
    print(f"[DEBUG] edge_functions shape: {edge_functions.shape}")
    print(f"[DEBUG] Value range: [{edge_functions.min():.4f}, {edge_functions.max():.4f}]")

    return x_segments, edge_functions


# Interpolate edge functions at inputs by projecting onto each segment direction.
def interpolate_edge_functions(x_segments, edge_functions, X, x_a_selected, x_b_selected):
    """
    Interpolate edge functions at input points X by projecting onto segment directions.
    Supports arbitrary input dimension D via scalar projection onto segment direction.

    For each edge i:
        1. Compute unit direction: d_i = (x_b_i - x_a_i) / ||x_b_i - x_a_i||
        2. Project x_segments onto direction → t_grid (scalar grid positions)
        3. Project X onto direction → t (scalar positions for query points)
        4. Interpolate f(t) using precomputed GP mean values

    Args:
        x_segments:     (layer_width, G_sample, D) — segment input points
        edge_functions: (layer_width, G_sample)    — GP mean values along segments
        X:              (N, D)                     — points to evaluate
        x_a_selected:   (layer_width, D)           — segment start points
        x_b_selected:   (layer_width, D)           — segment end points

    Returns:
        H: (N, layer_width) — feature matrix
    """
    layer_width = x_segments.shape[0]
    N           = X.shape[0]
    H           = torch.zeros(N, layer_width)

    for i in range(layer_width):
        # ── Step 1: Compute unit direction for segment i ───────────────────
        diff      = x_b_selected[i] - x_a_selected[i]   # (D,)
        length    = diff.norm()                           # scalar

        if length < 1e-8:
            # Degenerate segment — x_a and x_b are the same point
            H[:, i] = edge_functions[i].mean()
            continue

        direction = diff / length   # (D,) unit vector

        # ── Step 2: Project x_segments onto direction → t_grid ────────────
        # (G_sample, D) - (D,) → (G_sample, D) @ (D,) → (G_sample,)
        t_grid = (x_segments[i] - x_a_selected[i]) @ direction   # (G_sample,)

        # ── Step 3: Project X onto direction → t ──────────────────────────
        # (N, D) - (D,) → (N, D) @ (D,) → (N,)
        t = (X - x_a_selected[i]) @ direction   # (N,)

        # ── Step 4: Interpolate f(t) for each point ────────────────────────
        t_np      = t.detach().numpy()                   # (N,)
        t_grid_np = t_grid.detach().numpy()              # (G_sample,)
        seg_f_np  = edge_functions[i].detach().numpy()   # (G_sample,)

        # np.interp clamps extrapolation to boundary values automatically
        H[:, i] = torch.tensor(np.interp(t_np, t_grid_np, seg_f_np))

    print(f"[DEBUG] H shape: {H.shape}")
    print(f"[DEBUG] H value range: [{H.min():.4f}, {H.max():.4f}]")
    print(f"[DEBUG] H rank (approx): {torch.linalg.matrix_rank(H)}")

    return H


# Build a surrogate-guided KAN layer by layer using GP fitting, pair selection, and edge function construction
def build_sgkan(X_train, y_train, layer_configs, activation=torch.tanh):
    """
    Build a surrogate-guided KAN with multiple layers.

    Args:
        X_train:       (N, D) training inputs
        y_train:       (N,)   training targets
        layer_configs: list of dicts, one per layer:
                           width, M, G, T
        activation:    nonlinearity between layers

    Returns:
        layers: list of layer dicts (for prediction)
        W_out:  (final_width+1, 1) output weights
    """
    layers        = []
    current_input = X_train

    for l, cfg in enumerate(layer_configs):
        print(f"\n─── Layer {l+1} | input: {current_input.shape} ───")

        # Fit GP
        gp_model, likelihood = gp.init_gp(current_input, y_train)
        gp_model, likelihood = gp.train_gp(gp_model, likelihood, current_input, y_train)

        # Pair selection
        x_a, x_b, _, _      = gs.sample_candidate_pairs(current_input, y_train, M=cfg["M"])
        scores, probs        = gs.compute_score_g(gp_model, x_a, x_b, T=cfg["T"])
        x_a_sel, x_b_sel, _ = gs.select_pairs(x_a, x_b, probs, layer_width=cfg["width"])

        # Edge functions
        x_segs, edge_fns = sample_edge_functions(gp_model, x_a_sel, x_b_sel, G_sample=cfg["G"])

        # Interpolate
        H = interpolate_edge_functions(x_segs, edge_fns, current_input, x_a_sel, x_b_sel)

        layers.append({
            "x_a_sel":  x_a_sel,
            "x_b_sel":  x_b_sel,
            "x_segs":   x_segs,
            "edge_fns": edge_fns,
        })

        current_input = activation(H)

    W_out = solve_output_layer(current_input, y_train)

    return layers, W_out


# Predict outputs for new inputs by passing through all constructed layers and the output layer.
def predict_sgkan(layers, W_out, X, activation=torch.tanh):
    """
    Args:
        layers: output of build_sgkan
        W_out:  output weights
        X:      (N, D) input points

    Returns:
        y_pred: (N,)
    """
    current = X

    for layer in layers:
        H = interpolate_edge_functions(
            layer["x_segs"], layer["edge_fns"],
            current,
            layer["x_a_sel"], layer["x_b_sel"],
        )
        current = activation(H)

    N      = current.shape[0]
    H_b    = torch.cat([current, torch.ones(N, 1)], dim=1)
    y_pred = (H_b @ W_out).squeeze()

    return y_pred


# Fit the output layer weights via least squares on the feature matrix.
def solve_output_layer(H_train, y_train):
    """
    Solve for output layer weights using OLS (least squares).
    Adds a bias column to H_train before solving.

    Args:
        H_train: (N_train, layer_width) — training feature matrix
        y_train: (N_train,)             — training targets

    Returns:
        W_out: (layer_width+1, 1) — output weights including bias
    """
    N_train = H_train.shape[0]

    # Add bias column
    H_train_b = torch.cat([H_train, torch.ones(N_train, 1)], dim=1)  # (N_train, layer_width+1)

    result = torch.linalg.lstsq(H_train_b, y_train.unsqueeze(1))
    W_out  = result.solution  # (layer_width+1, 1)

    print(f"[DEBUG] W_out shape: {W_out.shape}")

    return W_out


# Predict targets and report metrics using the learned output layer.
def predict_and_evaluate(H, y, W_out, split_name="Test"):
    """
    Predict and evaluate using solved output weights.

    Args:
        H:          (N, layer_width)      — feature matrix
        y:          (N,)                  — ground truth targets
        W_out:      (layer_width+1, 1)    — output weights including bias
        split_name: label for printing (default: "Test")

    Returns:
        y_pred: (N,)  — predictions
        mse:    scalar
        rel_l2: scalar
    """
    N = H.shape[0]

    # Add bias column
    H_b    = torch.cat([H, torch.ones(N, 1)], dim=1)  # (N, layer_width+1)
    y_pred = (H_b @ W_out).squeeze()                   # (N,)

    mse    = em.compute_mse(y_pred, y)
    rel_l2 = em.compute_relative_l2(y_pred, y)

    print(f"[{split_name}] MSE: {mse.item():.6f} | Relative L2: {rel_l2.item():.6f}")

    return y_pred, mse, rel_l2