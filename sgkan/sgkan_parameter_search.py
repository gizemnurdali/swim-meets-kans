"""
Optuna TPE hyperparameter search for a 2-layer SGKAN model.

Searches:
  - layer_width   (first layer's width; layer 2 is fixed at width=1, i.e. output)
  - num_inducing  (shared by both layers)
  - alpha_edge    (edge-level ridge regularization, shared by both layers)
  - alpha_neuron  (neuron-level ridge regularization, shared by both layers)

Everything else (kernel_type, sigma_scale, max_local_points,
pair_selection_strategy) is held fixed at the values used in the
tf1_sgkan_layer_swim_configs example.

Usage:
    from sgkan_parameter_search import study_optuna_sgkan

    result = study_optuna_sgkan(
        "tf1", tf1_x_train, tf1_y_train, tf1_x_test, tf1_y_test, n_trials=60
    )
    print(result["layer_width"], result["num_inducing"],
          result["alpha_edge"], result["alpha_neuron"], result["test_rmse"])

Also includes diagnose_edge_conditioning(), which directly checks whether
K_local (the per-edge kernel feature matrix, before ridge regularization) is
ill-conditioned — i.e. whether a large alpha_edge is masking numerical
instability rather than reflecting a genuinely better bias-variance tradeoff.
"""
import os
import numpy as np
import pandas as pd
import optuna

# Same import convention used elsewhere in this project: sgkan_model.py sits
# next to this file (both inside the 'sgkan' folder), so a plain import works
# once that folder is on sys.path (as utils.py already arranges).
from sgkan.sgkan_model import (
    SGKANModel, select_swim_pairs_gen, collect_local_gen, build_edge_features,
)
import swim as ss


# ASSUMPTIONS (edit these if you want a different search space):
LAYER_WIDTH_OPTIONS = [100, 250, 500, 750, 1000]
NUM_INDUCING_OPTIONS = [10, 20, 30, 50, 75, 100]
ALPHA_EDGE_RANGE = (1e-6, 1e1)      # searched log-uniform
ALPHA_NEURON_RANGE = (1e-10, 1e1)   # searched log-uniform


def _make_two_layer_sgkan_configs(layer_width, num_inducing, alpha_edge=1e-1,
                                   alpha_neuron=1e-6, seed=0):
    """Build the 2-layer config list, layer 2 fixed at width=1, given
    candidate hyperparameters. kernel_type/sigma_scale/max_local_points/
    pair_selection_strategy stay fixed to the tf1_sgkan_layer_swim_configs
    example values."""
    shared = {
        "num_inducing": num_inducing,
        "pair_selection_strategy": "swim",
        "alpha_edge": alpha_edge,
        "alpha_neuron": alpha_neuron,
        "max_local_points": 1000,
        "kernel_type": "rbf",
        "seed": seed,
        "sigma_scale": 1,
    }
    return [
        {"layer_width": layer_width, **shared},
        {"layer_width": 1, **shared},
    ]


def objective_sgkan(trial, X_train, y_train, val_size=0.2, seed=0):
    """
    Optuna objective: pick the best (layer_width, num_inducing, alpha_edge,
    alpha_neuron) combination for a 2-layer SGKAN, via a single train/
    validation split (same pattern used for the KAN search in utils.py).
    kernel_type/sigma_scale/max_local_points/pair_selection_strategy are
    held fixed. Returns validation RMSE. Test-set evaluation happens once,
    after the search, for the winning config only (see study_optuna_sgkan).
    """
    layer_width = trial.suggest_categorical("layer_width", LAYER_WIDTH_OPTIONS)
    num_inducing = trial.suggest_categorical("num_inducing", NUM_INDUCING_OPTIONS)
    alpha_edge = trial.suggest_float("alpha_edge", *ALPHA_EDGE_RANGE, log=True)
    alpha_neuron = trial.suggest_float("alpha_neuron", *ALPHA_NEURON_RANGE, log=True)

    layer_configs = _make_two_layer_sgkan_configs(
        layer_width, num_inducing, alpha_edge=alpha_edge, alpha_neuron=alpha_neuron, seed=seed
    )

    # Single train/validation split (same approach as the KAN search)
    n_samples = X_train.shape[0]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    n_val = int(round(n_samples * val_size))
    val_idx, sub_train_idx = perm[:n_val], perm[n_val:]

    X_sub_train, X_val = X_train[sub_train_idx], X_train[val_idx]
    y_sub_train, y_val = y_train[sub_train_idx], y_train[val_idx]

    model = SGKANModel(layer_configs).fit(X_sub_train, y_sub_train)
    y_val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))

    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("layer_width", layer_width)
    trial.set_user_attr("num_inducing", num_inducing)
    trial.set_user_attr("alpha_edge", alpha_edge)
    trial.set_user_attr("alpha_neuron", alpha_neuron)

    return val_rmse


def study_optuna_sgkan(dataset_name, X_train, y_train, X_test, y_test,
                        n_trials=60, val_size=0.2, seed=0):
    """
    Run a TPE-based Optuna search over (layer_width, num_inducing,
    alpha_edge, alpha_neuron) for a 2-layer SGKAN model on a dataset.
    Layer 2's width is fixed at 1; kernel_type/sigma_scale/max_local_points/
    pair_selection_strategy are fixed (see _make_two_layer_sgkan_configs).

    n_trials=60 default: now 4 hyperparameters (2 categorical, 2 continuous
    log-uniform) instead of 2 — bumped up from the earlier 40 since the
    search space is meaningfully larger.

    Caches results to data/{dataset_name}_sgkan_optuna_search.csv.

    Returns:
        dict with keys 'layer_width', 'num_inducing', 'alpha_edge',
        'alpha_neuron', 'layer_configs', 'test_rmse'
    """
    csv_path = f"data/{dataset_name.lower()}_sgkan_optuna_search.csv"

    # # Load cached results if they exist
    # if os.path.exists(csv_path):
    #     print("=" * 70)
    #     print(f"Results found for {dataset_name}. Loading from {csv_path}")
    #     print("=" * 70)

    #     trials_df = pd.read_csv(csv_path)
    #     best_idx = trials_df['value'].idxmin()
    #     best_row = trials_df.loc[best_idx]

    #     best_layer_width = int(best_row['params_layer_width'])
    #     best_num_inducing = int(best_row['params_num_inducing'])
    #     best_alpha_edge = float(best_row['params_alpha_edge'])
    #     best_alpha_neuron = float(best_row['params_alpha_neuron'])

    #     print(f"Best Validation RMSE:   {best_row['value']:.6f}")
    #     print(f"Best Test RMSE:          {best_row['test_rmse']:.6f}")
    #     print(f"Best layer_width:        {best_layer_width}")
    #     print(f"Best num_inducing:       {best_num_inducing}")
    #     print(f"Best alpha_edge:         {best_alpha_edge:.3e}")
    #     print(f"Best alpha_neuron:       {best_alpha_neuron:.3e}")

    #     print(f"\nAll {len(trials_df)} trials:")
    #     print(trials_df[['number', 'value', 'params_layer_width', 'params_num_inducing',
    #                       'params_alpha_edge', 'params_alpha_neuron',
    #                       'val_rmse', 'test_rmse']].to_string())

    #     return {
    #         "layer_width": best_layer_width,
    #         "num_inducing": best_num_inducing,
    #         "alpha_edge": best_alpha_edge,
    #         "alpha_neuron": best_alpha_neuron,
    #         "layer_configs": _make_two_layer_sgkan_configs(
    #             best_layer_width, best_num_inducing,
    #             alpha_edge=best_alpha_edge, alpha_neuron=best_alpha_neuron, seed=seed
    #         ),
    #         "test_rmse": float(best_row['test_rmse']),
        # }

    # Run optimization if results don't exist
    print("=" * 70)
    print(f"Optimizing SGKAN (layer_width, num_inducing, alpha_edge, alpha_neuron) "
          f"via TPE on {dataset_name}")
    print("=" * 70)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective_sgkan(trial, X_train, y_train, val_size=val_size, seed=seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print best trial results
    print("\n" + "=" * 70)
    print("Best trial results:")
    print("=" * 70)
    print(f"Best Validation RMSE:   {study.best_value:.6f}")
    best_layer_width = study.best_params['layer_width']
    best_num_inducing = study.best_params['num_inducing']
    best_alpha_edge = study.best_params['alpha_edge']
    best_alpha_neuron = study.best_params['alpha_neuron']
    print(f"Best layer_width:        {best_layer_width}")
    print(f"Best num_inducing:       {best_num_inducing}")
    print(f"Best alpha_edge:         {best_alpha_edge:.3e}")
    print(f"Best alpha_neuron:       {best_alpha_neuron:.3e}")

    # Retrain ONCE on the full training set with the winning combination,
    # and evaluate on the held-out test set — only place a full-data
    # training happens; it does not happen per trial.
    final_layer_configs = _make_two_layer_sgkan_configs(
        best_layer_width, best_num_inducing,
        alpha_edge=best_alpha_edge, alpha_neuron=best_alpha_neuron, seed=seed
    )
    final_model = SGKANModel(final_layer_configs).fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    best_test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test) ** 2)))
    print(f"Best Test RMSE:          {best_test_rmse:.6f}")

    trials_df = study.trials_dataframe()
    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[['number', 'value', 'params_layer_width', 'params_num_inducing',
                      'params_alpha_edge', 'params_alpha_neuron',
                      'user_attrs_val_rmse']].to_string())

    trials_df = trials_df.rename(columns={"user_attrs_val_rmse": "val_rmse"})
    trials_df['duration'] = trials_df['duration'].dt.total_seconds()

    # test_rmse only known for the winning trial (computed once above)
    trials_df['test_rmse'] = np.nan
    trials_df.loc[trials_df['number'] == study.best_trial.number, 'test_rmse'] = best_test_rmse

    os.makedirs("data", exist_ok=True)
    trials_df.to_csv(csv_path, index=False)

    return {
        "layer_width": best_layer_width,
        "num_inducing": best_num_inducing,
        "alpha_edge": best_alpha_edge,
        "alpha_neuron": best_alpha_neuron,
        "layer_configs": final_layer_configs,
        "test_rmse": best_test_rmse,
    }


# ─── Diagnostic: is K_local actually ill-conditioned? ──────────────────
# Directly tests the "large alpha_edge is masking ill-conditioning, not
# reflecting a genuinely better bias-variance tradeoff" hypothesis, by
# computing the condition number of every edge's kernel feature matrix
# BEFORE any ridge regularization is applied.

def diagnose_edge_conditioning(X, y, layer_width, num_inducing, sigma_scale=1.0,
                                seed=0, pair_selection_strategy="swim",
                                kernel_type="rbf", period=1.0, max_local_points=1000):
    """
    Computes the condition number of K_local (the per-edge kernel feature
    matrix) for every (neuron, dimension) edge in a single layer, before any
    ridge regularization. High condition numbers (rule of thumb: >1e6 is
    concerning, >1e10 is severe) mean the unregularized least-squares
    problem is numerically unstable — in that case, needing a large
    alpha_edge to get good results is consistent with "masking instability"
    rather than "genuinely wanting a smoother function."

    If condition numbers come back small (e.g. <1e3) even without
    regularization, that's evidence AGAINST the ill-conditioning hypothesis —
    the large alpha_edge would then more likely reflect an actual smoothness
    preference (reducing variance from a well-posed but still noisy fit).

    Returns:
        dict with median/max condition number, fraction of edges above
        1e6 and 1e10, total edge count, and the raw per-edge array.
    """
    N, D_in = X.shape[0], X.shape[1]

    if pair_selection_strategy == "swim":
        x_a, x_b, y_a, y_b = select_swim_pairs_gen(X, y, layer_width, random_seed=seed)
    elif pair_selection_strategy == "random":
        x_a, x_b, y_a, y_b = ss.sample_candidate_pairs(X, y, layer_width, random_seed=seed)
    else:
        raise ValueError("pair_selection_strategy must be 'swim' or 'random'")

    cond_numbers = []
    for q in range(layer_width):
        for p in range(D_in):
            lo = min(x_a[q, p], x_b[q, p])
            hi = max(x_a[q, p], x_b[q, p])

            local_x_p, local_y, method = collect_local_gen(
                X, y, p, lo, hi, max_points=max_local_points,
                cap_seed=seed + q * D_in + p
            )
            k_local, z_p, l_p = build_edge_features(
                local_x_p, lo, hi, num_inducing=num_inducing,
                sigma_scale=sigma_scale, seed=seed + q,
                kernel_type=kernel_type, period=period,
            )
            cond_numbers.append(np.linalg.cond(k_local))

    cond_numbers = np.array(cond_numbers)
    summary = {
        "median_cond": float(np.median(cond_numbers)),
        "max_cond": float(np.max(cond_numbers)),
        "frac_above_1e6": float(np.mean(cond_numbers > 1e6)),
        "frac_above_1e10": float(np.mean(cond_numbers > 1e10)),
        "n_edges": int(len(cond_numbers)),
        "all_cond_numbers": cond_numbers,
    }

    print(f"[diagnose_edge_conditioning] {summary['n_edges']} edges, "
          f"median cond={summary['median_cond']:.2e}, max cond={summary['max_cond']:.2e}, "
          f"frac>1e6={summary['frac_above_1e6']:.1%}, frac>1e10={summary['frac_above_1e10']:.1%}")

    return summary