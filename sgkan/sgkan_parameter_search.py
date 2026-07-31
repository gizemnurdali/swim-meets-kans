"""
Optuna hyperparameter search (TPE or exhaustive grid) for a 2-layer SGKAN model.
"""
import os
import time
import numpy as np
import pandas as pd
import optuna

from sgkan.sgkan_model import (
    SGKANModel, select_swim_pairs_gen, collect_local_gen, build_edge_features,
)
import swim as ss

pd.set_option("display.precision", 10)


# ASSUMPTIONS
LAYER_WIDTH_OPTIONS = [100, 250, 500, 750, 1000]
NUM_INDUCING_OPTIONS = [10, 25, 50, 75, 100]
ALPHA_EDGE_OPTIONS = [0, 1e-3, 1e-1, 1, 10]
ALPHA_NEURON_OPTIONS = [0]
KERNEL_TYPE_OPTIONS = ["rbf", "matern", "periodic"]
MAX_LOCAL_POINTS = 1000


def _make_two_layer_sgkan_configs(layer_width, num_inducing, alpha_edge=1e-1,
                                   alpha_neuron=1e-1, seed=0, sigma_scale=1, kernel_type="rbf"):
    """Build the 2-layer config list, layer 2 fixed at width=1, given
    candidate hyperparameters. kernel_type/sigma_scale/max_local_points/
    pair_selection_strategy stay fixed to the tf1_sgkan_layer_swim_configs
    example values."""
    shared = {
        "num_inducing": num_inducing,
        "pair_selection_strategy": "swim",
        "alpha_edge": alpha_edge,
        "alpha_neuron": alpha_neuron,
        "max_local_points": MAX_LOCAL_POINTS,
        "kernel_type": kernel_type,
        "seed": seed,
        "sigma_scale": sigma_scale,
    }
    return [
        {"layer_width": layer_width, **shared},
        {"layer_width": 1, **shared},
    ]


def objective_sgkan(trial, X_train, y_train, val_size=0.2, seed=0, sigma_scale=1, kernel_type="rbf"):
    """
    Optuna objective: pick the best hyperparameter combination
    (layer_width, num_inducing, alpha_edge, alpha_neuron, kernel_type) for a 2-layer SGKAN,
    via a single train/validation split (same pattern used for the KAN search in utils.py).
    Returns validation RMSE. Test-set evaluation happens once, after the search,
    for the winning config only (see study_optuna_sgkan).
    """
    layer_width = trial.suggest_categorical("layer_width", LAYER_WIDTH_OPTIONS)
    num_inducing = trial.suggest_categorical("num_inducing", NUM_INDUCING_OPTIONS)
    alpha_edge = trial.suggest_categorical("alpha_edge", ALPHA_EDGE_OPTIONS)
    alpha_neuron = trial.suggest_categorical("alpha_neuron", ALPHA_NEURON_OPTIONS)
    kernel_type = trial.suggest_categorical("kernel_type", KERNEL_TYPE_OPTIONS)

    layer_configs = _make_two_layer_sgkan_configs(
        layer_width, num_inducing, alpha_edge=alpha_edge, alpha_neuron=alpha_neuron,
        seed=seed, sigma_scale=sigma_scale, kernel_type=kernel_type
    )

    # Single train/validation split (same approach as the KAN search)
    n_samples = X_train.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_val = int(round(n_samples * val_size))
    val_idx, sub_train_idx = perm[:n_val], perm[n_val:]

    X_sub_train, X_val = X_train[sub_train_idx], X_train[val_idx]
    y_sub_train, y_val = y_train[sub_train_idx], y_train[val_idx]

    trial.set_user_attr("layer_width", layer_width)
    trial.set_user_attr("num_inducing", num_inducing)
    trial.set_user_attr("alpha_edge", alpha_edge)
    trial.set_user_attr("alpha_neuron", alpha_neuron)
    trial.set_user_attr("kernel_type", kernel_type)
    trial.set_user_attr("sigma_scale", sigma_scale)

    # alpha_edge=0 on a near-singular local kernel matrix can be numerically unstable. 
    # Catch that here so one bad trial doesn't kill the whole study
    try:
        fit_start = time.perf_counter()
        model = SGKANModel(layer_configs).fit(X_sub_train, y_sub_train)
        fit_duration_sec = time.perf_counter() - fit_start

        y_val_pred = model.predict(X_val)
        val_rmse = float(np.sqrt(np.mean((y_val_pred - y_val) ** 2)))
        if not np.isfinite(val_rmse):
            raise ValueError("non-finite val_rmse")
    except Exception as e:
        trial.set_user_attr("failed", True)
        trial.set_user_attr("error", str(e))
        return float("inf")

    trial.set_user_attr("val_rmse", val_rmse)
    trial.set_user_attr("fit_duration_sec", fit_duration_sec)

    return val_rmse


def study_optuna_sgkan(dataset_name, X_train, y_train, X_test, y_test,
                        n_trials=60, val_size=0.2, seed=0, sigma_scale=1,
                        sampler="grid", kernel_type="rbf"):
    """
    Run an Optuna search (TPE-based or exhaustive grid, via `sampler`) over
    (layer_width, num_inducing, alpha_edge, alpha_neuron, kernel_type) for a 2-layer SGKAN
    model on a dataset. Layer 2's width is fixed at 1. kernel_type parameter is ignored
    as it is suggested per trial; kernel_type_rbf is used as fallback for API consistency.
    """
    csv_path = f"data/{dataset_name.lower()}_sgkan_optuna_search.csv"

    # Load cached results if they exist
    if os.path.exists(csv_path):
        print("=" * 70)
        print(f"Results found for {dataset_name}. Loading from {csv_path}")
        print("=" * 70)

        trials_df = pd.read_csv(csv_path)
        best_idx = trials_df['value'].idxmin()
        best_row = trials_df.loc[best_idx]

        best_layer_width = int(best_row['params_layer_width']) # type: ignore
        best_num_inducing = int(best_row['params_num_inducing']) # type: ignore
        best_alpha_edge = float(best_row['params_alpha_edge']) # type: ignore
        best_alpha_neuron = float(best_row['params_alpha_neuron']) # type: ignore
        best_kernel_type = best_row['params_kernel_type'] # type: ignore

        print(f"Best Validation RMSE:   {best_row['value']:.15f}")
        print(f"Best Test RMSE:          {best_row['test_rmse']:.15f}")
        print(f"Best layer_width:        {best_layer_width}")
        print(f"Best num_inducing:       {best_num_inducing}")
        print(f"Best alpha_edge:         {best_alpha_edge:.3e}")
        print(f"Best alpha_neuron:       {best_alpha_neuron:.3e}")
        print(f"Best kernel_type:        {best_kernel_type}")

        print(f"\nAll {len(trials_df)} trials:")
        print(trials_df[['number', 'value', 'params_layer_width', 'params_num_inducing',
                          'params_alpha_edge', 'params_alpha_neuron', 'params_kernel_type',
                          'val_rmse', 'test_rmse']].to_string())

        return {
            "layer_width": best_layer_width,
            "num_inducing": best_num_inducing,
            "alpha_edge": best_alpha_edge,
            "alpha_neuron": best_alpha_neuron,
            "kernel_type": best_kernel_type,
            "layer_configs": _make_two_layer_sgkan_configs(
                best_layer_width, best_num_inducing,
                alpha_edge=best_alpha_edge, alpha_neuron=best_alpha_neuron, seed=seed,
                kernel_type=best_kernel_type # type: ignore
            ),
            "test_rmse": float(best_row['test_rmse']), # type: ignore
        }

    # Run optimization if results don't exist
    print("=" * 70)
    print(f"Optimizing SGKAN (layer_width, num_inducing, alpha_edge, alpha_neuron, kernel_type) "
          f"via {sampler.upper()} on {dataset_name}")
    print("=" * 70)

    if sampler == "tpe":
        sampler=optuna.samplers.TPESampler(seed=42)
    else:
        search_space = {
            "layer_width": LAYER_WIDTH_OPTIONS,
            "num_inducing": NUM_INDUCING_OPTIONS,
            "alpha_edge": ALPHA_EDGE_OPTIONS,
            "alpha_neuron": ALPHA_NEURON_OPTIONS,
            "kernel_type": KERNEL_TYPE_OPTIONS,
        }
        sampler=optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective_sgkan(trial, X_train, y_train, val_size=val_size,
                                      seed=seed, sigma_scale=sigma_scale, kernel_type=kernel_type),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Print best trial results
    print("\n" + "=" * 70)
    print("Best trial results:")
    print("=" * 70)
    print(f"Best Validation RMSE:   {study.best_value:.15f}")
    best_layer_width = study.best_params['layer_width']
    best_num_inducing = study.best_params['num_inducing']
    best_alpha_edge = study.best_params['alpha_edge']
    best_alpha_neuron = study.best_params['alpha_neuron']
    best_kernel_type = study.best_params['kernel_type']
    print(f"Best layer_width:        {best_layer_width}")
    print(f"Best num_inducing:       {best_num_inducing}")
    print(f"Best alpha_edge:         {best_alpha_edge:.3e}")
    print(f"Best alpha_neuron:       {best_alpha_neuron:.3e}")
    print(f"Best kernel_type:        {best_kernel_type}")

    # Retrain ONCE on the full training set with the winning combination,
    # and evaluate on the held-out test set — only place a full-data
    # training happens; it does not happen per trial.
    final_layer_configs = _make_two_layer_sgkan_configs(
        best_layer_width, best_num_inducing,
        alpha_edge=best_alpha_edge, alpha_neuron=best_alpha_neuron,
        seed=seed, sigma_scale=sigma_scale, kernel_type=best_kernel_type
    )
    final_model = SGKANModel(final_layer_configs).fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
    best_test_rmse = float(np.sqrt(np.mean((y_test_pred - y_test) ** 2)))
    print(f"Best Test RMSE:          {best_test_rmse:.15f}")

    trials_df = study.trials_dataframe()
    print(f"\nAll {len(trials_df)} trials:")
    print(trials_df[['number', 'value', 'params_layer_width', 'params_num_inducing',
                      'params_alpha_edge', 'params_alpha_neuron', 'params_kernel_type',
                      'user_attrs_val_rmse', 'user_attrs_sigma_scale',
                      'user_attrs_fit_duration_sec']].to_string())

    trials_df = trials_df.rename(columns={
        "user_attrs_val_rmse": "val_rmse",
        "user_attrs_sigma_scale": "sigma_scale",
        "user_attrs_kernel_type": "kernel_type",
        "user_attrs_fit_duration_sec": "fit_duration_sec",
    })
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
        "kernel_type": best_kernel_type,
        "sigma_scale": sigma_scale,
        "layer_configs": final_layer_configs,
        "test_rmse": best_test_rmse,
    }