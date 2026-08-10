import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_hypothesis_1_results(results_df, kan_best_train_rmse=None, kan_best_test_rmse=None, title=None):
    """Plot Hypothesis 1 comparison results for SGKAN (swim), SGKAN (random), and HKAN.
    kan_best_train_rmse, kan_best_test_rmse: KAN's best train/test RMSE (from the
    winning architecture found via the full search), shown as horizontal reference
    lines rather than width-dependent curves, since KAN's meaningful width range
    (up to 2n+1) is incomparable in scale to SG-KAN/HKAN's swept range.
    title: optional plot title; if omitted, a default explanatory title is used."""
    plt.figure(figsize=(12, 6))

    plt.plot(results_df['layer_width'], results_df['sgkan_swim_train_rmse'], marker='o', label='SGKAN SWIM Train', color='#1f77b4')
    plt.plot(results_df['layer_width'], results_df['sgkan_swim_test_rmse'], marker='s', label='SGKAN SWIM Test', color='#1f77b4', linestyle='--')

    plt.plot(results_df['layer_width'], results_df['sgkan_random_train_rmse'], marker='o', label='SGKAN Random Train', color='#ff7f0e')
    plt.plot(results_df['layer_width'], results_df['sgkan_random_test_rmse'], marker='s', label='SGKAN Random Test', color='#ff7f0e', linestyle='--')

    plt.plot(results_df['layer_width'], results_df['hkan_train_rmse'], marker='o', label='HKAN Train', color='#2ca02c')
    plt.plot(results_df['layer_width'], results_df['hkan_test_rmse'], marker='s', label='HKAN Test', color='#2ca02c', linestyle='--')

    if kan_best_train_rmse is not None:
        plt.axhline(kan_best_train_rmse, color='#d62728', linestyle='-', linewidth=2,
                    label=f'KAN Train (best, RMSE={kan_best_train_rmse:.2e})')
    if kan_best_test_rmse is not None:
        plt.axhline(kan_best_test_rmse, color='#d62728', linestyle='--', linewidth=2,
                    label=f'KAN Test (best, RMSE={kan_best_test_rmse:.2e})')

    if title is None:
        title = ("Effect of layer width on RMSE for SG-KAN (SWIM vs. random) and HKAN;\n"
                  "KAN shown as a fixed reference (best architecture, width capped at $2n{+}1$)")
    plt.title(title, fontsize=11)

    plt.xlabel('Layer Width')
    plt.ylabel('RMSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hypothesis_2_results(results_df, kan_fit_time=None, title=None):
    """Plot Hypothesis 2 comparison results: training time vs. layer width for
    SGKAN (swim), SGKAN (random), and HKAN. KAN shown as a fixed horizontal
    reference, since its meaningful width range (up to 2n+1) is incomparable
    in scale to SG-KAN/HKAN's swept range."""
    plt.figure(figsize=(12, 6))

    plt.plot(results_df['layer_width'], results_df['sgkan_swim_fit_time'], marker='o',
             label='SGKAN SWIM', color='#1f77b4')
    plt.plot(results_df['layer_width'], results_df['sgkan_random_fit_time'], marker='o',
             label='SGKAN Random', color='#ff7f0e')
    plt.plot(results_df['layer_width'], results_df['hkan_fit_time'], marker='o',
             label='HKAN', color='#2ca02c')

    if kan_fit_time is not None:
        plt.axhline(kan_fit_time, color='#d62728', linestyle='-', linewidth=2,
                    label=f'KAN (best architecture, {kan_fit_time:.1f}s)')

    if title is None:
        title = "Layer Width vs. Training Time"
    plt.title(title, fontsize=11)

    plt.xlabel('Layer Width')
    plt.ylabel('Training Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hypothesis_3_results(results_df, kan_train_inference_time=None, kan_test_inference_time=None, title=None):
    """Plot Hypothesis 3 comparison results: inference time vs. layer width for
    SGKAN (swim), SGKAN (random), and HKAN. KAN shown as fixed horizontal
    references, for the same reason as Hypotheses 1 and 2."""
    plt.figure(figsize=(12, 6))

    plt.plot(results_df['layer_width'], results_df['sgkan_swim_train_inf_time'], marker='o', label='SGKAN SWIM Train', color='#1f77b4')
    plt.plot(results_df['layer_width'], results_df['sgkan_swim_test_inf_time'], marker='s', label='SGKAN SWIM Test', color='#1f77b4', linestyle='--')

    plt.plot(results_df['layer_width'], results_df['sgkan_random_train_inf_time'], marker='o', label='SGKAN Random Train', color='#ff7f0e')
    plt.plot(results_df['layer_width'], results_df['sgkan_random_test_inf_time'], marker='s', label='SGKAN Random Test', color='#ff7f0e', linestyle='--')

    plt.plot(results_df['layer_width'], results_df['hkan_train_inf_time'], marker='o', label='HKAN Train', color='#2ca02c')
    plt.plot(results_df['layer_width'], results_df['hkan_test_inf_time'], marker='s', label='HKAN Test', color='#2ca02c', linestyle='--')

    if kan_train_inference_time is not None:
        plt.axhline(kan_train_inference_time, color='#d62728', linestyle='-', linewidth=2,
                    label=f'KAN Train (best, {kan_train_inference_time:.4f}s)')
    if kan_test_inference_time is not None:
        plt.axhline(kan_test_inference_time, color='#d62728', linestyle='--', linewidth=2,
                    label=f'KAN Test (best, {kan_test_inference_time:.4f}s)')

    if title is None:
        title = "Layer Width vs. Inference Time"
    plt.title(title, fontsize=11)

    plt.xlabel('Layer Width')
    plt.ylabel('Inference Time (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_model_stages(
        X_train, y_train, stages, block_idx="random", cmap="inferno",
        seed=None, feature_idx=(0, 1)):
    """
    stages: list of (layer_idx, phi, h)
    block_idx: which (q, p) block to show for each layer's phi, or "random".
    feature_idx: which two columns of X_train to use as the x1/x2 axes in the
        3D surface plots. Only matters for datasets with more than 2 inputs such as TF4, TF5-5.
    """
    f1, f2 = feature_idx
    x1, x2 = X_train[:, f1], X_train[:, f2]
    plot_items = []
    rng = np.random.default_rng(seed)

    # build one (phi-block, h-node) entry pair per layer
    for layer_idx, phi, h in stages:
        n_vars_out = phi.shape[0] # q
        n_vars_in = phi.shape[1] # p

        # pick which (q, p) block to display for this layer
        if block_idx == "random":
            qq = rng.integers(0, n_vars_out)
            pp = rng.integers(0, n_vars_in)
        else:
            q, p = block_idx # set (q, p)
            qq, pp = min(q, n_vars_out - 1), min(p, n_vars_in - 1)

        plot_items.append(
            (rf"$\phi^{{({layer_idx})}}_{{q_{{{qq}}},p_{{{pp}}}}}$", phi[qq, pp, :],
             f"Layer {layer_idx} Univariate Edge function (q={qq}, p={pp})", cmap)
        )
        plot_items.append(
            (rf"$h^{{({layer_idx})}}_{{q_{{{qq}}}}}$", h[:, qq],
             f"Layer {layer_idx} Neuron (q={qq})", cmap)
        )

    plot_items.append((r"$y$ (target)", y_train, "Ground truth", cmap))
    n = len(plot_items)
    fig = plt.figure(figsize=(3 * n, 10))

    # render each stage as a 3D surface (top) + predicted-vs-target scatter (bottom)
    for i, (title, values, subtitle, cm) in enumerate(plot_items):
        ax = fig.add_subplot(2, n, i + 1, projection="3d")
        ax.plot_trisurf(x1, x2, values, cmap=cm, linewidth=0, antialiased=True)
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_xlabel(f"x{f1+1}", fontsize=7); ax.set_ylabel(f"x{f2+1}", fontsize=7)

        # scatter this stage's fitted values against the true target, with diagonal reference
        ax2 = fig.add_subplot(2, n, n + i + 1)
        ax2.scatter(y_train, values, s=3, alpha=0.6, c=values, cmap=cm)
        lims = [min(y_train.min(), values.min()), max(y_train.max(), values.max())]
        ax2.plot(lims, lims, linestyle="--", linewidth=1, color="gray")
        r2 = r2_score(y_train, values)
        ax2.set_title(f"{title} vs y,  " + r"$R^2={:.2f}$".format(r2), fontsize=8)
        ax2.set_xlabel("target y", fontsize=7); ax2.set_ylabel("fitted value", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_first_layer_featurewise_r2_summary(stages, y_train):
    """
    Boxplot of R^2 across all nodes q, one box per input p, for the FIRST layer only.
    """
    layer_idx, phi, _ = stages[0]
    n_vars_out, n_vars_in, _ = phi.shape

    data, labels = [], []
    for p in range(n_vars_in):
        # Collect each dimension prediction for all nodes
        r2_vals = [r2_score(y_train, phi[q, p, :]) for q in range(n_vars_out)]
        data.append(r2_vals)
        labels.append(f"p={p}")

    _, ax = plt.subplots(figsize=(1.5 * n_vars_in + 2, 4))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title(f"Layer {layer_idx}: Global R² distribution per input variable ({n_vars_out} nodes)")
    ax.set_ylabel(r"$R^2$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()


def _trim_outliers_iqr(values, k=1.5):
    values = np.asarray(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return values[(values >= lo) & (values <= hi)]


# Use for SGKAN model 
def plot_first_layer_local_edge_r2_summary(layer_neurons):
    """
    Boxplot of train-local R^2 across all nodes q, one box per input p,
    for the FIRST layer only. Unlike plot_first_layer_featurewise_r2_summary
    (which scores each edge against the full y_train), this scores each
    edge only against the local points it was actually fit on.
    """
    neurons = layer_neurons[0]
    n_vars_out = len(neurons)
    n_vars_in = len(neurons[0]["edge_params"])

    data, labels = [], []
    for p in range(n_vars_in):
        r2_vals = [
            r2_score(neurons[q]["edge_params"][p]["local_y"],
                     neurons[q]["edge_params"][p]["local_pred"])
            for q in range(n_vars_out)
        ]
        data.append(_trim_outliers_iqr(r2_vals))
        labels.append(f"p={p}")

    _, ax = plt.subplots(figsize=(1.5 * n_vars_in + 2, 4))
    ax.boxplot(data, tick_labels=labels, showmeans=True, showfliers=True)
    ax.set_title(f"Layer 1: Local-window R² distribution per input variable ({n_vars_out} nodes)")
    ax.set_ylabel(r"$R^2$ (local)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()