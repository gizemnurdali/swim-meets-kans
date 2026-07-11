import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


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
             f"Layer {layer_idx} BlF (q={qq}, p={pp})", cmap)
        )
        plot_items.append(
            (rf"$h^{{({layer_idx})}}_{{q_{{{qq}}}}}$", h[:, qq],
             f"Layer {layer_idx} h-function (q={qq})", cmap)
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
    ax.set_title(f"Layer {layer_idx}: R² distribution per input variable ({n_vars_out} nodes)")
    ax.set_ylabel(r"$R^2$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()


# Use for SGKAN model only
def plot_first_layer_featurewise_r2_summary_biased(stages, y_train, neurons_list):
    """
    Same as plot_first_layer_featurewise_r2_summary, but adds each neuron's
    bias to its edge output before computing R^2.

    In SGKAN, edge-level Ridge is fit with fit_intercept=False (bias lives only
    at the neuron level), so a raw edge output is zero-centered while y is
    not. Comparing a zero-centered value directly to y makes R^2 look very
    negative even for a perfectly reasonable edge. Adding the neuron's bias back
    puts the edge on the same scale as y, giving a fairer R^2.
    """
    layer_idx, phi, h = stages[0]
    neurons = neurons_list[0]
    n_vars_out, n_vars_in, _ = phi.shape

    data, labels = [], []
    for p in range(n_vars_in):
        # Add the neuron's bias to each edge's raw output before scoring
        r2_vals = [r2_score(y_train, phi[q, p, :] + neurons[q]["bias"]) for q in range(n_vars_out)]
        data.append(r2_vals)
        labels.append(f"p={p}")

    _, ax = plt.subplots(figsize=(1.5 * n_vars_in + 2, 4))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title(f"Layer {layer_idx}: R^2 distribution (bias-adjusted), per input variable ({n_vars_out} nodes)")
    ax.set_ylabel(r"$R^2$ (bias + edge)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()