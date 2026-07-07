import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# --- daha once yazdigimiz core fonksiyonlar (fit_layer, edge_function, vs.) ---
def select_pairs_gen(X, y, layer_width, random_seed=None, n_candidates=2000):
    M = X.shape[0]
    rng = np.random.default_rng(random_seed)
    idx_a = rng.integers(0, M, size=n_candidates)
    idx_b = rng.integers(0, M, size=n_candidates)
    x_a_cand, x_b_cand = X[idx_a], X[idx_b]
    y_a_cand, y_b_cand = y[idx_a], y[idx_b]
    dist = np.maximum(np.linalg.norm(x_a_cand - x_b_cand, axis=1), 1e-8)
    score = np.abs(y_a_cand - y_b_cand) / dist
    probs = score / score.sum()
    selected_idx = rng.choice(n_candidates, size=layer_width, replace=False, p=probs)
    return x_a_cand[selected_idx], x_b_cand[selected_idx]

def collect_local_gen(X, y, p, lo, hi, min_points=5, max_points=1000, k_fallback=10):
    x_p = X[:, p]
    mask = (x_p >= lo) & (x_p <= hi)
    n_inside = int(mask.sum())
    midpoint = (lo + hi) / 2
    if n_inside >= min_points:
        x_inside, y_inside = x_p[mask], y[mask]
        if n_inside <= max_points:
            return x_inside, y_inside, "interval"
        order = np.argsort(np.abs(x_inside - midpoint))
        cap_idx = order[:max_points]
        return x_inside[cap_idx], y_inside[cap_idx], "interval_capped"
    order = np.argsort(np.abs(x_p - midpoint))
    knn_idx = order[:k_fallback]
    return x_p[knn_idx], y[knn_idx], "knn_fallback"

def build_edge_features(X, p, local_x_p, num_inducing=15, sigma_scale=1.0, seed=0):
    n = local_x_p.shape[0]
    n_ind = min(num_inducing, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_ind, replace=False)
    z_p = np.sort(local_x_p[idx])
    diffs = np.abs(local_x_p[:, None] - local_x_p[None, :])
    mask_diag = ~np.eye(n, dtype=bool)
    l_p = sigma_scale * np.median(diffs[mask_diag]) if n > 1 else sigma_scale
    l_p = max(l_p, 1e-6)
    x_star = X[:, p]
    k_star = np.exp(-(x_star[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    return k_star, z_p, l_p

def edge_function(x_p_query, z_p, l_p, w_p):
    k = np.exp(-(x_p_query[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    return k @ w_p

def fit_layer(X, y, layer_width, num_inducing=15, sigma_scale=1.0, alpha=1e-3, seed=0):
    D_in = X.shape[1]
    x_a, x_b = select_pairs_gen(X, y, layer_width, random_seed=seed)
    H = np.zeros((len(y), layer_width))
    neurons = []
    for q in range(layer_width):
        feature_blocks, z_list, l_list = [], [], []
        for p in range(D_in):
            lo = min(x_a[q, p], x_b[q, p])
            hi = max(x_a[q, p], x_b[q, p])
            local_x_p, _, _ = collect_local_gen(X, y, p, lo, hi)
            k_feat, z_p, l_p = build_edge_features(X, p, local_x_p, num_inducing=num_inducing,
                                                     sigma_scale=sigma_scale, seed=seed + q)
            feature_blocks.append(k_feat)
            z_list.append(z_p)
            l_list.append(l_p)
        all_features = np.column_stack(feature_blocks)
        model_q = Ridge(alpha=alpha, fit_intercept=True)
        model_q.fit(all_features, y)
        H[:, q] = model_q.predict(all_features)
        w_full = model_q.coef_
        sizes = [fb.shape[1] for fb in feature_blocks]
        w_per_dim = np.split(w_full, np.cumsum(sizes)[:-1])
        neurons.append({"z_list": z_list, "l_list": l_list, "w_per_dim": w_per_dim, "bias": model_q.intercept_})
    return H, neurons


# ---------------------------------------------------------------------------
# YENI: HKAN'in intermediate-step API'siyle UYUMLU sarmalayicilar
# ---------------------------------------------------------------------------
def sgkan_layer_step(X, y, n_vars_out, num_inducing=15, sigma_scale=1.0, alpha=1e-3, seed=0):
    """
    HKAN'daki hkan_layer_step ile AYNI sozlesme: (phi, h, ...) dondurur.
    phi: (n_vars_out, n_vars_in, n_samples) -- her (q,p) icin kenar ciktisi
    h  : (n_samples, n_vars_out) -- katmanin ciktisi
    """
    H, neurons = fit_layer(X, y, n_vars_out, num_inducing=num_inducing,
                             sigma_scale=sigma_scale, alpha=alpha, seed=seed)
    D_in = X.shape[1]
    n_samples = X.shape[0]

    phi = np.zeros((n_vars_out, D_in, n_samples))
    for q, neuron in enumerate(neurons):
        for p in range(D_in):
            phi[q, p, :] = edge_function(X[:, p], neuron["z_list"][p],
                                          neuron["l_list"][p], neuron["w_per_dim"][p])

    return phi, H, neurons


def build_sgkan_stack(X_train, y_train, layer_configs):
    """HKAN'daki build_hkan_stack ile AYNI sozlesme."""
    X_current = X_train
    stages = []
    for i, cfg in enumerate(layer_configs, start=1):
        phi, h, neurons = sgkan_layer_step(X_current, y_train, **cfg)
        stages.append((i, phi, h))
        X_current = h
    yhat = stages[-1][2].ravel()
    return stages, yhat


# ---------------------------------------------------------------------------
# TEST: TF1-benzeri veri ile calistir, HKAN'in gorsellestirme fonksiyonlarinin
# calisip calismadigini (shape uyumu) kontrol et
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
M = 2000
X_train = rng.uniform(0, 1, size=(M, 2))
y_train = (0.3*X_train[:,0] - 0.2*X_train[:,1] + 0.8*X_train[:,0]*X_train[:,1] + 0.1)

layer_configs = [
    dict(n_vars_out=50, num_inducing=15, alpha=1e-3, seed=0),
    dict(n_vars_out=1, num_inducing=15, alpha=1e-3, seed=100),
]

stages, yhat = build_sgkan_stack(X_train, y_train, layer_configs)

print("Number of stages:", len(stages))
for layer_idx, phi, h in stages:
    print(f"Layer {layer_idx}: phi shape={phi.shape}, h shape={h.shape}")

rmse = np.sqrt(np.mean((yhat - y_train)**2))
print(f"\nTrain RMSE: {rmse:.6f}")
print(f"Train R2: {r2_score(y_train, yhat):.6f}")

# HKAN'in plot_first_layer_blf_r2_summary'sinin BEKLEDIGI seyle ayni mi kontrol edelim
layer_idx, phi, h = stages[0]
n_vars_out, n_vars_in, _ = phi.shape
for p in range(n_vars_in):
    r2_vals = [r2_score(y_train, phi[q, p, :]) for q in range(n_vars_out)]
    print(f"p={p}: R2 ortalama={np.mean(r2_vals):.4f}, max={np.max(r2_vals):.4f}")
