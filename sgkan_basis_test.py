import numpy as np
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------------------
# Sentetik TF1-benzeri veri: saf bilineer/etkilesim fonksiyonu, gurultusuz
# (kullanicinin TF1'de bulduguyla ayni ruhta: y ~ a*x0 + b*x1 + c*x0*x1 + d)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
M = 5000
tf1_x_train = rng.uniform(0, 1, size=(M, 2))
tf1_y_train = (0.3 * tf1_x_train[:, 0] - 0.2 * tf1_x_train[:, 1]
               + 0.8 * tf1_x_train[:, 0] * tf1_x_train[:, 1] + 0.1)
# NOT: gercek TF1 gurultusuzdu, biz de gurultu eklemiyoruz


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar (daha once tanimladiklarimizla ayni)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# YENI: sadece kernel-ozellik matrisi donduren fonksiyon (alpha_z YOK, y'ye
# hic bakmiyor -- HKAN'daki "merkezler y-bagimsiz" mantigi)
# ---------------------------------------------------------------------------
def fit_and_predict_basis_only(X, p, local_x_p, num_inducing=15, sigma_scale=1, seed=0):
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
    k_star = np.exp(-(x_star[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))  # (M, n_inducing)
    return k_star


def run_layer_basis(X, y, layer_width, num_inducing=15, seed=0, alpha=1.0, sigma_scale=1.0):
    D_in = X.shape[1]
    x_a, x_b = select_pairs_gen(X, y, layer_width, random_seed=seed)

    H = np.zeros((len(y), layer_width))
    models = []

    for q in range(layer_width):
        feature_blocks = []
        for p in range(D_in):
            lo = min(x_a[q, p], x_b[q, p])
            hi = max(x_a[q, p], x_b[q, p])
            local_x_p, local_y, _ = collect_local_gen(X, y, p, lo, hi)
            k_feat = fit_and_predict_basis_only(X, p, local_x_p, num_inducing=num_inducing,
                                                  sigma_scale=sigma_scale, seed=seed + q)
            feature_blocks.append(k_feat)

        all_features = np.column_stack(feature_blocks)   # (M, D_in * n_inducing)

        model_q = Ridge(alpha=alpha, fit_intercept=True)
        model_q.fit(all_features, y)

        H[:, q] = model_q.predict(all_features)
        models.append(model_q)

    return H, models


# ---------------------------------------------------------------------------
# TEST 1: eski yontem (GP alpha_z + Ridge, iki-asamali) -- karsilastirma icin
# ---------------------------------------------------------------------------
def fit_and_predict_gen(X, p, local_x_p, local_y, num_inducing=15,
                         sigma_scale=1, noise_ratio=0.1, seed=0):
    n = local_x_p.shape[0]
    n_ind = min(num_inducing, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_ind, replace=False)
    z_p = np.sort(local_x_p[idx])
    diffs = np.abs(local_x_p[:, None] - local_x_p[None, :])
    mask_diag = ~np.eye(n, dtype=bool)
    l_p = max(sigma_scale * np.median(diffs[mask_diag]) if n > 1 else sigma_scale, 1e-6)
    s2_p = np.var(local_y) + 1e-8
    sn2_p = noise_ratio * s2_p
    K_mm = s2_p * np.exp(-(z_p[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    K_nm = s2_p * np.exp(-(local_x_p[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    A = K_nm.T @ K_nm + sn2_p * K_mm
    b = K_nm.T @ local_y
    alpha_p = np.linalg.solve(A + 1e-10 * np.eye(n_ind), b)
    x_star = X[:, p]
    k_star = s2_p * np.exp(-(x_star[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    return k_star @ alpha_p


def run_layer_gp(X, y, layer_width, num_inducing=15, seed=0, ridge_lambda=1.0):
    D_in = X.shape[1]
    x_a, x_b = select_pairs_gen(X, y, layer_width, random_seed=seed)
    H = np.zeros((len(y), layer_width))
    models = []
    for q in range(layer_width):
        edge_outs = []
        for p in range(D_in):
            lo = min(x_a[q, p], x_b[q, p])
            hi = max(x_a[q, p], x_b[q, p])
            local_x_p, local_y, _ = collect_local_gen(X, y, p, lo, hi)
            edge_out = fit_and_predict_gen(X, p, local_x_p, local_y,
                                             num_inducing=num_inducing, seed=seed + q)
            edge_outs.append(edge_out)
        edge_outputs_q = np.column_stack(edge_outs)
        model_q = Ridge(alpha=ridge_lambda, fit_intercept=True)
        model_q.fit(edge_outputs_q, y)
        H[:, q] = model_q.predict(edge_outputs_q)
        models.append(model_q)
    return H, models


# ---------------------------------------------------------------------------
# KARSILASTIRMA
# ---------------------------------------------------------------------------
layer_width_1 = 200
layer_width_2 = 1

print("=" * 60)
print("YONTEM A: GP (alpha_z) + Ridge, iki-asamali y-fitting")
print("=" * 60)
H1_a, _ = run_layer_gp(tf1_x_train, tf1_y_train, layer_width_1, seed=0)
H2_a, _ = run_layer_gp(H1_a, tf1_y_train, layer_width_2, seed=100)
model_final_a = Ridge(alpha=1.0, fit_intercept=True)
model_final_a.fit(H2_a, tf1_y_train)
pred_a = model_final_a.predict(H2_a)
rmse_a = np.sqrt(np.mean((pred_a - tf1_y_train) ** 2))
print(f"RMSE: {rmse_a:.8f}")

print()
print("=" * 60)
print("YONTEM B: Sadece kernel-basis + TEK Ridge (y'ye tek bakis)")
print("=" * 60)
H1_b, _ = run_layer_basis(tf1_x_train, tf1_y_train, layer_width_1, seed=0, alpha=1e-3)
H2_b, _ = run_layer_basis(H1_b, tf1_y_train, layer_width_2, seed=100, alpha=1e-3)
model_final_b = Ridge(alpha=1e-3, fit_intercept=True)
model_final_b.fit(H2_b, tf1_y_train)
pred_b = model_final_b.predict(H2_b)
rmse_b = np.sqrt(np.mean((pred_b - tf1_y_train) ** 2))
print(f"RMSE: {rmse_b:.8f}")

print()
print(f"y std: {tf1_y_train.std():.5f}")
