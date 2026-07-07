"""
SGKAN - Final Version
======================
Her noron icin:
  - D boyutun her biri icin ayri inducing noktalar + kernel ozellikleri hesaplanir (y-bagimsiz)
  - TUM boyutlarin ozellikleri yan yana konup TEK bir Ridge ile y'ye fit edilir (ortak/joint fit,
    additive modellerin standart yontemi -- circularity yok, y'ye sadece BIR kez bakiliyor)
  - Ridge'in ogrendigi agirliklar, boyutlara gore dilimlenir -> her boyut icin GERCEK,
    bagimsiz bir univariate kenar fonksiyonu (phi_{q,p}(x_p)) elde edilir
  - Katmanlar zincirlenebilir (H, bir sonraki katmanin girdisi olur)
"""

import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# SWIM cift secimi (degismedi)
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


# ---------------------------------------------------------------------------
# Yerel veri toplama (degismedi)
# ---------------------------------------------------------------------------
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
# Bir boyut icin inducing noktalari + lengthscale + kernel ozellik matrisi
# (y-BAGIMSIZ -- sadece x'ten turetiliyor)
# ---------------------------------------------------------------------------
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
    k_star = np.exp(-(x_star[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))  # (M, n_ind)

    return k_star, z_p, l_p


# ---------------------------------------------------------------------------
# Gercek univariate kenar fonksiyonu: phi_{q,p}(x_p) = k(x_p, z_p) @ w_p
# ---------------------------------------------------------------------------
def edge_function(x_p_query, z_p, l_p, w_p):
    """
    Tek bir boyutun (p) fonksiyonu -- sadece x_p_query'ye bagli.
    x_p_query: (n,) herhangi bir sayida sorgu noktasi (egitimde gorulmemis olabilir)
    z_p, l_p, w_p: build/fit sirasinda o kenar icin saklanan parametreler
    """
    k = np.exp(-(x_p_query[:, None] - z_p[None, :]) ** 2 / (2 * l_p ** 2))
    return k @ w_p


# ---------------------------------------------------------------------------
# Bir katmanin tamami: N noron, her biri D kenar, ORTAK (joint) Ridge fit
# ---------------------------------------------------------------------------
def fit_layer(X, y, layer_width, num_inducing=15, sigma_scale=1.0, alpha=1e-3, seed=0):
    """
    Donus:
        H: (M, layer_width) -- katmanin egitim verisi uzerindeki ciktisi
        neurons: her noron icin {"z_list", "l_list", "w_per_dim", "bias"}
                 -- bu bilgiyle transform_layer, yeni (test) verisinde tahmin yapabilir
    """
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

            k_feat, z_p, l_p = build_edge_features(
                X, p, local_x_p, num_inducing=num_inducing,
                sigma_scale=sigma_scale, seed=seed + q)

            feature_blocks.append(k_feat)
            z_list.append(z_p)
            l_list.append(l_p)

        all_features = np.column_stack(feature_blocks)   # (M, D_in * n_ind) -- hala y-bagimsiz

        # --- TEK ORTAK (joint) Ridge fit -- y'ye SADECE burada bakiliyor ---
        model_q = Ridge(alpha=alpha, fit_intercept=True)
        model_q.fit(all_features, y)

        H[:, q] = model_q.predict(all_features)

        # --- agirliklari boyutlara gore dilimle -> GERCEK univariate kenarlar ---
        w_full = model_q.coef_
        sizes = [fb.shape[1] for fb in feature_blocks]
        w_per_dim = np.split(w_full, np.cumsum(sizes)[:-1])

        neurons.append({
            "z_list": z_list, "l_list": l_list,
            "w_per_dim": w_per_dim, "bias": model_q.intercept_,
        })

    return H, neurons


# ---------------------------------------------------------------------------
# Kurulu bir katmani YENI (goruunmemis) veriye uygulama
# ---------------------------------------------------------------------------
def transform_layer(X_new, neurons):
    M_new = X_new.shape[0]
    layer_width = len(neurons)
    H_new = np.zeros((M_new, layer_width))

    for q, neuron in enumerate(neurons):
        D_in = len(neuron["z_list"])
        total = neuron["bias"] * np.ones(M_new)
        for p in range(D_in):
            edge_out = edge_function(X_new[:, p], neuron["z_list"][p],
                                       neuron["l_list"][p], neuron["w_per_dim"][p])
            total += edge_out
        H_new[:, q] = total

    return H_new


# ---------------------------------------------------------------------------
# DOGRULAMA: kenarlarin toplami + bias, gercekten noron ciktisina esit mi?
# ---------------------------------------------------------------------------
def verify_edge_decomposition(X, y, neurons, H, tol=1e-8):
    H_recon = transform_layer(X, neurons)
    max_diff = np.max(np.abs(H_recon - H))
    print(f"Kenarlarin toplami ile Ridge ciktisi arasindaki max fark: {max_diff:.2e}")
    print("Tutarli mi (tol icinde)?", max_diff < tol)
    return max_diff


if __name__ == "__main__":
    # --- Sentetik TF1-benzeri (bilineer, gurultusuz) test verisi ---
    rng = np.random.default_rng(0)
    M = 5000
    X_train = rng.uniform(0, 1, size=(M, 2))
    y_train = (0.3 * X_train[:, 0] - 0.2 * X_train[:, 1]
               + 0.8 * X_train[:, 0] * X_train[:, 1] + 0.1)

    layer_width_1 = 200
    layer_width_2 = 1

    print("=" * 60)
    print("Katman 1")
    print("=" * 60)
    H1, neurons1 = fit_layer(X_train, y_train, layer_width_1, num_inducing=50, alpha=1e-3, seed=0)
    verify_edge_decomposition(X_train, y_train, neurons1, H1)

    print()
    print("=" * 60)
    print("Katman 2")
    print("=" * 60)
    H2, neurons2 = fit_layer(H1, y_train, layer_width_2, num_inducing=50, alpha=1e-3, seed=100)
    verify_edge_decomposition(H1, y_train, neurons2, H2)

    # --- Final: H2'yi Ridge ile skaler y'ye indir ---
    model_final = Ridge(alpha=1e-3, fit_intercept=True)
    model_final.fit(H2, y_train)
    y_pred = model_final.predict(H2)

    rmse = np.sqrt(np.mean((y_pred - y_train) ** 2))
    print()
    print(f"2 katmanli SGKAN (univariate kenarlarla) RMSE: {rmse:.8f}")
    print(f"y std: {y_train.std():.5f}")

    # --- Ornek: tek bir kenari cizdirmeye hazir hale getirme ---
    q_example, p_example = 0, 0
    z_p = neurons1[q_example]["z_list"][p_example]
    l_p = neurons1[q_example]["l_list"][p_example]
    w_p = neurons1[q_example]["w_per_dim"][p_example]

    x_dense = np.linspace(0, 1, 100)
    edge_curve = edge_function(x_dense, z_p, l_p, w_p)
    print(f"\nOrnek kenar (q={q_example}, p={p_example}) ilk 5 deger:", np.round(edge_curve[:5], 5))
