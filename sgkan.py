"""
SGKAN: SWIM + Yerel (Sparse/Inducing-Point) GP ile Gradient-Free KAN Katmani
=============================================================================

Tum konusmada ustune calistigimiz butun parcalari birlestiren, tek bastan
sona calistirilabilir bir versiyon. Bir "katman" (layer) su adimlari yapar:

  1) select_pairs        : n_vars_out kadar SWIM ciftini skor agirlikli sec
  2) collect_local_points: her (nöron, boyut) icin yerel veri topla
  3) fit_local_gp_sparse  : kapasitesi n_inducing ile sinirli GP kur (kenar)
  4) combine_edges_ols    : bir nöronun D kenarini OLS ile birlestir (h-function)

Katman ciktisi, bir sonraki katmanin girdisi olarak kullanilabilir (zincirleme).
"""

import numpy as np


# ---------------------------------------------------------------------------
# ADIM 1: SWIM ile cift secimi
# ---------------------------------------------------------------------------
def select_pairs(X, y, layer_width, random_seed=None, n_candidates=2000):
    """
    layer_width kadar cift secer (n_vars_out sayisi kadar -- her cift bir
    gizli notaya/norona karsilik gelecek).

    X: (M, D) egitim girdileri
    y: (M,)   egitim hedefleri (skaler cikti varsayiyoruz)
    layer_width: kac cift/noron secilecek
    n_candidates: skor hesaplanacak aday cift sayisi (M^2 yerine altornekleme)

    Donus:
        x_a, x_b: (layer_width, D) secilen ciftlerin baslangic/bitis noktalari
        y_a, y_b: (layer_width,)   bu noktalardaki hedef degerler
    """
    M = X.shape[0]
    rng = np.random.default_rng(random_seed)

    # aday ciftleri rastgele olustur (M^2 yerine altornekleme, SWIM'deki gibi)
    idx_a = rng.integers(0, M, size=n_candidates)
    idx_b = rng.integers(0, M, size=n_candidates)

    x_a_cand, x_b_cand = X[idx_a], X[idx_b]
    y_a_cand, y_b_cand = y[idx_a], y[idx_b]

    dist = np.linalg.norm(x_a_cand - x_b_cand, axis=1)
    dist = np.maximum(dist, 1e-8)  # sifira bolunmeyi engelle
    score = np.abs(y_a_cand - y_b_cand) / dist

    probs = score / score.sum()

    selected_idx = rng.choice(n_candidates, size=layer_width, replace=False, p=probs)

    return (x_a_cand[selected_idx], x_b_cand[selected_idx],
            y_a_cand[selected_idx], y_b_cand[selected_idx])


# ---------------------------------------------------------------------------
# ADIM 2: Yerel veri toplama (interval + capping + knn fallback)
# ---------------------------------------------------------------------------
def collect_local_points(X, y, p, lo, hi, min_points=5, max_points=100, k_fallback=10):
    """
    Bir boyut (p) icin [lo, hi] araligina dusen egitim noktalarini toplar.
    Cok azsa (knn_fallback) en yakin komsulara, cok fazlaysa (interval_capped)
    midpoint'e en yakin max_points taneye duser.
    """
    x_p = X[:, p]
    mask = (x_p >= lo) & (x_p <= hi)
    n_inside = int(mask.sum())
    midpoint = (lo + hi) / 2

    if n_inside >= min_points:
        x_inside, y_inside = x_p[mask], y[mask]
        if n_inside <= max_points:
            return x_inside, y_inside, "interval", n_inside
        order = np.argsort(np.abs(x_inside - midpoint))
        cap_idx = order[:max_points]
        return x_inside[cap_idx], y_inside[cap_idx], "interval_capped", n_inside

    order = np.argsort(np.abs(x_p - midpoint))
    knn_idx = order[:k_fallback]
    return x_p[knn_idx], y[knn_idx], "knn_fallback", n_inside


# ---------------------------------------------------------------------------
# ADIM 3: Kapasitesi sinirli (inducing-point / sparse) yerel GP
# ---------------------------------------------------------------------------
def fit_local_gp_sparse(local_x, local_y, n_inducing=10, sigma_scale=1.0,
                         noise_ratio=0.05):
    """
    Yerel noktalarin TAMAMI yerine, sabit sayida (n_inducing) esit araliki
    temsilci nokta uzerinden GP kurar. Kapasite acikca n_inducing ile sinirli
    (HKAN'daki sabit n_basis mantigiyla ayni felsefe).
    """
    n = len(local_x)
    n_inducing = min(n_inducing, n)  # veri azsa inducing sayisini asma

    z = np.linspace(local_x.min(), local_x.max(), n_inducing)

    # lengthscale: medyan sezgisi (yerel verinin kendi olcegi)
    diffs = np.abs(local_x[:, None] - local_x[None, :])
    mask = ~np.eye(n, dtype=bool)
    lengthscale = sigma_scale * np.median(diffs[mask]) if n > 1 else sigma_scale
    lengthscale = max(lengthscale, 1e-6)

    signal_var = np.var(local_y) + 1e-8
    noise_var = noise_ratio * signal_var

    K_mm = signal_var * np.exp(-(z[:, None] - z[None, :]) ** 2 / (2 * lengthscale ** 2))
    K_nm = signal_var * np.exp(-(local_x[:, None] - z[None, :]) ** 2 / (2 * lengthscale ** 2))

    A = K_nm.T @ K_nm + noise_var * K_mm
    b = K_nm.T @ local_y
    alpha_z = np.linalg.solve(A + 1e-10 * np.eye(n_inducing), b)

    return {
        "z": z, "lengthscale": lengthscale, "signal_var": signal_var,
        "noise_var": noise_var, "alpha_z": alpha_z, "n_inducing": n_inducing,
    }


def gp_predict_sparse(x_star, gp_params):
    z, l, s2, alpha_z = (gp_params["z"], gp_params["lengthscale"],
                          gp_params["signal_var"], gp_params["alpha_z"])
    k_star = s2 * np.exp(-(x_star[:, None] - z[None, :]) ** 2 / (2 * l ** 2))
    return k_star @ alpha_z


# ---------------------------------------------------------------------------
# ADIM 4: Bir katmanin tamami -- N noron, her biri D kenar, OLS ile birlesim
# ---------------------------------------------------------------------------
class SGKANLayer:
    def __init__(self, n_vars_out, n_inducing=10, sigma_scale=1.0,
                 noise_ratio=0.05, max_points=100, random_seed=None):
        self.n_vars_out = n_vars_out
        self.n_inducing = n_inducing
        self.sigma_scale = sigma_scale
        self.noise_ratio = noise_ratio
        self.max_points = max_points
        self.random_seed = random_seed

        self.neurons_ = []  # her noron icin: {edges: [gp_params]*D, w: (D+1,)}

    def fit(self, X, y):
        M, D = X.shape

        x_a, x_b, _, _ = select_pairs(X, y, self.n_vars_out, self.random_seed)

        for q in range(self.n_vars_out):
            edges = []
            edge_outputs = np.zeros((M, D))  # her kenarin TUM egitim setindeki ciktisi

            for p in range(D):
                lo, hi = sorted([x_a[q, p], x_b[q, p]])
                local_x, local_y, method, n_inside = collect_local_points(
                    X, y, p, lo, hi, max_points=self.max_points)

                gp_params = fit_local_gp_sparse(
                    local_x, local_y, n_inducing=self.n_inducing,
                    sigma_scale=self.sigma_scale, noise_ratio=self.noise_ratio)
                edges.append(gp_params)

                edge_outputs[:, p] = gp_predict_sparse(X[:, p], gp_params)

            # h-function: D kenarin ciktisini OLS ile birlestir (+ bias)
            design = np.column_stack([edge_outputs, np.ones(M)])
            w, *_ = np.linalg.lstsq(design, y, rcond=None)

            self.neurons_.append({"edges": edges, "w": w})

        return self

    def transform(self, X):
        M, D = X.shape
        out = np.zeros((M, self.n_vars_out))

        for q, neuron in enumerate(self.neurons_):
            edge_outputs = np.zeros((M, D))
            for p, gp_params in enumerate(neuron["edges"]):
                edge_outputs[:, p] = gp_predict_sparse(X[:, p], gp_params)

            design = np.column_stack([edge_outputs, np.ones(M)])
            out[:, q] = design @ neuron["w"]

        return out


# ---------------------------------------------------------------------------
# TEST: sentetik veri uzerinde tek katmanli calistirma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M, D = 500, 2
    X_train = rng.uniform(0, 1, size=(M, D))
    y_train = np.sin(2 * np.pi * X_train[:, 0]) + 0.5 * X_train[:, 1] ** 2
    y_train += rng.normal(0, 0.05, size=M)  # kucuk gozlem gurultusu

    layer = SGKANLayer(n_vars_out=20, n_inducing=8, random_seed=42)
    layer.fit(X_train, y_train)

    X_test = rng.uniform(0, 1, size=(200, D))
    y_test = np.sin(2 * np.pi * X_test[:, 0]) + 0.5 * X_test[:, 1] ** 2

    layer_out = layer.transform(X_test)  # (200, 20) -- ilk katman ciktisi

    # Tek katmanli bir model icin: bu 20 ciktiyi TEKRAR bir OLS ile
    # skaler y'ye indirgeyelim (cok-katmanli mimaride bu, bir sonraki
    # SGKANLayer'in girdisi olurdu)
    layer_out_train = layer.transform(X_train)
    design_final = np.column_stack([layer_out_train, np.ones(M)])
    w_final, *_ = np.linalg.lstsq(design_final, y_train, rcond=None)

    design_test = np.column_stack([layer_out, np.ones(len(X_test))])
    y_pred = design_test @ w_final

    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"Test RMSE: {rmse:.4f}")
    print(f"y_test std: {y_test.std():.4f}  (RMSE'nin karsilastirma noktasi)")
