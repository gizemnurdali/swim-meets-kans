# Report: Sanity Check of `hkan.py` — Analysis of the `re-running-of-hkan.ipynb` Notebook

## 1. Overview

The notebook `re-running-of-hkan.ipynb` serves as a **sanity check** for the HKAN (Hierarchical Kolmogorov-Arnold Network) implementation in `hkan.py`. It manually walks through every building block of the HKAN pipeline—center generation, basis-function application, expanding layers, connecting layers, layer stacking, and hyperparameter search—to verify correctness and explore the model's behaviour on regression datasets.

---

## 2. Key Components of `hkan.py`

### 2.1 Activation / Basis Functions

Six callable classes are defined, each parameterised by a slope/bandwidth scalar `s`:

| Class | Formula | Role |
|---|---|---|
| `Sigmoid(s)` | $\sigma(x) = \frac{1}{1+e^{-sx}}$ | Smooth saturating basis |
| `Gaussian(s)` | $g(x) = e^{-(sx)^2}$ | Localised radial basis |
| `ReLU(s)` | $r(x) = \max(0, sx)$ | Piece-wise linear basis |
| `Tanh(s)` | $t(x) = \tanh(sx)$ | Smooth odd basis |
| `Softplus(s)` | $\text{sp}(x) = \ln(1+e^{sx})$ | Smooth ReLU approximation |
| `Identity` | $f(x) = x$ | Pass-through (no nonlinearity) |

The parameter `s` is **not learned**; it is a hyperparameter that controls the slope or bandwidth of the basis function.

### 2.2 `make_centers`

```
make_centers(n_vars_out, n_vars_in, n_basis, centers, X=None)
→ ndarray of shape (n_vars_out, n_vars_in, n_basis)
```

**Purpose:** Generates the center locations $c_{q,p,k}$ against which input values are shifted before the basis function is applied. Each "edge" connecting input variable $p$ to output node $q$ gets its own set of `n_basis` centers.

**Three strategies:**

| `centers` value | Logic |
|---|---|
| `"random"` | Uniform $\mathcal{U}(0,1)$ samples, independent of data. |
| `"equally_spaced"` | `np.linspace(0, 1, n_basis)` tiled identically across all edges. |
| `"random_data_points"` | For each edge $(q, p)$, randomly sample `n_basis` values from $X_{:,p}$. This is **data-dependent** and tends to place centers where real data lies, yielding better coverage. |

**Returned shape:** `(n_vars_out, n_vars_in, n_basis)` — one set of centers per edge in the bipartite graph between two layers.

The notebook manually calls `make_centers` with `centers="random_data_points"` on a toy $(10 \times 2)$ dataset and inspects the resulting array, confirming that:
- The shape is `(n_vars_out=2, n_vars_in=2, n_basis=3)`.
- `centers_arr_[0]` gives the two edges feeding into the first hidden node.
- `centers_arr_[0, 0]` gives the three center values on the edge from input 0 to hidden node 0.

### 2.3 `apply_basis_fn`

```
apply_basis_fn(X, centers_arr, basis_fn, q, p)
→ ndarray of shape (n_samples, n_basis)
```

**Purpose:** For a single edge $(q, p)$, compute the "transformed features" that will be fed into a per-edge linear regression.

**Logic (step by step):**

1. Extract the $p$-th input column: $\mathbf{x}_p \in \mathbb{R}^{n}$ (reshaped to column $(n, 1)$).
2. Extract the centers for this edge: $\mathbf{c}_{q,p} \in \mathbb{R}^{k}$ (reshaped to row $(1, k)$).
3. Compute the **difference matrix** via broadcasting:

$$D_{i,k} = x_{i,p} - c_{q,p,k} \quad \in \mathbb{R}^{n \times k}$$

4. Apply the basis function element-wise:

$$\Phi_{i,k} = \phi(D_{i,k})$$

The result $\Phi \in \mathbb{R}^{n \times k}$ is a matrix where each column $k$ represents basis function $\phi$ centred at $c_{q,p,k}$ evaluated on all samples.

### 2.4 `ExpandingLayer`

A scikit-learn `TransformerMixin` that represents the first half of a KAN layer.

**`fit(X, y)`:**
- Calls `make_centers` to initialise one set of centers per edge.
- Iterates over every edge $(q, p)$ in the bipartite graph ($q \in [0, n\_vars\_out)$, $p \in [0, n\_vars\_in)$).
- For each edge, computes $\Phi = \texttt{apply\_basis\_fn}(X, \text{centers}, \phi, q, p)$ and fits a `LinearRegression(fit_intercept=False)` from $\Phi$ to $y$.
- Stores the list of `(q, p, fitted_regressor)` tuples.

**`transform(X)`:**
- For every stored edge model, transforms $X$ through the same basis function and predicts with the fitted regressor.
- Produces a 3-D output tensor of shape `(n_vars_out, n_vars_in, n_samples)` — each slice `[q, p, :]` holds one edge's scalar predictions for all samples.

### 2.5 `ConnectingLayer`

A scikit-learn `TransformerMixin + RegressorMixin` that forms the second half of a KAN layer.

**`fit(X, y)`:** (receives the 3-D output of the `ExpandingLayer`)
- For each output node $q$, takes `X[q, :, :].T` — a matrix of shape `(n_samples, n_vars_in)` containing all edge predictions feeding into node $q$.
- Fits a `LinearRegression(fit_intercept=True)` to aggregate those edge predictions into a single output per node.

**`transform(X)`** (when `n_vars_out > 1`):
- Produces shape `(n_samples, n_vars_out)`, the activations of the hidden layer that become the input of the next HKAN layer.

**`predict(X)`** (when `n_vars_out == 1`):
- Returns the final scalar predictions for a single-output regression.

### 2.6 `make_hkan_layer`

```python
make_hkan_layer(*, layer_idx, n_vars_out, n_basis, centers, basis_fn,
                expanding_base_regressor, connecting_base_regressor)
→ sklearn.pipeline.Pipeline
```

Constructs a **single HKAN layer** as a scikit-learn `Pipeline` with two steps:
1. `expanding_layer_{layer_idx}` — an `ExpandingLayer`
2. `connecting_layer_{layer_idx}` — a `ConnectingLayer`

### 2.7 `extend_hkan`

```python
extend_hkan(model, *, layer_idx, n_vars_out, n_basis, centers, basis_fn, ...)
→ sklearn.pipeline.Pipeline
```

**Purpose:** Appends a new HKAN layer to an existing pipeline by concatenating the steps of the existing model with the steps of a newly created layer. This is how deeper HKAN architectures are built incrementally.

---

## 3. Notebook Walkthrough

### 3.1 Section: Debugging the Code (Cells 2–16)

The notebook creates a **minimal toy dataset** to manually trace the pipeline:

```
X = (10, 2)   — 10 samples, 2 input variables
y = X.sum(axis=1) * 2
n_vars_out = 2, n_basis = 3, centers = "random_data_points", basis_fn = ReLU(s=1)
```

**Step-by-step sanity checks performed:**

1. **Center creation** — Calls `make_centers` and prints the resulting `(2, 2, 3)` array, verifying that each edge's centers are sampled from the correct input column.

2. **Expanding layer (manual fit)** — Loops over all 4 edges ($2 \times 2$), calls `apply_basis_fn` to get transformed features of shape `(10, 3)`, and fits a `LinearRegression(fit_intercept=False)` per edge. Confirms 4 `(q, p, model)` tuples are stored.

3. **Expanding layer (manual transform)** — Applies each edge model to produce `out` of shape `(2, 2, 10)` — two hidden nodes, two input contributions each, ten samples.

4. **Connecting layer (manual fit)** — For each hidden node $q$, fits a `LinearRegression(fit_intercept=True)` on `out[q, :, :].T` (shape `(10, 2)`), aggregating the two edge outputs.

5. **Connecting layer (manual transform)** — Predicts from each node's model and collects results in `preds` of shape `(2, 10)`. Notes that a final connecting layer with `n_vars_out=1` would be needed to produce a scalar prediction.

### 3.2 Section: Paper Analysis — Assumptions (Cells 17–27)

Uses the **TF5** synthetic dataset (2-input, grid-structured) to test two hypotheses with 3D surface plots.

#### Assumption A.1: HKAN Depth = Stability, Not Expressivity

Two cases are compared:

| | Model 1 (1 layer) | Model 2 (2 layers) |
|---|---|---|
| **Case 1** | `centers="random_data_points"`, `Tanh(s=50)`, `n_basis=23`, `Ridge(α=0.01)` | L0: `"random"`, L1: `"random_data_points"`, same basis/alpha |
| **Case 2** | `centers="random"` (uniform), same otherwise | Same 2-layer model as Case 1 |

**Key observations documented in the notebook:**
- No new modes, ridges, or valleys appear in the 2-layer model.
- `random_data_points` is superior to `random` for 1-layer models — evidence that **smart center sampling matters**.
- The second layer mainly stabilises the approximation (lowering RMSE) but does not change the function's global shape, since it operates on **predictions from the previous layer** rather than raw inputs.
- Depth adds **stability and variance reduction**, not fundamentally new expressivity.

### 3.3 Section: Hyperparameter Selection (Cells 28–39)

#### `build_hkan_from_params`

A utility function that constructs multi-layer HKAN models from a list of per-layer parameter dictionaries. Each dictionary specifies: `layer`, `n_vars_out`, `basis_fn`, `n_basis`, `centers`, and `regressor`.

#### Optuna Search

An automated hyperparameter search using Optuna:

- **Search space per layer:** `n_vars_out` ∈ {128, 256, 512} (forced to 1 for final layer), `n_basis` ∈ [1, 50], basis function index (selecting among the 6 functions), center method, Ridge alpha ∈ [1e-4, 1.0] (log-uniform).
- **Up to 3 layers** (`max_layers=3`).
- **Objective:** RMSE on the validation set (e.g., the `compactive` dataset).
- After the study, the best trial's parameters are extracted and the final model is retrained.

#### Comparison with Baseline

A 1-layer baseline with `Tanh(s=50)`, `n_basis=28`, `centers="random_data_points"`, `Ridge(α=0.01)` is trained for comparison. The `plot_model_comparison_3d_new` function handles datasets whose test points don't form a perfect grid by using `scipy.interpolate.griddata`.

### 3.4 Section: Torch Implementation (Cells 40–46)

A brief exploration of the PyTorch-based reimplementation (`src/basis_functions.py`, `src/activations.py`, `src/hkan_torch.py`). The notebook verifies that both the NumPy-based and PyTorch-based `Sigmoid` classes produce consistent outputs on a test tensor.

---

## 4. Data Flow Summary

The complete HKAN forward pass for a single layer follows this data flow:

```
Input X                                  shape: (n_samples, n_vars_in)
    │
    ▼
┌─────────────────────────────────────┐
│          ExpandingLayer              │
│                                     │
│  For each edge (q, p):              │
│    1. diff = X[:, p] − c[q,p,:]    │  → (n_samples, n_basis)
│    2. Φ = basis_fn(diff)            │  → (n_samples, n_basis)
│    3. ŷ = LinearReg(Φ)             │  → (n_samples,)
│                                     │
│  Stack all edges:                   │
│    out[q, p, :] = ŷ                │  → (n_vars_out, n_vars_in, n_samples)
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         ConnectingLayer              │
│                                     │
│  For each output node q:            │
│    features = out[q, :, :].T        │  → (n_samples, n_vars_in)
│    ŷ_q = LinearReg(features)        │  → (n_samples,)
│                                     │
│  n_vars_out > 1 → transform:       │
│    result.T                         │  → (n_samples, n_vars_out)
│  n_vars_out = 1 → predict:         │
│    result                           │  → (n_samples,)
└─────────────────────────────────────┘
```

For multi-layer models, the output of a `ConnectingLayer` (with `n_vars_out > 1`) becomes the input $X$ for the next `ExpandingLayer`.

---

## 5. Mathematical Formulation

Given input $\mathbf{X} \in \mathbb{R}^{n \times d_{in}}$ and centers $\mathbf{c} \in \mathbb{R}^{d_{out} \times d_{in} \times k}$:

### Expanding Layer

For edge $(q, p)$:

$$\Phi^{(q,p)} = \phi\!\left(\mathbf{x}_p \mathbf{1}_k^\top - \mathbf{1}_n \mathbf{c}_{q,p}^\top\right) \in \mathbb{R}^{n \times k}$$

$$\hat{\mathbf{y}}^{(q,p)} = \Phi^{(q,p)} \mathbf{w}^{(q,p)}$$

where $\mathbf{w}^{(q,p)} \in \mathbb{R}^k$ are fitted via ordinary (or Ridge) least squares.

### Connecting Layer

For output node $q$:

$$\mathbf{F}_q = \left[\hat{\mathbf{y}}^{(q,0)}, \hat{\mathbf{y}}^{(q,1)}, \ldots, \hat{\mathbf{y}}^{(q,d_{in}-1)}\right] \in \mathbb{R}^{n \times d_{in}}$$

$$\hat{\mathbf{z}}_q = \mathbf{F}_q \boldsymbol{\beta}_q + \beta_{q,0}$$

where $\boldsymbol{\beta}_q$ and $\beta_{q,0}$ are fitted with intercept via linear regression.

---

## 6. Key Takeaways from the Sanity Check

1. **`make_centers` with `"random_data_points"`** places centers at actual observed input values, giving better coverage than uniform random sampling — especially important for 1-layer models.

2. **`apply_basis_fn`** correctly broadcasts the difference between all samples and all centers, producing the $(n \times k)$ design matrix for per-edge regression.

3. **The Expanding → Connecting pipeline** decomposes a multivariate regression into many univariate basis-expansion problems (one per edge), then recombines them linearly — a faithful implementation of the KAN architecture using closed-form least-squares rather than gradient descent.

4. **Depth adds stability, not expressivity**: the 2-layer model improves RMSE without introducing new surface features, since subsequent layers operate on previous-layer predictions.

5. **Hyperparameter sensitivity**: the notebook demonstrates that center method, basis function choice, number of basis functions, regularization strength, and number of hidden nodes all interact significantly — motivating the Optuna-based search.

6. **PyTorch parity**: the `src/` reimplementation mirrors the NumPy-based `hkan.py` logic using `torch.linalg.lstsq` instead of scikit-learn regressors, enabling future GPU acceleration and gradient-based fine-tuning via the `learnable` flag.
