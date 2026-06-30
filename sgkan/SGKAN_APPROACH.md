# Surrogate-Guided KAN (SGKAN) Approach

## Overview

SGKAN is a hybrid approach that combines **global Gaussian Process (GP) surrogates** with **local adaptive edge functions** to create an interpretable, data-driven neural network architecture inspired by Kolmogorov-Arnold Networks (KAN).

## Key Insight

Rather than learning edge functions directly from data (like standard KAN with B-splines), SGKAN uses a **global GP model** to guide the selection and construction of **local edge functions** along strategically chosen segments in the input space.

## Architecture

```
Training Data
    ↓
[1] Fit Global GP Surrogate
    (single model sees all training data)
    ↓
[2] SWIM-Style Pair Selection
    (sample candidate pairs, score with GP gradients + uncertainty)
    ↓
[3] Sample Edge Functions Locally
    (extract GP mean along selected segment directions)
    ↓
[4] Interpolate for New Inputs
    (project input onto segment directions, interpolate)
    ↓
[5] Build Layer Features
    (apply activation, stack into feature matrix)
    ↓
[6] Solve Output Layer via OLS
    (least squares on layer outputs)
```

## Pipeline Details

### Step 1: Global GP Training
- Train a single Gaussian Process on the entire training set
- Automatically selects ExactGP (N < 2000) or SparseGP (N ≥ 2000) for scalability
- Provides posterior mean and uncertainty estimates across input space

### Step 2: SWIM-Style Pair Scoring
From candidate pairs (x_a, x_b):
```
score = ||∇μ(x_a) - ∇μ(x_b)||∞  /  (σ(x_a) + Σσ(interior) + σ(x_b))
```
- **Numerator**: L-infinity norm of gradient difference (how much function varies along segment)
- **Denominator**: Sum of uncertainties (higher uncertainty → lower priority)
- **Interpretation**: Select pairs where GP is confident and function changes significantly

### Step 3: Sample Edge Functions Locally
For each selected pair (x_a, x_b):
- Create G_sample points uniformly along segment: x(t) = x_a + t·(x_b - x_a), t ∈ [0,1]
- Query GP posterior mean at these points: f(x(t))
- Store as 1D lookup table per edge

### Step 4: Interpolate for New Inputs
For input x and edge i with segment (x_a, x_b):
1. Compute unit direction: d = (x_b - x_a) / ||x_b - x_a||
2. Project x onto direction: t = (x - x_a) · d
3. Interpolate: output = interp(t, t_grid, f_grid)

### Step 5: Layer Construction
```
H[n, i] = edge_function_i(x_n)  →  (N, width) feature matrix
H_activated = activation(H)      →  input for next layer
```

### Step 6: Output Layer
```
y = [H | 1] @ W_out  (least squares solution)
```

## Local vs Global Nature

| Aspect | Scope | Mechanism |
|--------|-------|-----------|
| **GP Model** | GLOBAL | Single model trained on entire dataset |
| **Edge Definition** | LOCAL | Each edge defined between specific pair of points |
| **Function Sampling** | LOCAL | GP mean extracted only along segment directions |
| **Prediction** | LOCAL | New inputs projected onto segment, interpolated locally |
| **Guidance** | GLOBAL | Global GP gradients and uncertainty guide pair selection |

**Hybrid Interpretation:**
- The global GP provides a smooth, data-driven estimate of the function landscape
- Local edges extract the function along strategically important directions
- Result: Adaptive basis functions guided by global structure, but computed locally

## Advantages

1. **Interpretability**: Each edge has a clear 1D function (like KAN)
2. **Data Efficiency**: Leverages full GP uncertainty to guide edge selection
3. **Adaptive**: Pair selection focuses on regions of high gradient or uncertainty
4. **Scalability**: Sparse GP for large datasets
5. **Flexibility**: Multiple layers can compose complex functions

## Hyperparameters

```python
layer_config = {
    "width": int,      # number of edges/neurons per layer (32-200)
    "M": int,          # candidate pairs to sample (500-1000)
    "G": int,          # grid points per edge function (25-100)
    "T": int,          # interior points for uncertainty (5-50)
}

gp_params = {
    "kernel": str,          # "rbf" or "matern" (default: Matérn ν=2.5)
    "lr": float,            # GP training learning rate (default: 0.001)
    "num_inducing": int,    # sparse GP inducing points (default: 388)
    "num_iters": int,       # GP training iterations (default: 500)
}
```

## Usage Example

```python
from sgkan import surrogate_guided_kan as sgkan
from sgkan import gaussian_process_models as gp

# Train SGKAN
layer_configs = [{"width": 64, "M": 800, "G": 50, "T": 20}]
layers, W_out = sgkan.build_sgkan(
    X_train, y_train, 
    layer_configs,
    activation=torch.tanh,
    kernel=create_matern_kernel(X_train.shape[1]),
    lr=0.001, 
    num_inducing=400, 
    num_iters=500
)

# Predict
y_pred = sgkan.predict_sgkan(layers, W_out, X_test, activation=torch.tanh)
```

## Key References

- **KAN**: Kolmogorov-Arnold Networks - learn functions as sums of univariate functions on edges
- **SWIM**: SWIM networks use data-driven pair selection for network structure
- **GP Posterior Mean**: Used instead of samples to ensure smooth, consistent edge functions
- **GP-SWIM Scoring**: Combines gradient information with uncertainty for informative pair selection

## Implementation Notes

- Edge functions use GP **posterior mean** (not samples) for consistency across evaluations
- Segments can be degenerate (x_a = x_b); handled by using mean of edge function
- Inducing points in Sparse GP are learnable during training
- Output layer solved via stable least squares (torch.linalg.lstsq)