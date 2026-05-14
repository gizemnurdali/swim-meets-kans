import numpy as np
import sys
sys.path.insert(0, '/Users/gizemnurdal/Workspace/swim-meets-kans/swimnetworks-paper')

from swimnetworks.dense import Dense
from swimnetworks.linear import Linear

print("=" * 80)
print("SWIM SANITY CHECK: Single Hidden Layer Network")
print("=" * 80)

# Create data: y = 2x + 1
X_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([3, 5, 7, 9, 11]).reshape(-1, 1)

print(f"\n1. X_train: shape={X_train.shape}, first_2={X_train[:2].ravel()}")
print(f"2. y_train: shape={y_train.shape}, first_2={y_train[:2].ravel()}")

# Dense layer
dense = Dense(
    layer_width=3,
    activation="tanh",
    parameter_sampler="tanh",
    random_seed=1
)

print(f"\n[Dense Layer - Fitting...]")
dense.fit(X_train, y_train)

H1 = dense.transform(X_train)
print(f"21. H1 output: shape={H1.shape}, first_2_rows:")
print(H1[:2])

# Linear layer
linear = Linear(regularization_scale=1e-10)

print(f"\n[Linear Layer - Fitting...]")
linear.fit(H1, y_train)

print(f"24. Linear weights: shape={linear.weights.shape}")
print(f"25. Linear biases: shape={linear.biases.shape}")

# Predictions
y_pred = linear.transform(H1)
print(f"\n26. Final predictions: shape={y_pred.shape}, first_2={y_pred[:2].ravel()}")

# MSE
mse = np.mean((y_pred - y_train) ** 2)
print(f"    MSE: {mse}")

print("\n" + "=" * 80)
print("SANITY CHECK COMPLETE")
print("=" * 80)
