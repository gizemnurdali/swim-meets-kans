import numpy as np
import sys
sys.path.insert(0, '/Users/gizemnurdal/Workspace/swim-meets-kans/swimnetworks-paper')

from swimnetworks.dense import Dense
from swimnetworks.linear import Linear

# Set random seed
np.random.seed(42)

# Training data: 1000 samples, 2 dimensions
N_train = 1000
D = 2
X_train = np.random.uniform(0, 2*np.pi, size=(N_train, D))
y_train = (X_train[:, 0]**2 + np.sin(X_train[:, 1])).reshape(-1, 1)

# Test data: 200 samples, same distribution
# y = x1ˆ2 + sin(xˆ2)
N_test = 200
X_test = np.random.uniform(0, 2*np.pi, size=(N_test, D))
y_test = (X_test[:, 0]**2 + np.sin(X_test[:, 1])).reshape(-1, 1)

print(f"Training data:")
print(f"  X_train: shape={X_train.shape}")
print(f"  y_train: shape={y_train.shape}")
print(f"\nTest data:")
print(f"  X_test: shape={X_test.shape}")
print(f"  y_test: shape={y_test.shape}")
print(f"\nData ranges:")
print(f"  X_train min/max: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"  y_train min/max: [{y_train.min():.4f}, {y_train.max():.4f}]")

# First Dense layer
dense1 = Dense(
    layer_width=128,
    activation="tanh",
    parameter_sampler="tanh",
    random_seed=1
)

print("Fitting Dense layer 1...")
dense1.fit(X_train, y_train)
print("Done!")

# Transform with first Dense layer
H1 = dense1.transform(X_train)
print(f"\nH1 (hidden layer 1 output): shape={H1.shape}")
print(f"Dense1 weights: shape={dense1.weights.shape}")
print(f"Dense1 biases: shape={dense1.biases.shape}")


# Second Dense layer
dense2 = Dense(
    layer_width=128,
    activation="tanh",
    parameter_sampler="tanh",
    random_seed=2
)

print("Fitting Dense layer 2...")
dense2.fit(H1, y_train)
print("Done!")

# Transform with second Dense layer
H2 = dense2.transform(H1)
print(f"\nH2 (hidden layer 2 output): shape={H2.shape}")
print(f"Dense2 weights: shape={dense2.weights.shape}")
print(f"Dense2 biases: shape={dense2.biases.shape}")

# Linear output layer
linear = Linear(regularization_scale=1e-10)

print("Fitting Linear layer...")
linear.fit(H2, y_train)
print("Done!")

print(f"\nLinear weights: shape={linear.weights.shape}")
print(f"Linear biases: shape={linear.biases.shape}")