"""A module to represent a sampled dense layer."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .base import Base


@dataclass
class Dense(Base):
    """Dense fully connected layer class.

    This is a (supervised) dense layer implemented in the style of
    scikit-learn's BaseEstimator. "Dense" means that the layer has
    a linear transformation followed by a nonlinear activation.

    Attributes:
        parameter_sampler (Callable | str, optional): Parameter that
        determines sampling strategy.
        sample_uniformly (bool, optional): Flag to indicate if sampling
        should be uniform (data-agnostic).
        prune_duplicates (bool, optional): Flag to indicate if duplicate
        parameters should be ruled out.
        random_seed (int, optional): Seed used to create random number
        generator (for reproducibility).
        dist_min (np.float64, optional): Minimal allowed distance
        between pairs of points.
        repetition_scaler (int, optional): Indicates how many times
        candidates should be repeated
        (useful when more parameters need to be sampled than the
        available number of point pairs).

        idx_from (npt.NDArray, optional): Indices for first member of
        data pairs.
        idx_to (npt.NDArray, optional): Indices for second member of
        data pairs.

    """

    parameter_sampler: Callable | str = "relu"
    sample_uniformly: bool = False
    prune_duplicates: bool = False
    random_seed: int = 1
    dist_min: np.float64 = 1e-10
    repetition_scaler: int = 1

    idx_from: npt.NDArray | None = None
    idx_to: npt.NDArray | None = None

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        self.n_pruned_neurons = 0

        if not isinstance(self.parameter_sampler, Callable):
            if self.parameter_sampler == "relu":
                self.parameter_sampler = self.sample_parameters_relu
            elif self.parameter_sampler == "tanh":
                self.parameter_sampler = self.sample_parameters_tanh
            elif self.parameter_sampler == "random":
                self.parameter_sampler = self.sample_parameters_randomly
            else:
                msg = f"Unknown parameter sampler {self.parameter_sampler}."
                raise ValueError(msg)

    def fit(self, x: npt.NDArray, y: npt.NDArray = None) -> Dense:
        """Layer fitting procedure.

        For the passed data arrays x and y this function fits and
        stores the weight and bias attributes using a sampler.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.

        Returns:
            Dense

        """
        if self.layer_width is None:
            msg = "layer_width must be set before fitting."
            raise ValueError(msg)

        # Ensure compliant data arrays.
        x, y = self.clean_inputs(x, y)

        rng = np.random.default_rng(self.random_seed)
        weights, biases, idx_from, idx_to = self.parameter_sampler(x, y, rng)

        # Store sampled parameter data.
        self.idx_from = idx_from
        self.idx_to = idx_to
        self.weights = weights
        self.biases = biases

        self.n_parameters = np.prod(weights.shape) + np.prod(biases.shape)
        (self.input_shape, self.output_shape) = weights.shape

        return self

    def sample_parameters_tanh(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Sample parameters for tanh activation.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.
            rng (np.random.Generator): Random number generator.

        Returns:
            npt.NDArray: Sampled weights.
            npt.NDArray: Sampled biases.
            npt.NDArray: Indices for first members in ordered pairs of
            sampled datapoints used to construct the weights and biases.
            npt.NDArray: Indices for second members in ordered pairs of
            sampled datapoints used to construct the weights and biases.

        """
        # Define scaling hyperparameter (called s2 in Bolager, 2023).
        scale = 0.5 * (np.log(1 + 1 / 2) - np.log(1 - 1 / 2))

        # Sample relevant points and retain distances,
        # direction and indices.
        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)

        # Construct weights and biases.
        weights = (2 * scale * directions / dists).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

        return weights, biases, idx_from, idx_to

    def sample_parameters_relu(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Sample parameters for relu activation.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.
            rng (np.random.Generator): Random number generator.

        Returns:
            npt.NDArray: Sampled weights.
            npt.NDArray: Sampled biases.
            npt.NDArray: Indices for first members in ordered pairs of
            sampled datapoints used to construct the weights and biases.
            npt.NDArray: Indices for second members in ordered pairs of
            sampled datapoints used to construct the weights and biases.


        """
        # Define scaling hyperparameter (called s2 in Bolager, 2023).
        scale = 1.0

        # Sample relevant points and retain distances,
        # direction and indices.
        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)

        # Construct weights and biases.
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to

    def sample_parameters_randomly(
        self,
        x: npt.NDArray,
        _: npt.NDArray,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Sample parameters in a data-agnostic manner.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            _ (npt.NDArray): Ignored argument.
            rng (np.random.Generator): Random number generator.

        Returns:
            npt.NDArray: Sampled weights.
            npt.NDArray: Sampled biases.
            None
            None

        """
        # Sample weights from normal distribution,
        # and biases from uniform distribution.
        weights = rng.normal(loc=0, scale=1, size=(self.layer_width, x.shape[1])).T
        biases = rng.uniform(low=-np.pi, high=np.pi, size=(self.layer_width, 1)).T

        # Set indices to None, because there are no data pairs.
        idx_from = None
        idx_to = None

        return weights, biases, idx_from, idx_to

    def sample_parameters(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Sample directions between datapoints in the dataset (x, y).

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.
            rng (np.random.Generator): Random number generator.

        Returns:
            npt.NDArray: Sampled directions.
            npt.NDArray: Sampled distances.
            npt.NDArray: Indices for first members in ordered pairs of
            sampled datapoints used to construct the weights and biases.
            npt.NDArray: Indices for second members in ordered pairs of
            sampled datapoints used to construct the weights and biases.


        """
        # n_repetitions repeats the sampling procedure
        # to find better directions.
        # If we require more samples than data points,
        # the repetitions will cause more pairs to be drawn.
        n_repetitions = (
            max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler
        )

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus
        # the exact idx_from=idx_to case).
        # (b) no indices appear twice at the same position (never
        # idx0[k]==idx1[k] for all k).
        candidates_idx_from = rng.integers(
            low=0,
            high=x.shape[0],
            size=x.shape[0] * n_repetitions,
        )
        delta = rng.integers(low=1, high=x.shape[0], size=candidates_idx_from.shape[0])
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]

        # Construct directions and distances.
        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        if y is None:
            if not self.sample_uniformly:
                msg = (
                    "sample_uniformly is not specified, "
                    "but target (y) values are not given."
                )
                warnings.warn(msg, stacklevel=2)

            dy = None
        else:
            if self.sample_uniformly:
                msg = "sample_uniformly is specified, but target (y) values are given."
                warnings.warn(msg, stacklevel=2)
            dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]
            if self.is_classifier:
                dy[np.abs(dy) > 0] = 1

        # We always sample with replacement to avoid
        # forcing to sample low densities.
        probabilities = self.weight_probabilities(dy, dists)
        selected_idx = rng.choice(
            dists.shape[0],
            size=self.layer_width,
            replace=True,
            p=probabilities,
        )

        if self.prune_duplicates:
            encoded_candidates = [
                p[1] + x.shape[0] * p[0]
                for p in zip(
                    candidates_idx_from[selected_idx],
                    candidates_idx_to[selected_idx],
                    strict=True,
                )
            ]
            _, idx_unique = np.unique(encoded_candidates, return_index=True)
            selected_idx = selected_idx[idx_unique]
            self.n_pruned_neurons = self.layer_width - len(selected_idx)
            self.layer_width = len(selected_idx)

        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]

        return directions, dists, idx_from, idx_to

    def weight_probabilities(self, dy: npt.NDArray, dists: npt.NDArray) -> npt.NDArray:
        """Compute sampling probabilities for weights.

        Compute probability that a certain weight should be chosen
        as part of the network. This method computes all probabilities
        at once, without removing the new weights one by one.

        Args:
            dy (npt.NDArray): Difference in output pairs.
            dists (npt.NDArray): Distance between the base points.

        Returns:
            npt.NDArray: Probabilities for the weights.

        """
        if self.sample_uniformly:
            probabilities = np.ones(dists.shape[0]) / len(dists)
        elif dy is not None:
            # Compute the maximum over all changes in all y
            # directions to sample good gradients for all outputs.
            gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()
            if np.sum(gradients) < self.dist_min:
                # Fall back to uniform sampling.
                probabilities = np.ones(dists.shape[0]) / len(dists)
            else:
                probabilities = gradients / np.sum(gradients)
        else:
            msg = "Cannot compute gradients without function values."
            raise ValueError(msg)

        return probabilities
