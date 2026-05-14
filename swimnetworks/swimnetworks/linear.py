"""A module to represent a linear layer trained with least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .base import Base


@dataclass
class Linear(Base):
    """Linear fully connected layer class.

    This is a (supervised) linear layer implemented in the style of
    scikit-learn's BaseEstimator.

    Attributes:
        regularization_scale (float): parameter for least squares
        cut-off for singular values.

    """

    regularization_scale: float = 1e-8

    def fit(self, x: npt.NDArray, y: npt.NDArray = None) -> Linear:
        """Layer fitting procedure.

        For the data arrays x and y this function fits and stores the
        weight and bias attributes using a least squares solver.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.

        Returns:
            Linear: self.

        """
        # Ensure compliant data arrays.
        x, y = self.clean_inputs(x, y)

        # Prepare to fit the bias as well.
        x = np.column_stack([x, np.ones((x.shape[0], 1))])
        self.weights = np.linalg.lstsq(x, y, rcond=self.regularization_scale)[0]

        # Separate weights and biases back.
        self.biases = self.weights[-1:, :]
        self.weights = self.weights[:-1, :]

        self.layer_width = self.weights.shape[1]
        self.n_parameters = np.prod(self.weights.shape) + np.prod(self.biases.shape)
        (self.input_shape, self.output_shape) = self.weights.shape

        return self

    def transform(self, x: npt.NDArray, y: npt.NDArray = None) -> npt.NDArray:
        """Transform data.

        The provided data array x is passed through the layer to
        provide predictions.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.

        Returns:
            npt.NDArray: Predicted data of shape (N x D_target).

        """
        y_predict = super().transform(x, y)

        # Undo preprocessing on outputs.
        return self.prepare_y_inverse(y_predict)
