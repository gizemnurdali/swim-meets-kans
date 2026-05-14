"""A base module to represent a layer in a feedforward network."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator


@dataclass
class Base(BaseEstimator, ABC):
    """Base class for layers in the network.

    This class serves as a base (or superclass) for all layers in
    swimnetworks. It inherits from scikit-learn's BaseEstimator, making
    sure all layers in the networks can be fit through scikit-learn's
    Pipeline.

    Attributes:
        is_classifier (bool, optional): True if the underlying problem,
            is classification, False if regression. Default is False.
        layer_width (int, optional): The dimension D_out, which
            specifies the output dimension of the layer.
            Default is None.
        activation (Callable[[npt.NDArray], npt.NDArray] | str,
            optional): A function applied elementwise, or string with
            name of already implemented activation functions.
            See __post__init__ for list of implemented functions.
            Default is "none".
        weights_ (npt.NDArray, optional): Weight matrix of the layer,
            of shape (D_in x D_out). Default is None.
        biases_ (npt.NDArray, optional): Biases vector of the layer,
            of shape (1 x D_out).
        n_parameters (int, optional): Total number of parameters in the
            layer, which is counted when model is fitted. Default is 0.
        input_shape (Tuple[int, ...], optional): A tuple of integers,
            which defines the shape of the layer. Default is None.
        output_shape (Tuple[int, ...], optional): A tuple of integers,
            which defines the output shape of the layer.
            Default is None.

    """

    is_classifier: bool = False
    layer_width: int | None = None

    activation: Callable[[npt.NDArray], npt.NDArray] | str = "none"
    weights_: npt.NDArray = field(default_factory=lambda: np.empty((0, 0)))
    biases_: npt.NDArray = field(default_factory=lambda: np.empty((0, 0)))
    n_parameters: int = 0

    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None

    @property
    def weights(self) -> npt.NDArray:
        """Getter for weights.

        "weights_" with underscore is required for scikit-learn to
        check if the Pipeline is fitted.

        Returns:
            npt.NDArray: The weights, with the necessary underscore
                required by scikit-learn, of shape (D_in x D_out).

        """
        return self.weights_

    @property
    def biases(self) -> npt.NDArray:
        """Getter for biases.

        "biases_" with underscore is required for scikit-learn to check
        if the Pipeline is fitted.

        Returns:
            npt.NDArray: The bias vector, with the necessary underscore
                required by scikit-learn, of shape (1 x D_out).

        """
        return self.biases_

    @weights.setter
    def weights(self, w: npt.NDArray) -> None:
        """Setter for weights, setting weights to equal w.

        "weights_" with underscore is required for scikit-learn to
        check if the Pipeline is fitted.

        Args:
            w (npt.NDArray): The matrix weights are set to equal,
                of shape (D_in x D_out).

        """
        self.weights_ = w

    @biases.setter
    def biases(self, b: npt.NDArray) -> None:
        """Setter for biases, setting biases to equal b.

        "biases_" with underscore is required for scikit-learn to check
        if the Pipeline is fitted.

        Args:
            b (npt.NDArray): The vector biases are set to equal,
                of shape (1, D_out).

        """
        self.biases_ = b

    @staticmethod
    def identity_activation(x: npt.NDArray) -> npt.NDArray:
        """Compute the identity function, f(x) = x.

        Args:
            x (npt.NDArray): The input to the identity function,
                of shape (N x D_out).

        Returns:
            nptNDArray: The output equals the input x,
                of shape (N x D_out).

        """
        return x

    @staticmethod
    def relu_activation(x: npt.NDArray) -> npt.NDArray:
        """Compute the ReLU activation function, f(x) = max(x,0).

        The maximum is applied elementwise.

        Args:
            x (npt.NDArray): The input to the ReLU activation function,
                of shape (N x D_out).

        Returns:
            npt.NDArray: The output of ReLU, of shape (N x D_out).

        """
        return np.maximum(x, 0)

    @staticmethod
    def tanh_activation(x: npt.NDArray) -> npt.NDArray:
        """Compute the tanh function, f(x) = tanh(x).

        The tanh is applied elementwise to input x.

        Args:
            x (npt.NDArray): The input to the tanh function,
                of shape (N x D_out).

        Returns:
            npt.NDArray: The output of the tanh function,
                of shape (N x D_out).

        """
        return np.tanh(x)

    def __post_init__(self) -> None:  # noqa: D105
        self._classes = None
        # convert string activation function names to actual functions
        if not isinstance(self.activation, Callable):
            if self.activation == "none" or self.activation is None:
                self.activation = Base.identity_activation
            elif self.activation == "relu":
                self.activation = Base.relu_activation
            elif self.activation == "tanh":
                self.activation = Base.tanh_activation
            else:
                msg = f"Unknown activation {self.activation}."
                raise ValueError(msg)

    @abstractmethod
    def fit(self, x: npt.NDArray, y: npt.NDArray | None = None) -> Base:
        """Abstract method for fitting the layer.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values of shape (N,)
                or (N x D_target). Default is None.

        Returns:
            Base: Returns the class instance, i.e., self.

        """

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Use weights, biases, and activation function to transform x.

        More specifically, the output equals f(x @ W + b),
        where f is the activation function, W are the weights,
        and b are the biases.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: The output of the layer, of shape (N x D_out).

        """
        if self.layer_width is None:
            msg = "The fit method did not set the number of outputs, i.e. layer_width."
            raise ValueError(msg)

        x = self.prepare_x(x)
        return self.activation(x @ self.weights + self.biases)

    def fit_transform(
        self,
        x: npt.NDArray,
        y: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Fit the model and then transform the input x.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray, optional): The target values, of shape (N,)
                or (N x D_target). Default is None.

        Returns:
            npt.NDArray: The output of the layer, of shape (N x D_out).

        """
        self.fit(x, y)
        return self.transform(x, y)

    def predict(self, x: npt.NDArray) -> npt.NDArray:
        """Use weights, biases, and activation function to transform x.

        More specifically, the output equals f(x @ W + b),
        where f is the activation function, W are the weights,
        and b are the biases.

        Function required by scikit-learn, simply calls transform.

        Args:
            x (npt.NDArray): Input to the layer of shape (N x ...).
                The input is reshaped to (N x D_in).

        Returns:
            npt.NDArray: The output of the layer applied to x,
                of shape (N x D_out).

        """
        return self.transform(x)

    def prepare_x(self, x: npt.NDArray) -> npt.NDArray:
        """Prepare input x before transforming it.

        If input x is of shape (N, D_1, D_2, ...), it returns a 2D
        array with shape (N, D_in), where D_in = D_1 + D_2 + ...

        Args:
            x (npt.NDArray): Input of the shape (N x ...).
                The input is reshaped to (N x D_in).

        Returns:
            npt.NDArray: The flattened output of shape (N, D_in).

        """
        max_dim = 2
        if len(x.shape) > max_dim:
            x = x.reshape(x.shape[0], -1)
        return x

    def prepare_y(self, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Prepare targets for model fit.

        If target y is of shape (N,), it is reshaped to (N x D_target).
        If targets are classes, it is turned to one-hot encoding.

        Args:
            y (npt.NDArray): Targets of shape (N,) or (N x D_target).
                If targets are labels, the shape is (N,).

        Returns:
            npt.NDArray: Targets of shape (N x D_target). If input y
                are labels, the output are one-hot encoded matrix.
            npt.NDArray: If regression, function returns targets of
                shape (N x D_target) again. If classification, the
                function returns the class index of shape (N x 1).

        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if not self.is_classifier:
            return y, y

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        y_encoded_index = np.argmax(y == self._classes, axis=1)
        y_encoded_onehot = np.eye(n_classes)[y_encoded_index]
        return y_encoded_onehot, y_encoded_index.reshape(-1, 1)

    def prepare_y_inverse(self, y: npt.NDArray) -> npt.NDArray:
        """Reverse the actions taken by prepare_y.

        Maps from one-hot encoding to class labels for input y.
        If regression, y is returned.

        Args:
            y (npt.NDArray): Targets after prepare_y,
                of shape (N x D_target).

        Returns:
            npt.NDArray: Returns y if regression, and the corresponding
                labels if y is one-hot encoded targets.

        """
        if not self.is_classifier:
            return y

        probability_max = np.argmax(y, axis=1)
        return self._classes[probability_max].reshape(-1, 1)

    def clean_inputs(self, x: npt.NDArray, y: npt.NDArray | None = None) -> npt.NDArray:
        """Run prepare_x and prepare_y on x and y respectively.

        Args:
            x (npt.NDArray): Input of shape (N x ...).
                The input is reshaped to (N x D_in).
            y (npt.NDArray): Targets of shape (N,) or (N x D_target).
                If targets are labels, the shape is (N,).

        Returns:
            npt.NDArray: prepare_x(x) of shape (N x D_target).
            npt.NDArray: prepare_y(y) of shape (N x D_target).

        """
        x = self.prepare_x(x)
        if y is not None:
            y, _ = self.prepare_y(y)
        return x, y
