"""A module to implement sampled Fourier Neural Operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .dense import Dense
from .linear import Linear

if TYPE_CHECKING:
    from collections.abc import Callable


def crop_rfft_modes(x: npt.NDArray, ks: list[int]) -> npt.NDArray:
    """Crop the RFFT modes along the specified axes.

    Args:
        x (npt.NDArray): Input array (RFFT output).
        ks (list[int]): Sequence of crop sizes for each axis.

    Returns:
        npt.NDArray: Cropped array.

    """
    shift = x.ndim - len(ks)
    slices = [slice(k, -k + 1) for k in ks[:-1]] + [slice(ks[-1], None)]
    cropped = x.copy()
    for i, s in enumerate(slices):
        cropped = np.delete(cropped, s, axis=shift + i)
    return cropped


def pad_rfft_modes(x: npt.NDArray, target_lengths: list[int]) -> npt.NDArray:
    """Pad the RFFT modes to match the target lengths.

    Args:
        x (npt.NDArray): Input array (RFFT output).
        target_lengths (list[int]): Target shape for each axis.

    Returns:
        npt.NDArray: Padded array.

    """
    shift = x.ndim - len(target_lengths)
    indices = [(d + 1) // 2 for d in x.shape[shift:-1]] + [x.shape[-1]]
    last_dim = target_lengths[-1] // 2 + 1
    deltas = [
        t - d for t, d in zip(target_lengths[:-1], x.shape[shift:-1], strict=True)
    ] + [
        last_dim - x.shape[-1],
    ]
    padded = x.copy()
    for i, delta in enumerate(deltas):
        ax = shift + i
        padding_shape = np.ones(x.ndim, dtype=np.int64)
        padding_shape[0] = delta
        padding = np.zeros(padding_shape)
        padded = np.insert(padded, indices[i], padding, axis=ax)
    return padded


def rfft(signal: npt.NDArray, ks_max: list[int], norm: str = "backward") -> npt.NDArray:
    """Compute the real FFT along the last axes and crop modes.

    Args:
        signal (npt.NDArray): Input array.
        ks_max (list[int]): Maximum number of modes to keep for axes.
        norm (str, optional): Normalization mode for FFT.
            Default is "backward".

    Returns:
        npt.NDArray: Cropped RFFT output.

    """
    shift = signal.ndim - len(ks_max)
    transformed = np.fft.rfftn(signal, s=signal.shape[shift:], norm=norm)
    return crop_rfft_modes(transformed, ks_max)


def irfft(
    modes: npt.NDArray,
    target_lengths: list[int],
    norm: str = "backward",
) -> npt.NDArray:
    """Compute the inverse real FFT along the last axes.

    Args:
        modes (npt.NDArray): Input array (cropped RFFT output).
        target_lengths (list[int]): Target shape for each axis.
        norm (str, optional): Normalization mode for FFT.
            Default is "backward".

    Returns:
        npt.NDArray: Restored signal in the time domain.

    """
    padded = pad_rfft_modes(modes, target_lengths)
    return np.fft.irfftn(padded, s=target_lengths, norm=norm)


def _split_to_real(x: npt.NDArray) -> npt.NDArray:
    """Split a complex array into real and imaginary parts.

    Args:
        x (npt.NDArray): Complex input array with last dimension K.

    Returns:
        npt.NDArray: Real array with doubled last dimension 2*K.

    """
    return np.concatenate([x.real, x.imag], axis=-1)


def _merge_to_complex(x: npt.NDArray) -> npt.NDArray:
    """Merge real and imaginary parts from the two last axis.

    Args:
        x (npt.NDArray): Real array with last dimension 2*K.

    Returns:
        npt.NDArray: Complex array with last dimension K.

    """
    half = x.shape[-1] // 2
    real = x[..., :half]
    imag = x[..., half:]
    return real + 1j * imag


def _to_int_sequence(parameter: int | list[int]) -> list[int]:
    """Ensure the parameter is a list of ints.

    Args:
        parameter (int | list[int]): An int or a sequence of ints.

    Returns:
        Sequence of ints.

    """
    if isinstance(parameter, int):
        return [parameter]
    return parameter


@dataclass
class FFT(BaseEstimator):
    """RFFT transformer for input arrays.

    Attributes:
        ks_max (tuple[int | None], optional): Maximum number of Fourier
            modes to keep for each axis. If int is provided, it is
            converted to a list. Default is (None,).
        norm (str, optional): Normalization mode for FFT.
            Default is "backward".
        avoid_complex (bool, optional): If True, splits the output into
            real and imaginary parts to avoid complex numbers.
            Default is True.

    """

    ks_max: tuple[int | None] = (None,)
    norm: str = "backward"
    avoid_complex: bool = True

    def __post_init__(self) -> None:  # noqa: D105
        self.ks_max = _to_int_sequence(self.ks_max)

    def fit(self, _: npt.NDArray, __: npt.NDArray | None = None) -> FFT:  # noqa: D102
        return self

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Apply RFFT to the input.

        Args:
            x (npt.NDArray): Input array.
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: Transformed array.

        """
        transformed = rfft(x, self.ks_max, norm=self.norm)
        if self.avoid_complex:
            transformed = _split_to_real(transformed)
        return transformed


@dataclass
class IFFT(BaseEstimator):
    """Inverse RFFT transformer for input arrays.

    Attributes:
        target_lengths (tuple[int | None], optional): Target shape for
            each axis in the output. If int is provided, it is
            converted to a list. Default is (None,).
        norm (str, optional): Normalization mode for FFT.
            Default is "backward".
        avoid_complex (bool, optional): If True, merges real and
            imaginary parts before applying the inverse FFT.
            Default is True.
        real (bool, optional):
            If True, ensures the output is real-valued.
            Default is False.

    """

    target_lengths: tuple[int] = (None,)
    norm: str = "backward"
    avoid_complex: bool = True
    real: bool = False

    def __post_init__(self) -> None:  # noqa: D105
        self.target_lengths = _to_int_sequence(self.target_lengths)

    def fit(self, _: npt.NDArray, __: npt.NDArray | None = None) -> IFFT:  # noqa: D102
        return self

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Apply inverse RFFT, optionally merging real/imaginary parts.

        Args:
            x (npt.NDArray): Input array.
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: Restored array in the time domain.

        """
        if self.avoid_complex:
            x = _merge_to_complex(x)
        return irfft(x, self.target_lengths, norm=self.norm)


@dataclass
class InFourier(BaseEstimator):
    """Pipeline wrapper for learning in the Fourier domain.

    Attributes:
        pipeline (Pipeline): The scikit-learn pipeline to fit and
            transform in the Fourier domain.
        n_modes (tuple[int] | None, optional): Number of Fourier modes
            to keep for each axis. If None, all modes are used.
            Default is None.
        avoid_complex (bool, optional): If True, splits/merges real and
            imaginary parts to avoid complex numbers. Default is True.

    """

    pipeline: Pipeline
    n_modes: tuple[int] | None = None
    avoid_complex: bool = True
    _goal_shape: tuple[int] = None

    def __post_init__(self) -> None:  # noqa: D105
        self.n_modes = _to_int_sequence(self.n_modes)

    def fit(self, x: npt.NDArray, y: npt.NDArray) -> InFourier:
        """Fit the pipeline in the Fourier domain.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            y (npt.NDArray): Target array of shape (N x D_out).

        Returns:
            InFourier: self.

        """
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        fft_y = fft_transform.transform(y)
        self.pipeline.fit(fft_x, fft_y)
        self._goal_shape = x.shape[-len(self.n_modes) :]
        return self

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Transform input using the pipeline in the Fourier domain.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: Restored array in the time domain.

        """
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        prediction = self.pipeline.transform(fft_x)
        ifft_transform = IFFT(self._goal_shape, avoid_complex=self.avoid_complex)
        return ifft_transform.transform(prediction)


@dataclass
class Lifting(BaseEstimator):
    """Lifting transformer to increase input dimensionality.

    Attributes:
        n_hidden_channels (int):
            Number of hidden channels after lifting.
        random_seed (int): Seed for random number generation.
        grid_bounds (tuple[float, float], optional): Bounds for the
            space grid to be appended to the input. Default is (0, 1).
        grid (npt.NDArray | None, optional): The space grid appended to
            the input. If not provided, the grid is generated from
            grid_bounds. Default is None.

    """

    n_hidden_channels: int
    random_seed: int
    grid_bounds: tuple[float, float] = (0, 1)
    grid: npt.NDArray | None = None
    _weights: npt.NDArray | None = None
    _data_mean: npt.NDArray | None = None
    _data_std: npt.NDArray | None = None

    def _append_grid(self, x: npt.NDArray) -> npt.NDArray:
        """Append a grid to the input array.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).

        Returns:
            npt.NDArray: Array with grid appended as new features of
                shape (N x 2*D_in).

        """
        expanded_grid = np.repeat(self.grid.reshape(1, -1), len(x), axis=0)
        return np.stack([x, expanded_grid], axis=-1)

    def fit(self, x: npt.NDArray, _: npt.NDArray | None = None) -> Lifting:
        """Fit the lifting layer.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            Lifting: self

        """
        if self.grid is None:
            self.grid = np.linspace(*self.grid_bounds, x.shape[-1])

        rng = np.random.default_rng(self.random_seed)
        stacked_x = self._append_grid(x)
        stacked_x = stacked_x.reshape(-1, stacked_x.shape[-1])
        self._data_mean = np.mean(stacked_x, axis=0, keepdims=True)
        self._data_std = np.std(stacked_x, axis=0, keepdims=True)
        self._weights = rng.uniform(
            low=-1,
            high=1,
            size=(stacked_x.shape[-1], self.n_hidden_channels),
        )
        return self

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        """Apply the lifting transformation to the input.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).

        Returns:
            npt.NDArray: Lifted array of shape (N x n_hidden_channels).

        """
        stacked_x = self._append_grid(x)
        normalized_x = (stacked_x - self._data_mean) / self._data_std
        lifted = normalized_x @ self._weights
        return lifted.transpose(0, 2, 1)


@dataclass
class FourierBlock(Dense, Linear):
    """A block consisting of Dense and Linear layers.

    Attributes:
        n_hidden_channels (int | None, optional): Number of hidden
            channels. Default is None.
        n_modes (list[int] | None, optional): Number of Fourier modes
            to keep for each axis. If None, all modes are used.
            Default is None.
        avoid_complex (bool | None, optional): If True, splits/merges
            real and imaginary parts to avoid complex numbers.
            Default is True.

    """

    n_hidden_channels: int | None = None
    n_modes: list[int] | None = None
    avoid_complex: bool = True
    _block_pipelines: list[Pipeline] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        rng = np.random.default_rng(self.random_seed)
        self._build_block_pipelines(rng)

    def _build_block_pipelines(self, rng: np.random.Generator) -> None:
        """Build pipelines for each hidden channel.

        Args:
            rng (np.random.Generator): Random number generator.

        """
        block_pipelines = []
        for _ in range(self.n_hidden_channels):
            random_seed = rng.integers(np.iinfo(np.int64).max)
            steps = [
                (
                    "dense",
                    Dense(
                        layer_width=self.layer_width,
                        activation=self.activation,
                        parameter_sampler=self.parameter_sampler,
                        random_seed=random_seed,
                    ),
                ),
                ("linear", Linear(regularization_scale=self.regularization_scale)),
            ]
            block_pipelines.append(Pipeline(steps))
        self._block_pipelines = block_pipelines

    def fit(self, x: npt.NDArray, y: npt.NDArray) -> FourierBlock:
        """Fit each block pipeline in the Fourier domain.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            y (npt.NDArray): Target array of shape (N x D_in).

        Returns:
            FourierBlock: self.

        """
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        fft_x = fft_transform.transform(x)
        fft_residual = fft_transform.transform(y - x)
        for channel, pipeline in enumerate(self._block_pipelines):
            pipeline.fit(fft_x[:, channel], fft_residual[:, channel])
        return self

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Apply Fourier blocks to the input data.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: Output array of shape (N x D_in).

        """
        fft_transform = FFT(self.n_modes, avoid_complex=self.avoid_complex)
        goal_shape = x.shape[-self.n_modes :]
        fft_x = fft_transform.transform(x)
        for channel, pipeline in enumerate(self._block_pipelines):
            fft_x[:, channel] = pipeline.transform(fft_x[:, channel])
        ifft_transform = IFFT(goal_shape, avoid_complex=self.avoid_complex)
        restored = ifft_transform.transform(fft_x)
        return restored + x


@dataclass
class FNO1D(BaseEstimator):
    """Sampled 1D Fourier Neural Operator.

    Either fourier_pipeline, lifting_pipeline, and projection_pipeline
    must be provided, or the other attributes must be set to initialize
    the pipelines.

    Attributes:
        n_blocks (int | None, optional): Number of Fourier blocks to
            apply. Default is None.
        layer_width (int | None, optional): Width of each dense layer
            in the Fourier blocks. Default is None.
        n_hidden_channels (int | None, optional): Number of hidden
            channels for the lifting. Default is None.
        n_modes (int | list[int] | None, optional): Number of Fourier
            modes to keep for each axis. If None, all modes are used.
            Default is None.
        random_seed (int, optional): Seed for random number generation.
            Default is 1.
        activation (Callable[[npt.NDArray], npt.NDArray] | str,
            optional): Activation function. Default is "none".
        parameter_sampler (Callable | str, optional):
            Sampler for layer parameters. Default is "relu".
        regularization_scale (float, optional): Regularization scale.
            Default is 1e-8.
        avoid_complex (bool, optional): If True, splits/merges real and
            imaginary parts to avoid complex numbers. Default is True.
        lifting_pipeline (Pipeline | None, optional): Pipeline for
            lifting. If None, it is created during initialization.
            Default is None.
        fourier_pipeline (Pipeline | None, optional): Pipeline for
            Fourier blocks. If None, it is created during
            initialization. Default is None.
        projection_pipeline (Pipeline | None, optional): Projection
            pipeline. If None, it is created during initialization.
            Default is None.

    """

    n_blocks: int | None = None
    layer_width: int | None = None
    n_hidden_channels: int | None = None
    n_modes: int | list[int] | None = None
    random_seed: int = 1
    activation: Callable[[npt.NDArray], npt.NDArray] | str = "none"
    parameter_sampler: Callable | str = "relu"
    regularization_scale: float = 1e-8
    avoid_complex: bool = True
    lifting_pipeline: Pipeline | None = None
    fourier_pipeline: Pipeline | None = None
    projection_pipeline: Pipeline | None = None

    def __post_init__(self) -> None:
        """Initialize the FNO1D model components.

        If self.lifting_pipeline, self.fourier_pipeline,
        or self.projection_pipeline are not specified, they are
        created based on the provided attributes.
        """
        rng = np.random.default_rng(self.random_seed)

        if self.lifting_pipeline is None:
            random_seed = rng.integers(np.iinfo(np.int64).max)
            self.lifting_pipeline = Lifting(
                self.n_hidden_channels,
                random_seed=random_seed,
            )
        if self.fourier_pipeline is None:
            fourier_steps = []
            for block_id in range(self.n_blocks):
                random_seed = rng.integers(np.iinfo(np.int64).max)
                block = FourierBlock(
                    n_hidden_channels=self.n_hidden_channels,
                    n_modes=self.n_modes,
                    layer_width=self.layer_width,
                    activation=self.activation,
                    parameter_sampler=self.parameter_sampler,
                    random_seed=random_seed,
                    avoid_complex=self.avoid_complex,
                )
                fourier_steps.append((f"fourier{block_id}", block))
            self.fourier_pipeline = Pipeline(fourier_steps)

        if self.projection_pipeline is None:
            random_seed = rng.integers(np.iinfo(np.int64).max)
            steps = [
                (
                    "dense",
                    Dense(
                        layer_width=self.layer_width,
                        activation=self.activation,
                        parameter_sampler=self.parameter_sampler,
                        random_seed=random_seed,
                    ),
                ),
                ("linear", Linear(regularization_scale=self.regularization_scale)),
            ]
            self.projection_pipeline = Pipeline(steps)

    def fit(self, x: npt.NDArray, y: npt.NDArray) -> FNO1D:
        """Fit the FNO model.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).
            y (npt.NDArray): Target array of shape (N x D_in).

        Returns:
            FNO1D: self.

        """
        self.lifting_pipeline.fit(x)
        lifted_x = self.lifting_pipeline.transform(x)
        lifted_y = self.lifting_pipeline.transform(y)
        if self.n_blocks > 0:
            self.fourier_pipeline.fit(lifted_x, lifted_y)
            lifted_x = self.fourier_pipeline.transform(lifted_x)
        lifted_x = lifted_x.reshape(lifted_x.shape[0], -1)
        self.projection_pipeline.fit(lifted_x, y)
        return self

    def transform(self, x: npt.NDArray) -> npt.NDArray:
        """Apply the FNO model to the input.

        Args:
            x (npt.NDArray): Input array of shape (N x D_in).

        Returns:
            npt.NDArray: Output array of shape (N x D_in).

        """
        lifted_x = self.lifting_pipeline.transform(x)
        if self.n_blocks > 0:
            lifted_x = self.fourier_pipeline.transform(lifted_x)
        lifted_x = lifted_x.reshape(lifted_x.shape[0], -1)
        return self.projection_pipeline.transform(lifted_x)
