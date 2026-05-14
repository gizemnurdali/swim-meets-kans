"""A module to implement sampled Fourier Neural Operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


@dataclass
class DeepONetPOD(BaseEstimator):
    """Sampled DeepONet-POD model.

    This class implements a sampled Deep Operator Network (DeepONet)
    with POD components for the trunk network.

    Attributes:
        pipeline (Pipeline): The scikit-learn pipeline for
            the branch network.
        n_modes (int, optional): Number of POD modes to retain for
            dimensionality reduction. If None, all modes are used.
            Default is None.

    """

    pipeline: Pipeline
    n_modes: int = None

    def __post_init__(self) -> None:  # noqa: D105
        self._pod_modes = None
        self._pod_mean = None

    def fit(self, x: npt.NDArray, y: npt.NDArray) -> DeepONetPOD:
        """Fit the DeepONet-POD model.

        Args:
            x (npt.NDArray): Input data of shape (N x D_in).
            y (npt.NDArray): Target data of shape (N x D_out).

        Returns:
            DeepONetPOD: self.

        """
        self._set_pod(y)
        pod_y = self._apply_pod(y)
        self.pipeline.fit(x, pod_y)
        return self

    def transform(self, x: npt.NDArray, _: npt.NDArray | None = None) -> npt.NDArray:
        """Transform input data using the trained pipeline.

        Args:
            x (npt.NDArray): Input data for transformation.
            _ (npt.NDArray, optional): Ignored argument.

        Returns:
            npt.NDArray: Transformed output data.

        """
        prediction = self.pipeline.transform(x)
        return self._restore_output(prediction)

    def _set_pod(self, y: npt.NDArray) -> None:
        """Compute and store the POD.

        Args:
            y (npt.NDArray): Target data to compute POD.

        """
        mean = y.mean(axis=0)
        shifted = y - mean
        _, _, vh = np.linalg.svd(shifted)
        self._pod_mean = mean
        self._pod_modes = vh.T[:, : self.n_modes]

    def _apply_pod(self, y: npt.NDArray) -> npt.NDArray:
        """Project output data onto the POD modes.

        Args:
            y (npt.NDArray): Target data to project.

        Returns:
            npt.NDArray: POD-reduced output data.

        """
        return (y - self._pod_mean) @ self._pod_modes

    def _restore_output(self, pod_y: npt.NDArray) -> npt.NDArray:
        """Restore the POD-reduced output to the original space.

        Args:
            pod_y (npt.NDArray): POD-reduced output data.

        Returns:
            npt.NDArray: Output data in the original space.

        """
        return pod_y @ self._pod_modes.T + self._pod_mean
