"""Basis functions."""


import torch


class Sigmoid:
    """
    Sigmoid basis function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter for the sigmoid. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-self.sigma * x))

class Gaussian:
    """
    Gaussian basis function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter controlling Gaussian width. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((self.sigma * x) ** 2))

class ReLU:
    """
    ReLU basis function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Scaling parameter applied before ReLU. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.zeros_like(x), self.sigma * x)

class Tanh:
    """
    Tanh basis function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter for tanh. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.sigma * x)

class Softplus:
    """
    Softplus basis function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Scaling parameter applied inside softplus. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(self.sigma * x))
    
class Identity:
    """
    Identity basis function that returns the input unchanged.

    Args:
        x (array_like): Input value(s) passed to the callable.
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x