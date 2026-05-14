import numpy as np


class Sigmoid:
    """
    Sigmoid activation function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter for the sigmoid. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return 1 / (1 + np.exp(-self.sigma * x))

class Gaussian:
    """
    Gaussian activation function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter controlling Gaussian width. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-((self.sigma * x) ** 2))

class ReLU:
    """
    ReLU activation function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Scaling parameter applied before ReLU. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return np.maximum(0, self.sigma * x)

class Tanh:
    """
    Tanh activation function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Smoothing/slope parameter for tanh. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return np.tanh(self.sigma * x)

class Softplus:
    """
    Softplus activation function with smoothing parameter corresponding to the slope or bandwidth of the basis functions.
    It is not learned, it is a hyperparameter.

    Args:
        sigma (float): Scaling parameter applied inside softplus. Default: 1.
        x (array_like): Input value(s) passed to the callable.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, x):
        return np.log(1 + np.exp(self.sigma * x))
    
class Identity:
    """
    Identity activation function that returns the input unchanged.

    Args:
        x (array_like): Input value(s) passed to the callable.
    """
    def __call__(self, x):
        return x